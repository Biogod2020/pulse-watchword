"""The heartbeat: a cheap read on every event decides which capabilities wake up.

Each event (a transcript segment, a revision, a camera frame, a timer) produces one tick:
1. the event is written into working memory;
2. one DecisionProvider call answers every capability's trigger question over the same context;
3. capabilities whose trigger fires run concurrently (agents, vision, writers);
4. deterministic rules run last as a backstop, independent of the model's intuition.
Every tick is traced, so the UI can show what was asked, what fired and why.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from .memory import Evidence, WorkingMemory
from .providers import Decision, DecisionProvider, Question


@dataclass
class Event:
    kind: str                   # speech | vision | timer
    t: float
    seg_id: str = ""
    rev: int = 1
    speaker: str = ""
    text: str = ""
    data: dict = field(default_factory=dict)


@dataclass
class TickContext:
    memory: WorkingMemory
    event: Event

    @property
    def evidence(self) -> Evidence:
        if self.event.kind == "speech":
            return Evidence("speech", self.event.seg_id, self.event.rev, self.event.text)
        return Evidence(self.event.kind, self.event.data.get("frame", self.event.kind), 1)


@dataclass
class Capability:
    name: str
    triggers: list[Question]    # the capability wakes if any trigger fires
    run: Callable[[TickContext], Awaitable[dict]]
    threshold: float = 0.5      # minimum P(yes) when the provider is calibrated
    on: tuple[str, ...] = ("speech",)
    kind: str = "agent"         # agent | vision | state — shown in the UI


@dataclass
class Alert:
    id: str
    severity: str
    text: str
    evidence: list[str]


@dataclass
class Rule:
    name: str
    check: Callable[[WorkingMemory, Event], list[Alert]]


class Heartbeat:
    def __init__(self, decider: DecisionProvider, memory: WorkingMemory, capabilities: list[Capability],
                 rules: list[Rule], context: Callable[[WorkingMemory, Event, dict[str, Alert]], str]):
        self.decider = decider
        self.memory = memory
        self.capabilities = capabilities
        self.rules = rules
        self.context = context
        self.alerts: dict[str, Alert] = {}
        self.trace: list[dict] = []

    @staticmethod
    def fires(cap: Capability, q: Question, d: Decision | None) -> tuple[bool, float | None]:
        if d is None:
            return False, None
        if d.probs:
            p = d.probs.get(q.options[0], 0.0)
            return p >= cap.threshold, p
        return d.choice == q.options[0], None

    def _apply(self, ev: Event) -> list[str]:
        if ev.kind == "speech":
            return self.memory.add_segment(ev.seg_id, ev.rev, ev.speaker, ev.text, ev.t)
        if ev.kind == "vision":
            self.memory.put("camera.latest_frame", ev.data, [Evidence("vision", ev.data.get("frame", ""))],
                            "observed", ev.t)
        return []

    async def on_event(self, ev: Event) -> dict:
        t0 = time.perf_counter()
        invalidated = self._apply(ev)
        eligible = [c for c in self.capabilities if ev.kind in c.on]
        decisions: dict[str, Decision] = {}
        read_ms, provider, calibrated, error = 0.0, self.decider.name, None, None
        if eligible:
            try:
                batch = await self.decider.decide(self.context(self.memory, ev, self.alerts),
                                                  [q for c in eligible for q in c.triggers])
                decisions, read_ms, calibrated = batch.decisions, batch.latency_ms, batch.calibrated
            except Exception as e:  # the backstop rules still run when the intuition layer is down
                error = f"{type(e).__name__}: {e}"
        asked, fired = [], []
        for c in eligible:
            woke = False
            for q in c.triggers:
                go, p = self.fires(c, q, decisions.get(q.id))
                d = decisions.get(q.id)
                asked.append({"capability": c.name, "question": q.id, "choice": d.choice if d else None,
                              "p_yes": p, "fired": go})
                woke = woke or go
            if woke:
                fired.append(c)
        ctx = TickContext(self.memory, ev)
        outs = await asyncio.gather(*[c.run(ctx) for c in fired], return_exceptions=True)
        results = {c.name: (o if not isinstance(o, BaseException) else {"error": f"{type(o).__name__}: {o}"})
                   for c, o in zip(fired, outs)}
        raised, cleared = self._run_rules(ev)
        tick = {"t": ev.t, "event": {"kind": ev.kind, "seg_id": ev.seg_id, "rev": ev.rev, "speaker": ev.speaker,
                                     "text": ev.text, "data": ev.data},
                "invalidated": invalidated, "provider": provider, "calibrated": calibrated, "read_ms": read_ms,
                "decider_error": error, "asked": asked, "results": results,
                "alerts_raised": raised, "alerts_cleared": cleared,
                "active_alerts": [a.__dict__ for a in self.alerts.values()],
                "tick_ms": (time.perf_counter() - t0) * 1e3, "memory_version": self.memory.version}
        self.trace.append(tick)
        return tick

    def _run_rules(self, ev: Event) -> tuple[list[dict], list[str]]:
        current: dict[str, Alert] = {}
        for rule in self.rules:
            for a in rule.check(self.memory, ev):
                current[a.id] = a
        raised = [a.__dict__ for k, a in current.items() if k not in self.alerts or self.alerts[k].text != a.text]
        cleared = [k for k in self.alerts if k not in current]
        self.alerts = current
        return raised, cleared
