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
from .providers import Decision, DecisionProvider, Question, image_url


@dataclass
class Event:
    kind: str                   # speech | vision | clip | timer
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
    dynamic: Callable[[WorkingMemory], list[Question]] | None = None   # extra triggers built from memory


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
                 rules: list[Rule], context: Callable[[WorkingMemory, Event, dict[str, Alert]], str],
                 perceive: Callable[[Event], Awaitable[None]] | None = None,
                 remember: Callable[[Event], Awaitable[None]] | None = None, background_remember: bool = False):
        """perceive: enrich a raw event before the read (e.g. caption a frame).
        remember: write the event to long-term memory after the tick."""
        self.decider = decider
        self.memory = memory
        self.capabilities = capabilities
        self.rules = rules
        self.context = context
        self.perceive = perceive
        self.remember = remember
        self.background_remember = background_remember
        self._bg: set[asyncio.Task] = set()
        self.remember_errors: list[str] = []
        self.alerts: dict[str, Alert] = {}
        self.trace: list[dict] = []
        self.pending: list[tuple[Capability, Event, int]] = []   # failed runs waiting for a retry
        self.max_attempts = 3

    @staticmethod
    def fires(cap: Capability, q: Question, d: Decision | None) -> tuple[bool, float | None]:
        if d is None:
            return False, None
        if d.probs:
            p = d.probs.get(q.options[0], 0.0)
            return p >= (q.threshold if q.threshold is not None else cap.threshold), p
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
        perceive_ms = 0.0
        if self.perceive:
            tp = time.perf_counter()
            await self.perceive(ev)
            perceive_ms = (time.perf_counter() - tp) * 1e3
        invalidated = self._apply(ev)
        eligible = [c for c in self.capabilities if ev.kind in c.on]
        triggers = {c.name: c.triggers + (c.dynamic(self.memory) if c.dynamic else []) for c in eligible}
        decisions: dict[str, Decision] = {}
        read_ms, provider, calibrated, error = 0.0, self.decider.name, None, None
        if eligible:
            try:
                images = [image_url(ev.data["image"])] if ev.kind == "vision" and ev.data.get("image") else None
                videos = [ev.data["video_url"]] if ev.kind == "clip" and ev.data.get("video_url") else None
                batch = await self.decider.decide(self.context(self.memory, ev, self.alerts),
                                                  [q for c in eligible for q in triggers[c.name]], images, videos)
                decisions, read_ms, calibrated = batch.decisions, batch.latency_ms, batch.calibrated
            except Exception as e:  # the backstop rules still run when the intuition layer is down
                error = f"{type(e).__name__}: {e}"
        asked, fired = [], []
        for c in eligible:
            woke = False
            for q in triggers[c.name]:
                go, p = self.fires(c, q, decisions.get(q.id))
                d = decisions.get(q.id)
                asked.append({"capability": c.name, "question": q.id, "choice": d.choice if d else None,
                              "p_yes": p, "fired": go})
                woke = woke or go
            if woke:
                fired.append(c)
        jobs = [(c, ev, 1) for c in fired] + self._due_retries()
        outs = await asyncio.gather(*[c.run(TickContext(self.memory, e)) for c, e, _ in jobs], return_exceptions=True)
        results = {}
        for (c, e, attempt), o in zip(jobs, outs):
            name = c.name if e is ev else f"{c.name} (retry {attempt - 1} for {e.seg_id or e.kind})"
            failed = isinstance(o, BaseException) or (isinstance(o, dict) and "error" in o)
            results[name] = {"error": f"{type(o).__name__}: {o}"} if isinstance(o, BaseException) else o
            self._track(c, e, attempt, failed)
        raised, cleared = self._run_rules(ev)
        remember_ms, remember_error = 0.0, None
        if self.remember and self.background_remember:
            task = asyncio.create_task(self._remember_bg(ev))
            self._bg.add(task)
            task.add_done_callback(self._bg.discard)
        elif self.remember:
            tr = time.perf_counter()
            try:
                await self.remember(ev)
            except Exception as e:  # long-term memory is best effort; the tick's decisions still stand
                remember_error = f"{type(e).__name__}: {e}"
            remember_ms = (time.perf_counter() - tr) * 1e3
        tick = {"t": ev.t, "event": {"kind": ev.kind, "seg_id": ev.seg_id, "rev": ev.rev, "speaker": ev.speaker,
                                     "text": ev.text, "data": ev.data},
                "perceive_ms": perceive_ms, "remember_ms": remember_ms, "remember_error": remember_error,
                "invalidated": invalidated, "provider": provider, "calibrated": calibrated, "read_ms": read_ms,
                "decider_error": error, "asked": asked, "results": results,
                "alerts_raised": raised, "alerts_cleared": cleared,
                "active_alerts": [a.__dict__ for a in self.alerts.values()],
                "tick_ms": (time.perf_counter() - t0) * 1e3, "memory_version": self.memory.version}
        self.trace.append(tick)
        return tick

    async def _remember_bg(self, ev: Event) -> None:
        try:
            await self.remember(ev)
        except Exception as e:
            self.remember_errors.append(f"{ev.seg_id or ev.data.get('frame')}: {type(e).__name__}: {e}")

    async def drain(self) -> None:
        """Wait for background long-term memory writes."""
        while self._bg:
            await asyncio.gather(*list(self._bg), return_exceptions=True)

    def _due_retries(self) -> list[tuple[Capability, Event, int]]:
        """Re-run failed capabilities whose input is still current; a revised segment drops its old retry."""
        due, self.pending = self.pending, []
        return [(c, e, n + 1) for c, e, n in due
                if e.kind != "speech" or self.memory.is_current(Evidence("speech", e.seg_id, e.rev))]

    def _track(self, cap: Capability, ev: Event, attempt: int, failed: bool) -> None:
        key = f"pending.{cap.name}.{ev.seg_id or ev.kind}"
        if not failed:
            self.memory.facts.pop(key, None)
            return
        evidence = [Evidence("speech", ev.seg_id, ev.rev, ev.text)] if ev.kind == "speech" else []
        self.memory.put(key, {"capability": cap.name, "attempts": attempt, "text": ev.text,
                              "gave_up": attempt >= self.max_attempts}, evidence, "derived", ev.t)
        if attempt < self.max_attempts:
            self.pending.append((cap, ev, attempt))

    def _run_rules(self, ev: Event) -> tuple[list[dict], list[str]]:
        current: dict[str, Alert] = {}
        for rule in self.rules:
            for a in rule.check(self.memory, ev):
                current[a.id] = a
        raised = [a.__dict__ for k, a in current.items() if k not in self.alerts or self.alerts[k].text != a.text]
        cleared = [k for k in self.alerts if k not in current]
        self.alerts = current
        return raised, cleared
