"""Operating-room sponge count: a capability pack for the heartbeat.

This is record keeping and process checking (the count every OR team already performs), not
clinical advice: the assistant keeps the ledger, cross-checks the spoken call-outs against what
the camera sees in the counter bag, and flags an unreconciled count before closure.
"""
from __future__ import annotations

import re
from typing import Protocol

from .heartbeat import Alert, Capability, Event, Rule, TickContext
from .memory import Evidence, WorkingMemory
from .providers import AgentProvider, Question, Tool


class VisionCounter(Protocol):
    name: str

    async def count(self, frame: dict) -> dict: ...   # {"count": int, "frame": str, "detail": ...}


class FixtureVision:
    """Stand-in for SAM 3.1: returns the labelled count stored with each scenario frame."""
    name = "fixture"

    async def count(self, frame: dict) -> dict:
        return {"count": int(frame["sponges_in_bag"]), "frame": frame.get("frame", ""), "detail": "fixture label"}


# Question design follows what worked on djev (see eval_triggers.py and runs/trigger_eval*.json):
# native yes/no reads, one fixed prefix that pins the subject to the utterance being judged, narrow
# questions with explicit exclusions, and per-question thresholds. A broad "is this about sponges?"
# gate was the weakest question in every design, so the pack asks its specific questions directly.
PREFIX = "Does this utterance itself "


def _q(qid: str, text: str, threshold: float | None = None) -> Question:
    return Question(qid, PREFIX + text, kind="noul", threshold=threshold)


TRIGGERS = {
    "added": _q("added", "report new sponges being opened, handed onto the field or added to the count, including "
                         "an initial count? Not using or asking for a sponge already on the field, and not a "
                         "restated total."),
    "out": _q("out", "report used sponges being taken off the field into the counter bag? Not sponges packed or "
                     "left inside the patient, and not a restated total."),
    "verify": _q("verify", "ask, as a question or request, for the sponge count to be checked or reported? Not a "
                           "statement that the count is correct."),
    "closing": _q("closing", "say the surgical team is starting to close the incision (fascia, wound or skin "
                             "closure)? Not closing a door, and not 'close' meaning near."),
    "assistant": _q("assistant", "directly address the assistant, asking it for information or to record "
                                 "something? Not a question or request to another team member.", threshold=0.8),
    "recount": Question("recount", "Do the LINES SINCE THE LAST CAMERA COUNT report used sponges going into the "
                                   "counter bag?", kind="noul", threshold=0.3),  # a missed recount costs more than a spare one
}


def ledger(m: WorkingMemory) -> dict:
    events = m.facts_with_prefix("count.event.")
    added = sum(f.value["quantity"] for f in events if f.value["direction"] == "added")
    removed = sum(f.value["quantity"] for f in events if f.value["direction"] == "removed")
    return {"added": added, "called_out": removed, "on_field_by_callout": added - removed,
            "counter_bag_camera": m.get("count.vision.bag"), "phase": m.get("phase", "operating"),
            "events": [{"key": f.key, **f.value, "from": f.evidence[0].ref, "rev": f.evidence[0].rev}
                       for f in events]}


def context(m: WorkingMemory, ev: Event, alerts: dict[str, Alert]) -> str:
    led = ledger(m)
    bag = led["counter_bag_camera"]
    lines = [f"PHASE: {led['phase']}",
             f"SPONGE LEDGER: added {led['added']}, called out {led['called_out']}, counter bag seen by camera: "
             f"{bag if bag is not None else 'not yet counted'}",
             "OPEN ALERTS: " + ("; ".join(a.text for a in alerts.values()) or "none")]
    if ev.kind == "speech":
        earlier = [s for s in m.recent(7) if s.seg_id != ev.seg_id][-6:]
        lines += ["", "EARLIER TRANSCRIPT (context only):"]
        lines += [f"{s.speaker}: {s.text}" for s in earlier] or ["(none)"]
        lines += ["", f"THIS UTTERANCE: {ev.speaker}: {ev.text}"]
    else:
        seen = m.facts.get("count.vision.bag")
        since = [m.segments[x] for x in m.order if seen is None or m.segments[x].t > seen.t]
        lines += ["", "LINES SINCE THE LAST CAMERA COUNT:"]
        lines += [f"{s.speaker}: {s.text}" for s in since[-6:]] or ["(none)"]
        lines += ["", f"THIS EVENT: new camera frame {ev.data.get('frame', '')} of the counter bag (no speech)"]
    return "\n".join(lines)


COUNT_SYSTEM = (
    "You keep the operating-room sponge ledger. Record only what the given line explicitly states, with "
    "record_count_event once per statement. 'added' = new sponges opened onto the field. 'removed' = sponges "
    "taken out and placed in the counter bag. Sponges packed or left inside the patient are NOT removed. "
    "A line that only restates a total is not an event. If nothing should be recorded, reply 'none'.")


def pack(agent: AgentProvider, vision: VisionCounter) -> tuple[list[Capability], list[Rule]]:
    async def extract_count(ctx: TickContext) -> dict:
        ev, committed = ctx.event, []
        evidence = ctx.evidence

        async def record(args: dict) -> dict:
            direction = args.get("direction")
            quantity = args.get("quantity")
            if direction not in ("added", "removed") or not isinstance(quantity, int) or quantity <= 0:
                return {"ok": False, "error": "direction must be added|removed and quantity a positive integer"}
            key = f"count.event.{ev.seg_id}.{len(committed)}"
            ok = ctx.memory.put(key, {"direction": direction, "quantity": quantity, "item": "sponge"},
                                [evidence], "proposed", ev.t)
            if ok:
                committed.append(key)
            return {"ok": ok, "key": key, "stale": not ok}

        tool = Tool("record_count_event", "Record one sponge count event stated in the line.",
                    {"type": "object", "properties": {
                        "direction": {"type": "string", "enum": ["added", "removed"]},
                        "quantity": {"type": "integer", "minimum": 1}},
                     "required": ["direction", "quantity"]}, record)
        run = await agent.run(COUNT_SYSTEM, f"{ev.speaker}: {ev.text}", [tool], stop_after_tools=True)
        return {"agent": agent.name, "committed": committed, "agent_ms": run.latency_ms, "steps": run.steps,
                "calls": run.tool_calls}

    async def vision_recount(ctx: TickContext) -> dict:
        frame = ctx.memory.get("camera.latest_frame")
        if not frame:
            return {"skipped": "no frame yet"}
        counted = ctx.memory.facts.get("count.vision.bag")
        if counted and counted.evidence[0].ref == frame.get("frame"):
            return {"skipped": f"frame {frame.get('frame')} already counted"}
        out = await vision.count(frame)
        ctx.memory.put("count.vision.bag", out["count"], [Evidence("vision", out["frame"])], "observed", ctx.event.t)
        return {"vision": vision.name, **out}

    async def set_closing(ctx: TickContext) -> dict:
        ok = ctx.memory.put("phase", "closing", [ctx.evidence], "proposed", ctx.event.t)
        return {"phase": "closing" if ok else "stale"}

    async def answer_query(ctx: TickContext) -> dict:
        async def read_ledger(_: dict) -> dict:
            led = ledger(ctx.memory)
            led.pop("events")
            return led

        async def search(args: dict) -> dict:
            return {"lines": [{"seg": s.seg_id, "speaker": s.speaker, "text": s.text}
                              for s in ctx.memory.search(str(args.get("query", "")))]}

        tools = [Tool("read_count_ledger", "Current sponge ledger and phase.", {"type": "object", "properties": {}},
                      read_ledger),
                 Tool("search_transcript", "Find earlier transcript lines by keywords.",
                      {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
                      search)]
        run = await agent.run("You answer the OR team's questions about the sponge count and the case record. "
                              "Use the tools; cite transcript segment ids; do not give clinical advice. "
                              "Two sentences at most.", f"{ctx.event.speaker}: {ctx.event.text}", tools)
        ctx.memory.put(f"answer.{ctx.event.seg_id}", run.text, [ctx.evidence], "proposed", ctx.event.t)
        return {"agent": agent.name, "answer": run.text, "agent_ms": run.latency_ms, "calls": run.tool_calls}

    caps = [
        Capability("extract_count", [TRIGGERS["added"], TRIGGERS["out"]], extract_count, kind="agent"),
        Capability("vision_recount", [TRIGGERS["recount"]], vision_recount, on=("vision",), kind="vision"),
        Capability("verify_recount", [TRIGGERS["verify"]], vision_recount, kind="vision"),
        Capability("set_closing", [TRIGGERS["closing"]], set_closing, kind="state"),
        Capability("answer_query", [TRIGGERS["assistant"]], answer_query, kind="agent"),
    ]

    closing_words = re.compile(r"\b(clos(e|ing)|fascia|skin staples?)\b", re.I)

    def closure_keyword(m: WorkingMemory, ev: Event) -> list[Alert]:
        # Deterministic backstop: closure is detected even if the intuition read misses it.
        if ev.kind == "speech" and closing_words.search(ev.text) and m.get("phase") != "closing":
            m.put("phase", "closing", [Evidence("speech", ev.seg_id, ev.rev, ev.text)], "derived", ev.t)
        return []

    def reconcile(m: WorkingMemory, ev: Event) -> list[Alert]:
        led = ledger(m)
        alerts = []
        bag = led["counter_bag_camera"]
        seen = m.facts.get("count.vision.bag")
        last_out = max((f.t for f in m.facts_with_prefix("count.event.") if f.value["direction"] == "removed"),
                       default=None)
        fresh = seen is not None and (last_out is None or seen.t >= last_out)
        if fresh and bag != led["called_out"]:
            alerts.append(Alert("callout_vs_camera", "warning",
                                f"Call-outs say {led['called_out']} sponges in the counter bag; camera counts {bag}.",
                                [e["from"] for e in led["events"] if e["direction"] == "removed"]))
        if led["phase"] == "closing" and (led["on_field_by_callout"] != 0 or alerts):
            alerts.append(Alert("unreconciled_at_closure", "high",
                                f"Closing with count not reconciled: {led['added']} added, {led['called_out']} called "
                                f"out, camera sees {bag if bag is not None else 'no recount'} in the bag.",
                                [e["from"] for e in led["events"]]))
        return alerts

    return caps, [Rule("closure_keyword", closure_keyword), Rule("reconcile", reconcile)]
