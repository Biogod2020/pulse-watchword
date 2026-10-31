import asyncio
import json
import re
from pathlib import Path

from pulse.heartbeat import Event, Heartbeat
from pulse.memory import Evidence, WorkingMemory
from pulse.providers import AgentRun, ScriptedDecider
from pulse.surgical_count import FixtureVision, context, ledger, pack

SCENARIO = json.loads((Path(__file__).parent.parent / "scenarios/or_sponge_count.json").read_text())
WORDS = {"two": 2, "three": 3, "four": 4, "ten": 10}


def newest(ctx: str) -> str:
    m = re.search(r"THIS (UTTERANCE|EVENT): (.*)", ctx)
    return m.group(2).lower() if m else ""


def intuition(ctx: str, q) -> dict:
    line = newest(ctx)
    yes = {"added": any(w in line for w in ("count, ten", "opening")),
           "out": " out" in line, "recount": "frame" in line, "verify": "verify" in line,
           "closing": "closing" in line, "assistant": "assistant" in line}[q.id]
    return {"yes": 0.9, "no": 0.1} if yes else {"yes": 0.05, "no": 0.95}


class RuleAgent:
    """Offline stand-in for the slow agent: parses the line and calls the tool like a model would."""
    name = "rule-agent"

    async def run(self, system, user, tools, max_steps=4, stop_after_tools=False):
        calls = []
        by = {t.name: t for t in tools}
        if "record_count_event" in by:
            text = user.lower()
            n = next((v for k, v in WORDS.items() if re.search(rf"\b{k}\b", text)), None)
            if n and "pack" not in text:
                direction = "removed" if " out" in text else "added"
                args = {"direction": direction, "quantity": n}
                calls.append({"tool": "record_count_event", "args": args,
                              "result": await by["record_count_event"].handler(args)})
            return AgentRun("", calls, 0.0, 1)
        led = await by["read_count_ledger"].handler({})
        return AgentRun(f"{led['added']} added, {led['called_out']} called out.", calls, 0.0, 1)


def replay():
    caps, rules = pack(RuleAgent(), FixtureVision())
    hb = Heartbeat(ScriptedDecider(intuition), WorkingMemory(), caps, rules, context)

    async def go():
        for raw in SCENARIO["events"]:
            await hb.on_event(Event(**raw))
    asyncio.run(go())
    return hb


def test_revision_replaces_count_and_closure_alert_fires():
    hb = replay()
    led = ledger(hb.memory)
    assert (led["added"], led["called_out"], led["counter_bag_camera"]) == (12, 6, 5)
    assert {"callout_vs_camera", "unreconciled_at_closure"} <= set(hb.alerts)
    revised = next(t for t in hb.trace if t["event"]["seg_id"] == "s4" and t["event"]["rev"] == 2)
    assert revised["invalidated"] == ["count.event.s4.0"]


def test_vision_runs_only_when_heartbeat_fires():
    hb = replay()
    vision_ticks = [t for t in hb.trace if "vision_recount" in t["results"]]
    assert [t["event"]["data"]["frame"] for t in vision_ticks] == ["f1", "f2"]


def test_stale_proposal_is_rejected():
    m = WorkingMemory()
    m.add_segment("s1", 1, "A", "three in", 0)
    old = Evidence("speech", "s1", 1)
    m.add_segment("s1", 2, "A", "two in", 1)
    assert not m.put("count.event.s1.0", {"direction": "added", "quantity": 3}, [old], "proposed", 2)


def test_backstop_runs_when_decider_fails():
    class Down:
        name = "down"

        async def decide(self, context, questions):
            raise RuntimeError("timeout")

    caps, rules = pack(RuleAgent(), FixtureVision())
    hb = Heartbeat(Down(), WorkingMemory(), caps, rules, context)
    tick = asyncio.run(hb.on_event(Event("speech", 0, "s1", 1, "Surgeon", "Start closing the fascia.")))
    assert tick["decider_error"] and hb.memory.get("phase") == "closing"


def test_failed_capability_is_flagged_and_retried():
    class Flaky(RuleAgent):
        calls = 0

        async def run(self, system, user, tools, max_steps=4, stop_after_tools=False):
            if "record_count_event" in {x.name for x in tools}:
                Flaky.calls += 1
                if Flaky.calls == 1:
                    raise RuntimeError("429 free usage limit")
            return await super().run(system, user, tools, max_steps, stop_after_tools)

    caps, rules = pack(Flaky(), FixtureVision())
    hb = Heartbeat(ScriptedDecider(intuition), WorkingMemory(), caps, rules, context)

    async def go():
        await hb.on_event(Event("speech", 0, "s1", 1, "Scrub nurse", "Initial count, ten lap sponges on the field."))
        assert "unrecorded_count_lines" in hb.alerts
        await hb.on_event(Event("speech", 1, "s2", 1, "Surgeon", "Suction, please."))
    asyncio.run(go())
    assert ledger(hb.memory)["added"] == 10 and "unrecorded_count_lines" not in hb.alerts
