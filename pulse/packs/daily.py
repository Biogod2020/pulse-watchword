"""Everyday life pack: commitments, where things were left, tasks seen being done, and questions.

Speech and first-person camera frames go through the same heartbeat. Camera frames are captioned
by the vision decider, read with yes/no image questions, and kept in long-term memory so later
questions can find them by meaning.
"""
from __future__ import annotations

import httpx

from ..heartbeat import Capability, Event, Rule, TickContext
from ..longterm import Episode, LongTermMemory
from ..memory import Evidence, WorkingMemory
from ..providers import AgentProvider, DecisionProvider, Question, Tool, image_url

SAY = "Does this utterance itself "
SEE = "Does this image itself show "

TRIGGERS = {
    "task_new": Question("task_new", SAY + "state a plan, promise or request to do something later, or ask to be "
                                           "reminded? Not something already done, and not a question.", kind="noul"),
    "task_change": Question("task_change", SAY + "change, postpone, cancel or report finishing one of the OPEN "
                                                 "TASKS?", kind="noul"),
    "assistant": Question("assistant", SAY + "directly address the assistant, asking it for information or to "
                                             "record something? Not a question or request to another person.",
                          kind="noul", threshold=0.8),
    "put_down": Question("put_down", SEE + "a hand placing, setting down or holding a personal item (keys, a "
                                           "phone, glasses or a wallet) over a surface such as a table, shelf, "
                                           "counter, desk or bowl?", kind="noul"),
}
ITEMS = ["keys", "phone", "glasses", "wallet", "other"]


def open_tasks(m: WorkingMemory) -> list[dict]:
    return [{"id": f.key, **f.value} for f in m.facts_with_prefix("task.") if f.value.get("status") == "open"]


def item_locations(m: WorkingMemory) -> list[dict]:
    return [{"item": f.key.split(".", 1)[1], **f.value} for f in m.facts_with_prefix("item.")]


def task_questions(m: WorkingMemory) -> list[Question]:
    """One narrow question per open task: broad 'any open task?' questions read poorly on djev."""
    return [Question(f"doing_task_{i}", SEE + f"someone doing this: '{t['what']}'?", kind="noul")
            for i, t in enumerate(open_tasks(m)[:6])]


def context(m: WorkingMemory, ev: Event, alerts) -> str:
    tasks = open_tasks(m)
    lines = ["OPEN TASKS:"] + ([f"- {t['what']} (due {t.get('due') or 'unspecified'})" for t in tasks] or ["(none)"])
    if ev.kind == "speech":
        earlier = [s for s in m.recent(6) if s.seg_id != ev.seg_id][-5:]
        lines += ["", "EARLIER CONVERSATION (context only):"]
        lines += [f"{s.speaker}: {s.text}" for s in earlier] or ["(none)"]
        lines += ["", f"THIS UTTERANCE: {ev.speaker}: {ev.text}"]
    else:
        lines += ["", f"THIS IMAGE: first-person camera frame {ev.data.get('frame')}. "
                      f"Caption: {ev.data.get('caption', '(none)')}"]
    return "\n".join(lines)


def make_perceive(vision_url: str):
    """Caption each frame with the vision decider's generation endpoint (djev raw passthrough)."""
    client = httpx.AsyncClient(timeout=60, trust_env=False)

    async def perceive(ev: Event) -> None:
        if ev.kind != "vision" or not ev.data.get("image") or ev.data.get("caption"):
            return
        r = await client.post(vision_url.rstrip("/") + "/v1/raw/chat/completions", json={
            "model": "dgemma", "max_tokens": 40,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_url(ev.data["image"])}},
                {"type": "text", "text": "In one short sentence: what is the hand doing, and where (name the "
                                         "furniture or room)?"}]}]})
        r.raise_for_status()
        ev.data["caption"] = r.json()["choices"][0]["message"]["content"].strip()
    return perceive


def make_remember(ltm: LongTermMemory):
    async def remember(ev: Event) -> None:
        if ev.kind == "speech":
            await ltm.add(Episode(ev.seg_id, "speech", ev.t, f"{ev.speaker}: {ev.text}"))
        elif ev.kind == "vision":
            await ltm.add(Episode(ev.data["frame"], "frame", ev.t, ev.data.get("caption", ""), ev.data.get("image")))
    return remember


TASK_SYSTEM = ("You keep the user's task list. Use the tools for what the line explicitly states. Keep 'what' short "
               "(an imperative phrase); copy any due time as said. Never add a duplicate of an open task: update it instead "
               "(for example to add a due time). If nothing applies, reply 'none'.")


def pack(agent: AgentProvider, vision: DecisionProvider, ltm: LongTermMemory) -> tuple[list[Capability], list[Rule]]:
    def task_tools(ctx: TickContext, out: list) -> list[Tool]:
        ev = ctx.event
        evidence = [ctx.evidence]

        async def add(args: dict) -> dict:
            key = f"task.{ev.seg_id or ev.data.get('frame')}.{len(out)}"
            ok = ctx.memory.put(key, {"what": args.get("what", ""), "due": args.get("due"), "status": "open"},
                                evidence, "proposed", ev.t)
            out.append({"add": key, "ok": ok})
            return {"ok": ok, "id": key}

        async def update(args: dict) -> dict:
            fact = ctx.memory.facts.get(args.get("id", ""))
            if not fact:
                return {"ok": False, "error": "unknown task id", "open": open_tasks(ctx.memory)}
            value = dict(fact.value)
            if args.get("status"):
                value["status"] = args["status"]
            if args.get("due"):
                value["due"] = args["due"]
            ok = ctx.memory.put(fact.key, value, fact.evidence + evidence, "proposed", ev.t)
            out.append({"update": fact.key, **{k: v for k, v in args.items() if k != "id"}, "ok": ok})
            return {"ok": ok}

        return [Tool("add_task", "Add a task the line states.",
                     {"type": "object", "properties": {"what": {"type": "string"}, "due": {"type": "string"}},
                      "required": ["what"]}, add),
                Tool("update_task", "Change an open task: new due time, or status done/cancelled.",
                     {"type": "object", "properties": {"id": {"type": "string"},
                                                        "status": {"type": "string", "enum": ["done", "cancelled"]},
                                                        "due": {"type": "string"}}, "required": ["id"]}, update)]

    async def handle_tasks(ctx: TickContext) -> dict:
        out: list = []
        listing = "\n".join(f"{t['id']}: {t['what']} (due {t.get('due') or 'unspecified'})"
                            for t in open_tasks(ctx.memory)) or "(none)"
        what = (f"{ctx.event.speaker}: {ctx.event.text}" if ctx.event.kind == "speech"
                else f"Camera shows: {ctx.event.data.get('caption')}")
        run = await agent.run(TASK_SYSTEM, f"OPEN TASKS:\n{listing}\n\nLINE: {what}", task_tools(ctx, out),
                              stop_after_tools=True)
        return {"agent": agent.name, "changes": out, "agent_ms": round(run.latency_ms)}

    async def note_item(ctx: TickContext) -> dict:
        ev = ctx.event
        b = await vision.decide(f"Caption: {ev.data.get('caption', '')}",
                                [Question("item", "Which personal item is being put down?", ITEMS)],
                                [image_url(ev.data["image"])])
        item = b.decisions["item"].choice
        if item == "other":
            return {"item": "other"}
        ctx.memory.put(f"item.{item}", {"where": ev.data.get("caption", ""), "frame": ev.data["frame"], "t": ev.t},
                       [Evidence("vision", ev.data["frame"])], "observed", ev.t)
        return {"item": item, "where": ev.data.get("caption"), "read_ms": round(b.latency_ms)}

    async def answer(ctx: TickContext) -> dict:
        async def tasks(_: dict) -> dict:
            return {"open": open_tasks(ctx.memory)}

        async def items(_: dict) -> dict:
            return {"last_seen": item_locations(ctx.memory)}

        async def search(args: dict) -> dict:
            return {"matches": await ltm.search(str(args.get("query", "")), k=3)}

        tools = [Tool("list_open_tasks", "The user's open tasks.", {"type": "object", "properties": {}}, tasks),
                 Tool("item_locations", "Where personal items were last seen being put down.",
                      {"type": "object", "properties": {}}, items),
                 Tool("search_memory", "Search everything heard and seen today (utterances and camera frames) by "
                                       "meaning.", {"type": "object", "properties": {"query": {"type": "string"}},
                                                    "required": ["query"]}, search)]
        run = await agent.run("You are the user's always-on assistant. Answer from the tools only, in one or two "
                              "short sentences, and mention when something was seen or said.",
                              f"{ctx.event.speaker}: {ctx.event.text}", tools)
        ctx.memory.put(f"answer.{ctx.event.seg_id}", run.text, [ctx.evidence], "proposed", ctx.event.t)
        return {"answer": run.text, "tools": [c["tool"] for c in run.tool_calls], "agent_ms": round(run.latency_ms)}

    caps = [
        Capability("tasks", [TRIGGERS["task_new"], TRIGGERS["task_change"]], handle_tasks, kind="agent"),
        Capability("note_item", [TRIGGERS["put_down"]], note_item, on=("vision",), kind="vision"),
        Capability("task_seen", [], handle_tasks, on=("vision",), kind="agent", dynamic=task_questions),
        Capability("answer", [TRIGGERS["assistant"]], answer, kind="agent"),
    ]
    return caps, []
