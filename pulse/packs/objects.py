"""Where-did-I-put-it pack for first-person video: any object, any place.

Each kept frame gets one narrow image question. When it fires, the vision decider's generation
endpoint names the object and the place in a fixed "object | place" form, and the fact
`item.<object>` is updated with the frame as evidence (the frame's blob is pinned in storage).
"""
from __future__ import annotations

import re

import httpx

from ..heartbeat import Capability, Event, TickContext
from ..longterm import Episode, LongTermMemory
from ..memory import Evidence, WorkingMemory
from ..providers import Question, image_url

PUT_DOWN = Question(
    "put_down", "Does this image itself show a hand putting down, placing or leaving an object somewhere (on a "
                "surface, into a container, a drawer, a cupboard or the fridge)? Not picking an object up, and not "
                "just holding or using it.", kind="noul")

NAME_PROMPT = ("Look at the hand in this first-person image. Which object is it putting down, and where? Reply in "
               "exactly this form: object | place. Use a short common noun for the object (e.g. knife, plate, "
               "pan) and a short place (e.g. chopping board, sink, counter, drawer, fridge).")


def norm(noun: str) -> str:
    n = re.sub(r"[^a-z ]", "", noun.lower()).strip()
    n = re.sub(r"^(a|an|the|some) ", "", n)
    return n[:-1] if n.endswith("s") and not n.endswith("ss") else n


def context(m: WorkingMemory, ev: Event, alerts) -> str:
    return f"A first-person camera frame at {ev.t:.0f} s."


def item_locations(m: WorkingMemory) -> dict[str, dict]:
    return {f.key.split(".", 1)[1]: f.value for f in m.facts_with_prefix("item.")}


def make_remember(ltm: LongTermMemory):
    async def remember(ev: Event) -> None:
        if ev.kind == "vision":
            await ltm.add(Episode(ev.data["frame"], "frame", ev.t, ev.data.get("caption", ""), ev.data.get("image")))
    return remember


def pack(vision_url: str, store=None) -> list[Capability]:
    client = httpx.AsyncClient(timeout=60, trust_env=False)

    async def note_put(ctx: TickContext) -> dict:
        ev = ctx.event
        r = await client.post(vision_url.rstrip("/") + "/v1/raw/chat/completions", json={
            "model": "dgemma", "max_tokens": 24,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_url(ev.data["image"])}},
                {"type": "text", "text": NAME_PROMPT}]}]})
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip().splitlines()[0]
        obj, _, place = text.partition("|")
        obj, place = norm(obj), place.strip().strip(".").lower()
        if not obj or not place:
            return {"unparsed": text}
        value = {"object": obj, "place": place, "frame": ev.data["frame"], "t": ev.t, "raw": text}
        if store:
            value["blob"] = store.put_blob(ev.data["image"])   # pin the evidence frame (L2)
        ctx.memory.put(f"item.{obj}", value, [Evidence("vision", ev.data["frame"])], "observed", ev.t)
        return {"object": obj, "place": place}

    return [Capability("note_put", [PUT_DOWN], note_put, on=("vision",), kind="vision")]
