"""Look-back recall: answer "where is the X?" by scanning recent frames with the fast judge.

Validated on EPIC-KITCHENS P12_101 (docs/validation_epic_P12_101.md): 28/53 places correct over 34 minutes.
Steps: ask "is an X clearly visible?" on each frame in the window (newest first, run concurrently), take the
most recent confident sighting, ask where the X is; if it is still in a hand, step back to an earlier sighting.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import httpx

from .providers import DjevDecider, Question, image_url


@dataclass
class Sighting:
    found: bool
    place: str | None
    t: float | None
    frame: str | None
    visible_p: float | None
    frames_scanned: int
    place_checks: int


async def where_is(noun: str, frames: list[tuple[float, str]], now: float, djev: DjevDecider, djev_url: str | None,
                   window_s: float = 60, threshold: float = 0.6, max_checks: int = 4, concurrency: int = 6,
                   where_fn=None) -> Sighting:
    """frames: (t, path) seen so far; only frames within [now - window_s, now] are scanned."""
    pool = sorted(((t, p) for t, p in frames if now - window_s <= t <= now), reverse=True)
    sem = asyncio.Semaphore(concurrency)
    client = httpx.AsyncClient(timeout=60, trust_env=False)

    async def visible(path: str) -> float:
        async with sem:
            q = Question("v", f"Does this image itself clearly show a {noun} (the object itself, not just a similar "
                              f"object)?", kind="noul")
            b = await djev.decide("A first-person camera image.", [q], [image_url(path)])
            return b.decisions["v"].probs["yes"]

    async def where(path: str) -> str:
        if where_fn is not None:            # any vision chat model (e.g. Muse Glimmer) answers the short place
            return (await where_fn(path, f"Where is the {noun} in this image? Answer with only a short place, e.g. "
                                         f"countertop, cutting board, stove, sink, shelf, hand.")).splitlines()[0].strip(" .").lower()
        r = await client.post(djev_url.rstrip("/") + "/v1/raw/chat/completions", json={
            "model": "dgemma", "max_tokens": 16, "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_url(path)}},
                {"type": "text", "text": f"Where is the {noun} in this image? Answer with only a short place, e.g. "
                                         f"countertop, cutting board, stove, sink, shelf, hand."}]}]})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip().splitlines()[0].strip(" .").lower()

    try:
        scores = await asyncio.gather(*[visible(p) for _, p in pool])
        confident = [(t, p, s) for (t, p), s in zip(pool, scores) if s >= threshold]
        best, place, checks = None, None, 0
        for t, p, s in confident[:max_checks]:
            checks += 1
            answer = await where(p)
            if best is None:
                best, place = (t, p, s), answer
            if "hand" not in answer:
                best, place = (t, p, s), answer
                break
        if best is None:
            return Sighting(False, None, None, None, max(scores) if scores else None, len(pool), checks)
        t, p, s = best
        return Sighting(True, place, t, Path(p).stem, round(s, 3), len(pool), checks)
    finally:
        await client.aclose()
