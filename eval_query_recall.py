"""Query-time recall: answer "where is the X?" by scanning back through remembered frames.

For each narrated "put X on/in Y", at stop+20s: scan frames backwards over a window, ask the judge
"is an X clearly visible?" on each, take the most recent confident frame, then ask where the X is.
The remembered place is graded against the narrated place by an LLM (grades saved).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

from pulse.providers import DjevDecider, Question, image_url
from run_video import grade_places, load_puts


async def main(a):
    frames = sorted(Path(a.frames).glob("f_*.jpg"))[: int(a.minutes * 60)]
    by_t = {int(f.stem.split("_")[1]) - 1: f for f in frames}
    puts = [p for p in load_puts(a.annotations, a.video_id, 0, len(frames)) if p["place"]]
    djev = DjevDecider(a.djev_url)
    client = httpx.AsyncClient(timeout=60, trust_env=False)
    sem = asyncio.Semaphore(6)

    async def visible(noun: str, f: Path) -> float:
        async with sem:
            q = Question("v", f"Does this image itself clearly show a {noun} (the object itself, not just a similar "
                              f"object)?", kind="noul")
            b = await djev.decide("A first-person kitchen camera image.", [q], [image_url(str(f))])
            return b.decisions["v"].probs["yes"]

    async def where(noun: str, f: Path) -> str:
        r = await client.post(a.djev_url + "/v1/raw/chat/completions", json={"model": "dgemma", "max_tokens": 16,
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url(str(f))}},
                {"type": "text", "text": f"Where is the {noun} in this image? Answer with only a short place, e.g. "
                                         f"countertop, cutting board, stove, sink, shelf, hand."}]}]})
        return r.json()["choices"][0]["message"]["content"].strip().splitlines()[0].strip(" .").lower()

    results = []
    started = time.perf_counter()
    for p in puts:
        T = int(p["stop"] + a.delay)
        ts = [t for t in range(min(T, len(frames) - 1), max(T - a.window, -1), -1) if t in by_t]
        scores = await asyncio.gather(*[visible(p["noun"], by_t[t]) for t in ts])
        confident = [(t, s) for t, s in zip(ts, scores) if s >= a.threshold]
        best, place, tried = None, None, 0
        for t, s in confident[: a.max_checks]:      # most recent first; skip sightings where it is held
            tried += 1
            answer = await where(p["noun"], by_t[t])
            if "hand" not in answer:
                best, place = (t, s), answer
                break
            best, place = best or (t, s), place or answer
        if best is None and scores:
            best = max(zip(ts, scores), key=lambda x: x[1])
            place = await where(p["noun"], by_t[best[0]])
        results.append({"narration": p["narration"], "noun": p["noun"], "gt_place": p["place"], "asked_at": T,
                        "frame_t": best[0] if best else None, "visible_p": round(best[1], 3) if best else None,
                        "confident": bool(best and best[1] >= a.threshold), "answer_place": place,
                        "frames_scanned": len(ts), "place_checks": tried})
    graded = [r for r in results if r["answer_place"]]
    marks = await grade_places([{"gt": r["gt_place"], "pred": r["answer_place"]} for r in graded], a.grader_url,
                               "gpt-6-luna", Path(a.grader_key_file).read_text().strip())
    for r, m in zip(graded, marks):
        r["place_match"] = m
    summary = {"queries": len(results), "place_correct": sum(1 for r in results if r.get("place_match")),
               "confident": sum(r["confident"] for r in results),
               "correct_when_confident": sum(1 for r in results if r["confident"] and r.get("place_match")),
               "frames_scanned_per_query": a.window, "wall_s": round(time.perf_counter() - started),
               "settings": {"window_s": a.window, "delay_s": a.delay, "threshold": a.threshold, "max_checks": a.max_checks}}
    Path(a.out).write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=1))
    print(json.dumps(summary, indent=1))
    for r in results:
        print(f"{r['narration']:36} -> {r['answer_place']!s:18} p={r['visible_p']} match={r.get('place_match')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video-id", default="P12_101")
    p.add_argument("--frames", default="/cwStorage/home/jijh/pulse_data/epic/frames/P12_101")
    p.add_argument("--annotations", default="/cwStorage/home/jijh/pulse_data/epic/EPIC_100_train.csv")
    p.add_argument("--minutes", type=float, default=5)
    p.add_argument("--window", type=int, default=60)
    p.add_argument("--delay", type=float, default=20)
    p.add_argument("--threshold", type=float, default=0.6)
    p.add_argument("--max-checks", type=int, default=4)
    p.add_argument("--djev-url", default="http://127.0.0.1:18811")
    p.add_argument("--grader-url", default="http://127.0.0.1:8317/v1")
    p.add_argument("--grader-key-file", default="/home1/jijh/tools/cliproxyapi/local_api_key")
    p.add_argument("--out", default="runs/video_query_recall_5min.json")
    asyncio.run(main(p.parse_args()))
