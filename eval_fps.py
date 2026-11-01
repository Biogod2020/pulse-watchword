"""Does a higher frame rate help the heartbeat? Event-level comparison of 1, 2 and 4 fps.

Frames are extracted once at 4 fps and scored once (all activity cues in one read); 2 and 1 fps are subsets,
so every rate sees identical images. For each cue, an annotated action window scores the max probability of
the sampled frames inside it; negatives are equally long windows without that action. Reports event AUC,
how many actions get no sampled frame at all, and judge reads per minute.
    python eval_fps.py --frames .../P12_101_4fps_10min --minutes 10
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
import time
from pathlib import Path

from eval_activity_cues import CUES, auc
from pulse.providers import DjevDecider, Question, image_url
from run_video import ts


async def score_all(frames: list[Path], url: str, concurrency: int) -> list[dict]:
    djev, sem = DjevDecider(url), asyncio.Semaphore(concurrency)
    qs = [Question(k, v[0], kind="noul") for k, v in CUES.items()]

    async def one(f):
        async with sem:
            b = await djev.decide("A first-person kitchen camera image.", qs, [image_url(str(f))])
            return {k: d.probs["yes"] for k, d in b.decisions.items()}
    return await asyncio.gather(*[one(f) for f in frames])


def main(a):
    frames = sorted(Path(a.frames).glob("f_*.jpg"))
    frames = frames[: int(a.minutes * 60 * a.base_fps)]
    times = [(int(f.stem.split("_")[1]) - 1) / a.base_fps for f in frames]
    cache = Path(a.cache)
    if cache.exists():
        scores = json.loads(cache.read_text())
    else:
        t0 = time.perf_counter()
        scores = asyncio.run(score_all(frames, a.djev_url, a.concurrency))
        cache.write_text(json.dumps(scores))
        print(f"scored {len(frames)} frames in {time.perf_counter()-t0:.0f}s")
    rows = [r for r in csv.DictReader(open(a.annotations)) if r["video_id"] == a.video_id]
    end = times[-1]
    report = {}
    for k, (_, verbs) in CUES.items():
        wins = [(ts(r["start_timestamp"]), ts(r["stop_timestamp"])) for r in rows
                if r["verb"] in verbs and ts(r["stop_timestamp"]) <= end]
        if len(wins) < 3:
            continue
        length = statistics.median(b - a_ for a_, b in wins) + 1.0
        busy = [(a_ - 1, b + 1) for a_, b in wins]
        negs, t = [], 0.0
        while t + length <= end:
            if not any(t < b and t + length > a_ for a_, b in busy):
                negs.append((t, t + length))
            t += length
        report[k] = {"events": len(wins), "negatives": len(negs), "median_event_s": round(length - 1, 2)}
        for fps in (1, 2, 4):
            step = int(a.base_fps / fps)
            idx = [i for i in range(0, len(frames), step)]

            def window_max(w):
                vals = [scores[i][k] for i in idx if w[0] - 0.5 <= times[i] <= w[1] + 0.5]
                return max(vals) if vals else None
            pos = [window_max(w) for w in wins]
            neg = [window_max(w) for w in negs]
            empty = sum(v is None for v in pos)
            pv, nv = [v if v is not None else 0.0 for v in pos], [v if v is not None else 0.0 for v in neg]
            report[k][f"{fps}fps"] = {"event_auc": round(auc(pv + nv, [True] * len(pv) + [False] * len(nv)), 3),
                                      "events_without_a_frame": empty, "reads_per_min": 60 * fps}
        print(k, json.dumps(report[k], ensure_ascii=False))
    Path(a.out).write_text(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video-id", default="P12_101")
    p.add_argument("--frames", default="/cwStorage/home/jijh/pulse_data/epic/frames/P12_101_4fps_10min")
    p.add_argument("--annotations", default="/cwStorage/home/jijh/pulse_data/epic/EPIC_100_train.csv")
    p.add_argument("--minutes", type=float, default=10)
    p.add_argument("--base-fps", type=float, default=4)
    p.add_argument("--djev-url", default="http://127.0.0.1:18811")
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--cache", default="runs/fps_scores_4fps_10min.json")
    p.add_argument("--out", default="runs/fps_eval_10min.json")
    main(p.parse_args())
