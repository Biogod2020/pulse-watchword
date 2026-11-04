"""Muse Glimmer as the always-on judge on EPIC-KITCHENS P12_101: the same 2-second clips every second as
eval_video_windows.py (djev), but each clip goes to Glimmer as 4 frames (it takes images, not video) and each
activity cue is its own yes/no read (log-probabilities of the first answer token, reasoning strength low).
Output has the same layout as runs/video_windows_full_c2_s1.json, so eval_heldout.py can score both.
    python eval_glimmer_windows.py [--url http://127.0.0.1:18850/v1] [--limit N]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from eval_activity_cues import CUES
from pulse.providers import ChatYesNoDecider, Question, image_url

S = "Does this video itself show "


async def main(a):
    frames = sorted(Path(a.frames).glob("f_*.jpg"))
    n = (len(frames) - 8) // 4 + 1 if a.limit is None else a.limit
    judge = ChatYesNoDecider(a.url, "muse-glimmer")
    qs = [Question(k, v[0].replace("Does this image itself show ", S), kind="noul") for k, v in CUES.items()]
    sem = asyncio.Semaphore(a.concurrency)
    lat: list[float] = []
    done = 0

    async def read(i: int) -> dict:
        nonlocal done
        imgs = [image_url(str(f)) for f in frames[4 * i:4 * i + 8][1::2]]      # 4 of the 8 frames in [i, i+2) s
        async with sem:
            b = await judge.decide("Four frames, in order, covering 2 seconds of first-person kitchen video.", qs, imgs)
        lat.append(b.latency_ms); done += 1
        if done % 200 == 0:
            print(f"{done}/{n} clips", flush=True)
        return {k: round(d.probs["yes"], 4) for k, d in b.decisions.items()}

    t0 = time.perf_counter()
    scores = await asyncio.gather(*[read(i) for i in range(n)])
    wall = time.perf_counter() - t0
    report = {"judge": "muse-glimmer (Muse Glimmer 30B, BF16, vLLM, 1x H100)", "clips": n, "clip_s": 2.0, "stride_s": 1.0,
              "wall_s": round(wall), "read_ms_p50": round(sorted(lat)[len(lat) // 2]), "questions_per_clip": len(qs)}
    print(report)
    Path(a.out).write_text(json.dumps({"report": report, "scores": scores,
                                       "spans": [[float(i), float(i) + 2.0] for i in range(n)]}, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--frames", default="/cwStorage/home/jijh/pulse_data/epic/frames/P12_101_4fps")
    p.add_argument("--url", default="http://127.0.0.1:18850/v1")
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default="runs/glimmer_windows_full_c2_s1.json")
    asyncio.run(main(p.parse_args()))
