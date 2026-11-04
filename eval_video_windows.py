"""Video-clip judging vs frame judging: sliding short clips through the video-capable djev.

Clips are built from the 4 fps frames (same images as eval_fps.py): `--clip-s` seconds every `--stride-s`.
Each clip gets one structured read with every activity cue ("Does this video itself show ..."); an annotated
action scores the max over clips overlapping it, negatives are equally long action-free windows (as in
eval_fps.py), so event AUCs are directly comparable.
    python eval_video_windows.py --clip-s 4 --stride-s 2
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import json
import statistics
import subprocess
import time
from pathlib import Path

import httpx

from eval_activity_cues import CUES, auc
from run_video import ts

FFMPEG = "/home1/jijh/micromamba/envs/imgprocess/bin/ffmpeg"


def make_clip(frames_dir: Path, start_index: int, n: int, fps: float, out: Path) -> Path:
    if not out.exists():
        subprocess.run([FFMPEG, "-v", "error", "-y", "-framerate", str(fps), "-start_number", str(start_index),
                        "-i", str(frames_dir / "f_%06d.jpg"), "-frames:v", str(n), "-vf", "scale=448:-2",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "30", "-pix_fmt", "yuv420p", str(out)],
                       check=True)
    return out


async def main(a):
    frames_dir = Path(a.frames)
    total = len(list(frames_dir.glob("f_*.jpg")))
    per = int(a.clip_s * a.base_fps)
    step = int(a.stride_s * a.base_fps)
    clips_dir = Path(a.clips) / f"c{a.clip_s}_s{a.stride_s}"
    clips_dir.mkdir(parents=True, exist_ok=True)
    starts = list(range(1, total - per + 2, step))
    t0 = time.perf_counter()
    paths = [make_clip(frames_dir, s, per, a.base_fps, clips_dir / f"clip_{s:06d}.mp4") for s in starts]
    print(f"{len(paths)} clips ready in {time.perf_counter()-t0:.0f}s", flush=True)

    S = "Does this video itself show "
    qs = {k: {"type": "noul", "instructions": v[0].replace("Does this image itself show ", S)} for k, v in CUES.items()}
    sem = asyncio.Semaphore(a.concurrency)
    client = httpx.AsyncClient(timeout=120, trust_env=False)
    lat, toks = [], []

    async def read(path: Path) -> dict:
        async with sem:
            url = "data:video/mp4;base64," + base64.b64encode(path.read_bytes()).decode()
            t = time.perf_counter()
            r = await client.post(a.djev_url + "/v1/systemone", json={
                "model": "dgemma", "state": f"A {a.clip_s:g}-second first-person kitchen video.", "samples": 1,
                "questions": qs, "videos": [url]})
            r.raise_for_status()
            d = r.json()
            lat.append((time.perf_counter() - t) * 1e3)
            toks.append(d["usage"]["input_tokens"])
            return {k: v["noul"] for k, v in d["answers"].items()}

    t0 = time.perf_counter()
    scores = await asyncio.gather(*[read(p) for p in paths])
    wall = time.perf_counter() - t0
    spans = [((s - 1) / a.base_fps, (s - 1) / a.base_fps + a.clip_s) for s in starts]
    rows = [r for r in csv.DictReader(open(a.annotations)) if r["video_id"] == a.video_id]
    end = spans[-1][1]
    report = {"clips": len(paths), "clip_s": a.clip_s, "stride_s": a.stride_s, "wall_s": round(wall),
              "read_ms_p50": round(sorted(lat)[len(lat) // 2]), "tokens_per_clip": statistics.median(toks),
              "reads_per_min": 60 / a.stride_s, "cues": {}}
    for k, (_, verbs) in CUES.items():
        wins = [(ts(r["start_timestamp"]), ts(r["stop_timestamp"])) for r in rows
                if r["verb"] in verbs and ts(r["stop_timestamp"]) <= end]
        if len(wins) < 3:
            continue
        length = statistics.median(b - x for x, b in wins) + 1.0
        busy = [(x - 1, b + 1) for x, b in wins]
        negs, t = [], 0.0
        while t + length <= end:
            if not any(t < b and t + length > x for x, b in busy):
                negs.append((t, t + length))
            t += length

        def wmax(w):
            vals = [sc[k] for sc, (s0, s1) in zip(scores, spans) if s0 < w[1] + 0.5 and s1 > w[0] - 0.5]
            return max(vals) if vals else 0.0
        pv, nv = [wmax(w) for w in wins], [wmax(w) for w in negs]
        report["cues"][k] = {"events": len(wins), "event_auc": round(auc(pv + nv, [True] * len(pv) + [False] * len(nv)), 3)}
        print(k, report["cues"][k], flush=True)
    print({x: report[x] for x in ("clips", "wall_s", "read_ms_p50", "tokens_per_clip", "reads_per_min")})
    Path(a.out).write_text(json.dumps({"report": report, "scores": scores, "spans": spans}, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video-id", default="P12_101")
    p.add_argument("--frames", default="/cwStorage/home/jijh/pulse_data/epic/frames/P12_101_4fps_10min")
    p.add_argument("--clips", default="/cwStorage/home/jijh/pulse_data/epic/clips/windows")
    p.add_argument("--annotations", default="eval_data/EPIC_100_P12_101.csv")
    p.add_argument("--base-fps", type=float, default=4)
    p.add_argument("--clip-s", type=float, default=4)
    p.add_argument("--stride-s", type=float, default=2)
    p.add_argument("--djev-url", default="http://127.0.0.1:18831")
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--out", default="runs/video_windows_c4_s2.json")
    asyncio.run(main(p.parse_args()))
