"""Compare visual trigger questions on real first-person frames against EPIC put/place narrations.

    python eval_video_questions.py --minutes 5
Each variant scores every frame; we report AUC, and recall/precision at the threshold giving ~10% fire rate.
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
from pathlib import Path

from PIL import Image

from pulse.providers import DjevDecider, Question, image_url
from run_video import load_puts

SINGLE = ("Does this image itself show a hand putting down, placing or leaving an object somewhere (on a surface, "
          "into a container, a drawer, a cupboard or the fridge)? Not picking an object up, and not just holding "
          "or using it.")
STATIC = ("Does this image itself show a hand holding an object that is touching or just above a surface such as a "
          "countertop, cutting board, stove or shelf, as if setting it down?")
PAIR = ("The LEFT and RIGHT halves are two moments one second apart from the same first-person camera. Does the pair "
        "itself show an object being put down: held by a hand on the LEFT, and resting on a surface or released on "
        "the RIGHT? Not an object being picked up.")


def tile(a: Path, b: Path, out: Path) -> Path:
    if not out.exists():
        ia, ib = Image.open(a).convert("RGB"), Image.open(b).convert("RGB")
        w, h = ia.size
        canvas = Image.new("RGB", (w * 2 + 8, h), "white")
        canvas.paste(ia, (0, 0))
        canvas.paste(ib, (w + 8, 0))
        canvas.save(out, quality=85)
    return out


def auc(scores: list[float], labels: list[bool]) -> float:
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / max(len(pos) * len(neg), 1)


async def main(a):
    frames = sorted(Path(a.frames).glob("f_*.jpg"))[: int(a.minutes * 60)]
    puts = load_puts(a.annotations, a.video_id, 0, len(frames))
    t_of = lambda f: int(f.stem.split("_")[1]) - 1
    label = [any(p["start"] - 2 <= t_of(f) <= p["stop"] + 2 for p in puts) for f in frames]
    djev = DjevDecider(a.djev_url)
    tiles = Path(a.tiles)
    tiles.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(6)

    async def score(prompt: str, img: Path) -> float:
        async with sem:
            b = await djev.decide("A first-person camera image.", [Question("q", prompt, kind="noul")], [image_url(str(img))])
            return b.decisions["q"].probs["yes"]

    variants = {
        "A_single_frame": [score(SINGLE, f) for f in frames],
        "B_static_cue": [score(STATIC, f) for f in frames],
        "C_two_frame_pair": [score(PAIR, tile(frames[max(i - 1, 0)], f, tiles / f"pair_{f.stem}.jpg"))
                             for i, f in enumerate(frames)],
    }
    report = {"frames": len(frames), "puts": len(puts), "base_rate": sum(label) / len(label), "variants": {}}
    for name, coros in variants.items():
        s = await asyncio.gather(*coros)
        th = sorted(s)[int(len(s) * 0.9)]
        fired = [i for i, v in enumerate(s) if v >= th]
        caught = sum(1 for p in puts if any(p["start"] - 2 <= t_of(frames[i]) <= p["stop"] + 2 for i in fired))
        prec = sum(label[i] for i in fired) / max(len(fired), 1)
        report["variants"][name] = {"auc": round(auc(s, label), 3), "threshold_at_10pct": round(th, 3),
                                    "recall_at_10pct": f"{caught}/{len(puts)}", "precision_at_10pct": round(prec, 3),
                                    "scores": [round(v, 4) for v in s]}
        print(name, {k: v for k, v in report["variants"][name].items() if k != "scores"}, flush=True)
    Path(a.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video-id", default="P12_101")
    p.add_argument("--frames", default="/cwStorage/home/jijh/pulse_data/epic/frames/P12_101")
    p.add_argument("--annotations", default="/cwStorage/home/jijh/pulse_data/epic/EPIC_100_train.csv")
    p.add_argument("--minutes", type=float, default=5)
    p.add_argument("--djev-url", default="http://127.0.0.1:18811")
    p.add_argument("--tiles", default="/cwStorage/home/jijh/pulse_data/epic/tiles/P12_101")
    p.add_argument("--out", default="runs/video_question_eval_5min.json")
    asyncio.run(main(p.parse_args()))
