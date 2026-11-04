"""Scan stock clips second by second with the Muse Glimmer judge (4 frames covering the last 2 s, one yes/no
question each) and print the yes-probability timeline, plus a cross question that should stay low.

  python tools/glimmer_scan.py [--url http://127.0.0.1:18850/v1] [--out runs/glimmer_scan.json]
"""
import argparse, asyncio, json, subprocess, sys, tempfile, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pulse.providers import ChatYesNoDecider, Question, image_url  # noqa: E402

FF = "/home1/jijh/micromamba/envs/imgprocess/bin/ffmpeg"
STOCK = Path("/cwStorage/home/jijh/pulse_data/film2/stock")
P = "Does this video itself show "
CLIPS = {  # clip, the question the brain wrote for that watch (from the recorded takes), a question that should stay "no"
    "catdesk": ("1_cat_desk/mixkit_46542.mp4", "a cat walking across the laptop, not just sitting on it or walking nearby?", "wave"),
    "cake": ("2_pet_naughty/pexels_4013641.mp4", "the cat moving toward or reaching for the cake, not just being near or looking at the cake?", "coffee"),
    "coffee": ("3_person_desk/pexels_6799663.mp4", "the person taking a sip of coffee from a cup or mug, not just holding, preparing, or standing near coffee?", "catdesk"),
    "glass": ("2_pet_naughty/pexels_6869665.mp4", "the cat on top of the dinner table, not just beside it or underneath it?", "wave"),
    "wave": ("3_person_desk/pexels_4492700.mp4", "the person raising a hand and moving it side to side toward the camera in a wave, not just holding up or moving a hand?", "coffee"),
    "pill": ("5_pill/pexels_5104289.mp4", "the person bringing a pill to their mouth or swallowing it, not just holding a pill or a pill container?", "catdesk"),
}


def frames(clip: Path, d: Path) -> list[Path]:
    subprocess.run([FF, "-v", "error", "-i", str(clip), "-vf", "fps=4,scale=448:-2", "-q:v", "4", str(d / "f_%05d.jpg")], check=True)
    return sorted(d.glob("f_*.jpg"))


async def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://127.0.0.1:18850/v1")
    ap.add_argument("--out", default="runs/glimmer_scan.json"); ap.add_argument("--only"); a = ap.parse_args()
    judge = ChatYesNoDecider(a.url, "muse-glimmer"); res = {}; lat = []
    for name, (clip, ask, cross) in CLIPS.items():
        if a.only and name not in a.only.split(","):
            continue
        with tempfile.TemporaryDirectory() as td:
            fs = frames(STOCK / clip, Path(td)); n = len(fs) // 4
            qs = [Question("own", P + ask, kind="noul"), Question("cross", P + CLIPS[cross][1], kind="noul")]
            sem = asyncio.Semaphore(8)

            async def at(s):
                win = fs[max(0, 4 * s - 8):4 * s][1::2]
                async with sem:
                    b = await judge.decide("Four frames, in order, covering 2 seconds from a webcam or a phone camera.", qs,
                                           [image_url(str(f)) for f in win])
                lat.append(b.latency_ms)
                return s, b.decisions["own"].probs["yes"], b.decisions["cross"].probs["yes"]

            t = time.perf_counter()
            rows = await asyncio.gather(*[at(s) for s in range(2, n + 1)])
            wall = time.perf_counter() - t
        res[name] = {"clip": clip, "ask": P + ask, "seconds": [r[0] for r in rows], "own": [round(r[1], 3) for r in rows],
                     "cross": [round(r[2], 3) for r in rows], "wall_s": round(wall, 1)}
        bar = "".join(" .:-=+*#%@"[min(9, int(p * 10))] for _, p, _ in rows)
        print(f"{name:8s} own   |{bar}|  max {max(r[1] for r in rows):.2f} at {max(rows, key=lambda r: r[1])[0]}s")
        print(f"{'':8s} cross |{''.join(' .:-=+*#%@'[min(9, int(c * 10))] for _, _, c in rows)}|  max {max(r[2] for r in rows):.2f}"
              f"   ({len(rows)} clips in {wall:.1f} s)", flush=True)
    lat.sort(); print("latency per clip (2 questions) p50", round(lat[len(lat) // 2]), "ms  p90", round(lat[int(len(lat) * .9)]), "ms")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
