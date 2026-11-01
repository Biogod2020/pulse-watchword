"""Which visual activities can the heartbeat track reliably on real first-person video?

One djev image read answers every cue at once. Each cue is scored against EPIC narration windows for its
verbs over the whole video: AUC, plus precision/recall of an online rising-edge trigger.
    python eval_activity_cues.py
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import time
from pathlib import Path

from pulse.providers import DjevDecider, Question, image_url
from run_video import ts

S = "Does this image itself show "
CUES = {
    "washing": (S + "hands washing or rinsing something under running water at a sink?", {"wash"}),
    "stirring": (S + "someone stirring food in a pot or pan with a spoon or spatula?", {"stir", "stir-in", "mix"}),
    "pouring": (S + "someone pouring a liquid or food from one container into another?", {"pour-into", "pour-from", "pour"}),
    "opening": (S + "a hand opening or closing a cupboard, drawer, fridge door or jar lid?", {"open", "close"}),
    "tap_or_knob": (S + "a hand turning a tap, a stove knob or a switch?", {"turn-on", "turn-off"}),
    "putting_down": (S + "a hand putting down or placing an object on a surface?", {"put-down", "put-into", "put-onto"}),
}


def auc(scores, labels):
    pos = sorted(s for s, y in zip(scores, labels) if y)
    neg = sorted(s for s, y in zip(scores, labels) if not y)
    if not pos or not neg:
        return None
    import bisect
    wins = sum(bisect.bisect_left(neg, p) + 0.5 * (bisect.bisect_right(neg, p) - bisect.bisect_left(neg, p)) for p in pos)
    return wins / (len(pos) * len(neg))


def edge(scores, th, refractory=5):
    out, last = [], -99
    for t in range(1, len(scores)):
        if scores[t] >= th and scores[t - 1] < th and t - last >= refractory:
            out.append(t)
            last = t
    return out


async def main(a):
    frames = sorted(Path(a.frames).glob("f_*.jpg"))
    n = len(frames)
    rows = [r for r in csv.DictReader(open(a.annotations)) if r["video_id"] == a.video_id]
    djev, sem = DjevDecider(a.djev_url), asyncio.Semaphore(6)
    qs = [Question(k, v[0], kind="noul") for k, v in CUES.items()]

    async def read(f):
        async with sem:
            b = await djev.decide("A first-person kitchen camera image.", qs, [image_url(str(f))])
            return {k: d.probs["yes"] for k, d in b.decisions.items()}, b.latency_ms

    t0 = time.perf_counter()
    out = await asyncio.gather(*[read(f) for f in frames])
    wall = time.perf_counter() - t0
    report = {"frames": n, "wall_s": round(wall), "read_ms_p50": sorted(x[1] for x in out)[n // 2], "cues": {}}
    for k, (q, verbs) in CUES.items():
        wins = [(ts(r["start_timestamp"]) - 1, ts(r["stop_timestamp"]) + 1) for r in rows if r["verb"] in verbs]
        label = [any(s0 <= t <= s1 for s0, s1 in wins) for t in range(n)]
        sc = [o[0][k] for o in out]
        base = sum(label) / n
        best = None
        for th in (0.3, 0.5, 0.7, 0.9):
            fires = edge(sc, th)
            if not fires:
                continue
            prec = sum(label[t] for t in fires) / len(fires)
            caught = sum(1 for s0, s1 in wins if any(s0 <= t <= s1 for t in fires))
            cand = {"threshold": th, "fires": len(fires), "precision": round(prec, 3), "lift": round(prec / base, 2) if base else None,
                    "events_caught": f"{caught}/{len(wins)}"}
            if best is None or (cand["lift"] or 0) > (best["lift"] or 0):
                best = cand
        report["cues"][k] = {"question": q, "verbs": sorted(verbs), "events": len(wins), "base_rate": round(base, 3),
                             "auc": round(auc(sc, label), 3) if auc(sc, label) else None, "best_edge_trigger": best,
                             "scores": [round(x, 4) for x in sc]}
        print(f"{k:13} events={len(wins):3d} base={base:.2f} auc={report['cues'][k]['auc']} best={best}", flush=True)
    Path(a.out).write_text(json.dumps(report, indent=1))
    print("wall", round(wall), "s; read p50", report["read_ms_p50"], "ms (6 cues per read)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video-id", default="P12_101")
    p.add_argument("--frames", default="/cwStorage/home/jijh/pulse_data/epic/frames/P12_101")
    p.add_argument("--annotations", default="/cwStorage/home/jijh/pulse_data/epic/EPIC_100_train.csv")
    p.add_argument("--djev-url", default="http://127.0.0.1:18811")
    p.add_argument("--out", default="runs/activity_cues_full.json")
    asyncio.run(main(p.parse_args()))
