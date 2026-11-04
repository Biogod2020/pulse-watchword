"""Held-out scoring of always-on judges on EPIC-KITCHENS P12_101 (2-second clips every second).
Trigger thresholds are picked on minutes 0-10 only (best event-level F1, at least 3 fires), then frozen and
scored on minutes 10-34. Same procedure for every judge, so the rows are comparable.
    python eval_heldout.py runs/video_windows_full_c2_s1.json runs/glimmer_windows_full_c2_s1.json
"""
from __future__ import annotations

import csv
import json
import statistics
import sys

from eval_activity_cues import CUES, auc
from pulse.triggers import EdgeTrigger, StateTracker
from run_video import ts

ANN = "/cwStorage/home/jijh/pulse_data/epic/EPIC_100_train.csv"
T0 = 600
GRID = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def score(path: str) -> dict:
    d = json.load(open(path)); sc, spans = d["scores"], d["spans"]
    rows = [r for r in csv.DictReader(open(ANN)) if r["video_id"] == "P12_101"]
    end = spans[-1][1]; out = {"judge": d["report"].get("judge", "djev (DiffusionGemma 26B-A4B AWQ, 1x H100)"),
                               "wall_s": d["report"]["wall_s"], "clips": len(sc), "cues": {}}
    for k, (_, verbs) in CUES.items():
        allw = [(ts(r["start_timestamp"]), ts(r["stop_timestamp"])) for r in rows if r["verb"] in verbs]

        def part(lo, hi):
            wins = [w for w in allw if lo <= w[0] < hi and w[1] <= end]
            idx = [i for i, (s0, _) in enumerate(spans) if lo <= s0 < hi]
            return wins, idx

        def hit(wins, s0, s1):
            return any(s0 < b + 0.5 and s1 > a - 0.5 for a, b in wins)

        def edge(th, wins, idx):
            e = EdgeTrigger(th, 4); fires = [spans[i] for i in idx if e.update(spans[i][1], sc[i][k])]
            good = sum(hit(wins, *f) for f in fires)
            caught = sum(1 for w in wins if any(f[0] < w[1] + 0.5 and f[1] > w[0] - 0.5 for f in fires))
            return len(fires), good, caught

        def f1(n, good, caught, nw):
            p = good / n if n else 0; r = caught / nw if nw else 0
            return 2 * p * r / (p + r) if p + r else 0

        tw, ti = part(0, T0); hw, hi_ = part(T0, 1e9)
        # event AUC on the held-out part (max score over clips overlapping each event vs. equal-length quiet windows)
        length = statistics.median(b - a for a, b in hw) + 1; busy = [(a - 1, b + 1) for a, b in hw]; negs, t = [], T0
        while t + length <= end:
            if not any(t < b and t + length > a for a, b in busy):
                negs.append((t, t + length))
            t += length
        wm = lambda w: max([x[k] for x, (s0, s1) in zip(sc, spans) if s0 < w[1] + 0.5 and s1 > w[0] - 0.5] or [0])
        pv, nv = [wm(w) for w in hw], [wm(w) for w in negs]
        base = sum(hit(hw, *spans[i]) for i in hi_) / len(hi_)
        c = {"events_heldout": len(hw), "event_auc_heldout": round(auc(pv + nv, [True] * len(pv) + [False] * len(nv)), 3),
             "random_precision": round(base, 3)}
        cands = [(f1(*edge(th, tw, ti), len(tw)), th) for th in GRID if edge(th, tw, ti)[0] >= 3]
        if cands:
            th = max(cands)[1]; n, good, caught = edge(th, hw, hi_)
            c.update(threshold=th, fires=n, precision=round(good / n, 3) if n else None, caught=f"{caught}/{len(hw)}")
        out["cues"][k] = c
    return out


if __name__ == "__main__":
    res = [score(p) for p in sys.argv[1:]]
    for r in res:
        print(f"\n{r['judge']}  ({r['clips']} clips, {r['wall_s']} s wall)")
        for k, c in r["cues"].items():
            print(f"  {k:13s} AUC {c['event_auc_heldout']:.2f}  th {c.get('threshold')}  fires {c.get('fires')}  "
                  f"precision {c.get('precision')} (random {c['random_precision']:.2f})  caught {c.get('caught')}")
    json.dump(res, open("runs/heldout_compare.json", "w"), indent=1)
