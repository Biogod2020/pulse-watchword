"""Automatic scenario pack, evaluated across recordings: calibrate on some, test on unseen others.

    python run_autopack_multi.py --questions runs/autopack_atv_sop_questions.json \
        --calib R093-27July-ATV R123-15Aug-ATV --test R200-7Nov-ATV R202-7Nov-ATV ...
No hand tuning: questions come from another recording's step list (and calibration recordings' mistake notes);
thresholds come from a teacher (gpt-6-luna Decisions, 3 frames per clip) on the calibration recordings only.
Metrics on the test recordings (pooled):
- step questions: event AUC vs the human action labels the brain maps them to; edge-trigger precision / recall
- any alert in the 10 s before the instructor corrects a wrong action, vs the rate expected by chance
- mistake questions: event AUC vs labelled wrong actions
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from eval_activity_cues import auc
from pulse.autopack import best_threshold, map_to_labels
from pulse.triggers import EdgeTrigger
from run_autopack import judge_clips, recording_labels, teacher


def windows_and_negatives(wins, spans, end):
    if not wins:
        return [], []
    length = statistics.median(b - x for x, b in wins) + 1.0
    busy = [(x - 1, b + 1) for x, b in wins]
    negs, t = [], 0.0
    while t + length <= end:
        if not any(t < b and t + length > x for x, b in busy):
            negs.append((t, t + length))
        t += length
    return wins, negs


def wmax(scores, spans, k, w):
    return max([s[k] for s, (s0, s1) in zip(scores, spans) if s0 < w[1] + 0.5 and s1 > w[0] - 0.5] or [0.0])


def main(a):
    pack = json.loads(Path(a.questions).read_text())
    questions = pack["questions"]
    tag = Path(a.questions).stem
    root = Path(a.frames_root)
    data = {}
    t0 = time.perf_counter()
    for rec in a.calib + a.test:
        cache = Path(f"runs/am_{tag}_{rec}_clips.json")
        if cache.exists():
            scores, spans = json.loads(cache.read_text())
        else:
            frames = sorted((root / rec).glob("f_*.jpg"))
            scores, spans = asyncio.run(judge_clips(frames, a.fps, questions, a.djev_url, Path(a.tmp) / rec, a.concurrency))
            cache.write_text(json.dumps([scores, spans]))
        data[rec] = {"scores": scores, "spans": spans, "labels": recording_labels(a.labels, rec)}
        print(f"{rec}: {len(spans)} clips judged", flush=True)
    judge_wall = time.perf_counter() - t0
    # thresholds from the teacher on calibration recordings only
    tcache = Path(f"runs/am_{tag}_teacher.json")
    if tcache.exists():
        tl = json.loads(tcache.read_text())
    else:
        tl, spent = {}, 0.0
        for rec in a.calib:
            frames = sorted((root / rec).glob("f_*.jpg"))
            sp = data[rec]["spans"]
            out, spent = asyncio.run(teacher(frames, a.fps, sp, list(range(len(sp))), questions, a.budget))
            tl[rec] = out
        tl["_spent"] = spent
        tcache.write_text(json.dumps(tl))
    thresholds = {}
    for q in questions:
        sc = [s[q["id"]] for rec in a.calib for s in data[rec]["scores"]]
        te = [x[q["id"]] >= 0.5 for rec in a.calib for x in tl[rec]]
        thresholds[q["id"]] = best_threshold(sc, te)
    # label mapping for evaluation (step questions only)
    vocab = sorted({x["label"] for rec in data for x in data[rec]["labels"]["actions"]})
    key = Path(a.brain_key_file).read_text().strip()
    steps = [q for q in questions if q.get("kind") != "mistake"]
    mapping = map_to_labels(steps, vocab, a.brain_url, a.brain_model, key)
    rows = []
    for q in questions:
        k, th = q["id"], thresholds[q["id"]][0]
        pv, nv, fires, hits, caught, events, inwin, total = [], [], 0, 0, 0, 0, 0, 0
        for rec in a.test:
            d = data[rec]
            sp, sc, end = d["spans"], d["scores"], d["spans"][-1][1]
            if q.get("kind") == "mistake":
                wins = [(x["start"], x["end"]) for x in d["labels"]["actions"] if x["wrong"]]
            else:
                wins = [(x["start"], x["end"]) for x in d["labels"]["actions"] if x["label"] in mapping.get(k, [])]
            wins, negs = windows_and_negatives(wins, sp, end)
            pv += [wmax(sc, sp, k, w) for w in wins]
            nv += [wmax(sc, sp, k, w) for w in negs]
            e = EdgeTrigger(th, 4)
            f = [s for x, s in zip(sc, sp) if e.update(s[1], x[k])]
            hit = lambda s0, s1: any(s0 < b + 0.5 and s1 > x0 - 0.5 for x0, b in wins)
            fires += len(f)
            hits += sum(hit(*s) for s in f)
            caught += sum(1 for x0, b in wins if any(s0 < b + 0.5 and s1 > x0 - 0.5 for s0, s1 in f))
            events += len(wins)
            inwin += sum(hit(*s) for s in sp)
            total += len(sp)
        row = {"id": k, "kind": q.get("kind", "step"), "mapped_labels": mapping.get(k) if q.get("kind") != "mistake" else "wrong actions",
               "threshold": th, "teacher_f1": round(thresholds[k][1], 2), "test_events": events}
        if events >= 3 and nv:
            row.update(event_auc=round(auc(pv + nv, [True] * len(pv) + [False] * len(nv)), 3), fires=fires,
                       precision=round(hits / max(fires, 1), 2), random_precision=round(inwin / max(total, 1), 2),
                       caught=f"{caught}/{events}")
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    # alerts before expert corrections
    corr_total = corr_hit = 0
    alert_seconds = test_seconds = 0.0
    for rec in a.test:
        d = data[rec]
        sp, sc = d["spans"], d["scores"]
        alerts = []
        for q in questions:
            e = EdgeTrigger(thresholds[q["id"]][0], 4)
            alerts += [s[1] for x, s in zip(sc, sp) if e.update(s[1], x[q["id"]])]
        alerts = sorted(alerts)
        corr = d["labels"]["corrections"]
        corr_total += len(corr)
        corr_hit += sum(1 for c in corr if any(c - 10 <= t <= c for t in alerts))
        covered, last = 0.0, -1e9
        for t in alerts:                      # seconds of the timeline that sit within 10 s after some alert
            covered += min(10.0, t - last) if t - last > 0 else 0
            last = t
        alert_seconds += covered
        test_seconds += sp[-1][1]
    summary = {"questions": len(questions), "calib": a.calib, "test": a.test,
               "test_minutes": round(test_seconds / 60, 1), "judge_wall_s": round(judge_wall),
               "teacher_spend_usd": round(tl.get("_spent", 0.0), 4),
               "corrections": corr_total, "corrections_with_alert_in_prior_10s": corr_hit,
               "share": round(corr_hit / max(corr_total, 1), 2),
               "chance_share": round(alert_seconds / max(test_seconds, 1), 2)}
    print(json.dumps(summary))
    Path(a.out).write_text(json.dumps({"summary": summary, "rows": rows, "mapping": mapping,
                                       "thresholds": thresholds, "pack": pack}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--questions", required=True)
    p.add_argument("--calib", nargs="+", required=True)
    p.add_argument("--test", nargs="+", required=True)
    p.add_argument("--frames-root", default="/cwStorage/home/jijh/pulse_data/holoassist/frames")
    p.add_argument("--labels", default="/cwStorage/home/jijh/pulse_data/holoassist/labels.json")
    p.add_argument("--fps", type=float, default=4)
    p.add_argument("--djev-url", default="http://127.0.0.1:18831")
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--tmp", default="/tmp/jijh-pulse-autopack")
    p.add_argument("--budget", type=float, default=8.0)
    p.add_argument("--brain-url", default="http://127.0.0.1:8317/v1")
    p.add_argument("--brain-model", default="gpt-6-luna")
    p.add_argument("--brain-key-file", default="/home1/jijh/tools/cliproxyapi/local_api_key")
    p.add_argument("--out", default="runs/autopack_multi_atv.json")
    main(p.parse_args())
