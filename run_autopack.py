"""End-to-end check of an automatic scenario pack on a HoloAssist recording (no hand tuning, no human labels).

    python run_autopack.py --recording R200-7Nov-ATV --questions runs/autopack_atv_questions.json
Steps: 4 fps frames -> 2 s clips every 1 s judged by djev (video) with the drafted questions -> the brain maps
questions to the recording's action vocabulary (evaluation only) -> a teacher (gpt-6-luna Decisions on 3 frames
per clip) labels the first `--calib-min` minutes -> per-question thresholds -> evaluation on the rest against
human labels: event AUC per question, edge-trigger precision/recall, and whether any alert fires before the
instructor corrects a wrong action.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import statistics
import subprocess
import time
from pathlib import Path

import httpx

from eval_activity_cues import auc
from pulse.autopack import best_threshold, map_to_labels
from pulse.providers import LunaDecider, Question, image_url
from pulse.triggers import EdgeTrigger

FFMPEG = "/home1/jijh/micromamba/envs/imgprocess/bin/ffmpeg"


def recording_labels(labels_path: str, name: str) -> dict:
    rec = next(r for r in json.load(open(labels_path)) if r["video_name"] == name)
    fine = [e for e in rec["events"] if e["label"] == "Fine grained action"]
    acts = [{"label": f"{e['attributes']['Verb']} {e['attributes']['Noun']}".strip(), "start": e["start"],
             "end": e["end"], "wrong": e["attributes"].get("Action Correctness") != "Correct Action",
             "why": e["attributes"].get("Incorrect Action Explanation")} for e in fine]
    corrections = [e["start"] for e in rec["events"] if e["label"] == "Conversation"
                   and "correct the wrong" in e["attributes"].get("Conversation Purpose", "")]
    return {"task": rec["taskType"], "actions": acts, "corrections": corrections}


async def judge_clips(frames: list[Path], fps: float, questions: list[dict], url: str, tmp: Path, conc: int):
    sem, client = asyncio.Semaphore(conc), httpx.AsyncClient(timeout=180, trust_env=False)
    per, qs = int(2 * fps), {q["id"]: {"type": "noul", "instructions": q["ask"]} for q in questions}
    starts = list(range(0, len(frames) - per + 1, int(fps)))

    async def one(i):
        clip = tmp / f"c_{i:06d}.mp4"
        if not clip.exists():
            listing = clip.with_suffix(".txt")
            listing.write_text("".join(f"file '{f.resolve()}'\nduration {1/fps:.4f}\n" for f in frames[i:i + per]))
            await asyncio.to_thread(subprocess.run, [FFMPEG, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i",
                                    str(listing), "-vf", f"fps={fps},scale=448:-2", "-c:v", "libx264", "-preset",
                                    "veryfast", "-crf", "30", "-pix_fmt", "yuv420p", str(clip)], check=True)
        data = "data:video/mp4;base64," + base64.b64encode(clip.read_bytes()).decode()
        async with sem:
            r = await client.post(url + "/v1/systemone", json={"model": "dgemma", "samples": 1, "questions": qs,
                                  "state": "A 2-second first-person video of a person doing a hands-on task.",
                                  "videos": [data]})
        r.raise_for_status()
        return {k: v["noul"] for k, v in r.json()["answers"].items()}
    tmp.mkdir(parents=True, exist_ok=True)
    scores = await asyncio.gather(*[one(i) for i in starts])
    spans = [(i / fps, i / fps + 2) for i in starts]
    return scores, spans


async def teacher(frames: list[Path], fps: float, spans, idx, questions, budget: float):
    luna, sem = LunaDecider(budget_usd=budget), asyncio.Semaphore(8)
    qs = [Question(q["id"], q["ask"].replace("Does this video itself show", "Do these three frames, taken over 2 "
                                             "seconds, show"), kind="noul") for q in questions]

    async def one(i):
        s0 = int(spans[i][0] * fps)
        picks = [frames[min(s0 + k, len(frames) - 1)] for k in (0, int(fps), int(2 * fps) - 1)]
        async with sem:
            b = await luna.decide("Three frames from a 2-second first-person video of a hands-on task, in order.",
                                  qs, [image_url(str(p)) for p in picks])
        return {k: d.probs.get("yes", 0.0) for k, d in b.decisions.items()}
    out = await asyncio.gather(*[one(i) for i in idx])
    return out, luna.spent


def main(a):
    labels = recording_labels(a.labels, a.recording)
    frames = sorted(Path(a.frames).glob("f_*.jpg"))
    pack = json.loads(Path(a.questions).read_text())
    questions = pack["questions"]
    cache = Path(f"runs/autopack_{a.recording}_clipscores.json")
    t0 = time.perf_counter()
    if cache.exists():
        scores, spans = json.loads(cache.read_text())
    else:
        scores, spans = asyncio.run(judge_clips(frames, a.fps, questions, a.djev_url, Path(a.tmp), a.concurrency))
        cache.write_text(json.dumps([scores, spans]))
    judge_s = time.perf_counter() - t0
    key = Path(a.brain_key_file).read_text().strip()
    vocab = sorted({x["label"] for x in labels["actions"]})
    mapping = map_to_labels(questions, vocab, a.brain_url, a.brain_model, key)
    calib = [i for i, (s0, _) in enumerate(spans) if s0 < a.calib_min * 60]
    test = [i for i, (s0, _) in enumerate(spans) if s0 >= a.calib_min * 60]
    tcache = Path(f"runs/autopack_{a.recording}_teacher.json")
    if tcache.exists():
        tlabels, spent = json.loads(tcache.read_text())
    else:
        tlabels, spent = asyncio.run(teacher(frames, a.fps, spans, calib, questions, a.budget))
        tcache.write_text(json.dumps([tlabels, spent]))
    report = {"recording": a.recording, "task": labels["task"], "scenario": pack["scenario"], "clips": len(spans),
              "judge_wall_s": round(judge_s), "calibration_clips": len(calib), "test_clips": len(test),
              "teacher_spend_usd": round(spent, 4), "questions": []}
    end = spans[-1][1]
    for q in questions:
        k = q["id"]
        th, f1 = best_threshold([scores[i][k] for i in calib], [tlabels[j][k] >= 0.5 for j in range(len(calib))])
        wins = [(x["start"], x["end"]) for x in labels["actions"] if x["label"] in mapping.get(k, [])
                and x["start"] >= a.calib_min * 60]
        row = {"id": k, "ask": q["ask"], "mapped_labels": mapping.get(k, []), "threshold": th,
               "teacher_agreement_f1": round(f1, 2), "test_events": len(wins)}
        if len(wins) >= 2:
            length = statistics.median(b - x for x, b in wins) + 1.0
            busy = [(x - 1, b + 1) for x, b in wins]
            negs, t = [], a.calib_min * 60
            while t + length <= end:
                if not any(t < b and t + length > x for x, b in busy):
                    negs.append((t, t + length))
                t += length
            wmax = lambda w: max([scores[i][k] for i in test if spans[i][0] < w[1] + 0.5 and spans[i][1] > w[0] - 0.5] or [0])
            pv, nv = [wmax(w) for w in wins], [wmax(w) for w in negs]
            row["event_auc"] = round(auc(pv + nv, [True] * len(pv) + [False] * len(nv)), 3)
            e = EdgeTrigger(th, 4)
            fires = [spans[i] for i in test if e.update(spans[i][1], scores[i][k])]
            hit = lambda s0, s1: any(s0 < b + 0.5 and s1 > x - 0.5 for x, b in wins)
            base = sum(hit(*spans[i]) for i in test) / len(test)
            row.update(fires=len(fires), precision=round(sum(hit(*f) for f in fires) / max(len(fires), 1), 2),
                       random_precision=round(base, 2),
                       caught=f"{sum(1 for x, b in wins if any(s0 < b + 0.5 and s1 > x - 0.5 for s0, s1 in fires))}/{len(wins)}")
        report["questions"].append(row)
        print(json.dumps(row, ensure_ascii=False))
    # alerts before expert corrections: any question firing in the 10 s before a correction, vs random 10 s windows
    alerts = []
    for q in report["questions"]:
        e = EdgeTrigger(q["threshold"], 4)
        alerts += [spans[i][1] for i in test if e.update(spans[i][1], scores[i][q["id"]])]
    corr = [c for c in labels["corrections"] if c >= a.calib_min * 60]
    wrong = [x for x in labels["actions"] if x["wrong"] and x["start"] >= a.calib_min * 60]
    before = sum(1 for c in corr if any(c - 10 <= t <= c for t in alerts))
    test_span = end - a.calib_min * 60
    report["expert_corrections"] = {"in_test_part": len(corr), "with_an_alert_in_prior_10s": before,
                                    "alerts_total": len(alerts),
                                    "chance_rate": round(min(1.0, len(alerts) * 10 / test_span), 2),
                                    "wrong_actions_in_test_part": len(wrong)}
    print(json.dumps(report["expert_corrections"]))
    Path(a.out).write_text(json.dumps({"report": report, "mapping": mapping}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--recording", default="R200-7Nov-ATV")
    p.add_argument("--frames", default="/cwStorage/home/jijh/pulse_data/holoassist/frames/R200-7Nov-ATV")
    p.add_argument("--labels", default="/cwStorage/home/jijh/pulse_data/holoassist/labels.json")
    p.add_argument("--questions", default="runs/autopack_atv_questions.json")
    p.add_argument("--fps", type=float, default=4)
    p.add_argument("--calib-min", type=float, default=3)
    p.add_argument("--djev-url", default="http://127.0.0.1:18831")
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--tmp", default="/tmp/jijh-pulse-autopack")
    p.add_argument("--budget", type=float, default=8.0)
    p.add_argument("--brain-url", default="http://127.0.0.1:8317/v1")
    p.add_argument("--brain-model", default="gpt-6-luna")
    p.add_argument("--brain-key-file", default="/home1/jijh/tools/cliproxyapi/local_api_key")
    p.add_argument("--out", default="runs/autopack_R200.json")
    main(p.parse_args())
