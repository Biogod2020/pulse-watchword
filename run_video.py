"""Validate the heartbeat on a real first-person video against EPIC-KITCHENS-100 narrations.

    python run_video.py --video-id P12_101 --frames /path/to/frames [--minutes 10]

Measures (all against human narrations, which are dense but not exhaustive):
- wake rate: frames sampled -> frames kept by the change filter -> frames that woke a capability
- trigger recall: annotated put/place actions with a heartbeat trigger inside [start-2s, stop+2s]
- trigger precision: triggers that land inside some annotated put/place window (+-2s)
- "where is X?": 20 s after each located put, look up the object; correct if the remembered fact comes from
  that same put and the remembered place matches the narrated place (place match graded by an LLM, saved).
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import time
from pathlib import Path

import httpx

from pulse.heartbeat import Heartbeat
from pulse.longterm import LongTermMemory, RemoteEmbedder
from pulse.memory import WorkingMemory
from pulse.packs.objects import context, item_locations, make_remember, norm, pack
from pulse.providers import DjevDecider
from pulse.store import Store
from pulse.video import ChangeFilter, frame_time, video_events

PUT_VERBS = {"put", "put-down", "place"}
PREP = re.compile(r"\b(?:back )?(?:on|in|into|onto|inside|under|to)\b (?:the )?(.+)$")


def ts(s: str) -> float:
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def load_puts(csv_path: str, video_id: str, t0: float, t1: float) -> list[dict]:
    out = []
    for r in csv.DictReader(open(csv_path)):
        if r["video_id"] != video_id or r["verb"] not in PUT_VERBS:
            continue
        a, b = ts(r["start_timestamp"]), ts(r["stop_timestamp"])
        if a < t0 or b > t1:
            continue
        m = PREP.search(r["narration"])
        out.append({"id": r["narration_id"], "narration": r["narration"], "start": a, "stop": b,
                    "noun": norm(r["noun"].split(":")[0]), "noun_raw": r["noun"],
                    "place": m.group(1) if m else None})
    return sorted(out, key=lambda x: x["start"])


def find_fact(items: dict, noun: str) -> tuple[str, dict] | None:
    if noun in items:
        return noun, items[noun]
    for k, v in items.items():          # loose head-word match: "frying pan" ~ "pan"
        if noun in k.split() or k in noun.split():
            return k, v
    return None


async def grade_places(pairs: list[dict], url: str, model: str, key: str) -> list[bool | None]:
    if not pairs:
        return []
    listing = "\n".join(f"{i}. narrated: '{p['gt']}' | remembered: '{p['pred']}'" for i, p in enumerate(pairs))
    body = {"model": model, "temperature": 0, "messages": [
        {"role": "system", "content": "You grade whether two short place descriptions in a kitchen refer to the "
                                      "same place (e.g. 'board' and 'chopping board' = same; 'sink' and 'counter' = "
                                      "different). Reply with a JSON array of true/false, one per line, nothing else."},
        {"role": "user", "content": listing}]}
    async with httpx.AsyncClient(timeout=120, trust_env=False, headers={"Authorization": f"Bearer {key}"}) as c:
        r = await c.post(url.rstrip("/") + "/chat/completions", json=body)
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    arr = json.loads(text[text.find("["): text.rfind("]") + 1])
    return [bool(x) for x in arr] + [None] * (len(pairs) - len(arr))


async def main(a):
    frames = sorted(Path(a.frames).glob("f_*.jpg"))
    t_end = a.start + a.minutes * 60 if a.minutes else 1e9
    frames = [f for f in frames if frame_time(f, a.fps, a.start) < t_end]
    t1 = frame_time(frames[-1], a.fps, a.start) + 1
    puts = load_puts(a.annotations, a.video_id, a.start, t1)
    store = Store(a.store) if a.store else None
    if store and a.fresh:
        store.wipe()
        store = Store(a.store)
    memory = WorkingMemory(store=store)
    ltm = LongTermMemory(RemoteEmbedder(a.embed_url) if a.embed_url else None, store=store)
    djev = DjevDecider(a.djev_url)
    hb = Heartbeat(djev, memory, pack(a.djev_url, store), [], context,
                   remember=make_remember(ltm) if a.embed_url else None, background_remember=True)

    queries = [{"put": p, "at": p["stop"] + a.query_delay} for p in puts if p["place"]]
    qi, records, started = 0, [], time.perf_counter()
    for ev, path, t, diff in video_events(frames, a.fps, a.start, ChangeFilter(a.threshold, a.floor)):
        rec = {"t": t, "frame": path.stem, "diff": round(diff, 2), "kept": ev is not None}
        if ev:
            tick = await hb.on_event(ev)
            q = tick["asked"][0] if tick["asked"] else {}
            rec.update(p_put=q.get("p_yes"), fired=bool(q.get("fired")), read_ms=round(tick["read_ms"]),
                       result=tick["results"].get("note_put"), error=tick["decider_error"])
        records.append(rec)
        while qi < len(queries) and queries[qi]["at"] <= t:
            q = queries[qi]
            hit = find_fact(item_locations(memory), q["put"]["noun"])
            q["answer"] = None if not hit else {"key": hit[0], **{k: hit[1][k] for k in ("place", "t", "frame", "raw")}}
            qi += 1
        if len(records) % 200 == 0:
            kept = sum(r["kept"] for r in records)
            print(f"t={t:6.0f}s frames={len(records)} kept={kept} fired={sum(r.get('fired', False) for r in records)} "
                  f"elapsed={time.perf_counter()-started:.0f}s", flush=True)
    await hb.drain()

    # ---- metrics -------------------------------------------------------------------------------
    fired = [r for r in records if r.get("fired")]
    win = 2.0
    caught = [p for p in puts if any(p["start"] - win <= r["t"] <= p["stop"] + win for r in fired)]
    on_put = [r for r in fired if any(p["start"] - win <= r["t"] <= p["stop"] + win for p in puts)]
    for q in queries:
        ans, p = q.get("answer"), q["put"]
        q["same_event"] = bool(ans and p["start"] - win <= ans["t"] <= p["stop"] + win)
    graded = [q for q in queries if q["same_event"]]
    marks = await grade_places([{"gt": q["put"]["place"], "pred": q["answer"]["place"]} for q in graded],
                               a.grader_url, a.grader_model, Path(a.grader_key_file).read_text().strip())
    for q, m in zip(graded, marks):
        q["place_match"] = m
    kept = [r for r in records if r["kept"]]
    reads = sorted(r["read_ms"] for r in kept if r.get("read_ms"))
    summary = {
        "video": a.video_id, "seconds": round(t1 - a.start), "frames_sampled": len(records), "frames_kept": len(kept),
        "frames_fired": len(fired), "keep_rate": round(len(kept) / len(records), 3),
        "fire_rate_of_sampled": round(len(fired) / len(records), 3),
        "annotated_puts": len(puts), "puts_caught": len(caught), "trigger_recall": round(len(caught) / max(len(puts), 1), 3),
        "trigger_precision": round(len(on_put) / max(len(fired), 1), 3),
        "queries": len(queries), "answered": sum(1 for q in queries if q.get("answer")),
        "answer_from_same_put": len(graded), "place_correct": sum(1 for q in graded if q.get("place_match")),
        "read_ms_p50": reads[len(reads) // 2] if reads else None, "read_ms_p95": reads[int(len(reads) * .95) - 1] if reads else None,
        "wall_s": round(time.perf_counter() - started), "storage": store.usage() if store else None,
        "remember_errors": len(hb.remember_errors),
        "settings": {"fps": a.fps, "threshold": a.threshold, "floor_s": a.floor, "query_delay_s": a.query_delay,
                     "match_window_s": win, "grader": a.grader_model},
        "caveats": ["EPIC narrations are dense but not exhaustive: some correct triggers land on unlabelled puts, "
                    "so precision is a lower bound", "Place match is graded by an LLM; per-query grades are saved"]}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "queries": queries, "puts": puts, "records": records},
                              ensure_ascii=False, indent=1, default=str))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video-id", default="P12_101")
    p.add_argument("--frames", required=True)
    p.add_argument("--annotations", default="eval_data/EPIC_100_P12_101.csv")
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--minutes", type=float, default=0.0)
    p.add_argument("--threshold", type=float, default=6.0)
    p.add_argument("--floor", type=float, default=30.0)
    p.add_argument("--query-delay", type=float, default=20.0)
    p.add_argument("--djev-url", default="http://127.0.0.1:18811")
    p.add_argument("--embed-url", default="")
    p.add_argument("--store", default="")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--grader-url", default="http://127.0.0.1:8317/v1")
    p.add_argument("--grader-model", default="gpt-6-luna")
    p.add_argument("--grader-key-file", default="/home1/jijh/tools/cliproxyapi/local_api_key")
    p.add_argument("--out", default="runs/video_eval.json")
    asyncio.run(main(p.parse_args()))
