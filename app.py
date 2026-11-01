"""Pulse MVP web app: a first-person video stream judged as short clips; only meaningful frames are stored.

    python app.py --frames /path/to/4fps_frames [--fps 4] [--port 18900]
Open http://127.0.0.1:18900 .

Every captured frame goes into an in-memory ring (the last 3 minutes). Each second, the newest 2 seconds are
encoded as a clip and read once by the video-capable judge (djev) with three activity questions; online
triggers turn the scores into events. Only evidence frames (something noticed, or used in an answer) and one
keyframe every 30 s are written to disk. "Where is X?" scans the ring first, then stored keyframes.
The brain is any OpenAI-compatible endpoint: PULSE_MODEL_BASE_URL / PULSE_MODEL / PULSE_MODEL_KEY_FILE
(default gpt-6-luna via CLIProxyAPI; final build muse-spark-1.3-contributor via the Meta Model API).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import deque
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from pulse.longterm import Episode
from pulse.providers import DjevDecider, LunaDecider, OpenAICompatAgent, Question, Tool
from pulse.recall import where_is
from pulse.store import Store
from pulse.stream import FrameRing, KeepPolicy, make_clip
from pulse.triggers import EdgeTrigger, StateTracker

ROOT = Path(__file__).parent
app = FastAPI()
CFG: dict = {}
STATE: dict = {"clients": set(), "task": None}

# Validated on EPIC P12_101 with 2 s clips every 1 s (eval_video_windows.py; thresholds tuned on minutes 0-10):
# washing event AUC 0.98 (edge 0.2: 7/7 correct), stirring 1.00 (state 0.95/0.7), tap/knob 0.84 (edge 0.4: 40%).
V = "Does this video itself show "
ACTIVITIES = {
    "washing": ("在水槽洗东西", V + "hands washing or rinsing something under running water at a sink?",
                lambda: EdgeTrigger(0.2, 4)),
    "stirring": ("在锅里搅拌", V + "someone stirring food in a pot or pan with a spoon or spatula?",
                 lambda: StateTracker(0.95, 0.7, 2)),
    "tap_or_knob": ("拧水龙头/灶台旋钮", V + "a hand turning a tap, a stove knob or a switch?",
                    lambda: EdgeTrigger(0.4, 4)),
}
QUESTIONS = [Question(k, q, kind="noul") for k, (_, q, _) in ACTIVITIES.items()]


def model_agent() -> OpenAICompatAgent:
    base = os.environ.get("PULSE_MODEL_BASE_URL", "http://127.0.0.1:8317/v1")
    model = os.environ.get("PULSE_MODEL", "gpt-6-luna")
    key_file = os.environ.get("PULSE_MODEL_KEY_FILE", "/home1/jijh/tools/cliproxyapi/local_api_key")
    return OpenAICompatAgent(base, model, api_key=Path(key_file).read_text().strip(), name=model)


def build_runtime() -> None:
    # Each replay gets a fresh store directory; the previous run's connection is closed first (on NFS an open
    # SQLite file blocks deleting its directory).
    if STATE.get("store") is not None:
        STATE["store"].db.close()
    import time
    store = Store(str(Path(CFG["store"]) / f"run_{int(time.time())}"))
    video_judge = DjevDecider(CFG["video_djev_url"])
    frame_judge = DjevDecider(CFG["djev_url"])
    STATE.update(store=store, video_judge=video_judge,
                 recall_judge=LunaDecider() if CFG.get("recall_judge") == "luna" else frame_judge,
                 ring=FrameRing(180), keep=KeepPolicy(30), now=0.0, activity_log=[], stored=[],
                 trackers={k: make() for k, (_, _, make) in ACTIVITIES.items()},
                 counters={"captured": 0, "clips": 0, "noticed": 0, "stored": 0, "brain_calls": 0, "judge_reads": 0})


async def broadcast(msg: dict) -> None:
    for ws in list(STATE["clients"]):
        try:
            await ws.send_text(json.dumps(msg, ensure_ascii=False, default=str))
        except Exception:
            STATE["clients"].discard(ws)


def store_frame(t: float, path: str, label: str) -> None:
    store = STATE["store"]
    store.save_episode(Episode(Path(path).stem, "frame", t, label, path), None, store.put_blob(path))
    STATE["stored"].append((t, str(store.blob_path(store.put_blob(path))), label))
    STATE["counters"]["stored"] += 1


async def judge_clip(t: float, frames: list[tuple[float, str]]) -> tuple[float, str, dict, float]:
    clip = Path(CFG["clips"]) / f"live_{t:08.2f}.mp4"
    url = await asyncio.to_thread(make_clip, frames, CFG["fps"], clip)
    clip.unlink(missing_ok=True)                      # clips are transient: judged once, never stored
    clip.with_suffix(".txt").unlink(missing_ok=True)
    b = await STATE["video_judge"].decide("A 2-second first-person kitchen video.", QUESTIONS, None, [url])
    return t, frames[len(frames) // 2][1], {k: d.probs["yes"] for k, d in b.decisions.items()}, b.latency_ms


def apply_result(t: float, mid: str, probs: dict, read_ms: float) -> dict:
    c = STATE["counters"]
    c["clips"] += 1
    c["judge_reads"] += 1
    events = []
    for k, tracker in STATE["trackers"].items():
        kind = tracker.update(t, probs.get(k, 0.0))
        if kind:
            entry = {"activity": k, "label": ACTIVITIES[k][0], "kind": kind, "t": t, "frame": Path(mid).name}
            STATE["activity_log"].append(entry)
            events.append(entry)
            if kind != "stop":
                c["noticed"] += 1
    noticed = any(e["kind"] != "stop" for e in events)
    why = STATE["keep"].decide(t, noticed)
    if why:
        store_frame(t, mid, ", ".join(e["label"] for e in events) or "关键帧")
    return {"type": "tick", "t": t, "frame": Path(mid).name, "probs": probs, "events": events, "stored": why,
            "active": {k: getattr(tr, "active", False) for k, tr in STATE["trackers"].items()},
            "read_ms": round(read_ms), "counters": c, "usage": STATE["store"].usage()}


async def replay(start: float, minutes: float, speed: float) -> None:
    fps = CFG["fps"]
    frames = sorted(Path(CFG["frames"]).glob("f_*.jpg"))
    t_of = lambda f: (int(f.stem.split("_")[1]) - 1) / fps
    end = start + minutes * 60 if minutes else 1e9
    frames = [f for f in frames if start <= t_of(f) < end]
    ring, c = STATE["ring"], STATE["counters"]
    inflight: deque = deque()
    loop = asyncio.get_running_loop()
    t_start = loop.time()
    for i, f in enumerate(frames):
        t = t_of(f)
        ring.add(t, str(f))
        STATE["now"] = t
        c["captured"] += 1
        if i % 2 == 0:
            await broadcast({"type": "frame", "t": t, "frame": f.name})
        if i % int(fps) == int(fps) - 1 and len(ring.last(2.0)) >= 2 * fps - 1:
            inflight.append(asyncio.create_task(judge_clip(t, ring.last(2.0))))
        while inflight and (inflight[0].done() or len(inflight) > 6):
            await broadcast(apply_result(*(await inflight.popleft())))
        wait = t_start + (t - start) / speed - loop.time()
        if wait > 0:
            await asyncio.sleep(wait)
    while inflight:
        await broadcast(apply_result(*(await inflight.popleft())))
    await broadcast({"type": "done", "counters": c, "usage": STATE["store"].usage()})


class ReplayIn(BaseModel):
    start: float = 0
    minutes: float = 0
    speed: float = 4


@app.post("/api/replay")
async def start_replay(req: ReplayIn):
    if STATE.get("task") and not STATE["task"].done():
        STATE["task"].cancel()
    build_runtime()
    STATE["task"] = asyncio.create_task(replay(req.start, req.minutes, req.speed))
    return {"ok": True}


class AskIn(BaseModel):
    question: str
    window_s: float = 90


@app.post("/api/ask")
async def ask(req: AskIn):
    found: list[dict] = []

    async def find_object(args: dict) -> dict:
        noun = str(args.get("object", "")).strip().lower()
        recent = STATE["ring"].sampled(1.0)
        s = await where_is(noun, recent, STATE["now"], STATE["recall_judge"], CFG["djev_url"], window_s=req.window_s)
        scanned, source = s.frames_scanned, "最近画面"
        if not s.found and STATE["stored"]:
            older = [(t, p) for t, p, _ in STATE["stored"] if t < STATE["now"] - req.window_s]
            if older:
                s2 = await where_is(noun, older, STATE["now"], STATE["recall_judge"], CFG["djev_url"], window_s=1e9)
                scanned += s2.frames_scanned
                if s2.found:
                    s, source = s2, "存下的关键帧"
        STATE["counters"]["judge_reads"] += scanned
        frame = None
        if s.found:
            hit = next((p for t, p in recent if Path(p).stem == s.frame), None) or \
                next((p for t, p, _ in STATE["stored"] if Path(p).stem == s.frame), None)
            if hit:
                if not any(Path(p).stem == s.frame for _, p, _ in STATE["stored"]):
                    store_frame(s.t, hit, f"回答依据：{noun}")       # evidence for an answer is kept
                frame = Path(hit).name if Path(hit).parent == Path(CFG["frames"]) else "blob/" + Path(hit).name
        out = {"object": noun, "found": s.found, "place": s.place, "seen_at_s": s.t, "frame": frame,
               "confidence": s.visible_p, "frames_scanned": scanned, "source": source if s.found else None,
               "now_s": STATE["now"]}
        found.append(out)
        return out

    async def recent_activities(args: dict) -> dict:
        kind = str(args.get("activity", "")).strip()
        log = [e for e in STATE["activity_log"] if not kind or e["activity"] == kind]
        return {"now_s": STATE["now"], "events": [{**e, "ago_s": round(STATE["now"] - e["t"])} for e in log[-12:]]}

    tools = [Tool("find_object", "Look back through what the camera saw for an object; returns where it was last "
                  "seen resting and when (seconds into the recording).",
                  {"type": "object", "properties": {"object": {"type": "string", "description": "short noun"}},
                   "required": ["object"]}, find_object),
             Tool("recent_activities", "What the camera noticed the user doing: washing at the sink, stirring a pot "
                  "(start/stop), turning a tap or stove knob. Optional filter: washing, stirring or tap_or_knob.",
                  {"type": "object", "properties": {"activity": {"type": "string"}}}, recent_activities)]
    STATE["counters"]["brain_calls"] += 1
    agent = model_agent()
    run = await agent.run("You are an always-on assistant that sees through the user's first-person camera. For "
                          "where-is questions call find_object with a short noun; for questions about what the user "
                          "did call recent_activities. Answer in one short sentence with the place or activity and "
                          "how long ago (now_s minus the time, in seconds).", req.question, tools)
    hit = next((f for f in reversed(found) if f["found"]), found[-1] if found else None)
    return JSONResponse({"answer": run.text.strip(), "evidence": hit, "model": agent.name, "ms": round(run.latency_ms),
                         "counters": STATE["counters"], "usage": STATE["store"].usage()})


@app.get("/frames/{name}")
async def frame(name: str):
    return FileResponse(Path(CFG["frames"]) / Path(name).name)


@app.get("/frames/blob/{name}")
async def blob(name: str):
    sha = Path(name).stem
    return FileResponse(STATE["store"].blob_path(sha))


@app.get("/")
async def index():
    return HTMLResponse((ROOT / "web/index.html").read_text())


@app.websocket("/ws")
async def ws(socket: WebSocket):
    await socket.accept()
    STATE["clients"].add(socket)
    try:
        while True:
            await socket.receive_text()
    except WebSocketDisconnect:
        STATE["clients"].discard(socket)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--frames", required=True)
    p.add_argument("--fps", type=float, default=4)
    p.add_argument("--djev-url", default="http://127.0.0.1:18811", help="frame judge (recall, place)")
    p.add_argument("--video-djev-url", default="http://127.0.0.1:18831", help="video-capable judge for clips")
    p.add_argument("--store", default="/cwStorage/home/jijh/pulse_data/store_live")
    p.add_argument("--clips", default="/tmp/jijh-pulse-clips")
    p.add_argument("--port", type=int, default=18900)
    p.add_argument("--recall-judge", choices=["djev", "luna"], default="djev")
    a = p.parse_args()
    CFG.update(frames=a.frames, fps=a.fps, djev_url=a.djev_url, video_djev_url=a.video_djev_url, store=a.store,
               clips=a.clips, recall_judge=a.recall_judge)
    build_runtime()
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
