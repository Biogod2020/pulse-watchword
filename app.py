"""Pulse MVP web app: replay a first-person video through the heartbeat and ask where things are.

    python app.py --frames /path/to/frames [--port 18900]
Open http://127.0.0.1:18900 . The brain is any OpenAI-compatible endpoint, set by environment:
PULSE_MODEL_BASE_URL / PULSE_MODEL / PULSE_MODEL_KEY_FILE (default: gpt-6-luna via CLIProxyAPI;
final build: muse-spark-1.3-contributor via the Meta Model API).

Every frame gets one fast judge read (the visual cue that scored best on real video). Questions go to the
brain, which calls find_object: a look-back scan of recent frames by the same judge (pulse/recall.py).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from pulse.heartbeat import Capability, Heartbeat
from pulse.memory import WorkingMemory
from pulse.providers import DjevDecider, OpenAICompatAgent, Question, Tool
from pulse.recall import where_is
from pulse.triggers import EdgeTrigger, StateTracker
from pulse.video import ChangeFilter, video_events

ROOT = Path(__file__).parent
app = FastAPI()
CFG: dict = {}
STATE: dict = {"clients": set(), "task": None}

# Activities the judge tracks reliably on real first-person video (eval_activity_cues.py, EPIC P12_101):
# washing AUC 0.82 (edge 86% precise), stirring AUC 0.89 (state), tap/knob AUC 0.80 (edge 58% precise).
# "Putting down" (AUC 0.62, below random as a trigger over the full video) is deliberately not tracked.
S = "Does this image itself show "
ACTIVITIES = {
    "washing": ("在水槽洗东西", Question("washing", S + "hands washing or rinsing something under running water at a "
                                                     "sink?", kind="noul"), lambda: EdgeTrigger(0.9, 5)),
    "stirring": ("在锅里搅拌", Question("stirring", S + "someone stirring food in a pot or pan with a spoon or "
                                                     "spatula?", kind="noul"), lambda: StateTracker(0.95, 0.7, 3)),
    "tap_or_knob": ("拧水龙头/灶台旋钮", Question("tap_or_knob", S + "a hand turning a tap, a stove knob or a "
                                                              "switch?", kind="noul"), lambda: EdgeTrigger(0.5, 5)),
}


def model_agent() -> OpenAICompatAgent:
    base = os.environ.get("PULSE_MODEL_BASE_URL", "http://127.0.0.1:8317/v1")
    model = os.environ.get("PULSE_MODEL", "gpt-6-luna")
    key_file = os.environ.get("PULSE_MODEL_KEY_FILE", "/home1/jijh/tools/cliproxyapi/local_api_key")
    return OpenAICompatAgent(base, model, api_key=Path(key_file).read_text().strip(), name=model)


def build_runtime() -> None:
    djev = DjevDecider(CFG["djev_url"])

    async def nothing(ctx) -> dict:
        return {}

    # threshold 0 so every score is reported; the temporal triggers below decide what counts as an event
    cues = [Question(q.id, q.ask, kind="noul", threshold=0.0) for _, q, _ in ACTIVITIES.values()]
    hb = Heartbeat(djev, WorkingMemory(), [Capability("activities", cues, nothing, on=("vision",), kind="vision")],
                   [], lambda m, ev, a: "A first-person kitchen camera image.")
    STATE.update(djev=djev, hb=hb, seen=[], now=0.0, answers=[], activity_log=[],
                 trackers={k: make() for k, (_, _, make) in ACTIVITIES.items()},
                 counters={"sampled": 0, "judged": 0, "noticed": 0, "brain_calls": 0, "judge_reads": 0})


async def broadcast(msg: dict) -> None:
    for ws in list(STATE["clients"]):
        try:
            await ws.send_text(json.dumps(msg, ensure_ascii=False, default=str))
        except Exception:
            STATE["clients"].discard(ws)


async def replay(start: float, minutes: float, speed: float) -> None:
    frames = sorted(Path(CFG["frames"]).glob("f_*.jpg"))
    end = start + minutes * 60 if minutes else 1e9
    frames = [f for f in frames if start <= int(f.stem.split("_")[1]) - 1 < end]
    c = STATE["counters"]
    flt = ChangeFilter()
    for ev, path, t, diff in video_events(frames, 1.0, 0.0, flt):
        loop_start = asyncio.get_running_loop().time()
        c["sampled"] += 1
        STATE["seen"].append((t, str(path)))
        STATE["now"] = t
        await broadcast({"type": "frame", "t": t, "frame": path.name, "judged": ev is not None})
        if ev:
            tick = await STATE["hb"].on_event(ev)
            c["judged"] += 1
            c["judge_reads"] += 1
            probs = {a["question"]: a["p_yes"] for a in tick["asked"]}
            events = []
            for k, tracker in STATE["trackers"].items():
                kind = tracker.update(t, probs.get(k) or 0.0)
                if kind:
                    entry = {"activity": k, "label": ACTIVITIES[k][0], "kind": kind, "t": t, "frame": path.stem}
                    STATE["activity_log"].append(entry)
                    events.append(entry)
                    if kind != "stop":
                        c["noticed"] += 1
            await broadcast({"type": "tick", "t": t, "frame": path.name, "probs": probs, "events": events,
                             "active": {k: getattr(tr, "active", False) for k, tr in STATE["trackers"].items()},
                             "read_ms": round(tick["read_ms"]), "counters": c})
        elapsed = asyncio.get_running_loop().time() - loop_start
        await asyncio.sleep(max(0.0, 1.0 / speed - elapsed))
    await broadcast({"type": "done", "counters": c})


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
        s = await where_is(noun, STATE["seen"], STATE["now"], STATE["djev"], CFG["djev_url"], window_s=req.window_s)
        STATE["counters"]["judge_reads"] += s.frames_scanned
        out = {"object": noun, "found": s.found, "place": s.place, "seen_at_s": s.t, "frame": s.frame,
               "confidence": s.visible_p, "frames_scanned": s.frames_scanned, "now_s": STATE["now"]}
        found.append(out)
        return out

    async def recent_activities(args: dict) -> dict:
        kind = str(args.get("activity", "")).strip()
        log = [e for e in STATE["activity_log"] if not kind or e["activity"] == kind]
        return {"now_s": STATE["now"], "events": [{**e, "ago_s": round(STATE["now"] - e["t"])} for e in log[-12:]]}

    activity_tool = Tool("recent_activities", "What the camera noticed the user doing: washing at the sink, stirring "
                         "a pot (start/stop), turning a tap or stove knob. Optional filter: washing, stirring or "
                         "tap_or_knob.", {"type": "object", "properties": {"activity": {"type": "string"}}},
                         recent_activities)
    tool = Tool("find_object", "Look back through the last minutes of camera frames for an object; returns where it "
                "was last seen resting and when (seconds into the recording).",
                {"type": "object", "properties": {"object": {"type": "string",
                                                             "description": "short common noun, e.g. knife"}},
                 "required": ["object"]}, find_object)
    STATE["counters"]["brain_calls"] += 1
    agent = model_agent()
    run = await agent.run("You are an always-on assistant that sees through the user's first-person camera. For "
                          "questions about where something is, call find_object with a short noun. Answer in one short "
                          "sentence: the place and how long ago (now_s minus seen_at_s, in seconds). If not found, say "
                          "you did not see it in the last minutes. For questions about what the user did (washing, "
                          "stirring, turning the stove or tap), call recent_activities and say how long ago.",
                          req.question, [tool, activity_tool])
    hit = next((f for f in reversed(found) if f["found"]), found[-1] if found else None)
    answer = {"answer": run.text.strip(), "evidence": hit, "model": agent.name, "ms": round(run.latency_ms),
              "counters": STATE["counters"]}
    STATE["answers"].append(answer)
    return JSONResponse(answer)


@app.get("/frames/{name}")
async def frame(name: str):
    path = Path(CFG["frames"]) / Path(name).name
    return FileResponse(path if path.suffix else path.with_suffix(".jpg"))


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
    p.add_argument("--djev-url", default="http://127.0.0.1:18811")
    p.add_argument("--port", type=int, default=18900)
    a = p.parse_args()
    CFG.update(frames=a.frames, djev_url=a.djev_url)
    build_runtime()
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
