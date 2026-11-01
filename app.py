"""Pulse MVP web app: replay a first-person video through the live heartbeat and ask where things are.

    python app.py --frames /path/to/frames --store /path/to/store [--port 18900]
Open http://127.0.0.1:18900 . The model ("brain") is any OpenAI-compatible endpoint: gpt-6-luna via
CLIProxyAPI during development, muse-spark-1.3-contributor via the Meta Model API for the final build.
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

from pulse.heartbeat import Heartbeat
from pulse.longterm import LongTermMemory, RemoteEmbedder
from pulse.memory import WorkingMemory
from pulse.packs.objects import context, item_locations, make_remember, pack
from pulse.providers import DjevDecider, OpenAICompatAgent, Tool
from pulse.store import Store
from pulse.video import ChangeFilter, video_events

ROOT = Path(__file__).parent
app = FastAPI()
CFG: dict = {}
STATE: dict = {"clients": set(), "task": None}


def model_agent() -> OpenAICompatAgent:
    """One switch for the brain: PULSE_MODEL_BASE_URL / PULSE_MODEL / PULSE_MODEL_KEY_FILE."""
    base = os.environ.get("PULSE_MODEL_BASE_URL", "http://127.0.0.1:8317/v1")
    model = os.environ.get("PULSE_MODEL", "gpt-6-luna")
    key_file = os.environ.get("PULSE_MODEL_KEY_FILE", "/home1/jijh/tools/cliproxyapi/local_api_key")
    return OpenAICompatAgent(base, model, api_key=Path(key_file).read_text().strip(), name=model)


def build_runtime(fresh: bool) -> None:
    store = Store(CFG["store"])
    if fresh:
        store.wipe()
        store = Store(CFG["store"])
    memory = WorkingMemory(store=store)
    ltm = LongTermMemory(RemoteEmbedder(CFG["embed_url"]) if CFG["embed_url"] else None, store=store)
    hb = Heartbeat(DjevDecider(CFG["djev_url"]), memory, pack(CFG["djev_url"], store), [], context,
                   remember=make_remember(ltm) if CFG["embed_url"] else None, background_remember=True)
    STATE.update(store=store, memory=memory, ltm=ltm, hb=hb,
                 counters={"sampled": 0, "kept": 0, "fired": 0, "brain_calls": 0})


async def broadcast(msg: dict) -> None:
    dead = []
    for ws in STATE["clients"]:
        try:
            await ws.send_text(json.dumps(msg, ensure_ascii=False, default=str))
        except Exception:
            dead.append(ws)
    for ws in dead:
        STATE["clients"].discard(ws)


def memory_view() -> dict:
    items = sorted(item_locations(STATE["memory"]).items(), key=lambda kv: -kv[1]["t"])
    return {"type": "memory", "items": [{"object": k, **v} for k, v in items],
            "usage": STATE["store"].usage(), "counters": STATE["counters"]}


async def replay(start: float, minutes: float, speed: float) -> None:
    frames = sorted(Path(CFG["frames"]).glob("f_*.jpg"))
    end = start + minutes * 60 if minutes else 1e9
    frames = [f for f in frames if start <= int(f.stem.split("_")[1]) - 1 < end]
    c = STATE["counters"]
    for ev, path, t, diff in video_events(frames, 1.0, 0.0, ChangeFilter()):
        c["sampled"] += 1
        await broadcast({"type": "frame", "t": t, "frame": path.name, "kept": ev is not None, "diff": round(diff, 1)})
        if ev:
            c["kept"] += 1
            tick = await STATE["hb"].on_event(ev)
            q = tick["asked"][0] if tick["asked"] else {}
            if q.get("fired"):
                c["fired"] += 1
            await broadcast({"type": "tick", "t": t, "frame": path.name, "p": q.get("p_yes"), "fired": q.get("fired"),
                             "read_ms": round(tick["read_ms"]), "result": tick["results"].get("note_put")})
            if q.get("fired"):
                await broadcast(memory_view())
        await asyncio.sleep(1.0 / speed)
    await STATE["hb"].drain()
    await broadcast(memory_view())
    await broadcast({"type": "done"})


class ReplayIn(BaseModel):
    start: float = 0
    minutes: float = 0
    speed: float = 4
    fresh: bool = True


@app.post("/api/replay")
async def start_replay(req: ReplayIn):
    if STATE.get("task") and not STATE["task"].done():
        STATE["task"].cancel()
    build_runtime(req.fresh)
    STATE["task"] = asyncio.create_task(replay(req.start, req.minutes, req.speed))
    return {"ok": True}


class AskIn(BaseModel):
    question: str


@app.post("/api/ask")
async def ask(req: AskIn):
    memory, ltm = STATE["memory"], STATE["ltm"]
    seen: list[dict] = []

    async def items(_: dict) -> dict:
        out = [{"object": k, "place": v["place"], "seen_at_s": round(v["t"]), "frame": v["frame"]}
               for k, v in item_locations(memory).items()]
        seen.extend(out)
        return {"last_seen": out}

    async def search(args: dict) -> dict:
        hits = await ltm.search(str(args.get("query", "")), k=3, kind="frame")
        return {"frames": [{"frame": h["id"], "seen_at_s": round(h["t"]), "score": h["score"]} for h in hits]}

    tools = [Tool("item_locations", "Where each object was last seen being put down.",
                  {"type": "object", "properties": {}}, items),
             Tool("search_frames", "Search remembered camera frames by meaning.",
                  {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}, search)]
    STATE["counters"]["brain_calls"] += 1
    run = await model_agent().run(
        "You are an always-on assistant answering from what the user's camera saw. Use the tools. Answer in one "
        "short sentence with the place and roughly when (mm:ss). Then on a new line write FRAME=<frame id> for the "
        "frame that shows it, or FRAME=none.", req.question, tools)
    text, frame = run.text, None
    if "FRAME=" in text:
        text, _, frame = text.rpartition("FRAME=")
        frame = frame.strip().split()[0] if frame.strip() else None
        frame = None if frame in (None, "none") else frame
    return JSONResponse({"answer": text.strip(), "frame": frame, "tools": [c["tool"] for c in run.tool_calls],
                         "model": run and model_agent().name, "ms": round(run.latency_ms)})


@app.get("/frames/{name}")
async def frame(name: str):
    path = Path(CFG["frames"]) / Path(name).name
    if not path.suffix:
        path = path.with_suffix(".jpg")
    return FileResponse(path)


@app.get("/")
async def index():
    return HTMLResponse((ROOT / "web/index.html").read_text())


@app.websocket("/ws")
async def ws(socket: WebSocket):
    await socket.accept()
    STATE["clients"].add(socket)
    if "memory" in STATE:
        await socket.send_text(json.dumps(memory_view(), default=str))
    try:
        while True:
            await socket.receive_text()
    except WebSocketDisconnect:
        STATE["clients"].discard(socket)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--frames", required=True)
    p.add_argument("--store", default="/cwStorage/home/jijh/pulse_data/store_app")
    p.add_argument("--djev-url", default="http://127.0.0.1:18811")
    p.add_argument("--embed-url", default="http://127.0.0.1:18820")
    p.add_argument("--port", type=int, default=18900)
    a = p.parse_args()
    CFG.update(frames=a.frames, store=a.store, djev_url=a.djev_url, embed_url=a.embed_url)
    build_runtime(fresh=False)
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
