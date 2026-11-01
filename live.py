"""Pulse Live: your browser camera, watched by a fast judge; tell it what to watch for in plain words.

    python live.py [--port 18900]
Open http://localhost:18900 on the same machine, or forward the port over SSH (`ssh -L 18900:127.0.0.1:18900 host`)
so the page is served from localhost: browsers only allow camera access on https or localhost.

Flow: the page sends ~4 JPEG frames/s over a WebSocket -> in-memory ring (last 3 min, frames on a temp disk
dir, evicted ones deleted) -> every second the newest 2 s become one clip -> one djev video read answers every
active watch -> online triggers -> a "moment" (clip + shareable card) when one fires. Only moments, answer
evidence and a keyframe every 30 s are kept. The brain (any OpenAI-compatible model) turns plain-language
requests into watch questions and answers "where did I leave ...?" by scanning recent frames with the judge.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import struct
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

from pulse.autopack import DRAFT_SYSTEM, PREFIX, _chat, _json
from pulse.cards import moment_card, moment_reel
from pulse.providers import DjevDecider, LunaDecider, OpenAICompatAgent, Question, Tool, image_url
from pulse.recall import where_is
from pulse.stream import FrameRing, KeepPolicy, make_clip
import base64

import httpx

from pulse.triggers import ConfirmedEdge, StateTracker

ROOT = Path(__file__).parent
app = FastAPI()
CFG: dict = {}


@dataclass
class Watch:
    id: str
    label: str
    ask: str
    threshold: float = 0.6
    mode: str = "edge"          # edge: brief action | state: ongoing activity
    builtin: bool = False
    trigger: object = None
    last_p: float = 0.0
    fires: int = 0
    candidate: float = 0.5      # cheap-judge score that proposes a moment to the verifier
    said: str = ""              # the user's own words, shown on the card
    dismissed: int = 0

    def make_trigger(self):
        # With a verifier: one clip above the candidate threshold proposes, the verifier decides.
        # Without one: brief actions need two agreeing overlapping clips (suppresses one-clip false alarms).
        if self.mode == "state":
            self.trigger = StateTracker(self.threshold, max(0.3, self.threshold - 0.3), 2)
        elif CFG.get("verifier"):
            self.trigger = ConfirmedEdge(self.candidate, 1, 8)
        else:
            self.trigger = ConfirmedEdge(self.threshold, 2, 8)

    def public(self) -> dict:
        return {"id": self.id, "label": self.label, "ask": self.ask, "threshold": self.threshold, "mode": self.mode,
                "builtin": self.builtin, "p": round(self.last_p, 3), "fires": self.fires, "dismissed": self.dismissed,
                "active": bool(getattr(self.trigger, "active", False))}


# Fun, camera-facing defaults that make the first minute delightful (and shareable).
DEFAULTS = [
    ("wave", "👋 有人挥手 · Waving", "a person waving a hand at the camera, not just moving the hand to do something else", 0.8, "edge"),
    ("thumbs", "👍 竖大拇指 · Thumbs up", "a person giving a clear thumbs-up gesture with one hand", 0.8, "edge"),
    ("drink", "🥤 喝水 · Drinking", "a person drinking from a cup, glass, mug or bottle held at the mouth, not just holding it", 0.8, "edge"),
    ("phone", "📱 看手机 · On the phone", "a person holding a mobile phone and looking at its screen", 0.8, "state"),
]


@dataclass
class Session:
    dir: Path
    ring: FrameRing = field(default_factory=lambda: FrameRing(180))
    keep: KeepPolicy = field(default_factory=lambda: KeepPolicy(30))
    watches: dict = field(default_factory=dict)
    moments: list = field(default_factory=list)
    kept: set = field(default_factory=set)
    keyframes: list = field(default_factory=list)      # (t, path)
    now: float = 0.0
    t0_wall: float = field(default_factory=time.time)
    stats: dict = field(default_factory=lambda: {"frames": 0, "clips": 0, "moments": 0, "brain": 0, "judge": 0,
                                                 "kept_frames": 0, "candidates": 0, "verified": 0, "dismissed": 0})
    clients: set = field(default_factory=set)
    inflight: deque = field(default_factory=deque)
    history: deque = field(default_factory=lambda: deque(maxlen=600))   # (t, {watch: p}) for sparklines / tuning
    last_clip_t: float = -1.0


S: Session | None = None


def new_session() -> Session:
    global S
    base = Path(CFG["data"]) / f"session_{time.strftime('%Y%m%d_%H%M%S')}"
    (base / "frames").mkdir(parents=True, exist_ok=True)
    (base / "moments").mkdir(parents=True, exist_ok=True)
    S = Session(base)
    for wid, label, ask, th, mode in DEFAULTS:
        w = Watch(wid, label, PREFIX + ask + "?", th, mode, builtin=True)
        w.make_trigger()
        S.watches[wid] = w
    return S


def brain() -> OpenAICompatAgent:
    base = os.environ.get("PULSE_MODEL_BASE_URL", "http://127.0.0.1:8317/v1")
    model = os.environ.get("PULSE_MODEL", "gpt-6-luna")
    key = Path(os.environ.get("PULSE_MODEL_KEY_FILE", "/home1/jijh/tools/cliproxyapi/local_api_key")).read_text().strip()
    return OpenAICompatAgent(base, model, api_key=key, name=model)


async def send_all(msg: dict) -> None:
    text = json.dumps(msg, ensure_ascii=False, default=str)
    for ws in list(S.clients):
        try:
            await ws.send_text(text)
        except Exception:
            S.clients.discard(ws)


def state_msg() -> dict:
    return {"type": "state", "watches": [w.public() for w in S.watches.values()],
            "moments": S.moments[-40:], "stats": S.stats, "verifier": CFG.get("verifier_name"), "storage_mb": round(dir_mb(S.dir / "moments"), 2)}


def dir_mb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6 if p.exists() else 0.0


def add_frame(t: float, jpeg: bytes) -> None:
    path = S.dir / "frames" / f"f_{int(t * 1000):010d}.jpg"
    path.write_bytes(jpeg)
    before = list(S.ring.frames)
    S.ring.add(t, str(path))
    alive = {p for _, p in S.ring.frames}
    for _, p in before:
        if p not in alive and p not in S.kept:
            Path(p).unlink(missing_ok=True)
    S.now = t
    S.stats["frames"] += 1
    if S.keep.decide(t, False) == "keyframe":
        keep_frame(t, str(path))


def keep_frame(t: float, path: str) -> str:
    dst = S.dir / "moments" / Path(path).name
    if not dst.exists():
        shutil.copy2(path, dst)
        S.stats["kept_frames"] += 1
    S.kept.add(path)
    S.keyframes.append((t, str(dst)))
    return str(dst)


async def judge(t: float, frames: list[tuple[float, str]], watches: list[Watch]):
    clip = S.dir / f"clip_{int(t * 1000)}.mp4"
    url = await asyncio.to_thread(make_clip, frames, 4, clip, 448)
    qs = [Question(w.id, w.ask, kind="noul") for w in watches]
    b = await CFG["judge"].decide("A 2-second video from a webcam or a phone camera.", qs, None, [url])
    return t, clip, frames, {k: d.probs["yes"] for k, d in b.decisions.items()}, b.latency_ms


async def apply(result) -> None:
    t, clip, frames, probs, ms = result
    S.stats["clips"] += 1
    S.stats["judge"] += 1
    S.history.append((round(t, 2), {k: round(v, 4) for k, v in probs.items()}))
    fired = []
    for wid, p in probs.items():
        w = S.watches.get(wid)
        if not w:
            continue
        w.last_p = p
        kind = w.trigger.update(t, p)
        if kind in ("event", "start"):
            w.fires += 1
            fired.append(w)
    confirmed = []
    for w in fired:
        if w.mode != "state" and CFG.get("verifier"):
            S.stats["candidates"] += 1
            if await verify(w, frames):
                S.stats["verified"] += 1
                confirmed.append(w)
            else:
                w.dismissed += 1
                w.fires -= 1
                S.stats["dismissed"] += 1
        else:
            confirmed.append(w)
    for w in confirmed:
        await make_moment(w, t, clip, frames, probs[w.id])
    if not confirmed:
        clip.unlink(missing_ok=True)
    clip.with_suffix(".txt").unlink(missing_ok=True)
    await send_all({"type": "tick", "t": t, "probs": probs, "ms": round(ms), "stats": S.stats,
                    "active": {w.id: bool(getattr(w.trigger, "active", False)) for w in S.watches.values()}})


async def verify(w: Watch, frames) -> bool:
    """A stronger judge looks at 3 frames of the candidate clip; only candidates ever reach it."""
    picks = [frames[0][1], frames[len(frames) // 2][1], frames[-1][1]]
    try:
        b = await CFG["verifier"].decide("Three frames, in order, from a 2-second camera clip.",
                                         [Question("v", "Do these frames show " + w.ask[len(PREFIX):], kind="noul")],
                                         [image_url(p) for p in picks])
        S.stats["brain"] += 1
        return b.decisions["v"].probs.get("yes", 0.0) >= CFG["verify_threshold"]
    except Exception:
        return False


async def make_moment(w: Watch, t: float, clip: Path, frames, p: float) -> None:
    mid = frames[len(frames) // 2][1]
    frame = keep_frame(t, mid)
    mid_id = uuid.uuid4().hex[:10]
    mdir = S.dir / "moments"
    clip_dst = mdir / f"{mid_id}.mp4"
    if clip.exists():
        shutil.copy2(clip, clip_dst)
    wall = time.strftime("%H:%M:%S", time.localtime(S.t0_wall + t))
    card = mdir / f"{mid_id}_card.jpg"
    title = re.sub(r"^\W+\s*", "", w.label)
    sub = f"你说：“{w.said}”" if w.said and not w.said.isascii() else (f"You said: “{w.said}”" if w.said else
                                                                     "Watching for: " + title)
    await asyncio.to_thread(moment_card, frame, title, sub, wall, str(card))
    reel = mdir / f"{mid_id}_reel.mp4"
    if clip_dst.exists():
        try:
            await asyncio.to_thread(moment_reel, str(clip_dst), title, wall, str(reel))
        except Exception:
            reel = None
    m = {"id": mid_id, "watch": w.id, "label": w.label, "t": round(t, 1), "wall": wall, "p": round(p, 2),
         "frame": f"/m/{Path(frame).name}", "clip": f"/m/{clip_dst.name}" if clip_dst.exists() else None,
         "card": f"/m/{card.name}", "reel": f"/m/{reel.name}" if reel and reel.exists() else None}
    S.moments.append(m)
    S.stats["moments"] += 1
    await send_all({"type": "moment", "moment": m})


async def pump() -> None:
    """Every second of client time, judge the newest 2 s; results are applied in order."""
    while True:
        await asyncio.sleep(0.25)
        if S is None:
            continue
        while S.inflight and S.inflight[0].done():
            try:
                await apply(S.inflight.popleft().result())
            except Exception as e:
                await send_all({"type": "error", "message": f"judge: {type(e).__name__}: {e}"})
        frames = S.ring.last(2.0)
        if not frames or S.now - S.last_clip_t < 1.0 or len(frames) < 4 or len(S.inflight) >= 6:
            continue
        watches = list(S.watches.values())
        if not watches:
            continue
        S.last_clip_t = S.now
        S.inflight.append(asyncio.create_task(judge(S.now, frames[-8:], watches)))


@app.on_event("startup")
async def startup():
    new_session()
    asyncio.create_task(pump())


@app.websocket("/ws")
async def ws(socket: WebSocket):
    await socket.accept()
    S.clients.add(socket)
    await socket.send_text(json.dumps(state_msg(), ensure_ascii=False, default=str))
    try:
        while True:
            msg = await socket.receive()
            if msg.get("bytes"):
                data = msg["bytes"]
                t = struct.unpack("<d", data[:8])[0]
                add_frame(t, data[8:])
            elif msg.get("text"):
                pass
            elif msg.get("type") == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass
    finally:
        S.clients.discard(socket)


class WatchIn(BaseModel):
    text: str


@app.post("/api/watch")
async def add_watch(req: WatchIn):
    """Plain words -> one precise yes/no question (the validated style) -> a live watch."""
    S.stats["brain"] += 1
    lang = "Simplified Chinese" if re.search(r"[\u3400-\u9fff]", req.text) else \
        "the exact language of the user's words (English if unsure)"
    sys = DRAFT_SYSTEM + ("\nThe camera is usually a laptop webcam or phone facing the user (not first-person). "
                          f"Also return \"label\": a short label (2-6 words) written in {lang}, with one fitting "
                          "emoji, and "
                          "\"mode\": \"state\" for ongoing activities (sleeping, typing, on the phone) or \"edge\" "
                          "for brief actions (waving, drinking, standing up).")
    try:
        out = _json(await asyncio.to_thread(_chat, CFG["brain_url"], CFG["brain_model"], CFG["brain_key"], sys,
                                            f"The user wants to be told when: {req.text}\nWrite 1 question."))
        q = out["questions"][0]
        label = out.get("label") or q.get("label") or req.text[:30]
        mode = out.get("mode") or q.get("mode") or "edge"
    except Exception as e:
        return JSONResponse({"error": f"could not turn that into a watch: {e}"}, status_code=400)
    ask = q["ask"].strip()
    ask = ask[len(PREFIX):] if ask.startswith(PREFIX) else ask[0].lower() + ask[1:]
    wid = re.sub(r"[^a-z0-9_]", "_", q.get("id", "watch").lower())[:24] + "_" + uuid.uuid4().hex[:4]
    # Custom watches always go through candidate -> verifier: real actions often show up as one strong clip,
    # which a sustained-state rule would miss (the brain's "state" guess is only used without a verifier).
    use_state = mode == "state" and not CFG.get("verifier")
    w = Watch(wid, label, PREFIX + ask.rstrip("?") + "?", 0.85 if use_state else 0.8, "state" if use_state else "edge")
    w.said = req.text.strip()
    w.make_trigger()
    S.watches[wid] = w
    await send_all(state_msg())
    return w.public()


class SensIn(BaseModel):
    threshold: float


@app.post("/api/watch/{wid}/threshold")
async def set_threshold(wid: str, req: SensIn):
    w = S.watches[wid]
    w.threshold = min(0.98, max(0.05, req.threshold))
    w.make_trigger()
    return w.public()


@app.delete("/api/watch/{wid}")
async def delete_watch(wid: str):
    S.watches.pop(wid, None)
    await send_all(state_msg())
    return {"ok": True}


class AskIn(BaseModel):
    question: str


@app.post("/api/ask")
async def ask(req: AskIn):
    found: list[dict] = []

    async def find_object(args: dict) -> dict:
        noun = str(args.get("object", "")).strip().lower()
        recent = S.ring.sampled(1.0)
        s = await where_is(noun, recent, S.now, CFG["frame_judge"], CFG["djev_url"], window_s=120)
        scanned = s.frames_scanned
        if not s.found and S.keyframes:
            s2 = await where_is(noun, S.keyframes, S.now, CFG["frame_judge"], CFG["djev_url"], window_s=1e9)
            scanned += s2.frames_scanned
            s = s2 if s2.found else s
        S.stats["judge"] += scanned
        frame = None
        if s.found:
            src = next((p for t, p in recent + S.keyframes if Path(p).stem == s.frame), None)
            if src:
                frame = "/m/" + Path(keep_frame(s.t, src)).name
        out = {"object": noun, "found": s.found, "place": s.place, "seconds_ago": round(S.now - s.t) if s.t else None,
               "frame": frame, "confidence": s.visible_p, "frames_scanned": scanned}
        found.append(out)
        return out

    async def describe_recent(args: dict) -> dict:
        secs = min(8.0, float(args.get("seconds", 6) or 6))
        frames = S.ring.last(secs)
        if len(frames) < 4:
            return {"description": None, "note": "not enough recent video"}
        frames = frames[:: max(1, len(frames) // 16)]
        clip = S.dir / f"describe_{int(S.now * 1000)}.mp4"
        url = await asyncio.to_thread(make_clip, frames, 2, clip, 448)
        clip.unlink(missing_ok=True)
        clip.with_suffix(".txt").unlink(missing_ok=True)
        async with httpx.AsyncClient(timeout=60, trust_env=False) as c:
            r = await c.post(CFG["video_djev_url"] + "/v1/raw/chat/completions", json={
                "model": "dgemma", "max_tokens": 60, "messages": [{"role": "user", "content": [
                    {"type": "video_url", "video_url": {"url": url}},
                    {"type": "text", "text": "In one or two sentences, what is the person doing in this video?"}]}]})
        S.stats["judge"] += 1
        return {"seconds": secs, "description": r.json()["choices"][0]["message"]["content"].strip() if r.status_code == 200 else None}

    async def recent_moments(_: dict) -> dict:
        return {"now_s": S.now, "moments": [{**m, "seconds_ago": round(S.now - m["t"])} for m in S.moments[-10:]]}

    tools = [Tool("find_object", "Look back through what the camera saw (last minutes, then kept keyframes) for an "
                  "object; returns where it was last seen and how long ago.",
                  {"type": "object", "properties": {"object": {"type": "string"}}, "required": ["object"]}, find_object),
             Tool("recent_moments", "What Pulse noticed recently (the user's watches that fired). These are fast "
                  "automatic triggers and can be wrong.", {"type": "object", "properties": {}}, recent_moments),
             Tool("describe_recent", "Look at the last few seconds of camera video and describe what is happening.",
                  {"type": "object", "properties": {"seconds": {"type": "number"}}}, describe_recent)]
    S.stats["brain"] += 1
    b = brain()
    run = await b.run("You are Pulse, a friendly camera assistant. Use the tools. For 'where is X' call find_object. "
                      "For 'what happened / what am I doing' call describe_recent first; use recent_moments only as "
                      "supporting hints, since they can be wrong. Answer in one or two short sentences in the "
                      "user's language, with the place or event and how long ago.", req.question, tools)
    hit = next((f for f in reversed(found) if f["found"]), found[-1] if found else None)
    return {"answer": run.text.strip(), "evidence": hit, "model": b.name, "ms": round(run.latency_ms)}


@app.post("/api/reset")
async def reset():
    old = S
    clients = old.clients
    new_session()
    S.clients = clients
    shutil.rmtree(old.dir, ignore_errors=True)        # "forget everything" really deletes
    await send_all(state_msg())
    return {"ok": True}


@app.get("/m/{name}")
async def media(name: str):
    path = S.dir / "moments" / Path(name).name
    if not path.exists():
        return Response(status_code=404)
    return FileResponse(path)


@app.get("/api/history")
async def history():
    return {"history": list(S.history)}


@app.get("/api/state")
async def get_state():
    return state_msg()


@app.get("/")
async def index():
    return HTMLResponse((ROOT / "web/live.html").read_text())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=18900)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--djev-url", default="http://127.0.0.1:18811", help="frame judge for where-is scans")
    p.add_argument("--video-djev-url", default="http://127.0.0.1:18831", help="video-capable judge for clips")
    p.add_argument("--data", default="/cwStorage/home/jijh/pulse_data/live")
    p.add_argument("--verifier", choices=["luna", "none"], default="luna",
                   help="stronger judge that checks candidate moments (OpenAI gpt-6-luna Decisions; needs a key)")
    p.add_argument("--verify-threshold", type=float, default=0.2)
    p.add_argument("--openai-key-file", default="~/.config/pulse/openai_key")
    a = p.parse_args()
    CFG.update(djev_url=a.djev_url, video_djev_url=a.video_djev_url, judge=DjevDecider(a.video_djev_url), frame_judge=DjevDecider(a.djev_url),
               data=a.data, brain_url=os.environ.get("PULSE_MODEL_BASE_URL", "http://127.0.0.1:8317/v1"),
               brain_model=os.environ.get("PULSE_MODEL", "gpt-6-luna"),
               brain_key=Path(os.environ.get("PULSE_MODEL_KEY_FILE", "/home1/jijh/tools/cliproxyapi/local_api_key")).read_text().strip())
    if a.verifier == "luna" and Path(a.openai_key_file).expanduser().exists():
        CFG.update(verifier=LunaDecider(key_file=a.openai_key_file), verifier_name="gpt-6-luna",
                   verify_threshold=a.verify_threshold)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
