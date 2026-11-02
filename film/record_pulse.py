"""Record real Pulse Live takes for the film (film instance on :18910, generated webcam feeds as the fake camera).

  record_pulse.py main|drink|walkin [--port 18910]

Each take: build the fake-camera y4m from H3 clips, open the app at 2560x1440 (CSS layout 1536x864), drive it with a
visible cursor, capture CDP screencast frames (JPEG q92, ~30 fps) with timestamps, log beats + element boxes, then
encode a constant-30-fps mp4. Output: /cwStorage/home/jijh/pulse_data/film/ui/<take>.{mp4,json}
Everything shown on screen is the app's real output on that feed; nothing is mocked.
"""
import argparse, asyncio, base64, json, shutil, subprocess, time, urllib.request
from pathlib import Path
from playwright.async_api import async_playwright

FF = "/home1/jijh/micromamba/envs/imgprocess/bin/ffmpeg"
H3 = Path("/cwStorage/home/jijh/pulse_data/film/h3"); OUT = Path("/cwStorage/home/jijh/pulse_data/film/ui")
TMP = Path("/tmp/jijh-pulse-film/rec"); W, H, ZOOM = 2560, 1440, 2560 / 1536

CURSOR_JS = """
(() => { const c = document.createElement('div'); c.id = '__cur';
 c.innerHTML = '<svg width="26" height="34" viewBox="0 0 26 34"><path d="M2 2 L2 26 L8.5 20 L13 31 L17.5 29 L13 18.5 L22 18.5 Z" fill="#fff" stroke="#111" stroke-width="2" stroke-linejoin="round"/></svg>';
 Object.assign(c.style, {position:'fixed', left:'0px', top:'0px', zIndex:99999, pointerEvents:'none', transform:'translate(-3px,-2px)', filter:'drop-shadow(0 2px 4px rgba(0,0,0,.45))', transition:'none'});
 document.body.appendChild(c); window.__cx = 760; window.__cy = 520; c.style.left = '760px'; c.style.top = '520px';
 window.__move = (x, y, ms) => new Promise(r => { const x0 = window.__cx, y0 = window.__cy, t0 = performance.now();
   const step = now => { const k = Math.min(1, (now - t0) / ms), e = k < .5 ? 4*k*k*k : 1 - Math.pow(-2*k + 2, 3) / 2;
     window.__cx = x0 + (x - x0) * e; window.__cy = y0 + (y - y0) * e; c.style.left = window.__cx + 'px'; c.style.top = window.__cy + 'px';
     k < 1 ? requestAnimationFrame(step) : r(); }; requestAnimationFrame(step); });
 window.__ripple = () => { const d = document.createElement('div'); Object.assign(d.style, {position:'fixed', left:(window.__cx-16)+'px', top:(window.__cy-16)+'px',
   width:'32px', height:'32px', borderRadius:'50%', border:'2px solid rgba(255,255,255,.9)', zIndex:99998, pointerEvents:'none',
   transition:'transform .45s ease-out, opacity .45s ease-out'}); document.body.appendChild(d);
   requestAnimationFrame(() => { d.style.transform = 'scale(1.9)'; d.style.opacity = '0'; }); setTimeout(() => d.remove(), 600); };
})();
"""


def api(port, path, body=None, method=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=None if body is None else json.dumps(body).encode(),
                                 headers={"content-type": "application/json"}, method=method or ("POST" if body is not None else "GET"))
    return json.loads(urllib.request.urlopen(req, timeout=30).read() or b"null")


def save_media(port, take):
    """Copy every moment's card / reel / clip before anything can delete them ("Forget everything")."""
    d = OUT / "media"; d.mkdir(parents=True, exist_ok=True); n = 0
    for m in api(port, "/api/state")["moments"]:
        for k in ("card", "reel", "clip", "frame"):
            if m.get(k):
                (d / f"{take}_{m['watch']}_{Path(m[k]).name}").write_bytes(
                    urllib.request.urlopen(f"http://127.0.0.1:{port}{m[k]}", timeout=30).read()); n += 1
    (d / f"{take}_moments.json").write_text(json.dumps(api(port, "/api/state")["moments"], indent=1))
    return n


def feed(name, parts, fps=24, size="1600:900"):
    """parts: list of (clip, start, dur, mode) with mode 'play' | 'hold' (first frame) | 'pingpong' (repeat k times)."""
    TMP.mkdir(parents=True, exist_ok=True)
    segs = []
    for i, (clip, ss, dur, mode, *k) in enumerate(parts):
        s = TMP / f"{name}_{i}.mp4"; src = str(H3 / clip / "video.mp4")
        vf = f"scale={size}:flags=lanczos,fps={fps},format=yuv420p"
        if mode == "play":
            cmd = [FF, "-v", "error", "-y", "-ss", str(ss), "-t", str(dur), "-i", src, "-vf", vf]
        elif mode == "hold":
            cmd = [FF, "-v", "error", "-y", "-ss", str(ss), "-i", src, "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"]
            png = subprocess.run(cmd, capture_output=True, check=True).stdout
            (TMP / "hold.png").write_bytes(png)
            cmd = [FF, "-v", "error", "-y", "-loop", "1", "-t", str(dur), "-i", str(TMP / "hold.png"), "-vf", vf]
        else:  # pingpong
            n = k[0] if k else 3
            cmd = [FF, "-v", "error", "-y", "-ss", str(ss), "-t", str(dur), "-i", src, "-filter_complex",
                   f"[0:v]{vf},split[a][b];[b]reverse[r];[a][r]concat=n=2:v=1,loop={n - 1}:32767:0"]
        subprocess.run(cmd + ["-an", "-c:v", "libx264", "-crf", "12", "-preset", "veryfast", str(s)], check=True)
        segs.append(s)
    lst = TMP / f"{name}.txt"; lst.write_text("".join(f"file '{s}'\n" for s in segs))
    y4m = TMP / f"{name}.y4m"
    subprocess.run([FF, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-pix_fmt", "yuv420p", str(y4m)], check=True)
    dur = float(subprocess.run([FF.replace("ffmpeg", "ffprobe"), "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                str(y4m)], capture_output=True, text=True).stdout or 0)
    return y4m, dur


class Take:
    def __init__(self, name, page, cdp):
        self.name, self.page, self.cdp = name, page, cdp
        self.frames, self.beats, self.t0 = [], [], None
        self.dir = TMP / name / "frames"; shutil.rmtree(self.dir, ignore_errors=True); self.dir.mkdir(parents=True)

    async def on_frame(self, f):
        ts = f["metadata"]["timestamp"]
        if self.t0 is None:
            self.t0 = ts
        p = self.dir / f"{len(self.frames):06d}.jpg"; p.write_bytes(base64.b64decode(f["data"]))
        self.frames.append((ts - self.t0, p.name))
        try:
            await self.cdp.send("Page.screencastFrameAck", {"sessionId": f["sessionId"]})
        except Exception:
            pass

    def now(self):
        return (self.frames[-1][0] if self.frames else 0.0)

    async def box(self, sel):
        b = await self.page.evaluate("""s => { const e = document.querySelector(s); if (!e) return null;
            const r = e.getBoundingClientRect(); return [r.left, r.top, r.width, r.height]; }""", sel)
        return None if b is None else [round(v * ZOOM) for v in b]   # device pixels in the 2560x1440 capture

    async def mark(self, name, sel=None):
        self.beats.append({"name": name, "t": round(self.now(), 3), "box": await self.box(sel) if sel else None, "sel": sel})
        print(f"[{self.name}] {self.now():6.2f}s {name}", flush=True)

    async def move_to(self, sel, ms=650, dx=0.5, dy=0.5):
        b = await self.page.evaluate("""s => { const r = document.querySelector(s).getBoundingClientRect(); return [r.left, r.top, r.width, r.height]; }""", sel)
        await self.page.evaluate("([x,y,ms]) => window.__move(x,y,ms)", [b[0] + b[2] * dx, b[1] + b[3] * dy, ms])

    async def click(self, sel, **kw):
        await self.page.evaluate("(()=>{const c=document.getElementById('__cur');c.style.transition='opacity .2s';c.style.opacity=1})()")
        await self.move_to(sel, **kw); await self.page.evaluate("window.__ripple()"); await asyncio.sleep(0.12)
        await self.page.click(sel)
        await self.page.evaluate("setTimeout(()=>{const c=document.getElementById('__cur');c.style.transition='opacity .45s';c.style.opacity=0},700)")

    async def wait(self, js, timeout=60):
        t = time.time()
        while time.time() - t < timeout:
            if await self.page.evaluate(js):
                return True
            await asyncio.sleep(0.1)
        raise TimeoutError(f"[{self.name}] gave up waiting for: {js}")


async def run(take, port):
    if take == "main":
        y4m, dur = feed("main", [("w_keys", 0, 10.1, "play"), ("w_cat", 0, 2.0, "hold"), ("w_cat", 0, 10.1, "play"),
                                 ("w_cat", 6.0, 4.0, "pingpong", 8)])
    elif take == "drink":
        y4m, dur = feed("drink", [("w_drink", 0, 1.6, "pingpong", 3), ("w_drink", 0, 10.1, "play"), ("w_drink", 8.0, 2.0, "pingpong", 10)])
    else:
        y4m, dur = feed("walkin", [("w_walkin", 0, 1.6, "pingpong", 3), ("w_walkin", 0, 10.1, "play"), ("w_walkin", 8.0, 2.0, "pingpong", 10)])
    print(take, "feed", round(dur, 1), "s", flush=True)
    if take == "main":   # clean session: no old moments, no built-in watches, English cards
        api(port, "/api/reset", {})
        for w in api(port, "/api/state")["watches"]:
            api(port, f"/api/watch/{w['id']}", method="DELETE")
    api(port, "/api/lang", {"lang": "en"})
    async with async_playwright() as pw:
        b = await pw.chromium.launch(args=["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
                                           f"--use-file-for-fake-video-capture={y4m}"])
        ctx = await b.new_context(viewport={"width": W, "height": H}, permissions=["camera"], locale="en-US")
        page = await ctx.new_page(); page.on("dialog", lambda d: asyncio.ensure_future(d.accept()))
        await page.goto(f"http://127.0.0.1:{port}")
        await page.add_style_tag(content=f"html{{zoom:{ZOOM}}} main{{height:800px}} .gate{{display:none!important}}")
        await page.evaluate(CURSOR_JS)
        cdp = await ctx.new_cdp_session(page); tk = Take(take, page, cdp)
        cdp.on("Page.screencastFrame", lambda f: asyncio.ensure_future(tk.on_frame(f)))
        await page.evaluate("start()")                                   # camera on: the fake feed starts at t=0 here
        await cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 92, "maxWidth": W, "maxHeight": H})
        await asyncio.sleep(0.6); await tk.mark("feed_start")
        if take == "main":
            await asyncio.sleep(0.6)
            await tk.click("#watchIn"); await tk.mark("type_start", "#watchIn")
            await page.type("#watchIn", "Tell me when the cat is on my desk", delay=48)
            await tk.click("#watchGo", ms=420); await tk.mark("add_click", "#watchGo")
            await tk.wait("document.querySelector('#wlist .w.new')!==null", 40); await asyncio.sleep(0.4)
            await tk.mark("watch_added", "#wlist .w.new")
            await tk.wait("document.querySelector('#check.show')!==null", 60); await tk.mark("candidate", "#check")
            await tk.wait("document.querySelectorAll('#feed .mo').length>=1", 60)
            await tk.mark("moment", "#feed .mo"); await asyncio.sleep(0.25); await tk.mark("toast", "#toast")
            await asyncio.sleep(4.0)
            await tk.click("#askIn"); await tk.mark("ask_type", "#askIn")
            await page.type("#askIn", "Where did I leave my keys?", delay=52)
            await tk.click("#askGo", ms=380); await tk.mark("ask_click", "#askGo")
            await tk.wait("(()=>{const a=document.querySelector('#answer');return a&&a.querySelector('b')&&a.querySelector('b').textContent.length>3})()", 90)
            await asyncio.sleep(0.6); await tk.mark("answer", "#answer")
            await asyncio.sleep(4.0)
            print("saved media files:", save_media(port, take), flush=True)
        else:
            chip = "I drink water" if take == "drink" else "someone walks behind me"
            await asyncio.sleep(0.8)
            sel = f"#ideas span:nth-child({1 if take == 'drink' else 3})"
            assert (await page.text_content(sel)).strip() == chip
            await tk.click(sel); await tk.mark("chip_click", sel)
            await tk.wait("document.querySelectorAll('#wlist .w.new').length>=" + ("2" if take == "drink" else "3"), 40)
            await asyncio.sleep(0.4); await tk.mark("watch_added", "#wlist .w.new")
            n0 = await page.evaluate("document.querySelectorAll('#feed .mo').length")
            await tk.wait("document.querySelector('#check.show')!==null", 60); await tk.mark("candidate", "#check")
            await tk.wait(f"document.querySelectorAll('#feed .mo').length>{n0}", 60)
            await tk.mark("moment", "#feed .mo"); await asyncio.sleep(0.25); await tk.mark("toast", "#toast")
            await asyncio.sleep(4.5)
            print("saved media files:", save_media(port, take), flush=True)
            if take == "walkin":
                await tk.mark("feed_box", "#feed")
                await tk.click("#forget"); await tk.mark("forget_click", "#forget")
                await asyncio.sleep(2.5); await tk.mark("forgotten", "#feed")
        await tk.mark("end")
        await cdp.send("Page.stopScreencast"); await asyncio.sleep(0.3)
        counter = await page.inner_text("#counter")
        await b.close()
    hist = api(port, "/api/history"); state = api(port, "/api/state")
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"take": take, "beats": tk.beats, "frames": len(tk.frames), "counter": counter, "feed_s": dur,
            "events": hist["events"], "history": hist["history"], "stats": state["stats"],
            "moments": state["moments"], "watches": [{k: w[k] for k in ("id", "label", "ask")} for w in state["watches"]]}
    (OUT / f"{take}.json").write_text(json.dumps(meta, indent=1))
    # constant 30 fps: concat demuxer with each screencast frame's real duration
    lst = TMP / take / "frames.txt"; fr = tk.frames
    with open(lst, "w") as f:
        for i, (t, n) in enumerate(fr):
            d = (fr[i + 1][0] - t) if i + 1 < len(fr) else 1 / 30
            f.write(f"file 'frames/{n}'\nduration {max(d, 1e-3):.4f}\n")
        f.write(f"file 'frames/{fr[-1][1]}'\n")
    subprocess.run([FF, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-vf", "fps=30,format=yuv420p",
                    "-c:v", "libx264", "-crf", "14", "-preset", "medium", str(OUT / f"{take}.mp4")], check=True, cwd=TMP / take)
    print(take, "frames", len(fr), "->", OUT / f"{take}.mp4", "| counter:", counter, flush=True)


def run_retry(take, port, tries=6):
    for k in range(tries):
        before = {w["id"] for w in api(port, "/api/state")["watches"]}
        try:
            return asyncio.run(run(take, port))
        except TimeoutError as e:
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / f"{take}_fail{k}.json").write_text(json.dumps(api(port, "/api/history"), indent=1))
            print("retry", take, k, e, [w["ask"] for w in api(port, "/api/state")["watches"] if w["id"] not in before], flush=True)
            for w in api(port, "/api/state")["watches"]:      # drop the watch this failed attempt added
                if w["id"] not in before:
                    api(port, f"/api/watch/{w['id']}", method="DELETE")
    raise SystemExit(f"{take} failed {tries} times")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("takes", nargs="+", choices=["main", "drink", "walkin"]); ap.add_argument("--port", type=int, default=18910)
    a = ap.parse_args()
    for t in a.takes:
        run_retry(t, a.port)
