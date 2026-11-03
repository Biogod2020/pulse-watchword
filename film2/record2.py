"""Record real Pulse Live takes on real stock footage, desktop (16:9) or mobile (9:16) layout.

  record2.py <scene> [<scene> ...] [--layout desk|mobile|both] [--port 18920]

Each scene: the stock clip becomes the fake camera (pre-roll = first seconds ping-ponged while the watch is set up),
a watch is typed (or not), we wait for the real moment (and optionally ask a real question), and the screen is
captured with CDP screencast. Output: /cwStorage/home/jijh/pulse_data/film2/ui/<scene>_<layout>.{mp4,json} + media/.
"""
import argparse, asyncio, base64, json, shutil, subprocess, time, urllib.request
from pathlib import Path
from playwright.async_api import async_playwright

FF = "/home1/jijh/micromamba/envs/imgprocess/bin/ffmpeg"
ST = Path("/cwStorage/home/jijh/pulse_data/film2/stock"); OUT = Path("/cwStorage/home/jijh/pulse_data/film2/ui")
TMP = Path("/tmp/jijh-pulse-film2")
LAYOUT = {"desk": dict(W=2560, H=1440, zoom=2560 / 1536, css="main{height:800px}"),
          "mobile": dict(W=864, H=1536, zoom=2.0, css="")}

# scene -> clip, segment, what is typed, what happens after
SCENES = {
    "wave":    dict(clip="3_person_desk/pexels_4492700.mp4", ss=0, dur=None, watch="Tell me when I wave at the camera"),
    "catdesk": dict(clip="1_cat_desk/mixkit_46542.mp4", ss=0, dur=None, watch="Tell me when the cat walks across my laptop"),
    "cake":    dict(clip="2_pet_naughty/pexels_4013641.mp4", ss=0, dur=None, watch="Tell me if the cat goes for the cake"),
    "glass":   dict(clip="2_pet_naughty/pexels_6869665.mp4", ss=0, dur=None, watch="Tell me when the cat is on the dinner table"),
    "pill":    dict(clip="5_pill/pexels_5104289.mp4", ss=0, dur=None, watch="Tell me when I take my pills", ask="Did I take my pills today?"),
    "package": dict(clip="6_delivery/pexels_8926524.mp4", ss=0, dur=None, watch="Tell me when a package is delivered"),
    "keys":    dict(clip="4_keys/pexels_4513069.mp4", ss=0, dur=None, watch=None, ask="Where did I leave my keys?", ask_after=9.0),
    "gate":    dict(clip="1_cat_desk/mixkit_46542.mp4", ss=0, dur=None, watch=None, gate=True),
    "coffee":  dict(clip="3_person_desk/pexels_6799663.mp4", ss=0, dur=None, watch="Tell me every time I drink coffee"),
}

CURSOR_JS = Path(__file__).with_name("cursor.js").read_text()


def api(port, path, body=None, method=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=None if body is None else json.dumps(body).encode(),
                                 headers={"content-type": "application/json"}, method=method or ("POST" if body is not None else "GET"))
    return json.loads(urllib.request.urlopen(req, timeout=60).read() or b"null")


def probe_dur(p):
    return float(subprocess.run([FF.replace("ffmpeg", "ffprobe"), "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                 str(p)], capture_output=True, text=True).stdout or 0)


def feed(name, sc, pre=7.0):
    """Fake camera: ping-pong of the first 1.2 s (while the watch is being set up), the clip, then a long ping-pong tail."""
    TMP.mkdir(parents=True, exist_ok=True)
    src = ST / sc["clip"]; d = sc["dur"] or probe_dur(src) - sc["ss"]
    vf = "scale=1600:900:force_original_aspect_ratio=increase,crop=1600:900,fps=24,format=yuv420p"
    parts = [(sc["ss"], 1.2, max(1, round(pre / 2.4))), (sc["ss"], d, 0), (sc["ss"] + max(0, d - 2.0), 2.0, 12)]
    segs = []
    for i, (ss, dd, loops) in enumerate(parts):
        s = TMP / f"{name}_{i}.mp4"
        filt = f"[0:v]{vf}" + (f",split[a][b];[b]reverse[r];[a][r]concat=n=2:v=1,loop={loops - 1}:32767:0" if loops else "")
        subprocess.run([FF, "-v", "error", "-y", "-ss", str(ss), "-t", str(dd), "-i", str(src), "-filter_complex", filt, "-an",
                        "-c:v", "libx264", "-crf", "12", "-preset", "veryfast", str(s)], check=True)
        segs.append(s)
    lst = TMP / f"{name}.txt"; lst.write_text("".join(f"file '{s}'\n" for s in segs))
    y4m = TMP / f"{name}.y4m"
    subprocess.run([FF, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-pix_fmt", "yuv420p", str(y4m)], check=True)
    return y4m, probe_dur(segs[0]), d


class Take:
    def __init__(self, name, page, cdp, zoom):
        self.name, self.page, self.cdp, self.zoom = name, page, cdp, zoom
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
        return self.frames[-1][0] if self.frames else 0.0

    async def box(self, sel):
        return await self.page.evaluate("""s => { const e = document.querySelector(s); if (!e) return null;
            const r = e.getBoundingClientRect(); return [r.left, r.top, r.width, r.height]; }""", sel)

    async def mark(self, name, sel=None):
        self.beats.append({"name": name, "t": round(self.now(), 3), "box": await self.box(sel) if sel else None})
        print(f"[{self.name}] {self.now():6.2f}s {name}", flush=True)

    async def click(self, sel, ms=600):
        await self.page.evaluate("(()=>{const c=document.getElementById('__cur');c.style.transition='opacity .2s';c.style.opacity=1})()")
        b = await self.box(sel)
        await self.page.evaluate("([x,y,ms]) => window.__move(x,y,ms)", [b[0] + b[2] / 2, b[1] + b[3] / 2, ms])
        await self.page.evaluate("window.__ripple()"); await asyncio.sleep(0.12); await self.page.click(sel)
        await self.page.evaluate("setTimeout(()=>{const c=document.getElementById('__cur');c.style.transition='opacity .45s';c.style.opacity=0},700)")

    async def wait(self, js, timeout=60):
        t = time.time()
        while time.time() - t < timeout:
            if await self.page.evaluate(js):
                return True
            await asyncio.sleep(0.1)
        raise TimeoutError(f"[{self.name}] gave up waiting for: {js}")


async def run(scene, layout, port, details=False):
    sc = SCENES[scene]; L = LAYOUT[layout]; name = f"{scene}_{layout}" + ("_det" if details else "")
    y4m, pre_s, clip_s = feed(name, sc, sc.get("pre", 10.0))
    api(port, "/api/reset", {})
    for w in api(port, "/api/state")["watches"]:
        api(port, f"/api/watch/{w['id']}", method="DELETE")
    api(port, "/api/lang", {"lang": "en"})
    async with async_playwright() as pw:
        b = await pw.chromium.launch(args=["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
                                           f"--use-file-for-fake-video-capture={y4m}"])
        ctx = await b.new_context(viewport={"width": L["W"], "height": L["H"]}, permissions=["camera"], locale="en-US")
        await ctx.add_init_script(f"localStorage.setItem('pulse_details','{'1' if details else '0'}')")
        page = await ctx.new_page(); page.on("dialog", lambda d: asyncio.ensure_future(d.accept()))
        await page.goto(f"http://127.0.0.1:{port}")
        await page.add_style_tag(content=f"html{{zoom:{L['zoom']}}} {L['css']}" + ("" if sc.get("gate") else " .gate{display:none!important}"))
        await page.evaluate(CURSOR_JS)
        cdp = await ctx.new_cdp_session(page); tk = Take(name, page, cdp, L["zoom"])
        cdp.on("Page.screencastFrame", lambda f: asyncio.ensure_future(tk.on_frame(f)))
        if sc.get("gate"):                       # the real landing screen, then "Turn on camera"
            await cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 92, "maxWidth": L["W"], "maxHeight": L["H"]})
            await asyncio.sleep(1.6); await tk.mark("gate", "#start")
            await tk.click("#start", ms=700); await tk.mark("camera_on", ".stage"); await asyncio.sleep(3.0)
        else:
            await page.evaluate("start()")
            await cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 92, "maxWidth": L["W"], "maxHeight": L["H"]})
        await asyncio.sleep(0.5); await tk.mark("feed_start", ".stage")
        if sc.get("watch"):
            await asyncio.sleep(0.4)
            await tk.click("#watchIn"); await tk.mark("type_start", "#watchIn")
            await page.type("#watchIn", sc["watch"], delay=45)
            await tk.click("#watchGo", ms=400); await tk.mark("add_click", "#watchGo")
            await tk.wait("document.querySelector('#wlist .w.new')!==null", 40); await asyncio.sleep(0.3)
            await tk.mark("watch_added", "#wlist .w.new")
            await tk.wait("document.querySelector('#check.show')!==null", pre_s + clip_s + 30); await tk.mark("candidate", "#check")
            await tk.wait("document.querySelectorAll('#feed .mo').length>=1", 40)
            await tk.mark("moment", "#feed .mo"); await asyncio.sleep(0.25); await tk.mark("toast", "#toast")
            await asyncio.sleep(3.5)
        if sc.get("ask"):
            if not sc.get("watch"):
                await asyncio.sleep(pre_s + sc.get("ask_after", 4.0))
            if layout == "mobile":
                await page.evaluate("document.querySelector('#askIn').scrollIntoView({block:'center',behavior:'smooth'})"); await asyncio.sleep(0.8)
            await tk.click("#askIn"); await tk.mark("ask_type", "#askIn")
            await page.type("#askIn", sc["ask"], delay=50)
            await tk.click("#askGo", ms=380); await tk.mark("ask_click", "#askGo")
            await tk.wait("(()=>{const a=document.querySelector('#answer');return a&&a.querySelector('b')&&a.querySelector('b').textContent.length>3})()", 90)
            await asyncio.sleep(0.5); await tk.mark("answer", "#answer")
            if layout == "mobile":
                await asyncio.sleep(2.5)
                await page.evaluate("window.scrollTo({top:0,behavior:'smooth'})"); await asyncio.sleep(1.0); await tk.mark("replay", ".stage")
            await asyncio.sleep(4.0)
        await tk.mark("end")
        await cdp.send("Page.stopScreencast"); await asyncio.sleep(0.3)
        st = api(port, "/api/state"); d = OUT / "media"; d.mkdir(parents=True, exist_ok=True)
        for m in st["moments"]:
            for k in ("card", "reel", "clip", "frame"):
                if m.get(k):
                    (d / f"{name}_{Path(m[k]).name}").write_bytes(urllib.request.urlopen(f"http://127.0.0.1:{port}{m[k]}", timeout=30).read())
        answer = await page.evaluate("(()=>{const a=document.querySelector('#answer b');return a?a.textContent:null})()")
        await b.close()
    hist = api(port, "/api/history")
    meta = {"scene": scene, "layout": layout, "clip": sc["clip"], "watch": sc.get("watch"), "ask": sc.get("ask"), "answer": answer,
            "pre_s": pre_s, "zoom": L["zoom"], "beats": tk.beats, "events": hist["events"], "history": hist["history"],
            "stats": st["stats"], "moments": st["moments"], "watches": [{k: w[k] for k in ("id", "label", "ask")} for w in st["watches"]]}
    OUT.mkdir(parents=True, exist_ok=True); (OUT / f"{name}.json").write_text(json.dumps(meta, indent=1))
    fr = tk.frames; lst = TMP / name / "frames.txt"
    with open(lst, "w") as f:
        for i, (t, n) in enumerate(fr):
            f.write(f"file 'frames/{n}'\nduration {max((fr[i + 1][0] - t) if i + 1 < len(fr) else 1 / 30, 1e-3):.4f}\n")
        f.write(f"file 'frames/{fr[-1][1]}'\n")
    subprocess.run([FF, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-vf", "fps=30,format=yuv420p",
                    "-c:v", "libx264", "-crf", "14", "-preset", "medium", str(OUT / f"{name}.mp4")], check=True, cwd=TMP / name)
    print(name, "->", OUT / f"{name}.mp4", "| answer:", answer, "| events:", [(e["watch"], round(e["p"], 2), e["verified"]) for e in hist["events"]], flush=True)


def run_retry(scene, layout, port, tries=4, details=False):
    for k in range(tries):
        try:
            return asyncio.run(run(scene, layout, port, details))
        except TimeoutError as e:
            print("retry", scene, layout, k, e, flush=True)
    print("FAILED", scene, layout, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("scenes", nargs="+"); ap.add_argument("--layout", default="both")
    ap.add_argument("--port", type=int, default=18920); ap.add_argument("--details", action="store_true")
    a = ap.parse_args()
    for s in a.scenes:
        for L in (["desk", "mobile"] if a.layout == "both" else [a.layout]):
            run_retry(s, L, a.port, details=a.details)
