"""Render motion4.html scenes (deterministic) to 24 fps mp4 in 16:9 (--o h) or 9:16 (--o v).
  render_motion2.py --o h|v [--scene push] [--preview --times 1,2]"""
import argparse, functools, http.server, shutil, subprocess, threading
from pathlib import Path
from playwright.sync_api import sync_playwright
HERE = Path(__file__).resolve().parent; ASSETS = Path("/cwStorage/home/jijh/pulse_data/film4/motion_assets")
OUT = Path("/cwStorage/home/jijh/pulse_data/film4/motion"); FF = "/home1/jijh/micromamba/envs/imgprocess/bin/ffmpeg"; FPS = 24
ap = argparse.ArgumentParser(); ap.add_argument("--o", default="h"); ap.add_argument("--scene", action="append")
ap.add_argument("--preview", action="store_true"); ap.add_argument("--times", default="")
a = ap.parse_args(); shutil.copy(HERE / "motion4.html", ASSETS / "motion4.html"); OUT.mkdir(parents=True, exist_ok=True)
class Q(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *x): pass
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Q, directory=str(ASSETS)))
threading.Thread(target=srv.serve_forever, daemon=True).start()
W, H = (1080, 1920) if a.o == "v" else (1920, 1080)
with sync_playwright() as pw:
    b = pw.chromium.launch(); pg = b.new_page(viewport={"width": W, "height": H})
    pg.goto(f"http://127.0.0.1:{srv.server_address[1]}/motion4.html?o={a.o}"); pg.wait_for_timeout(700)
    scenes = pg.evaluate("window.SCENES")
    for sid in a.scene or list(scenes):
        if a.preview:
            for t in [float(x) for x in a.times.split(",")]:
                pg.evaluate("([i,t])=>window.render(i,t)", [sid, t]); pg.screenshot(path=str(OUT / f"still_{sid}_{a.o}_{t}.png"))
            continue
        n = round(scenes[sid] * FPS)
        ff = subprocess.Popen([FF, "-v", "error", "-y", "-f", "image2pipe", "-framerate", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "14",
                               "-pix_fmt", "yuv420p", str(OUT / f"{sid}_{a.o}.mp4")], stdin=subprocess.PIPE)
        for i in range(n):
            pg.evaluate("([i,t])=>window.render(i,t)", [sid, i / FPS]); ff.stdin.write(pg.screenshot(type="png"))
        ff.stdin.close(); ff.wait(); print(sid, a.o, n, flush=True)
    b.close()
