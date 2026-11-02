"""Render the deterministic scenes in motion.html (frames are pure functions of t) to 24 fps mp4.

  render_motion.py [--scene how] [--preview --times 1,4,8]
Output: /cwStorage/home/jijh/pulse_data/film/motion/<scene>.mp4 (or stills/<scene>_<t>.png)
"""
import argparse, functools, http.server, shutil, subprocess, threading
from pathlib import Path
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
ASSETS = Path("/cwStorage/home/jijh/pulse_data/film/motion_assets"); OUT = Path("/cwStorage/home/jijh/pulse_data/film/motion")
FF = "/home1/jijh/micromamba/envs/imgprocess/bin/ffmpeg"; FPS = 24


def serve():
    shutil.copy(HERE / "motion.html", ASSETS / "motion.html")
    shutil.copy(HERE.parent / "web" / "fonts" / "InterVariable.ttf", ASSETS / "InterVariable.ttf")
    class Q(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Q, directory=str(ASSETS)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv.server_address[1]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scene", action="append"); ap.add_argument("--preview", action="store_true")
    ap.add_argument("--times", default="")
    a = ap.parse_args(); port = serve(); OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        b = pw.chromium.launch(); pg = b.new_page(viewport={"width": 1920, "height": 1080})
        pg.goto(f"http://127.0.0.1:{port}/motion.html"); pg.wait_for_timeout(600)
        scenes = pg.evaluate("window.SCENES")
        for sid in a.scene or list(scenes):
            dur = scenes[sid]
            if a.preview:
                ts = [float(x) for x in a.times.split(",")] if a.times else [round(dur * f, 2) for f in (0.2, 0.5, 0.8, 0.99)]
                (OUT / "stills").mkdir(exist_ok=True)
                for t in ts:
                    pg.evaluate("([id,t])=>window.render(id,t)", [sid, t]); pg.screenshot(path=str(OUT / "stills" / f"{sid}_{t}.png"))
                print(sid, "stills", ts); continue
            n = round(dur * FPS)
            ff = subprocess.Popen([FF, "-v", "error", "-y", "-f", "image2pipe", "-framerate", str(FPS), "-i", "-", "-c:v", "libx264",
                                   "-crf", "14", "-preset", "medium", "-pix_fmt", "yuv420p", str(OUT / f"{sid}.mp4")], stdin=subprocess.PIPE)
            for i in range(n):
                pg.evaluate("([id,t])=>window.render(id,t)", [sid, i / FPS])
                ff.stdin.write(pg.screenshot(type="png"))
            ff.stdin.close(); ff.wait(); print(sid, n, "frames ->", OUT / f"{sid}.mp4", flush=True)
        b.close()


if __name__ == "__main__":
    main()
