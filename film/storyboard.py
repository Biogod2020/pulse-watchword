"""Storyboard pages for the Pulse demo film (pre-render review).

UI shots are real Pulse Live screenshots (film instance on :18910) with the keyframe as a still fake camera;
motion shots are text placeholders. Output: /cwStorage/home/jijh/pulse_data/film/review/sb_*.jpg
"""
import json, subprocess, textwrap, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/cwStorage/home/jijh/pulse_data/film"); KF = ROOT / "keyframes"; OUT = ROOT / "review"
FF = "/home1/jijh/micromamba/envs/imgprocess/bin/ffmpeg"
FB = "/home1/jijh/.fonts/noto-cjk/NotoSansCJKsc-Bold.otf"; FR = "/home1/jijh/.fonts/noto-cjk/NotoSansCJKsc-Regular.otf"
S = json.load(open(Path(__file__).with_name("script.json")))
F = lambda p, s: ImageFont.truetype(p, s)


def still_y4m(kf):
    out = Path(f"/tmp/jijh-pulse-film/{kf}.y4m")
    if not out.exists():
        subprocess.run([FF, "-v", "error", "-y", "-loop", "1", "-i", str(KF / f"{kf}.png"), "-t", "2", "-r", "8",
                        "-vf", "scale=1280:720,format=yuv420p", str(out)], check=True)
    return out


def ui_shot(kf, out, watch=None, ask=None):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        b = pw.chromium.launch(args=["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
                                     f"--use-file-for-fake-video-capture={still_y4m(kf)}"])
        pg = b.new_context(viewport={"width": 1920, "height": 1080}, permissions=["camera"], locale="en-US").new_page()
        pg.goto("http://127.0.0.1:18910"); pg.wait_for_timeout(700); pg.click("#start"); pg.wait_for_timeout(1500)
        if watch:
            pg.fill("#watchIn", watch)
        if ask:
            pg.fill("#askIn", ask)
        pg.wait_for_timeout(500); pg.screenshot(path=str(out)); b.close()


def placeholder(shot, out):
    im = Image.new("RGB", (1920, 1080), "#0b1020"); d = ImageDraw.Draw(im)
    d.text((120, 120), f"[{shot['kind']}]  code-rendered motion graphic", font=F(FR, 40), fill="#22d3ee")
    y = 220
    for line in textwrap.wrap(shot["picture"], 62):
        d.text((120, y), line, font=F(FR, 46), fill="white"); y += 66
    if shot.get("super"):
        d.text((120, 900), shot["super"], font=F(FB, 64), fill="#22d3ee")
    im.save(out)


def frame_for(shot):
    out = OUT / f"frame_{shot['id']}.png"
    if out.exists():
        return out
    k, src = shot["kind"], shot.get("src", "")
    if k.startswith("h3"):
        Image.open(KF / f"{src}.png").convert("RGB").resize((1920, 1080)).save(out)
    elif k == "ui_live" and shot["id"] == "S04":
        ui_shot("kf_drink", out, watch="Tell me when Miso jumps on my desk")
    elif k == "ui_live" and shot["id"] == "S05":
        ui_shot("kf_cat_last", out)
    elif k == "ui_live_montage":
        ui_shot("kf_thumbs", out, watch="when I drink water")
    elif k == "ui_live" and shot["id"] == "S09":
        ui_shot("kf_keys_first2", out, ask="Where did I leave my keys?")
    else:
        placeholder(shot, out)
    return out


def page(shots, n):
    W, H = 1920, 1080; tw, th = 940, 529
    pg = Image.new("RGB", (W, 2 * (th + 300) + 40), "#111"); d = ImageDraw.Draw(pg)
    for i, sh in enumerate(shots):
        x, y = 20 + (i % 2) * (tw + 20), 20 + (i // 2) * (th + 300)
        pg.paste(Image.open(frame_for(sh)).convert("RGB").resize((tw, th)), (x, y))
        ty = y + th + 10
        d.text((x, ty), f"{sh['id']}  {sh['dur']}s  · {sh['kind']}", font=F(FB, 26), fill="#22d3ee"); ty += 38
        for line in textwrap.wrap("PICTURE: " + sh["picture"], 95)[:4]:
            d.text((x, ty), line, font=F(FR, 20), fill="#ddd"); ty += 27
        for line in textwrap.wrap("VO: " + (sh["vo"] or "(none)"), 80)[:3]:
            d.text((x, ty), line, font=F(FB, 22), fill="#fde68a"); ty += 30
        if sh.get("super"):
            d.text((x, ty), "SUPER: " + sh["super"], font=F(FB, 22), fill="#a7f3d0")
    p = OUT / f"sb_{n}.jpg"; pg.save(p, quality=88); return p


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    shots = S["shots"]
    for n in range(0, len(shots), 4):
        print(page(shots[n:n + 4], n // 4 + 1))
