"""Cut the real-footage Pulse films: 16:9 long/short (desktop takes) and 9:16 long/short (mobile takes).

  cut2.py hlong|hshort|vlong|vshort [...] [--preview]

Sources: real Pulse Live screen recordings on real stock footage (ui/<scene>_<layout>.mp4 + beats json), deterministic
motion scenes (motion/<scene>_<o>.mp4), a real stock reaction clip. Captions are drawn here (Inter, social style).
Output: /cwStorage/home/jijh/pulse_data/film2/cut/pulse_<version>.mp4 (24 fps, -14 LUFS).
"""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from scipy import signal

R = Path("/cwStorage/home/jijh/pulse_data/film2"); UI, MO, ST, MU = R / "ui", R / "motion", R / "stock", R / "music"
FF = "/home1/jijh/micromamba/envs/imgprocess/bin/ffmpeg"; FPS, SR = 24, 48000
FONT = "/cwStorage/home/jijh/playground/jev-diagnosis/pulse/web/fonts/InterVariable.ttf"


def beats(take):
    return {b["name"]: b["t"] for b in json.load(open(UI / f"{take}.json"))["beats"]}


# capture regions: desktop takes are 2560x1440 (centre x, centre y, crop width); mobile takes are 864x1536
FULL, STAGE = (1280, 720, 2560), (780, 522, 1500)
TIGHT = (780, 470, 1250)
PX = 2040
WATCH_IN, WATCH_NEW, MOMENTS, ASK, ANSWER = (PX, 330, 1040), (PX, 470, 1040), (PX, 850, 1040), (PX, 1150, 1040), (PX, 1180, 1000)
MFULL, MTOP, MSTAGE, MWATCH = (432, 768, 864), (432, 570, 640), (432, 470, 560), (432, 1010, 760)


def ui(take, a, b, cam, cap=None, pos="low", sfx=()):
    return {"kind": "ui", "src": UI / f"{take}.mp4", "a": a, "b": b, "cam": cam, "cap": cap, "pos": pos, "sfx": list(sfx)}


def clip(path, a, b, cap=None, pos="low", fit="cover"):
    return {"kind": "clip", "src": path, "a": a, "b": b, "cam": None, "cap": cap, "pos": pos, "sfx": [], "fit": fit}


def hook_h(take="catdesk_desk", cap="My laptop just caught my cat."):
    B = beats(take)
    return ui(take, B["toast"] - 2.6, B["toast"] + 1.4, [(0, *TIGHT), (1, TIGHT[0], TIGHT[1], 1150)], cap, "low", [("chime", 2.6)])


def stage_h(take, cap, before=2.1, after=0.9):
    B = beats(take)
    return ui(take, B["toast"] - before, B["toast"] + after, [(0, *STAGE[:2], 1500), (1, *STAGE[:2], 1400)], cap, "low", [("chime", before)])


def answer_h(take, cap, hold=3.0, replay=True):
    B = beats(take)
    segs = [ui(take, B["answer"] - 0.3, B["answer"] + hold, [(0, *ANSWER), (1, ANSWER[0], ANSWER[1], 940)], cap, "top", [("soft", 0.3)])]
    if replay:
        segs.append(ui(take, B["answer"] + hold, B["answer"] + hold + 1.7, [(0, *STAGE[:2], 1500), (1, *STAGE[:2], 1380)], None))
    return segs


def typing_h(take, cap, speed=1.6):
    B = beats(take)
    sg = ui(take, B["type_start"] - 0.2, B["watch_added"] + 1.4,
            [(0, *WATCH_IN), (0.6, *WATCH_IN), (0.82, *WATCH_NEW), (1, *WATCH_NEW)], cap, "low", [("keys", 0.1, 1.6)])
    sg["speed"] = speed; return sg


def gate_h(cap):
    B = beats("gate_desk")
    return ui("gate_desk", B["gate"] - 0.6, B["camera_on"] + 1.6, [(0, *FULL), (1, *FULL)], cap, "low", [("click", B["camera_on"] - B["gate"] + 0.5)])


def version(v):
    if v == "hlong":
        C = beats("catdesk_desk"); P = beats("pill_desk")
        return (1920, 1080, MU / "p2_hype70_xl_s12_00001.flac", [
            hook_h(cap="Left my laptop open. My cat had plans."),
            clip(ST / "8_phone_smile/pexels_9130363.mp4", 3.0, 5.0, "Then my phone buzzed in the other room.", "low"),
            clip(MO / "push_h.mp4", 0.8, 4.2),
            *answer_h("keys_desk", "It also remembers where I left my keys."),
            clip(MO / "title_h.mp4", 0, 2.4),
            gate_h("Open it in your browser. No app."),
            typing_h("catdesk_desk", "Tell it what to watch for. In plain words."),
            ui("catdesk_desk", C["candidate"] - 0.5, C["toast"] + 1.0, [(0, *STAGE[:2], 1500), (1, *STAGE[:2], 1380)],
               "It double-checks before it tells you.", "low", [("soft", 0.5), ("chime", C["toast"] - C["candidate"] + 0.5)]),
            stage_h("cake_desk", "“Tell me if the cat goes for the cake.”"),
            stage_h("coffee_desk", "“Tell me every time I drink coffee.”"),
            stage_h("glass_desk", "“Tell me when the cat is on the table.”"),
            ui("pill_desk", P["ask_type"] - 0.2, P["ask_click"] + 0.2, [(0, *ASK), (1, *ASK)], "“Did I take my pills today?”", "top", [("keys", 0.2, 1.8)]),
            *answer_h("pill_desk", None, 2.6, replay=False),
            clip(MO / "share_h.mp4", 0, 4.0),
            stage_h("wave_desk", "Try it in 5 seconds: open it and wave.", 2.2, 1.2),
            clip(MO / "end_h.mp4", 0, 4.6)])
    if v == "hshort":
        return (1920, 1080, MU / "p2_hype36_xl_s12_00001.flac", [
            hook_h(cap="Left my laptop open. My cat had plans."),
            clip(ST / "8_phone_smile/pexels_9130363.mp4", 3.0, 4.8, "Then my phone buzzed.", "low"),
            clip(MO / "push_h.mp4", 0.8, 3.6),
            *answer_h("keys_desk", "It also found my keys.", 2.6),
            gate_h("Open it in your browser."),
            typing_h("catdesk_desk", "Type one sentence.", 2.0),
            stage_h("cake_desk", "“Tell me if the cat goes for the cake.”", 1.8, 0.8),
            stage_h("glass_desk", "“Tell me when the cat is on the table.”", 1.8, 0.8),
            clip(MO / "end_h.mp4", 0, 4.6)])
    C = beats("catdesk_mobile"); K = beats("keys_mobile"); Wv = beats("wave_mobile"); G = beats("gate_mobile")

    def sb(take, cap, before=2.1, after=0.9):
        B = beats(take)
        return ui(take, B["toast"] - before, B["toast"] + after, "STAGEBOX", cap, "low", [("chime", before)])

    keys = [ui("keys_mobile", K["answer"] - 0.4, K["answer"] + 2.8, [(0, *MFULL), (1, *MFULL)], "It also remembers where I left my keys.", "top",
               [("soft", 0.4)]),
            ui("keys_mobile", K["replay"] - 0.1, K["replay"] + 1.6, "STAGEBOX", None)]
    gate = ui("gate_mobile", G["gate"] - 0.6, G["camera_on"] + 1.5, [(0, *MFULL), (1, *MFULL)], "Open it in your browser. No app.", "low",
              [("click", G["camera_on"] - G["gate"] + 0.5)])

    def typing(speed, cap):
        sg = ui("catdesk_mobile", C["type_start"] - 0.2, C["watch_added"] + 1.2, [(0, *MFULL), (1, *MFULL)], cap, "low", [("keys", 0.1, 1.6)])
        sg["speed"] = speed; return sg

    if v == "vlong":
        P = beats("pill_mobile")
        return (1080, 1920, MU / "p2_hype52_xl_s12_00001.flac", [
            ui("catdesk_mobile", C["toast"] - 2.6, C["toast"] + 1.4, "STAGEBOX", "Left my laptop open. My cat had plans.", "low", [("chime", 2.6)]),
            clip(ST / "8_phone_smile/pexels_7710868.mp4", 1.0, 3.0, "Then my phone buzzed.", "low"),
            clip(MO / "push_v.mp4", 0.8, 4.2), *keys, clip(MO / "title_v.mp4", 0, 2.4), gate,
            typing(1.6, "Tell it what to watch for."),
            ui("catdesk_mobile", C["candidate"] - 0.5, C["toast"] + 1.0, "STAGEBOX", "It double-checks before it tells you.", "low",
               [("soft", 0.5), ("chime", C["toast"] - C["candidate"] + 0.5)]),
            sb("glass_mobile", "“Tell me when the cat is on the table.”"),
            sb("coffee_mobile", "“Tell me every time I drink coffee.”"),
            ui("pill_mobile", P["answer"] - 0.5, P["answer"] + 3.0, [(0, *MFULL), (1, *MFULL)], "“Did I take my pills today?”", "top", [("soft", 0.5)]),
            clip(MO / "share_v.mp4", 0, 4.0),
            ui("wave_mobile", Wv["toast"] - 2.2, Wv["toast"] + 1.2, "STAGEBOX", "Try it now: open it and wave.", "low", [("chime", 2.2)]),
            clip(MO / "end_v.mp4", 0, 4.6)])
    return (1080, 1920, MU / "p2_hype24_xl_s12_00001.flac", [
        ui("catdesk_mobile", C["toast"] - 2.2, C["toast"] + 1.2, "STAGEBOX", "Left my laptop open. My cat had plans.", "low", [("chime", 2.2)]),
        clip(ST / "8_phone_smile/pexels_7710868.mp4", 1.0, 2.8, "Then my phone buzzed.", "low"),
        clip(MO / "push_v.mp4", 0.8, 3.4),
        typing(2.0, "All I typed was one sentence."),
        sb("glass_mobile", "“Tell me when the cat is on the table.”", 1.8, 0.8),
        clip(MO / "end_v.mp4", 0, 3.8)])


# ------------------------------------------------------------------ video
def probe(src):
    o = subprocess.run([FF.replace("ffmpeg", "ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                        "-of", "csv=p=0", str(src)], capture_output=True, text=True).stdout.strip().split(",")
    return int(o[0]), int(o[1])


def cam_at(cam, u):
    if u <= cam[0][0]:
        return cam[0][1:]
    for (u0, *p0), (u1, *p1) in zip(cam, cam[1:]):
        if u <= u1:
            k = (u - u0) / max(1e-6, u1 - u0); k = k * k * (3 - 2 * k)
            return [a + (b - a) * k for a, b in zip(p0, p1)]
    return cam[-1][1:]


MSTAGEBOX = (24, 115, 816, 459)      # stage rect in the 864x1536 mobile capture


def stagebox(fr, W, H, u):
    x, y, w, h = MSTAGEBOX; st = fr[y:y + h, x:x + w]
    sc = max(W / w, H / h); bg = cv2.resize(st, (int(w * sc) + 2, int(h * sc) + 2))
    oy, ox = (bg.shape[0] - H) // 2, (bg.shape[1] - W) // 2; bg = bg[oy:oy + H, ox:ox + W]
    bg = (cv2.GaussianBlur(bg, (0, 0), 40) * 0.38).astype(np.uint8)
    fw = int(W * (0.96 + 0.03 * u)); fh = int(fw * h / w); fg = cv2.resize(st, (fw, fh), interpolation=cv2.INTER_LANCZOS4)
    x0, y0 = (W - fw) // 2, int(H * 0.47 - fh / 2); out = bg.copy(); out[y0:y0 + fh, x0:x0 + fw] = fg
    return out


def frames(seg, W, H):
    sp = seg.get("speed", 1.0); n = round((seg["b"] - seg["a"]) / sp * FPS); sw, sh = probe(seg["src"])
    p = subprocess.Popen([FF, "-v", "quiet", "-ss", f"{seg['a']:.3f}", "-i", str(seg["src"]), "-t", f"{seg['b'] - seg['a'] + 0.3:.3f}", "-vf",
                          f"setpts=PTS/{sp},fps={FPS}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    last = None; ar = H / W
    for i in range(n):
        buf = p.stdout.read(sw * sh * 3)
        fr = np.frombuffer(buf, np.uint8).reshape(sh, sw, 3) if len(buf) == sw * sh * 3 else last
        last = fr
        if seg["cam"] == "STAGEBOX":                  # vertical: the real camera view, enlarged, over a blurred copy
            yield stagebox(fr, W, H, i / max(1, n - 1)); continue
        if seg["cam"]:
            cx, cy, cw = cam_at(seg["cam"], i / max(1, n - 1))
        else:                                         # cover-fit the whole source
            cw = min(sw, sh / ar); cx, cy = sw / 2, sh / 2
        ch = cw * ar
        if ch > sh:
            ch = sh; cw = ch / ar
        cx = min(max(cx, cw / 2), sw - cw / 2); cy = min(max(cy, ch / 2), sh - ch / 2)
        s = W / cw; M = np.float32([[s, 0, W / 2 - s * cx], [0, s, H / 2 - s * cy]])
        yield cv2.warpAffine(fr, M, (W, H), flags=cv2.INTER_LANCZOS4 if s > 1 else cv2.INTER_AREA)
    p.kill(); p.wait()


def caption(text, W, H, pos):
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    size = 64 if W > H else 70
    f = ImageFont.truetype(FONT, size); f.set_variation_by_axes([800])
    maxw = W * (0.8 if W > H else 0.86); words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if d.textlength(t, font=f) > maxw and cur:
            lines.append(cur); cur = w
        else:
            cur = t
    lines.append(cur)
    lh = size * 1.22; bh = lh * len(lines) + 44
    y0 = {"low": H - bh - (90 if W > H else 360), "top": 60 if W > H else 150, "mid": H * 0.5 - bh / 2 + (0 if W > H else 40)}[pos]
    bw = max(d.textlength(l, font=f) for l in lines) + 72
    d.rounded_rectangle(((W - bw) / 2, y0, (W + bw) / 2, y0 + bh), 34, fill=(8, 10, 20, 215))
    for k, l in enumerate(lines):
        d.text(((W - d.textlength(l, font=f)) / 2, y0 + 22 + k * lh), l, font=f, fill=(255, 255, 255, 255))
    a = np.asarray(im).astype(np.float32) / 255.0
    return a[..., :3][..., ::-1] * 255.0, a[..., 3:4]


def render_video(v, path, preview):
    W, H, _, segs = version(v)
    enc = subprocess.Popen([FF, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libx264", "-preset", "veryfast" if preview else "slow", "-crf", "18" if preview else "16",
                            "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
    total = sum((s["b"] - s["a"]) / s.get("speed", 1.0) for s in segs); written = 0
    for seg in segs:
        n = round((seg["b"] - seg["a"]) / seg.get("speed", 1.0) * FPS); cap = caption(seg["cap"], W, H, seg["pos"]) if seg.get("cap") else None
        for i, fr in enumerate(frames(seg, W, H)):
            if cap is not None:
                k = min(1.0, (i + 1) / 5, (n - i) / 4); rgb, a = cap
                fr = np.clip(fr.astype(np.float32) * (1 - a * k) + rgb * a * k, 0, 255).astype(np.uint8)
            t = written / FPS
            if t > total - 0.6:
                fr = (fr.astype(np.float32) * max(0.0, (total - t) / 0.6)).astype(np.uint8)
            enc.stdin.write(np.ascontiguousarray(fr).tobytes()); written += 1
    enc.stdin.close(); enc.wait()
    return total


# ------------------------------------------------------------------ audio
def load(path, start=0.0, dur=None):
    cmd = [FF, "-v", "error", "-ss", f"{start:.3f}", "-i", str(path)] + (["-t", f"{dur:.3f}"] if dur else []) + ["-f", "f32le", "-ac", "2", "-ar", str(SR), "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.float32).reshape(-1, 2).copy()


def sfx(kind, dur=1.0):
    t = np.arange(int(SR * max(dur, 0.05))) / SR
    if kind == "chime":
        x = (np.sin(2 * np.pi * 1318.5 * t) * .6 + np.sin(2 * np.pi * 1975.5 * t) * .4) * np.exp(-t * 4) * np.minimum(1, t / .004); g = -15
    elif kind == "soft":
        x = np.sin(2 * np.pi * 880 * t) * np.exp(-t * 6) * np.minimum(1, t / .006); g = -19
    elif kind == "keys":
        rng = np.random.default_rng(7); x = np.zeros(len(t)); bb = signal.butter(2, [1200, 5000], "band", fs=SR)
        for k in np.cumsum(rng.uniform(.07, .15, 60)):
            if k >= dur - .05:
                break
            i = int(k * SR); m = min(len(t) - i, int(.03 * SR)); tt = np.arange(m) / SR
            x[i:i + m] += signal.lfilter(*bb, rng.standard_normal(m)) * np.exp(-tt * 260) * rng.uniform(.5, 1)
        g = -25
    elif kind == "click":
        x = signal.lfilter(*signal.butter(2, [1800, 6000], "band", fs=SR), np.random.default_rng(3).standard_normal(len(t))) * np.exp(-t * 180); g = -22
    else:
        x = signal.lfilter(*signal.butter(2, 900, "low", fs=SR), np.random.default_rng(5).standard_normal(len(t))) * np.sin(np.pi * t / dur) ** 2; g = -23
    x = x / (np.abs(x).max() + 1e-9) * 10 ** (g / 20)
    return np.stack([x, x], 1).astype(np.float32)


def render_audio(v, path, total):
    W, H, music, segs = version(v)
    n = int((total + .5) * SR); mix = np.zeros((n, 2), np.float32)
    m = load(music); m = m[:n]; fo = int(1.2 * SR); m[-fo:] *= np.linspace(1, 0, fo)[:, None]
    act = m[np.abs(m).max(1) > 1e-3]; m *= 10 ** (-17 / 20) / np.sqrt(np.mean(act ** 2)); mix[:len(m)] += m
    t0 = 0.0
    for k, seg in enumerate(segs):
        for kind, at, *d in seg["sfx"]:
            x = sfx(kind, d[0] if d else 1.2); s = int((t0 + at) * SR); e = min(n, s + len(x)); mix[s:e] += x[:e - s]
        if k:
            x = sfx("whoosh", .5); s = max(0, int((t0 - .3) * SR)); e = min(n, s + len(x)); mix[s:e] += x[:e - s] * .7
        t0 += (seg["b"] - seg["a"]) / seg.get("speed", 1.0)
    mix = mix[:int(total * SR)]; mix *= min(1.0, 10 ** (-1 / 20) / np.abs(mix).max())
    subprocess.run([FF, "-v", "error", "-y", "-f", "f32le", "-ac", "2", "-ar", str(SR), "-i", "-", "-c:a", "pcm_s24le", str(path)],
                   input=mix.astype(np.float32).tobytes(), check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("versions", nargs="+"); ap.add_argument("--preview", action="store_true")
    a = ap.parse_args(); out = R / "cut"; out.mkdir(parents=True, exist_ok=True)
    for v in a.versions:
        total = render_video(v, out / f"_{v}_video.mp4", a.preview)
        render_audio(v, out / f"_{v}.wav", total)
        subprocess.run([FF, "-v", "error", "-y", "-i", str(out / f"_{v}_video.mp4"), "-i", str(out / f"_{v}.wav"), "-map", "0:v", "-map", "1:a",
                        "-c:v", "copy", "-af", "loudnorm=I=-14:TP=-1.5:LRA=7", "-ar", "48000", "-c:a", "aac", "-b:a", "256k", "-shortest",
                        "-movflags", "+faststart", str(out / f"pulse_{v}.mp4")], check=True)
        print(v, round(total, 1), "s ->", out / f"pulse_{v}.mp4", flush=True)
