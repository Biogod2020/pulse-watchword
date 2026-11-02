"""Assemble the Pulse demo film (1920x1080, 24 fps) from real UI takes, H3 shots, motion scenes, narration, music, sfx.

  cut.py [--preview] [--only-audio] [--out NAME]

UI takes (2560x1440 screen recordings) get a virtual camera: per-segment crop keyframes (centre x/y + width, in capture
pixels), eased, resampled with OpenCV. Beat times come from ui/<take>.json, so cuts land on the recorded events.
Every UI shot carries a small "LIVE APP · camera feed generated" tag; generated cinematic shots carry "Generated footage".
"""
import argparse, json, subprocess
from pathlib import Path
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from scipy import signal

R = Path("/cwStorage/home/jijh/pulse_data/film"); UI, H3, MO, AU = R / "ui", R / "h3", R / "motion", R / "audio"
FF = "/home1/jijh/micromamba/envs/imgprocess/bin/ffmpeg"; FPS, W, H, SR = 24, 1920, 1080, 48000
FONT = "/cwStorage/home/jijh/playground/jev-diagnosis/pulse/web/fonts/InterVariable.ttf"
B = {k: {b["name"]: b["t"] for b in json.load(open(UI / f"{k}.json"))["beats"]} for k in ("main", "drink", "walkin")}
ZOOMF = 2560 / 1536

# capture-pixel regions (2560x1440): stage = camera view, right column = panels (x 1540..2540)
FULL = (1280, 720, 2560)
STAGE = (780, 522, 1500)
PX = 2040
WATCH_IN = (PX, 330, 1040)
WATCH_NEW = (PX, 470, 1040)
IDEAS = (PX, 330, 1040)
WLIST = (PX, 560, 1040)
RIGHTNOW = (780, 1150, 1500)
MOMENTS = (PX, 850, 1040)
FOOTER = (PX, 1150, 1040)
ASK = (PX, 1150, 1040)
ANSWER = (PX, 1180, 1000)



def ui(sid, take, a, b, cam, sfx=(), sup=None, tag="br", xf=0.0):
    """tag: 'br' bottom-right (camera-view shots), 'tc' top-centre (whole app), None for panel close-ups."""
    return {"id": sid, "kind": "ui", "src": UI / f"{take}.mp4", "a": a, "b": b, "cam": cam, "tag": tag and "live_" + tag,
            "sfx": list(sfx), "super": sup, "xf": xf}


def clip(sid, path, a, b, tag="gen", xf=0.0):
    return {"id": sid, "kind": "clip", "src": path, "a": a, "b": b, "cam": None, "tag": tag, "xf": xf, "sfx": [], "super": None}


M, D, Wk = B["main"], B["drink"], B["walkin"]
SEGS = [
    # A. cold open: the cat leaps into view (real take), then the detection lands
    ui("cold1", "main", 14.3, 16.5, [(0, 945, 470, 1170), (1, 945, 470, 1110)]),
    ui("cold2", "main", M["toast"] - 0.5, M["toast"] + 0.9, [(0, 945, 470, 1110), (1, 945, 470, 1080)], sfx=[("chime", 0.5)]),
    # B. lens macro + C. title
    clip("lens", H3 / "c_lens3/video.mp4", 0.0, 3.6),
    clip("title", MO / "title.mp4", 0.0, 3.0, tag=None),
    # D. plain words -> one question
    ui("wide", "main", M["type_start"] - 0.9, M["type_start"] + 0.1, [(0, *FULL), (1, *FULL)]),
    ui("type", "main", M["type_start"] + 0.1, M["watch_added"] + 1.8,
       [(0, *FULL), (0.16, *WATCH_IN), (0.6, *WATCH_IN), (0.8, *WATCH_NEW), (1, *WATCH_NEW)], sfx=[("keys", 0.15, 2.2)], tag=None),
    # E. keys down, Maya leaves; the cat lands; the reading spikes; Pulse notices
    ui("keys", "main", 3.6, 9.4, [(0, *STAGE[:2], 1500), (1, *STAGE[:2], 1420)]),
    ui("land", "main", 15.0, 16.3, [(0, *FULL), (1, 1000, 760, 2200)]),
    ui("spike", "main", M["candidate"] - 0.7, M["candidate"] + 1.1, [(0, *RIGHTNOW), (1, RIGHTNOW[0], RIGHTNOW[1], 1400)], tag=None, xf=0.2),
    ui("second", "main", M["candidate"] + 0.2, M["toast"] + 1.0, [(0, *STAGE[:2], 1500), (1, *STAGE[:2], 1400)],
       sfx=[("soft", 0.15), ("chime", M["toast"] - M["candidate"] - 0.2)], xf=0.2),
    ui("card", "main", M["toast"] + 1.0, M["toast"] + 2.6, [(0, *MOMENTS), (1, MOMENTS[0], MOMENTS[1], 980)], tag=None, xf=0.2),
    # F. how it keeps up
    clip("how", MO / "how.mp4", 0.0, 11.1, tag=None),
    # G. anything you can put into words
    ui("chip1", "drink", D["chip_click"] - 0.55, D["chip_click"] + 0.45, [(0, *IDEAS), (1, *IDEAS)], sfx=[("click", 0.55)], tag=None),
    ui("sip", "drink", D["moment"] - 4.4, D["moment"] - 2.4, [(0, *STAGE[:2], 1480), (1, *STAGE[:2], 1420)]),
    ui("sipnote", "drink", D["moment"] - 0.3, D["moment"] + 0.9, [(0, *STAGE[:2], 1420), (1, *STAGE[:2], 1400)], sfx=[("chime", 0.55)]),
    ui("chip2", "walkin", Wk["chip_click"] - 0.55, Wk["chip_click"] + 0.45, [(0, *IDEAS), (1, *IDEAS)], sfx=[("click", 0.55)], tag=None),
    ui("walk", "walkin", Wk["moment"] - 2.4, Wk["moment"] + 0.9, [(0, *STAGE[:2], 1480), (1, *STAGE[:2], 1420)], sfx=[("chime", 2.65)]),
    ui("list", "walkin", Wk["moment"] + 1.2, Wk["moment"] + 3.6, [(0, *WLIST), (1, WLIST[0], WLIST[1] + 10, 1000)], tag=None),
    # H. just ask -> the answer, then the evidence frame replays on the stage
    ui("ask", "main", M["ask_type"] - 0.25, M["ask_click"] + 0.45, [(0, *ASK), (1, *ASK)], sfx=[("keys", 0.25, 1.9), ("click", 2.2)], tag=None),
    ui("answer", "main", M["answer"] - 0.6, M["answer"] + 5.2,
       [(0, *ASK), (0.18, *ANSWER), (0.5, *ANSWER), (0.7, *STAGE[:2], 1500), (1, 640, 640, 1300)], sfx=[("soft", 0.6)], tag=None),
    # I. share
    clip("share", MO / "share.mp4", 0.0, 7.0, tag=None),
    # K. memory + the real "Forget everything"
    clip("memory", MO / "memory.mp4", 0.0, 4.6, tag=None),
    ui("kept", "walkin", Wk["forget_click"] - 5.6, Wk["forget_click"] - 1.3, [(0, MOMENTS[0], MOMENTS[1], 1100), (1, *MOMENTS)], tag=None),
    ui("forget", "walkin", Wk["forget_click"] - 1.3, Wk["forgotten"] + 1.0,
       [(0, *MOMENTS), (0.18, *FOOTER), (0.5, *FOOTER), (0.7, *MOMENTS), (1, MOMENTS[0], MOMENTS[1], 980)],
       sfx=[("click", 1.3)], tag=None),
    # L. close, M. end card
    clip("close", H3 / "c_close/video.mp4", 0.0, 5.17),
    clip("end", MO / "end.mp4", 0.0, 5.0, tag=None, xf=0.5),
]
IDX = {sg["id"]: i for i, sg in enumerate(SEGS)}

# narration: blind-picked Qwen3-TTS clones (take index per line)
VO_PICK = {"L1": 0, "L2": 2, "L3": 2, "L3b": 1, "L4": 2, "L5": 2, "L6": 1, "L7": 2, "L8": 1, "L9": 1, "L10": 2}


def seg_starts():
    t, out = 0.0, []
    for s in SEGS:
        t -= s.get("xf", 0.0)
        out.append(t); t += s["b"] - s["a"]
    return out, t


def tag_overlay(kind):
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, 22); f.set_variation_by_axes([600])
    if kind.startswith("live"):
        txt = "LIVE APP  ·  camera feed generated for this demo"
        w = d.textlength(txt, font=f) + 60
        x, y = (W - 44 - w, H - 74)
        d.rounded_rectangle((x, y, x + w, y + 40), 20, fill=(6, 10, 20, 170), outline=(255, 255, 255, 40))
        d.ellipse((x + 18, y + 14, x + 30, y + 26), fill=(244, 63, 94, 255))
        d.text((x + 42, y + 7), txt, font=f, fill=(238, 242, 255, 230))
    else:
        txt = "Generated footage"; w = d.textlength(txt, font=f)
        d.text((W - 44 - w, H - 62), txt, font=f, fill=(255, 255, 255, 120))
    a = np.asarray(im).astype(np.float32) / 255.0
    return a[..., :3][..., ::-1] * 255.0, a[..., 3:4]          # BGR premultiplied-ready, alpha


def super_overlay(text):
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, 68); f.set_variation_by_axes([720])
    w = d.textlength(text, font=f); x = (W - w) / 2; y = H - 210
    d.rounded_rectangle((x - 44, y - 20, x + w + 44, y + 98), 46, fill=(6, 10, 20, 200), outline=(255, 255, 255, 36))
    d.text((x, y), text, font=f, fill=(255, 255, 255, 255))
    a = np.asarray(im).astype(np.float32) / 255.0
    return a[..., :3][..., ::-1] * 255.0, a[..., 3:4]


def cam_at(cam, u):
    if u <= cam[0][0]:
        return cam[0][1:]
    for (u0, *p0), (u1, *p1) in zip(cam, cam[1:]):
        if u <= u1:
            k = (u - u0) / max(1e-6, u1 - u0); k = k * k * (3 - 2 * k)
            return [a + (b - a) * k for a, b in zip(p0, p1)]
    return cam[-1][1:]


def frames(seg):
    """Yield 1920x1080 BGR frames for one segment at 24 fps."""
    n = round((seg["b"] - seg["a"]) * FPS)
    src = str(seg["src"]); probe = subprocess.run([FF.replace("ffmpeg", "ffprobe"), "-v", "error", "-select_streams", "v:0",
                                                    "-show_entries", "stream=width,height", "-of", "csv=p=0", src],
                                                   capture_output=True, text=True).stdout.strip().split(",")
    sw, sh = int(probe[0]), int(probe[1])
    p = subprocess.Popen([FF, "-v", "error", "-ss", f"{seg['a']:.3f}", "-i", src, "-t", f"{n / FPS + 0.2:.3f}", "-vf", f"fps={FPS}",
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    last = None
    for i in range(n):
        buf = p.stdout.read(sw * sh * 3)
        if len(buf) < sw * sh * 3:
            fr = last
        else:
            fr = np.frombuffer(buf, np.uint8).reshape(sh, sw, 3); last = fr
        if seg["cam"]:
            cx, cy, cw = cam_at(seg["cam"], i / max(1, n - 1)); ch = cw * 9 / 16
            cx = min(max(cx, cw / 2), sw - cw / 2); cy = min(max(cy, ch / 2), sh - ch / 2)
            s = W / cw; M = np.float32([[s, 0, W / 2 - s * cx], [0, s, H / 2 - s * cy]])
            out = cv2.warpAffine(fr, M, (W, H), flags=cv2.INTER_LANCZOS4 if s > 1 else cv2.INTER_AREA)
        else:
            out = fr if (sw, sh) == (W, H) else cv2.resize(fr, (W, H), interpolation=cv2.INTER_LANCZOS4)
        yield out
    p.stdout.close(); p.wait()


def render_video(path, preview):
    tags = {k: tag_overlay(k) for k in ("live_br", "live_tc", "gen")}
    starts, total = seg_starts(); N = round(total * FPS)
    enc = subprocess.Popen([FF, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libx264", "-preset", "veryfast" if preview else "slow", "-crf", "18" if preview else "15",
                            "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
    tail, written = [], 0
    for si, seg in enumerate(SEGS):
        xfn = round(seg.get("xf", 0) * FPS); nseg = round((seg["b"] - seg["a"]) * FPS)
        nxt = round(SEGS[si + 1].get("xf", 0) * FPS) if si + 1 < len(SEGS) else 0
        new_tail = []
        sup = super_overlay(seg["super"][0]) if seg.get("super") else None
        for i, fr in enumerate(frames(seg)):
            fr = fr.astype(np.float32)
            if seg["tag"]:
                rgb, a = tags[seg["tag"]]; fr = fr * (1 - a) + rgb * a
            if sup is not None:
                ts = i / FPS - seg["super"][1]; k = min(1.0, max(0.0, ts / 0.3), max(0.0, (nseg - i) / FPS / 0.3))
                if k > 0:
                    rgb, a = sup; fr = fr * (1 - a * k) + rgb * (a * k)
            if i < xfn and tail:                                    # crossfade from the held tail of the previous segment
                k = (i + 1) / (xfn + 1); fr = tail[i] * (1 - k) + fr * k
            fr = np.clip(fr, 0, 255).astype(np.uint8)
            if nxt and i >= nseg - nxt:
                new_tail.append(fr.astype(np.float32)); continue    # these frames are blended into the next segment
            # fade the very end to black
            t = written / FPS
            if t > total - 1.0:
                fr = (fr.astype(np.float32) * max(0.0, (total - t) / 1.0)).astype(np.uint8)
            enc.stdin.write(fr.tobytes()); written += 1
        tail = new_tail
        print(f"seg {si:2d} {seg['kind']:4s} start {starts[si]:6.2f}s  {seg['b'] - seg['a']:5.2f}s", flush=True)
    enc.stdin.close(); enc.wait()
    return starts, total


# ---------------------------------------------------------------- audio
def load(path, start=0.0, dur=None):
    """Decode any audio file to 48 kHz stereo float32 through ffmpeg (no soundfile dependency)."""
    cmd = [FF, "-v", "error", "-ss", f"{start:.3f}", "-i", str(path)] + (["-t", f"{dur:.3f}"] if dur else []) + \
          ["-f", "f32le", "-ac", "2", "-ar", str(SR), "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.float32).reshape(-1, 2).copy()


def save_wav(path, x):
    subprocess.run([FF, "-v", "error", "-y", "-f", "f32le", "-ac", "2", "-ar", str(SR), "-i", "-", "-c:a", "pcm_s24le", str(path)],
                   input=np.ascontiguousarray(x, np.float32).tobytes(), check=True)


def place(bus, x, t, gain_db=0.0):
    s = int(t * SR); e = min(len(bus), s + len(x))
    if e > s:
        bus[s:e] += x[:e - s] * 10 ** (gain_db / 20)


def fade(x, fi, fo):
    x = x.copy(); n_in, n_out = int(fi * SR), int(fo * SR)
    if n_in:
        x[:n_in] *= np.linspace(0, 1, n_in)[:, None]
    if n_out:
        x[-n_out:] *= np.linspace(1, 0, n_out)[:, None]
    return x


def rms_norm(x, db=-20.0):
    act = x[np.abs(x).max(1) > 1e-3]; r = np.sqrt(np.mean(act ** 2)) if len(act) else 1.0
    return x * (10 ** (db / 20) / max(r, 1e-6))


def sfx(kind, dur=1.0):
    t = np.arange(int(SR * max(dur, 0.05))) / SR
    if kind == "chime":                       # two bright partials, soft attack, long tail
        x = (np.sin(2 * np.pi * 1318.5 * t) * 0.6 + np.sin(2 * np.pi * 1975.5 * t) * 0.4) * np.exp(-t * 4.0)
        x += np.sin(2 * np.pi * 2637 * t) * 0.15 * np.exp(-t * 7); x *= np.minimum(1, t / 0.004); g = -16.5
    elif kind == "soft":
        x = np.sin(2 * np.pi * 880 * t) * np.exp(-t * 6) * np.minimum(1, t / 0.006); g = -20
    elif kind == "click":
        n = np.random.default_rng(3).standard_normal(len(t)); x = signal.lfilter(*signal.butter(2, [1800, 6000], "band", fs=SR), n)
        x *= np.exp(-t * 180); g = -22
    elif kind == "keys":                      # soft laptop typing for `dur` seconds
        rng = np.random.default_rng(7); x = np.zeros(len(t)); bb = signal.butter(2, [1200, 5000], "band", fs=SR)
        for k in np.cumsum(rng.uniform(0.07, 0.16, 60)):
            if k >= dur - 0.05:
                break
            i = int(k * SR); m = min(len(t) - i, int(0.03 * SR)); tt = np.arange(m) / SR
            x[i:i + m] += signal.lfilter(*bb, rng.standard_normal(m)) * np.exp(-tt * 260) * rng.uniform(0.5, 1)
        g = -26
    elif kind == "whoosh":
        n = np.random.default_rng(5).standard_normal(len(t)); env = np.sin(np.pi * np.minimum(1, t / dur)) ** 2
        x = signal.lfilter(*signal.butter(2, 900, "low", fs=SR), n) * env; g = -24
    x = x / (np.abs(x).max() + 1e-9) * 10 ** (g / 20)
    return np.stack([x, x], 1).astype(np.float32)


def quiet_runs(x, floor_db=-42):
    env = np.convolve(np.abs(x[:, 0]), np.ones(960) / 960, "same"); quiet = env < 10 ** (floor_db / 20)
    runs, i = [], 0
    while i < len(quiet):
        if quiet[i]:
            j = i
            while j < len(quiet) and quiet[j]:
                j += 1
            if i > SR * 0.3 and j < len(quiet) - SR * 0.3:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def split_at_pauses(x, targets):
    """Cut at the pause nearest each sentence boundary (targets = boundary position as a fraction of the line)."""
    runs = [r for r in quiet_runs(x) if r[1] - r[0] > 0.08 * SR]
    cuts = sorted(min(((a + b) // 2 for a, b in runs), key=lambda m: abs(m - f * len(x))) for f in targets)
    return [x[a:b] for a, b in zip([0] + cuts, cuts + [len(x)])]


def tighten(x, keep_s):
    """Shorten every internal pause longer than keep_s to keep_s."""
    out, last = [], 0
    for a, b in quiet_runs(x):
        if (b - a) / SR > keep_s:
            m = (a + b) // 2; h = int(keep_s * SR / 2)
            out.append(x[last:m - h]); last = m + h
    out.append(x[last:])
    return np.concatenate(out)


def render_audio(path, starts, total):
    n = int((total + 0.5) * SR); mus, vo, fx = (np.zeros((n, 2), np.float32) for _ in range(3))
    T = {sg["id"]: starts[i] for i, sg in enumerate(SEGS)}
    # music: intro (cold open .. title), main (demo, ~47 s of music), outro2 (share .. end card)
    mdir = AU / "music"
    intro = rms_norm(load(mdir / "pulse_intro_xl_s2_00001.flac")); main = rms_norm(load(mdir / "pulse_main3_xl_s8_00001.flac"))
    outro = rms_norm(load(mdir / "pulse_outro3_xl_s10_00001.flac"))
    t_main = T["wide"] - 0.6
    t_outro = min(T["share"] - 0.8, t_main + 55.0)
    place(mus, fade(intro[:int((t_main + 1.5) * SR)], 0.3, 1.5), 0.0, -2)
    place(mus, fade(main[:int((t_outro - t_main + 1.5) * SR)], 1.5, 1.5), t_main, 0)
    out_len = total - t_outro
    place(mus, fade(outro[:int(out_len * SR)], 1.5, 2.5), t_outro, 0)
    # narration
    VO = {"L1": T["lens"] - 0.25, "L2": T["title"] + 1.25, "L3": T["type"] + 0.2, "L3b": T["keys"] + 0.9, "L4": T["how"] + 0.35,
          "L5a": T["sip"] + 0.15, "L5b": T["walk"] + 0.2, "L5c": T["list"] + 0.1, "L6": T["ask"] + 0.1,
          "L7": T["share"] + 0.45, "L8a": T["memory"] + 0.3, "L8b": T["kept"] + 0.2, "L10": T["forget"] + 1.75, "L9": T["end"] + 0.3}
    lines = {}
    for L, k in VO_PICK.items():
        lines[L] = load(AU / "vo_lines" / f"line_{L}_{k}.wav")
    # multi-sentence lines are split at their longest pauses so each sentence lands on its own picture
    for L, k in (("L5", (4 / 15, 9 / 15)), ("L8", (10 / 21,))):      # word share of each sentence
        x = lines.pop(L); parts = split_at_pauses(x, k)
        for name, part in zip("abc", parts):
            lines[L + name] = part
    lines["L2"] = tighten(lines["L2"], 0.10)          # "Meet ... Pulse": shorten the internal pause
    voice_on = np.zeros(n, bool)
    for L, t0 in VO.items():
        y = rms_norm(lines[L], -19.0); place(vo, y, t0, 0)
        voice_on[int(t0 * SR):int(t0 * SR) + len(y)] = True
    # duck music under the voice (-7 dB, 250 ms ramps)
    duck = np.where(voice_on, 10 ** (-7 / 20), 1.0).astype(np.float32)
    duck = np.convolve(duck, np.ones(int(0.25 * SR)) / int(0.25 * SR), "same")[:, None]
    mus *= duck
    # sfx and the generated footage's own sound
    for si, seg in enumerate(SEGS):
        for e in seg["sfx"]:
            kind, at, *d = e; place(fx, sfx(kind, d[0] if d else 1.2), starts[si] + at)
    for sid in ("lens", "land", "how", "share"):          # soft whooshes on the big transitions
        place(fx, sfx("whoosh", 0.7), T[sid] - 0.45)
    # the generated footage's own sound (feed time = take time - 12.0 inside the cat clip; +0.1 in the keys clip)
    amb = {"cold1": ("w_cat", 14.3 - 12.0), "land": ("w_cat", 14.9 - 12.0), "keys": ("w_keys", 3.6 + 0.1),
           "close": ("c_close", 0.0), "lens": ("c_lens3", 0.0)}
    for sid, (name, a) in amb.items():
        sg = SEGS[IDX[sid]]; dur = sg["b"] - sg["a"]
        y = load(H3 / name / "audio.wav", max(0.0, a), dur); place(fx, fade(rms_norm(y, -30), 0.08, 0.15), T[sid], -2)
    mix = mus * 0.9 + vo + fx
    mix[int((total - 1.1) * SR):] *= np.linspace(1, 0, n - int((total - 1.1) * SR))[:, None]
    peak = np.abs(mix).max(); mix *= min(1.0, 10 ** (-1 / 20) / peak)
    save_wav(path, mix[:int(total * SR)])
    return VO


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--preview", action="store_true"); ap.add_argument("--only-audio", action="store_true")
    ap.add_argument("--out", default="pulse_demo_v1")
    a = ap.parse_args(); out = R / "cut"; out.mkdir(exist_ok=True)
    starts, total = seg_starts()
    print("total", round(total, 2), "s", flush=True)
    vo = render_audio(out / f"{a.out}.wav", starts, total)
    json.dump({"starts": starts, "total": total, "vo": vo}, open(out / f"{a.out}_timeline.json", "w"), indent=1)
    if not a.only_audio:
        render_video(out / f"{a.out}_video.mp4", a.preview)
        subprocess.run([FF, "-v", "error", "-y", "-i", str(out / f"{a.out}_video.mp4"), "-i", str(out / f"{a.out}.wav"), "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", str(out / f"{a.out}.mp4")], check=True)
        print("->", out / f"{a.out}.mp4")
