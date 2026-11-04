"""Meta hackathon submission film (16:9, narrated, <= 3 min): Pulse running on Muse Glimmer only.
hook -> problem -> how it works -> live demo (Glimmer takes) -> memory -> held-out evidence -> glasses concept
-> built during the hackathon -> limits -> end. Reuses cut2's renderer; waits inside takes are sped up and labelled.

  cut4.py [--preview]   -> /cwStorage/home/jijh/pulse_data/film4/cut/pulse_muse.mp4
"""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "film2"))
import cut2  # noqa: E402
from cut2 import ui, clip, STAGE, TIGHT, WATCH_IN, WATCH_NEW, ANSWER, ASK, FULL  # noqa: E402

R4 = Path("/cwStorage/home/jijh/pulse_data/film4"); MO4, MO3, MO2 = R4 / "motion", Path("/cwStorage/home/jijh/pulse_data/film3/motion"), \
    Path("/cwStorage/home/jijh/pulse_data/film2/motion")
cut2.UI = R4 / "ui"
VO_DIR = R4 / "vo_final"          # post-processed lines (film4/vo_post.py), one file per line id


def sped(sg, k):
    sg["speed"] = k; return sg


def segments():
    b = cut2.beats
    C, K, G, Co, Ca = b("catdesk_desk_det"), b("keys_desk_det"), b("glass_desk_det"), b("coffee_desk_det"), b("cake_desk_det")
    return [
        ("M1", sped(ui("catdesk_desk_det", C["candidate"] - 1.0, C["toast"] + 1.4, [(0, *STAGE[:2], 1500), (1, *TIGHT[:2], 1250)],
                       "Live · Muse Glimmer · 1.6× speed", "low", [("chime", (C["toast"] - C["candidate"] + 1.0) / 1.6)]), 1.6)),
        (None, clip(MO2 / "title_h.mp4", 0, 2.4)),
        ("M2", sped(clip(MO3 / "problem_h.mp4", 0, 15.5), 0.95)),   # 5% slower animation: room for the slower read
        ("M3", clip(MO4 / "arch_h.mp4", 0, 25.0)),
        ("M4", ui("catdesk_desk_det", C["type_start"] - 0.2, C["add_click"] + 0.4, [(0, *WATCH_IN), (1, *WATCH_IN)], "Live run · a stock video is the camera", "top",
                  [("keys", 0.1, 2.0)])),
        (None, sped(ui("catdesk_desk_det", C["add_click"] + 0.4, C["watch_added"] + 1.4, [(0, *WATCH_IN), (0.8, *WATCH_NEW), (1, *WATCH_NEW)],
                       "Glimmer writes the question · 4× speed", "top"), 4.0)),
        (None, sped(ui("catdesk_desk_det", C["candidate"] - 0.8, C["toast"] + 1.2, [(0, *FULL), (0.3, *FULL), (1, *STAGE[:2], 1500)],
                       "Candidate 0.33 → second look thinks · 2× speed", "low", [("soft", 0.4), ("chime", (C["toast"] - C["candidate"] + 0.8) / 2)]), 2.0)),
        ("M5", ui("cake_desk_det", Ca["toast"] - 2.3, Ca["toast"] + 1.5, [(0, *STAGE[:2], 1500), (1, *STAGE[:2], 1400)],
                  "“Tell me if the cat goes for the cake”", "low", [("chime", 2.3)])),
        (None, ui("glass_desk_det", G["toast"] - 2.3, G["toast"] + 1.5, [(0, *STAGE[:2], 1500), (1, *STAGE[:2], 1400)],
                  "“Tell me when the cat is on the dinner table”", "low", [("chime", 2.3)])),
        (None, ui("coffee_desk_det", Co["toast"] - 2.3, Co["toast"] + 1.5, [(0, *STAGE[:2], 1500), (1, *STAGE[:2], 1400)],
                  "“Tell me every time I drink coffee”", "low", [("chime", 2.3)])),
        ("M6", ui("keys_desk_det", K["ask_type"] - 0.2, K["ask_click"] + 0.3, [(0, *ASK), (1, *ASK)], "“Where did I leave my keys?”", "top",
                  [("keys", 0.2, 1.8)])),
        (None, sped(ui("keys_desk_det", K["ask_click"] + 0.3, K["answer"] - 0.3, [(0, *ASK), (1, *ASK)],
                       "Glimmer calls find_object, scans back · 3× speed", "top"), 3.0)),
        (None, ui("keys_desk_det", K["answer"] - 0.3, K["answer"] + 4.1, [(0, *ANSWER), (1, ANSWER[0], ANSWER[1], 940)], None, "top", [("soft", 0.3)])),
        ("M7", clip(MO4 / "evidence_h.mp4", 0, 27.5)),
        ("M8", clip(MO4 / "glasses_h.mp4", 0, 20.0)),
        ("M9", clip(MO4 / "built_h.mp4", 0, 12.0)),
        ("M10", clip(MO4 / "limits_h.mp4", 0, 16.5)),
        ("M11", clip(MO4 / "end_h.mp4", 0, 5.0)),
    ]


def duck_curve(on, SR, depth_db=-11.0, bridge_s=2.5, attack_s=0.25, release_s=0.9):
    """Music gain under the narration: stay down through pauses shorter than bridge_s (no pumping between
    sentences), go down in attack_s before speech and come back up slowly (release_s) after it."""
    on = on.copy(); idx = np.flatnonzero(np.diff(on.astype(np.int8)))
    for a, b in zip(idx[1::2], idx[2::2]):                 # off-runs between two speech runs
        if (b - a) / SR < bridge_s:
            on[a:b + 1] = True
    hop = int(.01 * SR); tgt = np.where(on[::hop], 10 ** (depth_db / 20), 1.0)
    pre = int(attack_s / .01)                               # start ducking just before the voice
    tgt = np.minimum(tgt, np.concatenate([tgt[pre:], np.ones(pre)]))
    g = np.empty_like(tgt); cur = 1.0; ka, kr = 1 - np.exp(-1 / (attack_s / .01 / 3)), 1 - np.exp(-1 / (release_s / .01 / 3))
    for i, t in enumerate(tgt):
        cur += (t - cur) * (ka if t < cur else kr); g[i] = cur
    return np.interp(np.arange(len(on)), np.arange(len(g)) * hop, g).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--preview", action="store_true"); ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    S = segments(); segs = [s for _, s in S]
    out = R4 / "cut"; out.mkdir(parents=True, exist_ok=True)
    vo = lambda line: cut2.load(str(VO_DIR / f"{line}.wav"))
    t0 = 0.0
    for line, seg in S:
        d = (seg["b"] - seg["a"]) / seg.get("speed", 1.0)
        if line:
            print(f"{line:4s} at {t0:6.1f}s  voice {len(vo(line)) / cut2.SR:5.1f}s  picture {d:5.1f}s")
        t0 += d
    print("total", round(t0, 1))
    if a.check:
        return
    music = out / "_bed_long.wav"
    if not music.exists():                   # the 145 s bed, played twice with a 4 s crossfade
        src = "/cwStorage/home/jijh/pulse_data/film3/music/p3_bed_xl_s22_00001.flac"
        subprocess.run([cut2.FF, "-v", "error", "-y", "-i", src, "-i", src, "-filter_complex", "acrossfade=d=4", "-ar", str(cut2.SR), str(music)], check=True)
    cut2.version = lambda v: (1920, 1080, music, segs)
    total = cut2.render_video("muse", out / "_muse_video.mp4", a.preview)
    cut2.render_audio("muse", out / "_muse_music.wav", total)
    SR = cut2.SR; n = int(total * SR); voa = np.zeros((n, 2), np.float32); on = np.zeros(n, bool); t0 = 0.0
    for line, seg in S:
        if line:
            x = vo(line) * 10 ** (3 / 20)    # lines are already -20 LUFS each; keep their relative level
            s = int((t0 + 0.3) * SR); e = min(n, s + len(x)); voa[s:e] += x[:e - s]; on[s:e] = True
        t0 += (seg["b"] - seg["a"]) / seg.get("speed", 1.0)
    mus = cut2.load(out / "_muse_music.wav")[:n]
    duck = duck_curve(on, SR)[:len(mus), None]
    mix = mus * duck.astype(np.float32) + voa[:len(mus)]; mix *= min(1.0, 10 ** (-1 / 20) / np.abs(mix).max())
    subprocess.run([cut2.FF, "-v", "error", "-y", "-f", "f32le", "-ac", "2", "-ar", str(SR), "-i", "-", "-c:a", "pcm_s24le",
                    str(out / "_muse.wav")], input=mix.astype(np.float32).tobytes(), check=True)
    subprocess.run([cut2.FF, "-v", "error", "-y", "-i", str(out / "_muse_video.mp4"), "-i", str(out / "_muse.wav"), "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-af", "loudnorm=I=-14:TP=-1.5:LRA=9", "-ar", "48000", "-c:a", "aac", "-b:a", "256k", "-shortest",
                    "-movflags", "+faststart", str(out / "pulse_muse.mp4")], check=True)
    print("muse", round(total, 1), "s ->", out / "pulse_muse.mp4")


if __name__ == "__main__":
    main()
