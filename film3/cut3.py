"""Hackathon judges' cut (16:9, narrated): problem -> idea -> live demo (cascade, generality, memory) -> held-out evidence
-> limits -> end. Reuses cut2's renderer (real Pulse takes on real stock footage) and adds narration with ducking.

  cut3.py [--preview]   -> /cwStorage/home/jijh/pulse_data/film3/cut/pulse_judges.mp4
"""
import argparse, subprocess, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "film2"))
import cut2  # noqa: E402
from cut2 import ui, clip, beats, STAGE, TIGHT, WATCH_IN, WATCH_NEW, ANSWER, ASK, FULL, MOMENTS  # noqa: E402

R3 = Path("/cwStorage/home/jijh/pulse_data/film3"); MO3, MO2 = R3 / "motion", Path("/cwStorage/home/jijh/pulse_data/film2/motion")
VO = {"J1": 1, "J2": 1, "J3": 1, "J4": 1, "J5": 2, "J6": 2, "J7": 0, "J8": 2, "J9": 0}


def segments():
    C = beats("catdesk_desk_det"); K = beats("keys_desk_det"); Co = beats("coffee_desk_det"); G = beats("glass_desk_det")
    Ca = beats("cake_desk")
    S = [
        ("J1", ui("catdesk_desk_det", C["toast"] - 7.2, C["toast"] + 1.3, [(0, *STAGE[:2], 1500), (1, *TIGHT[:2], 1250)], None, "low",
                  [("chime", 7.2)])),
        (None, clip(MO2 / "title_h.mp4", 0, 2.4)),
        ("J2", clip(MO3 / "problem_h.mp4", 0, 15.5)),
        ("J3", clip(MO3 / "arch_h.mp4", 0, 25.0)),
        ("J4", ui("catdesk_desk_det", C["type_start"] - 0.2, C["watch_added"] + 1.6,
                  [(0, *WATCH_IN), (0.55, *WATCH_IN), (0.8, *WATCH_NEW), (1, *WATCH_NEW)], "Live, on real footage", "top", [("keys", 0.1, 2.0)])),
        (None, ui("catdesk_desk_det", C["candidate"] - 1.2, C["toast"] + 1.4, [(0, *FULL), (0.3, *FULL), (1, *STAGE[:2], 1500)],
                  "Near miss rejected · real one confirmed", "low", [("soft", 0.9), ("chime", C["toast"] - C["candidate"] + 1.2)])),
        ("J5", ui("cake_desk", Ca["toast"] - 2.1, Ca["toast"] + 0.9, [(0, *STAGE[:2], 1500), (1, *STAGE[:2], 1400)],
                  "“Tell me if the cat goes for the cake”", "low", [("chime", 2.1)])),
        (None, ui("coffee_desk_det", Co["toast"] - 2.1, Co["toast"] + 0.9, [(0, *STAGE[:2], 1500), (1, *STAGE[:2], 1400)],
                  "“Tell me every time I drink coffee”", "low", [("chime", 2.1)])),
        (None, ui("glass_desk_det", G["toast"] - 2.1, G["toast"] + 1.1, [(0, *STAGE[:2], 1500), (1, *STAGE[:2], 1400)],
                  "“Tell me when the cat is on the table”", "low", [("chime", 2.1)])),
        ("J6", ui("keys_desk_det", K["ask_type"] - 0.2, K["ask_click"] + 0.3, [(0, *ASK), (1, *ASK)], "“Where did I leave my keys?”", "top",
                  [("keys", 0.2, 1.8)])),
        (None, ui("keys_desk_det", K["answer"] - 0.3, K["answer"] + 3.2, [(0, *ANSWER), (1, ANSWER[0], ANSWER[1], 940)], None, "top", [("soft", 0.3)])),
        (None, ui("keys_desk_det", K["answer"] + 3.2, K["answer"] + 5.9, [(0, *STAGE[:2], 1500), (1, *STAGE[:2], 1380)], None)),
        ("J7", clip(MO3 / "evidence_h.mp4", 0, 27.5)),
        ("J8", clip(MO3 / "limits_h.mp4", 0, 16.5)),
        ("J9", clip(MO3 / "end_h.mp4", 0, 5.0)),
    ]
    return S


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--preview", action="store_true"); a = ap.parse_args()
    S = segments(); segs = [s for _, s in S]
    cut2.version = lambda v: (1920, 1080, R3 / "music" / "p3_bed_xl_s22_00001.flac", segs)
    out = R3 / "cut"; out.mkdir(parents=True, exist_ok=True)
    total = cut2.render_video("judges", out / "_judges_video.mp4", a.preview)
    cut2.render_audio("judges", out / "_judges_music.wav", total)
    # narration: each line starts with its segment (0.3 s in); music ducked 9 dB under the voice
    SR = cut2.SR; n = int(total * SR); vo = np.zeros((n, 2), np.float32); on = np.zeros(n, bool); t0 = 0.0
    for line, seg in S:
        if line:
            x = cut2.load(f"/cwStorage/home/jijh/pulse_data/film3/vo/line_{line}_{VO[line]}.wav")
            act = x[np.abs(x).max(1) > 1e-3]; x *= 10 ** (-17 / 20) / np.sqrt(np.mean(act ** 2))
            s = int((t0 + 0.3) * SR); e = min(n, s + len(x)); vo[s:e] += x[:e - s]; on[s:e] = True
        t0 += (seg["b"] - seg["a"]) / seg.get("speed", 1.0)
    mus = cut2.load(out / "_judges_music.wav")[:n]
    duck = np.convolve(np.where(on, 10 ** (-9 / 20), 1.0), np.ones(int(.3 * SR)) / int(.3 * SR), "same")[:len(mus), None]
    mix = mus * duck.astype(np.float32) + vo[:len(mus)]; mix *= min(1.0, 10 ** (-1 / 20) / np.abs(mix).max())
    subprocess.run([cut2.FF, "-v", "error", "-y", "-f", "f32le", "-ac", "2", "-ar", str(SR), "-i", "-", "-c:a", "pcm_s24le",
                    str(out / "_judges.wav")], input=mix.astype(np.float32).tobytes(), check=True)
    subprocess.run([cut2.FF, "-v", "error", "-y", "-i", str(out / "_judges_video.mp4"), "-i", str(out / "_judges.wav"), "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-af", "loudnorm=I=-14:TP=-1.5:LRA=9", "-ar", "48000", "-c:a", "aac", "-b:a", "256k", "-shortest",
                    "-movflags", "+faststart", str(out / "pulse_judges.mp4")], check=True)
    # check: every narration line must fit inside its picture
    t0 = 0.0
    for line, seg in S:
        d = (seg["b"] - seg["a"]) / seg.get("speed", 1.0)
        if line:
            ln = len(cut2.load(f"/cwStorage/home/jijh/pulse_data/film3/vo/line_{line}_{VO[line]}.wav")) / SR
            print(f"{line} at {t0:6.1f}s  voice {ln:5.1f}s  picture {d:5.1f}s")
        t0 += d
    print("judges", round(total, 1), "s ->", out / "pulse_judges.mp4")


if __name__ == "__main__":
    main()
