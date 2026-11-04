"""Objective audit of narration takes: speaking rate, pitch, loudness, bandwidth, edge silence, DNSMOS.
Run with the audiolab venv (/cwStorage/home/jijh/.venvs/audiolab/run).
  audio_audit.py --vo vo4.json --pick DIR/pick.json DIR            # the chosen take per line
  audio_audit.py --vo vo4.json DIR --all                            # every line_*.wav in DIR
"""
import argparse, json, re
from pathlib import Path
import numpy as np, librosa, parselmouth, pyloudnorm as pyln, soundfile as sf
from speechmos import dnsmos


def audit(path: Path, text: str) -> dict:
    y, sr = sf.read(path, always_2d=True); y = y.mean(1)
    words = len(re.findall(r"[A-Za-z0-9']+", text))
    nz = np.flatnonzero(np.abs(y) > 10 ** (-45 / 20) * np.abs(y).max())
    lead, tail = nz[0] / sr, (len(y) - nz[-1]) / sr
    speech = y[nz[0]:nz[-1]]
    snd = parselmouth.Sound(speech, sr); f0 = snd.to_pitch(time_step=0.01, pitch_floor=70, pitch_ceiling=450).selected_array["frequency"]
    f0 = f0[f0 > 0]
    # pauses inside the line: gaps of >= 250 ms below -40 dB (relative), in 20 ms frames
    fr = librosa.feature.rms(y=speech, frame_length=int(.02 * sr), hop_length=int(.02 * sr))[0]
    quiet = fr < fr.max() * 10 ** (-40 / 20); runs, n = [], 0
    for q in quiet:
        n = n + 1 if q else (runs.append(n) or 0)
    pauses = [r * .02 for r in runs if r * .02 >= .25]
    S = np.abs(librosa.stft(speech, n_fft=2048)); roll = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=.99)[0]
    y16 = librosa.resample(speech, orig_sr=sr, target_sr=16000)
    mos = dnsmos.run(np.clip(y16, -1, 1).astype(np.float32), 16000)
    return {"file": path.name, "sr": sr, "dur": round(len(y) / sr, 2), "lead_s": round(lead, 2), "tail_s": round(tail, 2),
            "wps": round(words / (len(speech) / sr), 2), "f0_med": round(float(np.median(f0)), 1),
            "f0_p10_p90": [round(float(np.percentile(f0, 10)), 1), round(float(np.percentile(f0, 90)), 1)],
            "lufs": round(pyln.Meter(sr).integrated_loudness(speech), 1), "peak_db": round(20 * np.log10(np.abs(y).max() + 1e-9), 1),
            "rolloff99_hz": int(np.median(roll)), "pauses": [round(p, 2) for p in pauses],
            "dnsmos_ovrl": round(mos["ovrl_mos"], 2), "dnsmos_sig": round(mos["sig_mos"], 2), "dnsmos_bak": round(mos["bak_mos"], 2),
            "p808": round(mos["p808_mos"], 2)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--vo", required=True); ap.add_argument("--pick")
    ap.add_argument("--all", action="store_true"); ap.add_argument("--out")
    a = ap.parse_args(); D = Path(a.dir); text = {x["id"]: x["text"] for x in json.load(open(a.vo))}
    if a.pick:
        pick = json.load(open(a.pick)); files = [(k, D / f"line_{k}_{v}.wav") for k, v in pick.items()]
    else:
        files = [(re.match(r"line_(M\d+)_", p.name).group(1), p) for p in sorted(D.glob("line_*.wav"))]
    rows = []
    for k, p in sorted(files, key=lambda x: (int(x[0][1:]), x[1].name)):
        r = audit(p, text[k]); r["id"] = k; rows.append(r)
        print(f"{k:4s} {r['file']:16s} sr {r['sr']} dur {r['dur']:5.2f} lead {r['lead_s']:.2f} tail {r['tail_s']:.2f} "
              f"wps {r['wps']:.2f} f0 {r['f0_med']:.0f} [{r['f0_p10_p90'][0]:.0f}-{r['f0_p10_p90'][1]:.0f}] lufs {r['lufs']:6.1f} "
              f"roll99 {r['rolloff99_hz']:5d} mos {r['dnsmos_ovrl']:.2f}/{r['p808']:.2f} pauses {r['pauses']}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1, default=float))
