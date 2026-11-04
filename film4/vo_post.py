"""Narration post chain (audiolab venv): make TTS lines sound like one voice recorded in one booth.
  vo_post.py IN_DIR OUT_DIR [--pick pick.json] [--ids M1,M2]   (IN_DIR/line_<id>_<k>.wav or <id>.wav -> OUT_DIR/<id>.wav)

Per line, in order:
 1. 48 kHz, mono; trim edge silence to 60 ms lead / 150 ms tail (the film places lines itself).
 2. High-pass 70 Hz, gentle low-mid cleanup (-2 dB at 300 Hz); no presence or air boost (they brought the TTS sizzle back).
 3. De-esser: STFT gain on 5-11 kHz, only where that band rises more than 4 dB above its median (at most -9 dB).
 4. Compressor 2.5:1.
 5. A small, dry-ish room (6% wet) so the voice sits in a space instead of digital silence.
 6. Every line normalised to -20 LUFS (consistent level between lines); peaks kept <= -1 dBFS by gain, not a limiter.
"""
import argparse, json, re
from pathlib import Path
import numpy as np, librosa, soundfile as sf, pyloudnorm as pyln
from pedalboard import Pedalboard, HighpassFilter, PeakFilter, HighShelfFilter, Compressor, Reverb, Limiter

SR = 48000
import os
CHAIN = os.environ.get("VO_CHAIN", "hpf+eq+lightdeess+room")   # chosen by UTMOS/Distill-MOS ablation (docs/audio_narration.md)


def deess(y, sr, lo=5000, hi=11000, over_db=4.0, max_cut_db=9.0):
    n_fft, hop = 1024, 256
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    f = librosa.fft_frequencies(sr=sr, n_fft=n_fft); band = (f >= lo) & (f <= hi)
    e = 10 * np.log10((np.abs(S[band]) ** 2).sum(0) + 1e-12)
    voiced = e > e.max() - 50
    ref = np.median(e[voiced]) if voiced.any() else np.median(e)
    cut = np.clip(e - ref - over_db, 0, max_cut_db)
    cut = np.convolve(cut, np.ones(3) / 3, "same")                    # smooth the gain over ~16 ms
    g = np.ones_like(S, dtype=float); g[band] = 10 ** (-cut / 20)
    return librosa.istft(S * g, hop_length=hop, length=len(y))


def process(path: Path) -> np.ndarray:
    """VO_CHAIN steps (env): any of hpf, eq, deess, comp, room, separated by '+'; 'bright'/'dark' are the old presets."""
    steps = {"bright": "hpf+eq+bright+deess+comp+room", "dark": "hpf+eq+deess+comp+room"}.get(CHAIN, CHAIN).split("+")
    y, sr = sf.read(path, always_2d=True); y = y.mean(1)
    y = librosa.resample(y, orig_sr=sr, target_sr=SR) if sr != SR else y
    nz = np.flatnonzero(np.abs(y) > 10 ** (-48 / 20) * np.abs(y).max())
    y = y[max(0, nz[0] - int(.06 * SR)): nz[-1] + int(.15 * SR)].astype(np.float32)
    fx = []
    if "hpf" in steps: fx.append(HighpassFilter(70))
    if "eq" in steps: fx.append(PeakFilter(300, -2.0, 0.9))
    if "bright" in steps: fx += [PeakFilter(3000, 1.5, 1.0), HighShelfFilter(10000, 2.0)]
    if fx: y = Pedalboard(fx)(y, SR)
    if "deess" in steps:
        y = deess(y, SR, **({"hi": 10000, "over_db": 6.0, "max_cut_db": 7.0} if CHAIN in ("bright",) or "lightdeess" in steps else {}))
    if "comp" in steps: y = Pedalboard([Compressor(threshold_db=-24, ratio=2.5, attack_ms=8, release_ms=120)])(y.astype(np.float32), SR)
    y = np.concatenate([y, np.zeros(int(.25 * SR))]).astype(np.float32)
    if "room" in steps: y = Pedalboard([Reverb(room_size=0.18, damping=0.6, wet_level=0.06, dry_level=1.0, width=0.0)])(y, SR)
    m = pyln.Meter(SR); y = y * 10 ** ((-20 - m.integrated_loudness(y)) / 20)
    # no limiter: it cost ~0.14 Distill-MOS (measured); a plain gain change keeps peaks <= -1 dBFS instead
    return y * min(1.0, 10 ** (-1 / 20) / np.abs(y).max())

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("inp"); ap.add_argument("out"); ap.add_argument("--pick"); ap.add_argument("--ids")
    a = ap.parse_args(); I, O = Path(a.inp), Path(a.out); O.mkdir(parents=True, exist_ok=True)
    if a.pick:
        src = {k: I / f"line_{k}_{v}.wav" for k, v in json.load(open(a.pick)).items()}
    else:
        src = {re.sub(r"\.wav$", "", p.name): p for p in I.glob("M*.wav")}
    for k, p in sorted(src.items(), key=lambda x: int(re.sub(r"\D", "", x[0]) or 0)):
        if a.ids and k not in a.ids.split(","):
            continue
        y = process(p); sf.write(O / f"{k}.wav", y, SR, subtype="PCM_24")
        print(k, p.name, f"{len(y) / SR:.2f}s", flush=True)
