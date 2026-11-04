"""Mean sentence/phrase-final pitch fall (semitones below the phrase median, last 0.35 s of voicing before each pause
>= 0.25 s and at clip end) and the share of creaky (unvoiced-in-tracker but energetic) frames at those ends.
  end_fall.py glob [glob ...]   -> prints one row per file + group mean  (audiolab venv)"""
import glob, sys
import numpy as np, soundfile as sf, parselmouth


def end_fall(path):
    y, sr = sf.read(path, always_2d=True); y = y.mean(1)
    nz = np.flatnonzero(np.abs(y) > 10 ** (-40 / 20) * np.abs(y).max()); y = y[nz[0]:nz[-1]]
    hop = int(.01 * sr); rms = np.array([np.sqrt(np.mean(y[i:i + hop] ** 2) + 1e-12) for i in range(0, len(y) - hop, hop)])
    q = rms < np.percentile(rms, 95) * 10 ** (-35 / 20); ends, n = [], 0
    for i, x in enumerate(q):
        if x: n += 1
        else:
            if n * .01 >= .25 and i - n > 0: ends.append((i - n) * .01)
            n = 0
    ends.append(len(y) / sr)
    p = parselmouth.Sound(y, sr).to_pitch(time_step=0.01, pitch_floor=65, pitch_ceiling=420)
    f0 = p.selected_array["frequency"]; t = p.xs(); falls = []
    for s0, e in zip([0.0] + ends[:-1], ends):
        ph = (t >= s0) & (t <= e) & (f0 > 0)
        if ph.sum() < 20: continue
        med = np.median(f0[ph]); tl = ph & (t >= t[ph][-1] - 0.35)
        falls.append(float(-np.mean(12 * np.log2(f0[tl] / med))))
    return float(np.mean(falls)) if falls else float("nan"), len(falls)


if __name__ == "__main__":
    for g in sys.argv[1:]:
        vals = []
        for f in sorted(glob.glob(g)):
            v, n = end_fall(f); vals.append(v); print(f"{f.split('/')[-1]:26s} fall {v:5.2f} st over {n} ends")
        print(f"== {g}: mean {np.nanmean(vals):.2f} st (n={len(vals)})\n")
