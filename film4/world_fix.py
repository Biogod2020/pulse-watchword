"""Soften over-steep phrase-final pitch falls with the WORLD vocoder, touching only the phrase tails.
  world_fix.py IN_DIR OUT_DIR [--target 1.2] [--keep 0.5] [--creak]      (audiolab venv: pyworld)
1. WORLD analysis at 5 ms: harvest F0, cheaptrick spectral envelope, d4c aperiodicity.
2. Phrase ends: pauses >= 0.25 s (20 ms RMS < speech level - 35 dB) and the clip end.
3. In the last 0.45 s of voicing before each end, F0 more than --target semitones below the phrase median is pulled
   up: excess * --keep. With --creak, energetic frames the tracker calls unvoiced at a phrase end (vocal fry) get
   the softened F0 and a lower aperiodicity, so the ending is voiced instead of creaky.
4. The line is resynthesised, but only the edited tails are spliced into the ORIGINAL audio (30 ms crossfades), so
   everything else keeps the original waveform with no vocoder colouring."""
import argparse
from pathlib import Path
import numpy as np, soundfile as sf, pyworld as pw

FP = 5.0


def phrase_ends(y, sr):
    hop = int(.02 * sr); rms = np.array([np.sqrt(np.mean(y[i:i + hop] ** 2) + 1e-12) for i in range(0, len(y) - hop, hop)])
    q = rms < np.percentile(rms, 95) * 10 ** (-35 / 20); ends, n = [], 0
    for i, x in enumerate(q):
        if x: n += 1
        else:
            if n * .02 >= .25 and i - n > 0: ends.append((i - n) * .02)
            n = 0
    return ends + [len(y) / sr]


def fix(path, out, target, keep, creak):
    y, sr = sf.read(path, always_2d=True); y = y.mean(1).astype(np.float64)
    f0, t = pw.harvest(y, sr, f0_floor=65, f0_ceil=420, frame_period=FP)
    sp = pw.cheaptrick(y, f0, t, sr); ap = pw.d4c(y, f0, t, sr)
    en = 10 * np.log10(sp.sum(1) + 1e-12)
    f2, ap2 = f0.copy(), ap.copy(); regions = []
    ends = phrase_ends(y, sr)
    for s0, e in zip([0.0] + ends[:-1], ends):
        ph = (t >= s0) & (t <= e); v = ph & (f0 > 0)
        if v.sum() < 30: continue
        med = np.median(f0[v]); lastv = t[v][-1]
        tail = ph & (t >= lastv - 0.45) & (t <= e)
        if creak:                                          # energetic unvoiced frames right after the last voiced one
            hi = en > np.percentile(en[ph], 60)
            fry = tail & (f0 == 0) & hi & (t <= lastv + 0.15)
        else:
            fry = np.zeros_like(tail)
        tv = tail & (f0 > 0)
        st = np.full(len(f0), np.nan); st[tv] = 12 * np.log2(f0[tv] / med)
        if tv.any() and np.nanmin(st[tv]) >= -target and not fry.any(): continue
        new = np.where(st < -target, -target + (st + target) * keep, st)
        f2[tv] = med * 2 ** (new[tv] / 12)
        if fry.any():
            f2[fry] = med * 2 ** (-target / 12); ap2[fry] = np.minimum(ap2[fry], 0.3)
        regions.append((max(0, lastv - 0.5), min(len(y) / sr, lastv + 0.2)))
    if not regions:
        sf.write(out, y, sr, subtype="PCM_24"); return 0
    z = pw.synthesize(f2, sp, ap2, sr, frame_period=FP)[:len(y)]
    z = np.pad(z, (0, len(y) - len(z)))
    g = np.zeros(len(y)); xf = int(.03 * sr)
    for a, b in regions:
        A, B = int(a * sr), int(b * sr); g[A:B] = 1.0
        g[max(0, A - xf):A] = np.linspace(0, 1, A - max(0, A - xf)); g[B:B + xf] = np.linspace(1, 0, len(g[B:B + xf]))
    # match the resynthesis level to the original inside the regions before splicing
    m = g > 0.5; k = np.sqrt(np.mean(y[m] ** 2) / (np.mean(z[m] ** 2) + 1e-12)) if m.any() else 1.0
    sf.write(out, (1 - g) * y + g * z * k, sr, subtype="PCM_24")
    return len(regions)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--target", type=float, default=1.2); ap.add_argument("--keep", type=float, default=0.5)
    ap.add_argument("--creak", action="store_true"); a = ap.parse_args()
    O = Path(a.out); O.mkdir(parents=True, exist_ok=True)
    for f in sorted(Path(a.inp).glob("*.wav")):
        n = fix(f, O / f.name, a.target, a.keep, a.creak); print(f.name, "edited tails:", n, flush=True)
