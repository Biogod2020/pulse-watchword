"""Soften over-steep sentence-final pitch falls (the "flat, dropping endings" heard in TTS) with Praat PSOLA.
  prosody_fix.py IN_DIR OUT_DIR [--keep 0.5] [--target 1.2]       (audiolab venv: parselmouth)
For every phrase end (a pause >= 0.25 s or the end of the clip) the last 0.45 s of voiced F0 is examined; if it falls
more than --target semitones below the phrase's median F0, the part of the fall beyond the target is scaled by
--keep (0.5 = keep half of the excess). Only the pitch tier is changed (overlap-add resynthesis); timing is untouched.
Measured reference: human read speech (LibriSpeech) ends ~1.2 st below the median; the chosen TTS takes ~3.1 st."""
import argparse
from pathlib import Path
import numpy as np, soundfile as sf, parselmouth
from parselmouth.praat import call


def fix(path, out, keep, target):
    snd = parselmouth.Sound(str(path)); sr = snd.sampling_frequency; y = snd.values[0]
    hop = int(.01 * sr); rms = np.array([np.sqrt(np.mean(y[i:i + hop] ** 2) + 1e-12) for i in range(0, len(y) - hop, hop)])
    q = rms < np.percentile(rms, 95) * 10 ** (-35 / 20)
    ends, n = [], 0
    for i, x in enumerate(q):
        if x:
            n += 1
        else:
            if n * .01 >= .25 and i - n > 0:
                ends.append((i - n) * .01)
            n = 0
    ends.append(len(y) / sr)
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=65, pitch_ceiling=420)
    f0 = pitch.selected_array["frequency"]; t = pitch.xs()
    man = call(snd, "To Manipulation", 0.01, 65, 420); tier = call(man, "Extract pitch tier")
    starts = [0.0] + ends[:-1]; changed = 0
    for s0, e in zip(starts, ends):
        ph = (t >= s0) & (t <= e) & (f0 > 0)
        if ph.sum() < 20:
            continue
        med = np.median(f0[ph]); tail = ph & (t >= e - 0.45)
        if tail.sum() < 5:
            continue
        st = 12 * np.log2(f0[tail] / med)
        if -st.min() <= target:
            continue
        new = np.where(st < -target, -target + (st + target) * keep, st)
        # replace the tier points inside the tail with the softened contour
        call(tier, "Remove points between", float(t[tail][0]), float(t[tail][-1]))
        for tt, s in zip(t[tail], new):
            call(tier, "Add point", float(tt), float(med * 2 ** (s / 12)))
        changed += 1
    call([man, tier], "Replace pitch tier")
    res = call(man, "Get resynthesis (overlap-add)")
    sf.write(out, res.values[0], int(sr), subtype="PCM_24")
    return changed, len(ends)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--keep", type=float, default=0.5); ap.add_argument("--target", type=float, default=1.2); a = ap.parse_args()
    I, O = Path(a.inp), Path(a.out); O.mkdir(parents=True, exist_ok=True)
    for f in sorted(I.glob("*.wav")):
        c, n = fix(f, O / f.name, a.keep, a.target); print(f.name, f"softened {c}/{n} phrase ends", flush=True)
