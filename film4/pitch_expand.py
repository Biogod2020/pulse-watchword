"""Widen the pitch range of narration lines (Praat PSOLA via parselmouth), keeping timing untouched.
  pitch_expand.py IN_DIR OUT_DIR [--k 1.2]      (audiolab venv)
Every pitch-tier point is moved to median * 2**(k * st / 12), where st is its distance from the line median in
semitones. k = 1.2 widens a 3.7 st spread to about 4.4 st (professional narration measured 4.6)."""
import argparse
from pathlib import Path
import numpy as np, soundfile as sf, parselmouth
from parselmouth.praat import call


def expand(path, out, k):
    snd = parselmouth.Sound(str(path)); sr = snd.sampling_frequency
    man = call(snd, "To Manipulation", 0.01, 65, 420); tier = call(man, "Extract pitch tier")
    n = call(tier, "Get number of points")
    ts = [call(tier, "Get time from index", i) for i in range(1, n + 1)]
    fs = np.array([call(tier, "Get value at index", i) for i in range(1, n + 1)])
    med = np.median(fs); new = med * 2 ** (k * 12 * np.log2(fs / med) / 12)
    call(tier, "Remove points between", 0, snd.duration)
    for t, f in zip(ts, new):
        call(tier, "Add point", t, float(f))
    call([man, tier], "Replace pitch tier")
    res = call(man, "Get resynthesis (overlap-add)")
    sf.write(out, res.values[0], int(sr), subtype="PCM_24")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("inp"); ap.add_argument("out"); ap.add_argument("--k", type=float, default=1.2)
    a = ap.parse_args(); O = Path(a.out); O.mkdir(parents=True, exist_ok=True)
    for f in sorted(Path(a.inp).glob("*.wav")):
        expand(f, O / f.name, a.k); print(f.name, flush=True)
