"""Cut a (separated) narration track into utterance clips: split at pauses >= 0.5 s, keep 4-15 s pieces.
  split_utts.py IN.wav OUT_DIR PREFIX [--max 12]   (audiolab venv)"""
import sys
from pathlib import Path
import numpy as np, soundfile as sf

inp, out, pre = sys.argv[1], Path(sys.argv[2]), sys.argv[3]; mx = int(sys.argv[5]) if len(sys.argv) > 5 else 12
out.mkdir(parents=True, exist_ok=True)
y, sr = sf.read(inp, always_2d=True); y = y.mean(1); hop = int(.02 * sr)
rms = np.array([np.sqrt(np.mean(y[i:i + hop] ** 2) + 1e-12) for i in range(0, len(y) - hop, hop)])
on = rms > np.percentile(rms, 97) * 10 ** (-32 / 20)
segs, s, quiet = [], None, 0
for i, v in enumerate(on):
    if v:
        if s is None: s = i
        quiet = 0
    elif s is not None:
        quiet += 1
        if quiet * .02 >= .5:
            segs.append((s, i - quiet + 1)); s = None; quiet = 0
if s is not None: segs.append((s, len(on)))
# merge neighbours into 4-15 s utterance groups
groups, cur = [], None
for a, b in segs:
    if cur and (b - cur[0]) * .02 <= 15: cur = (cur[0], b)
    else:
        if cur: groups.append(cur)
        cur = (a, b)
if cur: groups.append(cur)
n = 0
for a, b in groups:
    if (b - a) * .02 < 4: continue
    sf.write(out / f"{pre}_{n:02d}.wav", y[max(0, a * hop - int(.05 * sr)): b * hop + int(.1 * sr)], sr); n += 1
    if n >= mx: break
print(pre, n, "clips")
