"""Paired comparison of strict single-clip Gemini scores: DIR/<variant>__<clip>_r<k>.md.
  paired_strict.py DIR base variant [variant ...]"""
import glob, json, re, sys
import numpy as np
D, base, vs = sys.argv[1], sys.argv[2], sys.argv[3:]
def load(v):
    per, hum, n = {}, 0, 0
    for f in glob.glob(f"{D}/{v}__*.md"):
        clip = f.split("__")[1].rsplit("_r", 1)[0]; m = re.findall(r'\{[^{}]*"naturalness"[^{}]*\}', open(f).read())
        try:
            d = json.loads(m[-1]); per.setdefault(clip, []).append(d["naturalness"]); hum += d["verdict"] == "human"; n += 1
        except Exception:
            pass
    return {k: float(np.mean(x)) for k, x in per.items()}, hum, n
B, bh, bn = load(base); print(f"{base:10s} naturalness {np.mean(list(B.values())):.2f}  human {bh}/{bn}")
rng = np.random.default_rng(0)
for v in vs:
    X, h, n = load(v); keys = [k for k in B if k in X]; d = np.array([X[k] - B[k] for k in keys])
    bs = [rng.choice(d, len(d)).mean() for _ in range(5000)]
    print(f"{v:10s} naturalness {np.mean(list(X.values())):.2f}  human {h}/{n}  paired diff vs {base} {d.mean():+.2f}  95% CI [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}]")
