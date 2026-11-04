"""Prosody features per take and distance to a reference set (professional narration).
  take_features.py REF_GLOB OUT.json take.wav [...]     (audiolab venv)
Features: f0 spread (st), phrase-end fall (st), syllable rate, mean pause; distance = mean |z| vs the reference."""
import glob, json, sys
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from prosody_compare import feats
from end_fall import end_fall

KEYS = ["f0_spread_st", "end_fall_st", "syll_rate", "pause_mean_s"]


def f(path):
    d = feats(path); return {"f0_spread_st": d["f0_spread_st"], "end_fall_st": end_fall(path)[0], "syll_rate": d["syll_rate"],
                             "pause_mean_s": d["pause_mean_s"]}


if __name__ == "__main__":
    ref = [f(p) for p in sorted(glob.glob(sys.argv[1]))]
    R = np.array([[r[k] for k in KEYS] for r in ref]); mu, sd = np.nanmean(R, 0), np.nanstd(R, 0) + 1e-9
    out = {"ref_mean": dict(zip(KEYS, mu.round(3).tolist())), "takes": {}}
    for p in sys.argv[3:]:
        x = f(p); v = np.array([x[k] for k in KEYS]); x["dist"] = float(np.nanmean(np.abs((v - mu) / sd)))
        out["takes"][p.rsplit("/", 1)[-1]] = {k: round(float(val), 3) for k, val in x.items()}
    json.dump(out, open(sys.argv[2], "w"), indent=1); print("ref", out["ref_mean"], "takes", len(out["takes"]))
