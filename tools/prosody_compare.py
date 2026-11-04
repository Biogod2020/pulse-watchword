"""Prosody statistics of narration clips vs a human reference set (audiolab venv: parselmouth, librosa).
  prosody_compare.py OUT.json ref=glob group=glob [...]
Per clip:
- f0 spread: std of F0 in semitones, voiced frames only;
- final fall: F0 drop over the last 0.4 s of voicing vs the clip median, in semitones;
- pause share: fraction of time in pauses >= 120 ms;
- mean internal pause (s);
- syllable rate: energy-envelope peaks per second of speech.
Each group is reported with mean values and its distance to the reference group: mean |z| over the features, using the
reference mean and std. A lower distance means the group is closer to human prosody."""
import glob, json, sys
import numpy as np, soundfile as sf, librosa, parselmouth
from scipy.signal import find_peaks


def feats(path):
    y, sr = sf.read(path, always_2d=True); y = y.mean(1); y = librosa.resample(y, orig_sr=sr, target_sr=16000); sr = 16000
    nz = np.flatnonzero(np.abs(y) > 10 ** (-40 / 20) * np.abs(y).max()); y = y[nz[0]:nz[-1]]
    p = parselmouth.Sound(y, sr).to_pitch(time_step=0.01, pitch_floor=65, pitch_ceiling=420)
    f0 = p.selected_array["frequency"]; t = p.xs(); v = f0 > 0
    st = 12 * np.log2(f0[v] / np.median(f0[v]))
    tv = t[v]; last = st[tv >= tv[-1] - 0.4]
    hop = int(.01 * sr); rms = np.array([np.sqrt(np.mean(y[i:i + hop] ** 2) + 1e-12) for i in range(0, len(y) - hop, hop)])
    q = rms < np.percentile(rms, 95) * 10 ** (-35 / 20); runs, n = [], 0
    for x in q:
        n = n + 1 if x else (runs.append(n) or 0)
    pauses = [r * .01 for r in runs if r * .01 >= .12]
    env = librosa.feature.rms(y=y, frame_length=400, hop_length=160)[0]; env = np.convolve(env, np.ones(5) / 5, "same")
    pk, _ = find_peaks(env, height=env.max() * 0.08, distance=8, prominence=env.max() * 0.04)
    speech_s = len(y) / sr - sum(pauses)
    return {"f0_spread_st": float(np.std(st)), "final_fall_st": float(-np.mean(last)), "pause_share": float(sum(pauses) / (len(y) / sr)),
            "pause_mean_s": float(np.mean(pauses)) if pauses else 0.0, "syll_rate": float(len(pk) / max(speech_s, .1))}


def main():
    out, gs = sys.argv[1], dict(a.split("=", 1) for a in sys.argv[2:])
    data = {g: [feats(f) for f in sorted(glob.glob(p))] for g, p in gs.items()}
    keys = list(data["ref"][0]); R = np.array([[d[k] for k in keys] for d in data["ref"]]); mu, sd = R.mean(0), R.std(0) + 1e-9
    res = {}
    for g, rows in data.items():
        X = np.array([[d[k] for k in keys] for d in rows]); m = X.mean(0)
        dist = float(np.mean(np.abs((m - mu) / sd)))
        res[g] = {"n": len(rows), **{k: round(float(x), 3) for k, x in zip(keys, m)}, "distance_to_ref": round(dist, 3)}
        print(f"{g:16s} n={len(rows):2d} " + " ".join(f"{k} {x:.3f}" for k, x in zip(keys, m)) + f"  | dist {dist:.3f}")
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
