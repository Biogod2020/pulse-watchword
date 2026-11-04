"""Objective naturalness (MOS) predictions for speech clips: UTMOS22 (VoiceMOS 2022 winner, `utmos` on PyPI) and
Distill-MOS (Microsoft 2024, `distillmos`). Run with /cwStorage/home/jijh/.venvs/mos/bin/python (CPU is fine).
  mos_score.py OUT.json group=glob [group=glob ...]
Long clips are scored in 10 s windows (2 s hop) and averaged, so a 45 s paragraph and a 7 s line are comparable."""
import glob, json, os, sys
import numpy as np, soundfile as sf, librosa, torch

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def windows(y, sr, w=10.0, hop=8.0):
    n, step = int(w * sr), int(hop * sr)
    if len(y) <= n:
        return [y]
    return [y[i:i + n] for i in range(0, len(y) - int(3 * sr), step)]


def main():
    out, groups = sys.argv[1], dict(a.split("=", 1) for a in sys.argv[2:])
    import utmos, distillmos
    um = utmos.Score(); dm = distillmos.ConvTransformerSQAModel().eval()
    res = {}
    for g, pat in groups.items():
        rows = []
        for f in sorted(glob.glob(pat)):
            y, sr = sf.read(f, always_2d=True); y = librosa.resample(y.mean(1), orig_sr=sr, target_sr=16000).astype(np.float32)
            nz = np.flatnonzero(np.abs(y) > 1e-3); y = y[nz[0]:nz[-1]]
            u, d = [], []
            for seg in windows(y, 16000):
                x = torch.from_numpy(seg)[None]
                u.append(float(um.calculate_wav(x, 16000)))
                with torch.no_grad():
                    d.append(float(dm(x).squeeze()))
            rows.append({"file": os.path.basename(f), "utmos": round(float(np.mean(u)), 3), "distillmos": round(float(np.mean(d)), 3)})
        res[g] = {"n": len(rows), "utmos": round(float(np.mean([r["utmos"] for r in rows])), 3) if rows else None,
                  "distillmos": round(float(np.mean([r["distillmos"] for r in rows])), 3) if rows else None, "clips": rows}
        print(f"{g:14s} n={len(rows):2d}  UTMOS {res[g]['utmos']}  Distill-MOS {res[g]['distillmos']}", flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
