"""Fix over-long pauses in English narration takes without re-synthesis (runs on gpucw1 with the qwen stack: sherpa-onnx).
  pause_fix_en.py IN_DIR OUT_DIR LINES_JSON [--report r.json]          IN_DIR/line_<id>_0.wav -> OUT_DIR/line_<id>_0.wav
  pause_fix_en.py remote IN_DIR OUT_DIR LINES_JSON                      (launch from headcw)
1. SenseVoice gives a start time for each recognised token; tokens are joined into words and aligned (Levenshtein) to
   the script words, so every silence can be placed between two script words and read the punctuation between them.
2. Silences (20 ms RMS < speech level - 35 dB, >= 120 ms) longer than the target for that boundary are shortened by
   cutting out their middle with 15 ms crossfades: no punctuation 0.10 s, comma 0.25 s, colon / dash 0.33 s,
   sentence end 0.45 s (tightened after comparing pause length with human read speech). Nothing is lengthened and the speech itself is never edited."""
import json, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve()
TARGET = {"": 0.10, ",": 0.25, ":": 0.33, ";": 0.33, "-": 0.33, ".": 0.45, "?": 0.45, "!": 0.45}


def script_words(text):
    out = []
    for m in re.finditer(r"([A-Za-z0-9']+(?:-[A-Za-z0-9']+)*)([^A-Za-z0-9']*)", text):
        w = m.group(1).lower().replace("-", " ").split(); p = m.group(2).strip()
        punct = next((c for c in ".?!:;," if c in p), "-" if "—" in p or " - " in m.group(2) else "")
        for i, x in enumerate(w):
            out.append((x, punct if i == len(w) - 1 else ""))
    return out


def align(rec, ref):
    """map each recognised word index -> script word index (or None)"""
    n, m = len(rec), len(ref); D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): D[i][0] = i
    for j in range(m + 1): D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (rec[i - 1] != ref[j - 1]))
    i, j, mp = n, m, {}
    while i > 0 and j > 0:
        if D[i][j] == D[i - 1][j - 1] + (rec[i - 1] != ref[j - 1]):
            mp[i - 1] = j - 1; i, j = i - 1, j - 1
        elif D[i][j] == D[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return mp


def fix(path, text, out, asr):
    import numpy as np, soundfile as sf, scipy.signal as ss
    y, sr = sf.read(path, always_2d=True); y = y.mean(1)
    s = asr.create_stream(); s.accept_waveform(16000, ss.resample_poly(y, 16000, sr).astype("float32")); asr.decode_stream(s)
    toks, ts = s.result.tokens, s.result.timestamps
    words, starts = [], []
    for t, st in zip(toks, ts):
        piece = t.replace("▁", " ")
        if t.startswith(("▁", " ")) or not words:
            if re.sub(r"[^A-Za-z0-9']", "", piece):
                words.append(re.sub(r"[^a-z0-9']", "", piece.lower())); starts.append(st)
        else:
            words[-1] += re.sub(r"[^a-z0-9']", "", piece.lower())
    ref = script_words(text); mp = align(words, [w for w, _ in ref])
    hop = int(.02 * sr); rms = np.array([np.sqrt(np.mean(y[i:i + hop] ** 2) + 1e-12) for i in range(0, len(y) - hop, hop)])
    quiet = rms < np.percentile(rms, 95) * 10 ** (-35 / 20)
    runs, i = [], 0
    while i < len(quiet):
        if quiet[i]:
            j = i
            while j < len(quiet) and quiet[j]: j += 1
            if (j - i) * .02 >= .12 and i > 0 and j < len(quiet): runs.append((i * .02, j * .02))
            i = j
        else:
            i += 1
    cuts, report = [], []
    for a, b in runs:
        nxt = next((k for k, st in enumerate(starts) if st > a + 0.02), None)   # CTC times lag the onset
        if nxt is None or nxt == 0:
            continue
        si = mp.get(nxt - 1); punct = ref[si][1] if si is not None else ","     # unknown boundary: treat as a comma
        tgt = TARGET.get(punct, 0.30); dur = b - a
        if punct in ".?!" and len(text.split()) < 8:   # a short slogan keeps a real beat ("Pulse. ... Always watching")
            tgt = 0.8
        report.append({"at": round(a, 2), "dur": round(dur, 2), "after": ref[si][0] if si is not None else "?", "punct": punct,
                       "target": tgt, "cut": round(max(0, dur - tgt), 2)})
        if dur > tgt + 0.03:
            mid = (a + b) / 2; cuts.append((mid - (dur - tgt) / 2, mid + (dur - tgt) / 2))
    xf = int(.015 * sr); outy = []; pos = 0
    for a, b in cuts:
        A, B = int(a * sr), int(b * sr); seg = y[pos:A]
        if outy and len(seg) > xf:
            ramp = np.linspace(0, 1, xf); prev = outy[-1]; prev[-xf:] = prev[-xf:] * (1 - ramp) + seg[:xf] * ramp; seg = seg[xf:]
        outy.append(seg.copy()); pos = B
    seg = y[pos:]
    if outy and len(seg) > xf:
        ramp = np.linspace(0, 1, xf); prev = outy[-1]; prev[-xf:] = prev[-xf:] * (1 - ramp) + seg[:xf] * ramp; seg = seg[xf:]
    outy.append(seg); z = np.concatenate(outy); sf.write(out, z, sr)
    return {"file": Path(path).name, "before": round(len(y) / sr, 2), "after": round(len(z) / sr, 2), "pauses": report}


if __name__ == "__main__":
    if sys.argv[1] == "remote":
        cmd = (f"cd /tmp && HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONPATH=/cwStorage/home/jijh/venvs/film2_qwen_tts_overlay "
               f"/tmp/jijh-djev/env234/bin/python {HERE} " + " ".join(sys.argv[2:]))
        sys.exit(subprocess.call(["ssh", "-o", "RemoteCommand=none", "-o", "RequestTTY=no", "-o", "BatchMode=yes",
                                  "-o", "ClearAllForwardings=yes", "gpucw1", cmd], stdin=subprocess.DEVNULL))
    import sherpa_onnx
    I, O, lines = Path(sys.argv[1]), Path(sys.argv[2]), {x["id"]: x["text"] for x in json.load(open(sys.argv[3]))}
    O.mkdir(parents=True, exist_ok=True)
    sv = "/tmp/jijh-tts/models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
    asr = sherpa_onnx.OfflineRecognizer.from_sense_voice(model=f"{sv}/model.int8.onnx", tokens=f"{sv}/tokens.txt", num_threads=8,
                                                         language="en", use_itn=False)
    rep = []
    for f in sorted(I.glob("line_*_0.wav")):
        lid = f.name.split("_")[1]
        r = fix(f, lines[lid], O / f.name, asr); rep.append(r)
        print(lid, r["before"], "->", r["after"], [(p["after"], p["punct"], p["dur"]) for p in r["pauses"] if p["cut"] > 0], flush=True)
    (O / "pause_report.json").write_text(json.dumps(rep, indent=1))
