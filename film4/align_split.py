"""Cut ONE continuous narration take into the script's lines (runs on gpucw1 with the qwen stack: sherpa-onnx).
  align_split.py TAKE.wav VO_JSON OUT_DIR            -> OUT_DIR/line_<id>_0.wav + OUT_DIR/split.json
  align_split.py remote TAKE.wav VO_JSON OUT_DIR     (launch from headcw)
1. SenseVoice token times -> words, aligned (Levenshtein) to the script words, so each line's first and last word
   have an approximate time (CTC times lag the onset slightly).
2. Each boundary is cut in the middle of the last silence >= 60 ms (20 ms RMS < speech level - 35 dB) after the
   last word of one line starts and before the first word of the next (else at the quietest frame there); every piece keeps 80 ms before and after its speech.
3. Each piece is transcribed again: word error rate vs. its line (digits spelled out)."""
import json, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve()


def norm(s):
    s = s.lower().replace("34", "thirty four").replace("-", " ")
    return re.findall(r"[a-z0-9']+", s)


def wer(a, b):
    a, b = norm(a), norm(b); d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (a[i - 1] != b[j - 1]))
    return d[len(b)] / max(1, len(a))


def main(take, vo, out):
    import numpy as np, soundfile as sf, scipy.signal as ss, sherpa_onnx
    sys.path.insert(0, str(HERE.parent)); from pause_fix_en import align
    sv = "/tmp/jijh-tts/models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
    asr = sherpa_onnx.OfflineRecognizer.from_sense_voice(model=f"{sv}/model.int8.onnx", tokens=f"{sv}/tokens.txt", num_threads=8,
                                                         language="en", use_itn=False)
    def rec(x, sr):
        s = asr.create_stream(); s.accept_waveform(16000, ss.resample_poly(x, 16000, sr).astype("float32")); asr.decode_stream(s)
        return s.result
    lines = json.load(open(vo)); O = Path(out); O.mkdir(parents=True, exist_ok=True)
    y, sr = sf.read(take, always_2d=True); y = y.mean(1)
    # SenseVoice works best on < ~30 s: transcribe in 25 s windows at silences, then merge token times
    hop = int(.02 * sr); rms = np.array([np.sqrt(np.mean(y[i:i + hop] ** 2) + 1e-12) for i in range(0, len(y) - hop, hop)])
    q = rms < np.percentile(rms, 95) * 10 ** (-35 / 20)
    words, starts, pos = [], [], 0
    while pos < len(y):
        end = min(len(y), pos + int(25 * sr))
        if end < len(y):                                   # move the window end to the middle of a quiet run
            a, b = (end - int(6 * sr)) // hop, end // hop
            qs = [i for i in range(a, b) if q[i]]
            if qs: end = qs[len(qs) // 2] * hop
        r = rec(y[pos:end], sr)
        for t, st in zip(r.tokens, r.timestamps):
            piece = re.sub(r"[^a-z0-9']", "", t.replace("▁", " ").lower())
            if (t.startswith(("▁", " ")) or not words) and piece:
                words.append(piece); starts.append(pos / sr + st)
            elif piece and words:
                words[-1] += piece
        pos = end
    ref = []; owner = []
    for li, ln in enumerate(lines):
        for w in norm(ln["text"]):
            ref.append(w); owner.append(li)
    mp = align(words, ref); inv = {v: k for k, v in mp.items()}
    first = {}; last = {}
    for ri, li in enumerate(owner):
        if ri in inv:
            first.setdefault(li, starts[inv[ri]]); last[li] = starts[inv[ri]]
    cuts = [0.0]
    for li in range(len(lines) - 1):
        # the LAST silence (>= 120 ms) before the next line's first word: the longest one can be a pause inside a
        # line ("Just | sentences."), which moved a word into the wrong clip
        a = last.get(li, cuts[-1]) + 0.25; b = max(first.get(li + 1, a + 1.0) + 0.08, a + 0.1)   # after the last word starts
        ia, ib = int(a / .02), min(int(b / .02), len(q)); best, i = None, ib - 1
        while i >= ia:                                    # last silence >= 60 ms before the next line's first word
            if q[i]:
                j = i
                while j >= ia and q[j]: j -= 1
                if (i - j) * .02 >= .06:
                    best = (i + j + 1) // 2; break
                i = j
            i -= 1
        if best is None and ib > ia:                       # run-on read: cut at the quietest frame instead
            best = ia + int(np.argmin(rms[ia:ib]))
        cuts.append(best * .02 if best is not None else (a + b) / 2)
    cuts.append(len(y) / sr)
    # verify each boundary by content: try every silence (>= 60 ms) within 1.5 s of the cut and keep the one whose two
    # neighbouring pieces transcribe closest to their own lines (word timing alone can put a cut inside a line)
    for bi in range(1, len(cuts) - 1):
        lo, hi = max(cuts[bi - 1] + 0.3, cuts[bi] - 1.5), min(cuts[bi + 1] - 0.3, cuts[bi] + 1.5)
        cands, i = [cuts[bi]], int(lo / .02)
        while i < min(int(hi / .02), len(q)):
            if q[i]:
                j = i
                while j < len(q) and q[j]: j += 1
                if (j - i) * .02 >= .06: cands.append((i + j) / 2 * .02)
                i = j
            i += 1
        def cost(c):
            l = rec(y[int(cuts[bi - 1] * sr):int(c * sr)], sr).text; r = rec(y[int(c * sr):int(cuts[bi + 1] * sr)], sr).text
            return wer(lines[bi - 1]["text"], l) + wer(lines[bi]["text"], r)
        scored = sorted((cost(c), c) for c in set(round(c, 2) for c in cands))
        cuts[bi] = scored[0][1]
    rep = []
    for li, ln in enumerate(lines):
        seg = y[int(cuts[li] * sr):int(cuts[li + 1] * sr)]
        nz = np.flatnonzero(np.abs(seg) > 10 ** (-40 / 20) * np.abs(seg).max())
        seg = seg[max(0, nz[0] - int(.08 * sr)): nz[-1] + int(.08 * sr)]
        sf.write(O / f"line_{ln['id']}_0.wav", seg, sr)
        h = rec(seg, sr).text
        rep.append({"id": ln["id"], "sec": round(len(seg) / sr, 2), "wer": round(wer(ln["text"], h), 3), "asr": h})
        print(rep[-1], flush=True)
    (O / "split.json").write_text(json.dumps({"take": take, "cuts": cuts, "lines": rep}, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "remote":
        cmd = (f"cd /tmp && HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONPATH=/cwStorage/home/jijh/venvs/film2_qwen_tts_overlay "
               f"/tmp/jijh-djev/env234/bin/python {HERE} " + " ".join(sys.argv[2:]))
        sys.exit(subprocess.call(["ssh", "-o", "RemoteCommand=none", "-o", "RequestTTY=no", "-o", "BatchMode=yes",
                                  "-o", "ClearAllForwardings=yes", "gpucw1", cmd], stdin=subprocess.DEVNULL))
    main(*sys.argv[1:])
