"""Choose one narration take per line: ASR/speaker-similarity gate, then a blind Gemini pick among the best three.
  pick_lines.py PROD_DIR VO_JSON NOTES_JSON [--budget budget.json] [--ids M1,M2] -> PROD_DIR/pick.json (+ pick_reviews/)
Gate: word error rate <= 0.08 and speaker similarity to the anchor >= 0.95 (relaxed to the top-3 by similarity if none
pass). Clips go to Gemini as 24 kHz mono at -20 LUFS, shuffled and labelled A/B/C, so format and level reveal nothing.
"""
import argparse, glob, json, random, re, subprocess, sys
from pathlib import Path
import numpy as np, librosa, soundfile as sf, pyloudnorm as pyln

GEM = ["/cwStorage/home/jijh/playground/jev-diagnosis/medcopilot/.venv/bin/python",
       "/cwStorage/home/jijh/playground/jev-diagnosis/medcopilot/tools/video/film2/gem.py", "ask", "--model", "gemini-3.8-flash-high"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("prod"); ap.add_argument("vo"); ap.add_argument("notes")
    ap.add_argument("--budget"); ap.add_argument("--ids"); a = ap.parse_args()
    P = Path(a.prod); text = {x["id"]: x["text"] for x in json.load(open(a.vo))}; notes = json.load(open(a.notes))
    budget = json.load(open(a.budget)) if a.budget else {}
    rows = sum([json.load(open(f)) for f in glob.glob(str(P / "manifest_*.json"))], [])
    pick_path = P / "pick.json"; pick = json.load(open(pick_path)) if pick_path.exists() else {}
    R = P / "pick_reviews"; R.mkdir(exist_ok=True)
    for lid in sorted({r["id"] for r in rows}, key=lambda x: int(x[1:])):
        if a.ids and lid not in a.ids.split(","):
            continue
        cand = [r for r in rows if r["id"] == lid]
        if lid in budget:                               # leave 0.4 s of picture after the voice
            cand = [r for r in cand if r["sec"] - 0.5 <= budget[lid] - 0.4] or cand
        ok = [r for r in cand if r["wer"] <= 0.08 and r["sim"] >= 0.95]
        top = sorted(ok or cand, key=lambda r: (-r["sim"] + 2 * r["wer"]))[:3]
        random.seed(lid); random.shuffle(top); labels = "ABC"[:len(top)]
        files = []
        for L, r in zip(labels, top):
            y, sr = sf.read(P / r["file"], always_2d=True); y = librosa.resample(y.mean(1), orig_sr=sr, target_sr=24000)
            y = y * 10 ** ((-20 - pyln.Meter(24000).integrated_loudness(y)) / 20)
            f = R / f"{lid}_{L}.wav"; sf.write(f, np.clip(y, -1, 1), 24000); files += ["--file", str(f)]
        prompt = (f"These are {len(top)} takes ({', '.join(labels)}, in that order) of one line of voice-over for a premium "
                  f"keynote-style English product film.\nLine: \"{text[lid]}\"\nDirection: {notes[lid]}\n"
                  "Pick the take that sounds most like a skilled human narrator: natural, warm, intelligent; follows the "
                  "direction without overacting; natural pauses (not choppy, not rushed); correct pronunciation of every "
                  "word, especially 'Pulse', 'Muse Glimmer', 'Meta', 'Ray-Ban', 'GPU'; no synthesis artifacts, breathiness, "
                  "mumbling or clipped words. Briefly note the main flaw of each take, then end with one JSON line: "
                  "{\"best\": \"A\", \"ranking\": [\"A\", ...], \"best_score\": <naturalness 1-10>, \"flaws\": {\"A\": \"...\"}}")
        (R / f"{lid}_prompt.txt").write_text(prompt)
        out = R / f"{lid}_review.md"
        subprocess.run(GEM + ["--prompt-file", str(R / f"{lid}_prompt.txt"), *files, "--out", str(out)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t = out.read_text(); i = max(t.rfind('{"best"'), t.rfind('{ "best"'))
        js = json.JSONDecoder().raw_decode(t[i:])[0]
        best = top[labels.index(js["best"])]
        pick[lid] = {"file": best["file"], "wer": best["wer"], "sim": best["sim"], "sec": best["sec"],
                     "gemini_score": js.get("best_score"), "flaws": js.get("flaws"), "others": [r["file"] for r in top if r is not best]}
        print(lid, pick[lid]["file"], "score", js.get("best_score"), "| flaws:", js.get("flaws"), flush=True)
        pick_path.write_text(json.dumps(pick, indent=1))


if __name__ == "__main__":
    main()
