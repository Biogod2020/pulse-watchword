"""Pairwise blind Gemini comparison of narration takes (more stable than ranking many clips at once).
  pairwise.py OUT_DIR SCRIPT_TXT name=file.wav [name=file.wav ...] [--workers 4]
Every pair is judged twice, once in each order; clips are 24 kHz mono at -20 LUFS. Writes OUT_DIR/pairwise.json
with per-pair verdicts and a win table (a win counts 1, a tie 0.5)."""
import argparse, itertools, json, re, subprocess, concurrent.futures as cf
from pathlib import Path
import numpy as np, librosa, soundfile as sf, pyloudnorm as pyln

GEM = ["/cwStorage/home/jijh/playground/jev-diagnosis/medcopilot/.venv/bin/python",
       "/cwStorage/home/jijh/playground/jev-diagnosis/medcopilot/tools/video/film2/gem.py", "ask", "--model", "gemini-3.8-flash-high"]
ASK = ("You will hear two voice-over takes of the same script for a premium keynote-style English product film (Apple/Google "
       "launch-video quality). First file = take 1, second file = take 2.\nScript: \"{script}\"\n"
       "Which take sounds more like a skilled, warm, intelligent HUMAN narrator? Consider: natural intonation (not sing-song, "
       "not flat, natural sentence endings), natural pacing and pauses, correct pronunciation (e.g. 'Pulse', 'Muse Glimmer', "
       "'GPU'), every word of the script read, no synthesis artifacts, breathiness or mumbling. Name the single biggest "
       "problem of each take, then end with one JSON line: {{\"winner\": 1 or 2 or 0 for a tie, \"margin\": \"slight\"|\"clear\"}}")


def prep(src, dst):
    y, sr = sf.read(src, always_2d=True); y = librosa.resample(y.mean(1), orig_sr=sr, target_sr=24000)
    y = y * 10 ** ((-20 - pyln.Meter(24000).integrated_loudness(y)) / 20); sf.write(dst, np.clip(y, -1, 1), 24000)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("script"); ap.add_argument("takes", nargs="+")
    ap.add_argument("--workers", type=int, default=4); a = ap.parse_args()
    O = Path(a.out); O.mkdir(parents=True, exist_ok=True); C = O / "clips"; C.mkdir(exist_ok=True)
    takes = dict(t.split("=", 1) for t in a.takes)
    for n, f in takes.items():
        prep(f, C / f"{n}.wav")
    prompt = O / "prompt.txt"; prompt.write_text(ASK.format(script=Path(a.script).read_text().strip()))
    jobs = [(x, y) for x, y in itertools.permutations(takes, 2)]
    res_path = O / "pairwise.json"; done = json.loads(res_path.read_text())["verdicts"] if res_path.exists() else {}

    def judge(pair):
        x, y = pair; key = f"{x}|{y}"
        if key in done:
            return key, done[key]
        out = O / f"r_{x}__{y}.md"
        subprocess.run(GEM + ["--prompt-file", str(prompt), "--file", str(C / f"{x}.wav"), "--file", str(C / f"{y}.wav"), "--out", str(out)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=900)
        t = out.read_text() if out.exists() else ""; m = re.findall(r"\{[^{}]*\"winner\"[^{}]*\}", t)
        try:
            v = json.loads(m[-1])
        except Exception:
            v = {"winner": None}
        return key, v

    with cf.ThreadPoolExecutor(a.workers) as ex:
        for key, v in ex.map(judge, jobs):
            done[key] = v; print(key, v, flush=True)
    wins = {n: 0.0 for n in takes}; games = {n: 0 for n in takes}
    for key, v in done.items():
        x, y = key.split("|")
        if x not in takes or y not in takes or v.get("winner") is None:
            continue
        w = v["winner"]; games[x] += 1; games[y] += 1
        if w == 1: wins[x] += 1
        elif w == 2: wins[y] += 1
        else: wins[x] += .5; wins[y] += .5
    table = sorted(((round(wins[n] / max(1, games[n]), 3), n) for n in takes), reverse=True)
    res_path.write_text(json.dumps({"verdicts": done, "win_rate": {n: r for r, n in table}}, indent=1))
    for r, n in table:
        print(f"{n:12s} win rate {r:.2f}  ({wins[n]:.1f}/{games[n]})")


if __name__ == "__main__":
    main()
