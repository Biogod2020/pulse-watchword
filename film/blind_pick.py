"""Blind Gemini ranking of audio candidates (shuffled letters, several runs, Borda count).

  blind_pick.py --brief "..." --runs 2 a.wav b.wav ...   -> prints the ranking and writes <first dir>/pick_<tag>.json
"""
import argparse, json, random, re, subprocess, sys
from pathlib import Path

sys.path.insert(0, "/cwStorage/home/jijh/playground/jev-diagnosis/medcopilot/tools/video/film2")
from gem import ask  # noqa: E402

FF = "/home1/jijh/micromamba/envs/imgprocess/bin/ffmpeg"


def mp3(p):
    out = Path("/tmp/jijh-pulse-film/blind") / (Path(p).stem + ".mp3"); out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([FF, "-v", "error", "-y", "-i", str(p), "-ac", "1", "-b:a", "96k", str(out)], check=True)
    return out


ap = argparse.ArgumentParser(); ap.add_argument("--brief", required=True); ap.add_argument("--runs", type=int, default=2)
ap.add_argument("--tag", default="pick"); ap.add_argument("files", nargs="+")
a = ap.parse_args()
files = [Path(f) for f in a.files]; enc = {f: mp3(f) for f in files}
score = {f.name: 0 for f in files}; notes = []
for r in range(a.runs):
    order = files[:]; random.Random(100 + r).shuffle(order)
    letters = [chr(65 + i) for i in range(len(order))]
    prompt = (a.brief + f"\n\nYou will hear {len(order)} candidates, in order: " + ", ".join(letters) +
              ". Listen to each completely. Reply with a short reason per candidate, then a final line exactly like "
              "'RANKING: B > A > C' (best first, every letter once).")
    txt = ask(prompt, [enc[f] for f in order], temperature=0.3)
    m = re.search(r"RANKING:\s*([A-Z](?:\s*>\s*[A-Z])+)", txt)
    if not m:
        print("no ranking in run", r, txt[-300:]); continue
    rank = [x.strip() for x in m.group(1).split(">")]
    for i, L in enumerate(rank):
        if L in letters:
            score[order[letters.index(L)].name] += len(rank) - i
    notes.append({"run": r, "order": [f.name for f in order], "ranking": rank, "reply": txt})
    print(f"run {r}: " + " > ".join(order[letters.index(L)].name for L in rank if L in letters), flush=True)
best = sorted(score.items(), key=lambda kv: -kv[1])
print("BORDA:", best)
Path(files[0].parent, f"pick_{a.tag}.json").write_text(json.dumps({"borda": best, "runs": notes}, indent=1))
