"""Per-take defect check (one clip per Gemini call, so no position bias), then pick one take per line.
  defect_check.py TAKES_DIR LINES_JSON BUDGET_JSON ASR_SCORES_JSON OUT_JSON [--reps 2]
TAKES_DIR holds <id>_ga_<voice>_t<k>.wav. A take is eligible if it fits the picture budget (minus 0.4 s) and its ASR
word error rate (digits spelled out) is <= 0.08; among eligible takes the one with the fewest defects wins, ties broken
by the mean naturalness score."""
import argparse, glob, json, re, subprocess, concurrent.futures as cf
from pathlib import Path

GEM = ["/cwStorage/home/jijh/playground/jev-diagnosis/medcopilot/.venv/bin/python",
       "/cwStorage/home/jijh/playground/jev-diagnosis/medcopilot/tools/video/film2/gem.py", "ask", "--model", "gemini-3.8-flash-high"]
ASK = ("Listen to this ONE voice-over take for a premium product film. Script: \"{text}\"\n"
       "List every concrete defect you can hear, with timestamps: a mispronounced word (say which, e.g. 'Pulse' without "
       "its L, 'live' said like 'give', 'Muse Glimmer', 'Ray-Ban', 'GPU'), a word added, dropped or changed vs the script, "
       "a glitch/click/metallic artifact, an unnatural pause inside a phrase, a sentence ending that drops flat or goes "
       "sing-song, breathiness, rushed or slurred words. Do not list matters of taste. Then rate naturalness 1-10. End "
       "with one JSON line: {{\"defects\": [\"...\"], \"n_defects\": n, \"naturalness\": n}}")
NUM = {"34": "thirty four", "thirty-four": "thirty four"}


def words(s):
    s = s.lower()
    for k, v in NUM.items():
        s = s.replace(k, v)
    return re.findall(r"[a-z0-9']+", s.replace("-", " "))


def wer(ref, hyp):
    a, b = words(ref), words(hyp); d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (a[i - 1] != b[j - 1]))
    return d[len(b)] / max(1, len(a))


def main():
    ap = argparse.ArgumentParser(); [ap.add_argument(x) for x in ("takes", "lines", "budget", "asr", "out")]
    ap.add_argument("--reps", type=int, default=2); a = ap.parse_args()
    T = Path(a.takes); lines = {x["id"]: x for x in json.load(open(a.lines))}; budget = json.load(open(a.budget))
    asr = {r["file"]: r for r in json.load(open(a.asr))}
    R = T / "defects"; R.mkdir(exist_ok=True)
    takes = []
    for f in sorted(T.glob("M*_t*.wav")):
        lid = f.name.split("_")[0]; k = f.stem.split("_t")[-1]; s = asr.get(f"ga_{lid}_s{k}.wav")
        if not s:
            continue
        takes.append(dict(id=lid, file=f.name, sec=s["sec"], wer=round(wer(lines[lid]["text"], s["asr"]), 3), asr=s["asr"]))

    def judge(job):
        t, r = job; out = R / f"{Path(t['file']).stem}_{r}.md"
        if not out.exists():
            p = R / f"{t['id']}_prompt.txt"; p.write_text(ASK.format(text=lines[t["id"]]["text"]))
            subprocess.run(GEM + ["--prompt-file", str(p), "--file", str(T / t["file"]), "--out", str(out)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=900)
        txt = out.read_text() if out.exists() else ""; i = txt.rfind('{"defects"')
        try:
            return t["file"], json.JSONDecoder().raw_decode(txt[i:])[0]
        except Exception:
            return t["file"], None

    with cf.ThreadPoolExecutor(8) as ex:
        got = list(ex.map(judge, [(t, r) for t in takes for r in range(a.reps)]))
    by = {}
    for f, v in got:
        if v: by.setdefault(f, []).append(v)
    for t in takes:
        v = by.get(t["file"], [])
        t["n_defects"] = round(sum(x.get("n_defects", len(x.get("defects", []))) for x in v) / max(1, len(v)), 2)
        t["naturalness"] = round(sum(float(x.get("naturalness", 0)) for x in v) / max(1, len(v)), 2)
        t["defects"] = [d for x in v for d in x.get("defects", [])]
    pick = {}
    for lid in sorted({t["id"] for t in takes}, key=lambda x: int(x[1:])):
        c = [t for t in takes if t["id"] == lid]
        ok = [t for t in c if t["sec"] - 0.5 <= budget.get(lid, 99) - 0.4 and t["wer"] <= 0.08] or \
             [t for t in c if t["wer"] <= 0.08] or c
        best = sorted(ok, key=lambda t: (t["n_defects"], -t["naturalness"], t["wer"]))[0]
        pick[lid] = best
        print(f"{lid:4s} {best['file']:20s} {best['sec']:5.1f}s wer {best['wer']:.3f} defects {best['n_defects']} nat {best['naturalness']}"
              f" | {'; '.join(best['defects'][:3])[:200]}", flush=True)
    Path(a.out).write_text(json.dumps({"pick": pick, "all": takes}, indent=1))


if __name__ == "__main__":
    main()
