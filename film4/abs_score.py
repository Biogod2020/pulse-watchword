"""Absolute, one-clip-at-a-time naturalness scoring by two audio judges (no position bias):
Gemini 3.8 Flash (local CLIProxyAPI) and OpenAI gpt-audio (key from ~/.config/pulse/openai_key).
  abs_score.py OUT_DIR SCRIPT_TXT name=file.wav [...] [--reps 3] [--judges gemini,openai]
Each clip is scored --reps times per judge on a fixed rubric (1-10 per dimension); writes OUT_DIR/abs_scores.json."""
import argparse, base64, json, re, subprocess, concurrent.futures as cf
from pathlib import Path
import httpx, numpy as np, librosa, soundfile as sf, pyloudnorm as pyln

GEM = ["/cwStorage/home/jijh/playground/jev-diagnosis/medcopilot/.venv/bin/python",
       "/cwStorage/home/jijh/playground/jev-diagnosis/medcopilot/tools/video/film2/gem.py", "ask", "--model", "gemini-3.8-flash-high"]
RUBRIC = ("You are a strict voice-over casting director for premium keynote-style product films (Apple, Google launch videos). "
          "Listen to this ONE narration take.\nScript: \"{script}\"\n"
          "Score 1-10 (10 = indistinguishable from a top human voice actor in a finished Apple film; 7 = good but noticeably "
          "synthetic in places; 5 = obviously AI) on: naturalness (does it sound like a real person), intonation (varied, "
          "natural sentence endings, not sing-song or flat), pacing (natural pauses, not choppy or rushed), warmth (engaging, "
          "conversational, intelligent), pronunciation (every word right, e.g. 'Pulse', 'Muse Glimmer', 'GPU'), "
          "artifacts (10 = none: no metallic sound, breathiness, mumbling, glitches). Be critical and use the full scale. "
          "Name the two most noticeable problems with timestamps. End with one JSON line: "
          "{{\"naturalness\": n, \"intonation\": n, \"pacing\": n, \"warmth\": n, \"pronunciation\": n, \"artifacts\": n}}")
DIMS = ["naturalness", "intonation", "pacing", "warmth", "pronunciation", "artifacts"]


def prep(src, dst):
    y, sr = sf.read(src, always_2d=True); y = librosa.resample(y.mean(1), orig_sr=sr, target_sr=24000)
    y = y * 10 ** ((-20 - pyln.Meter(24000).integrated_loudness(y)) / 20); sf.write(dst, np.clip(y, -1, 1), 24000, subtype="PCM_16")


def parse(t):
    m = re.findall(r"\{[^{}]*\"naturalness\"[^{}]*\}", t or "")
    try:
        d = json.loads(m[-1]); return {k: float(d[k]) for k in DIMS}
    except Exception:
        return None


def gemini(clip, prompt, out):
    subprocess.run(GEM + ["--prompt-file", str(prompt), "--file", str(clip), "--out", str(out)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=900)
    return out.read_text() if out.exists() else ""


def openai(clip, text, out):
    key = Path("~/.config/pulse/openai_key").expanduser().read_text().strip()
    body = {"model": "gpt-audio-1.5", "modalities": ["text"], "messages": [{"role": "user", "content": [
        {"type": "text", "text": text},
        {"type": "input_audio", "input_audio": {"data": base64.b64encode(clip.read_bytes()).decode(), "format": "wav"}}]}]}
    r = httpx.post("https://api.openai.com/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {key}"},
                   timeout=httpx.Timeout(300, connect=20))
    t = r.json()["choices"][0]["message"]["content"] if r.status_code == 200 else f"HTTP {r.status_code} {r.text[:200]}"
    out.write_text(t); return t


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("script"); ap.add_argument("takes", nargs="+")
    ap.add_argument("--reps", type=int, default=3); ap.add_argument("--judges", default="gemini,openai"); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(); O = Path(a.out); O.mkdir(parents=True, exist_ok=True); C = O / "clips"; C.mkdir(exist_ok=True)
    takes = dict(t.split("=", 1) for t in a.takes); text = RUBRIC.format(script=Path(a.script).read_text().strip())
    prompt = O / "rubric.txt"; prompt.write_text(text)
    for n, f in takes.items():
        prep(f, C / f"{n}.wav")
    P = O / "abs_scores.json"; res = json.loads(P.read_text()) if P.exists() else {}
    jobs = [(j, n, r) for j in a.judges.split(",") for n in takes for r in range(a.reps) if f"{j}|{n}|{r}" not in res]

    def run(job):
        j, n, r = job; out = O / f"{j}_{n}_{r}.md"
        t = gemini(C / f"{n}.wav", prompt, out) if j == "gemini" else openai(C / f"{n}.wav", text, out)
        return f"{j}|{n}|{r}", parse(t)

    with cf.ThreadPoolExecutor(a.workers) as ex:
        for k, v in ex.map(run, jobs):
            res[k] = v; print(k, v, flush=True)
    P.write_text(json.dumps(res, indent=1))
    print(f"\n{'take':12s} " + " ".join(f"{j:>8s}" for j in a.judges.split(",")) + "   mean  (naturalness, per judge)")
    rows = []
    for n in takes:
        per = []
        for j in a.judges.split(","):
            v = [res[f"{j}|{n}|{r}"]["naturalness"] for r in range(a.reps) if res.get(f"{j}|{n}|{r}")]
            per.append(np.mean(v) if v else float("nan"))
        rows.append((np.nanmean(per), n, per))
    for m, n, per in sorted(rows, reverse=True):
        print(f"{n:12s} " + " ".join(f"{p:8.2f}" for p in per) + f"   {m:.2f}")


if __name__ == "__main__":
    main()
