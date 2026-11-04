"""Position-debiased A/B preference between two audio clips, with two judges (Gemini via CLIProxyAPI, OpenAI gpt-audio).
  ab_debiased.py OUT_DIR X=x.wav Y=y.wav [--reps 8] [--judges gemini,openai] [--context "..."]
Each judge hears the pair --reps times in each order. Position bias cancels in the statistic
    pref(X) = 0.5 * (P(X chosen | X first) + P(X chosen | X second));
pref(X) > 0.5 means X is preferred regardless of order. A sign test over the 2*reps trials is reported too."""
import argparse, base64, json, math, re, subprocess, concurrent.futures as cf
from pathlib import Path
import httpx, numpy as np, librosa, soundfile as sf, pyloudnorm as pyln

GEM = ["/cwStorage/home/jijh/playground/jev-diagnosis/medcopilot/.venv/bin/python",
       "/cwStorage/home/jijh/playground/jev-diagnosis/medcopilot/tools/video/film2/gem.py", "ask", "--model", "gemini-3.8-flash-high"]
ASK = ("Two versions of the same {context} follow: the first audio is version 1, the second is version 2. Which version's "
       "narration sounds more like a skilled human narrator in a finished premium product film, and less strange or "
       "AI-generated (intonation, pacing and pauses, pronunciation, artifacts, how the voice sits with the music)? Answer "
       "with a short reason and end with exactly one line: WINNER: 1 or WINNER: 2")


def prep(src, dst):
    y, sr = sf.read(src, always_2d=True); y = librosa.resample(y.mean(1), orig_sr=sr, target_sr=24000)
    y = y * 10 ** ((-20 - pyln.Meter(24000).integrated_loudness(y)) / 20); sf.write(dst, np.clip(y, -1, 1), 24000, subtype="PCM_16")


def winner(t):
    m = re.findall(r"WINNER:\s*\**\s*([12])", t or ""); return int(m[-1]) if m else None


def gemini(a, b, prompt, out):
    if not out.exists():
        subprocess.run(GEM + ["--prompt-file", str(prompt), "--file", str(a), "--file", str(b), "--out", str(out)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=900)
    return out.read_text() if out.exists() else ""


def openai(a, b, text, out):
    if out.exists():
        return out.read_text()
    key = Path("~/.config/pulse/openai_key").expanduser().read_text().strip()
    enc = lambda p: {"type": "input_audio", "input_audio": {"data": base64.b64encode(p.read_bytes()).decode(), "format": "wav"}}
    body = {"model": "gpt-audio-1.5", "modalities": ["text"], "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Version 1:"}, enc(a), {"type": "text", "text": "Version 2:"}, enc(b), {"type": "text", "text": text}]}]}
    r = httpx.post("https://api.openai.com/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {key}"},
                   timeout=httpx.Timeout(300, connect=20))
    t = r.json()["choices"][0]["message"]["content"] if r.status_code == 200 else f"HTTP {r.status_code}"
    if r.status_code == 200:
        out.write_text(t)
    return t


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("x"); ap.add_argument("y")
    ap.add_argument("--reps", type=int, default=8); ap.add_argument("--judges", default="gemini,openai")
    ap.add_argument("--context", default="product-film narration excerpt (voice over music)"); a = ap.parse_args()
    O = Path(a.out); O.mkdir(parents=True, exist_ok=True)
    (xn, xf), (yn, yf) = a.x.split("=", 1), a.y.split("=", 1)
    X, Y = O / f"{xn}.wav", O / f"{yn}.wav"; prep(xf, X); prep(yf, Y)
    text = ASK.format(context=a.context); prompt = O / "prompt.txt"; prompt.write_text(text)
    jobs = [(j, o, r) for j in a.judges.split(",") for o in ("xy", "yx") for r in range(a.reps)]

    def run(job):
        j, o, r = job; first, second = (X, Y) if o == "xy" else (Y, X); out = O / f"{j}_{o}_{r}.md"
        t = gemini(first, second, prompt, out) if j == "gemini" else openai(first, second, text, out)
        w = winner(t)
        return j, o, None if w is None else (w == 1) == (o == "xy")       # True -> X chosen

    with cf.ThreadPoolExecutor(10) as ex:
        res = list(ex.map(run, jobs))
    summary = {}
    for j in a.judges.split(","):
        xf_ = [v for jj, o, v in res if jj == j and o == "xy" and v is not None]
        xs_ = [v for jj, o, v in res if jj == j and o == "yx" and v is not None]
        if not xf_ or not xs_:
            continue
        p1, p2 = sum(xf_) / len(xf_), sum(xs_) / len(xs_); pref = 0.5 * (p1 + p2)
        k, n = sum(xf_) + sum(xs_), len(xf_) + len(xs_)
        pval = sum(math.comb(n, i) for i in range(n + 1) if abs(i - n / 2) >= abs(k - n / 2)) / 2 ** n
        bias = 0.5 * (p1 + (1 - p2))                                  # share of "version 1" answers
        summary[j] = {"pref_" + xn: round(pref, 3), "P(X|X first)": round(p1, 2), "P(X|X second)": round(p2, 2),
                      "first_position_rate": round(bias, 2), "wins": f"{k}/{n}", "sign_test_p": round(pval, 4)}
        print(j, summary[j], flush=True)
    (O / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
