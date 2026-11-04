"""gpt-audio (chat completions with audio output) reading a narration script verbatim. Key from ~/.config/pulse/openai_key.
  oai_audio.py OUT_DIR ITEMS.json [--voice marin] [--model gpt-audio-1.5] [--takes 1]   -> OUT_DIR/<id>_ga_<voice>_t<k>.wav"""
import argparse, base64, json, sys, time
from pathlib import Path
import httpx
sys.path.insert(0, str(Path(__file__).parent)); from oai_tts import STYLE


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("items"); ap.add_argument("--voice", default="marin")
    ap.add_argument("--model", default="gpt-audio-1.5"); ap.add_argument("--takes", type=int, default=1); ap.add_argument("--start", type=int, default=0); a = ap.parse_args()
    O = Path(a.out); O.mkdir(parents=True, exist_ok=True); key = Path("~/.config/pulse/openai_key").expanduser().read_text().strip()
    with httpx.Client(timeout=httpx.Timeout(600, connect=20)) as c:
        for it in json.load(open(a.items)):
            for k in range(a.start, a.start + a.takes):
                f = O / f"{it['id']}_ga_{a.voice}_t{k}.wav"
                if f.exists():
                    continue
                sysmsg = ("You are a professional voice-over artist recording narration for a premium product film. Read the "
                          "user's script EXACTLY as written, word for word: do not add, drop or change any word, do not "
                          "greet, do not comment. " + STYLE + (" Delivery for this part: " + it["note"] if it.get("note") else ""))
                body = {"model": a.model, "modalities": ["text", "audio"], "audio": {"voice": a.voice, "format": "wav"},
                        "messages": [{"role": "system", "content": sysmsg}, {"role": "user", "content": "Script:\n" + it["text"]}]}
                r = c.post("https://api.openai.com/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {key}"})
                if r.status_code != 200:
                    print("HTTP", r.status_code, r.text[:300], file=sys.stderr); continue
                au = r.json()["choices"][0]["message"]["audio"]
                f.write_bytes(base64.b64decode(au["data"])); print("wrote", f.name, "|", au.get("transcript", "")[:120], flush=True)


if __name__ == "__main__":
    main()
