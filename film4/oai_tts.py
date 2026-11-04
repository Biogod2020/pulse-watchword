"""OpenAI steerable TTS (gpt-4o-mini-tts) for the Pulse narration. Key read from ~/.config/pulse/openai_key (never printed).
  oai_tts.py OUT_DIR ITEMS.json [--voices marin,cedar] [--model gpt-4o-mini-tts-2025-12-15] [--takes 1]
ITEMS.json = [{"id": "M1", "text": "...", "note": "delivery note"}] -> OUT_DIR/<id>_<voice>_t<k>.wav (24 kHz)
Spend is appended to OUT_DIR/spend.json (characters sent; price per OpenAI's published rate ~ $0.015 per audio minute)."""
import argparse, json, os, sys, time
from pathlib import Path
import httpx

STYLE = ("Voice: a warm, natural narrator in her early thirties; intelligent, relaxed and conversational, like a thoughtful "
         "founder explaining something she's proud of to a friend in a premium product film. Tone: genuine, quietly confident, "
         "with a light smile. Pacing: unhurried but not slow, natural pauses only where a person would breathe; lists flow. "
         "Intonation: varied and human, sentence ends land naturally, never sing-song, never announcer-like, not salesy. "
         "Pronunciation: 'Pulse' clearly with the L; 'Muse Glimmer' as 'mews GLIM-er'; 'live' as in 'live broadcast' (rhymes "
         "with 'five'); 'GPU' as letters.")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("items")
    ap.add_argument("--voices", default="marin"); ap.add_argument("--model", default="gpt-4o-mini-tts-2025-12-15")
    ap.add_argument("--takes", type=int, default=1); a = ap.parse_args()
    O = Path(a.out); O.mkdir(parents=True, exist_ok=True)
    key = Path("~/.config/pulse/openai_key").expanduser().read_text().strip()
    items = json.load(open(a.items)); chars = 0
    with httpx.Client(timeout=httpx.Timeout(600, connect=20)) as c:
        for it in items:
            for v in a.voices.split(","):
                for k in range(a.takes):
                    f = O / f"{it['id']}_{v}_t{k}.wav"
                    if f.exists():
                        continue
                    body = {"model": a.model, "voice": v, "input": it["text"], "response_format": "wav",
                            "instructions": STYLE + (" Delivery for this line: " + it["note"] if it.get("note") else "")}
                    for i in range(4):
                        r = c.post("https://api.openai.com/v1/audio/speech", json=body, headers={"Authorization": f"Bearer {key}"})
                        if r.status_code == 200:
                            break
                        print("HTTP", r.status_code, r.text[:200], file=sys.stderr); time.sleep(5 * (i + 1))
                    r.raise_for_status(); f.write_bytes(r.content); chars += len(it["text"])
                    print("wrote", f.name, len(r.content), flush=True)
    sp = O / "spend.json"; log = json.loads(sp.read_text()) if sp.exists() else []
    log.append({"t": time.strftime("%F %T"), "model": a.model, "chars": chars}); sp.write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
