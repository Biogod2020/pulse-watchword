"""Gemini TTS (gemini-3.8-flash-tts) for the Pulse narration. Key read from ~/.config/pulse/google_key (never printed).
  gemini_tts.py OUT_DIR ITEMS.json [--voices Kore,Charon] [--takes 1] [--start 0] [--model gemini-3.8-flash-tts]
ITEMS.json = [{"id": "M1", "text": "...", "note": "delivery note"}] -> OUT_DIR/<id>_<voice>_t<k>.wav (24 kHz mono)
Direction goes in an AUDIO PROFILE / DIRECTOR'S NOTES / TRANSCRIPT prompt; a plain instruction prefix is read aloud."""
import argparse, base64, json, sys, time, wave
from pathlib import Path
import httpx

NOTES = ("# AUDIO PROFILE: Documentary narrator\nA warm, intelligent narrator for a premium product film (think an Apple "
         "or NASA film).\n\n## DIRECTOR'S NOTES\nStyle: expressive and natural, a lively pitch range with real melody, "
         "never announcer-like or salesy.\nPace: unhurried.\nEndings: buoyant, landing gently, never sinking into a low "
         "creaky voice.\nPronunciation: 'Pulse' with a clear L; 'Muse Glimmer' as 'mews GLIM-er'; 'GPU' as letters.\n")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("items")
    ap.add_argument("--voices", default="Kore"); ap.add_argument("--takes", type=int, default=1); ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--model", default="gemini-3.8-flash-tts"); a = ap.parse_args()
    O = Path(a.out); O.mkdir(parents=True, exist_ok=True)
    key = Path("~/.config/pulse/google_key").expanduser().read_text().strip()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{a.model}:generateContent"
    with httpx.Client(timeout=httpx.Timeout(300, connect=20)) as c:
        for it in json.load(open(a.items)):
            for v in a.voices.split(","):
                for k in range(a.start, a.start + a.takes):
                    f = O / f"{it['id']}_{v}_t{k}.wav"
                    if f.exists():
                        continue
                    prompt = NOTES + ("This line: " + it["note"] + "\n" if it.get("note") else "") + "\n## TRANSCRIPT\n" + it["text"]
                    body = {"contents": [{"parts": [{"text": prompt}]}],
                            "generationConfig": {"responseModalities": ["AUDIO"], "temperature": 1.0,
                                                 "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": v}}}}}
                    for i in range(5):
                        r = c.post(url, json=body, headers={"x-goog-api-key": key})
                        if r.status_code == 200:
                            break
                        print("HTTP", r.status_code, r.text[:200], file=sys.stderr); time.sleep(8 * (i + 1))
                    if r.status_code != 200:
                        continue
                    cand = r.json().get("candidates", [{}])[0]
                    if "content" not in cand:
                        print("no audio for", f.name, "finishReason", cand.get("finishReason"), file=sys.stderr); continue
                    part = next(p for p in cand["content"]["parts"] if "inlineData" in p)
                    pcm = base64.b64decode(part["inlineData"]["data"]); mime = part["inlineData"].get("mimeType", "")
                    sr = int(mime.split("rate=")[1].split(";")[0]) if "rate=" in mime else 24000
                    with wave.open(str(f), "wb") as w:
                        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm)
                    print("wrote", f.name, round(len(pcm) / 2 / sr, 2), "s", flush=True)


if __name__ == "__main__":
    main()
