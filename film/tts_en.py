"""English narration for the Pulse film with Qwen3-TTS (runs on gpucw1 via env234 + the film2 qwen-tts overlay).

  tts_en.py design OUT              -> OUT/design_<voice>_<k>.wav : the whole script in one pass per candidate voice
  tts_en.py clone OUT REF.wav       -> OUT/line_<id>_<k>.wav       : each line cloned from the chosen one-pass take
  tts_en.py remote <cmd> ...        -> same, launched on gpucw1 GPU 1 over ssh
Every take is transcribed with SenseVoice (en) and its word error rate written to OUT/<cmd>_scores.json.
"""
import json, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve()
import os
LINES = json.load(open(os.environ.get("VO_FILE") or HERE.with_name("vo.json")))
REF_TEXT_LINES = json.load(open(HERE.with_name("vo_ref.json"))) if HERE.with_name("vo_ref.json").exists() else LINES
VOICES = {
    "warmF": "A warm, calm female narrator in her early thirties with a soft American accent. Intimate close-mic delivery, "
             "unhurried, quietly playful and confident, like a premium tech product film. Natural and human, not a radio "
             "announcer, not salesy.",
    "warmM": "A warm male narrator in his thirties, soft-spoken, gentle American accent, relaxed and friendly with a hint of "
             "a smile, close-mic, natural conversational pacing. Not a movie-trailer voice, not salesy.",
    "brightF": "A bright, friendly female voice in her late twenties with a light British accent, close-mic, natural and "
               "conversational, gently witty, calm pacing. Not a radio announcer.",
}
MODELS = ["/tmp/jijh-tts/models", "/cwStorage/home/jijh/models/tts"]


def mp(name):
    for d in MODELS:
        if Path(d, name).exists():
            return str(Path(d, name))
    raise FileNotFoundError(name)


def words(s):
    return re.findall(r"[a-z0-9']+", s.lower())


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(1, len(r))


def worker(cmd, out, ref=None):
    import numpy as np, soundfile as sf, torch, sherpa_onnx
    from qwen_tts import Qwen3TTSModel
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    sv = mp("sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17")
    asr = sherpa_onnx.OfflineRecognizer.from_sense_voice(model=f"{sv}/model.int8.onnx", tokens=f"{sv}/tokens.txt",
                                                          num_threads=4, language="en", use_itn=True)

    def tr(w, sr):
        import scipy.signal as ss
        x = ss.resample_poly(w, 16000, sr).astype("float32") if sr != 16000 else w.astype("float32")
        s = asr.create_stream(); s.accept_waveform(16000, x); asr.decode_stream(s); return s.result.text

    scores = []
    if cmd == "design":
        m = Qwen3TTSModel.from_pretrained(mp("Qwen3-TTS-12Hz-1.7B-VoiceDesign"), device_map="cuda:0", dtype=torch.bfloat16)
        text = "\n\n".join(l["text"] for l in LINES)
        for v, ins in VOICES.items():
            for k in range(2):
                torch.manual_seed(1000 + k)
                w, sr = m.generate_voice_design(text=text, language="English", instruct=ins, do_sample=True, temperature=0.8)
                p = out / f"design_{v}_{k}.wav"; sf.write(p, w[0], sr)
                h = tr(w[0], sr); scores.append({"file": p.name, "wer": round(wer(text, h), 3), "sec": round(len(w[0]) / sr, 1), "asr": h})
                print(scores[-1], flush=True)
    else:
        m = Qwen3TTSModel.from_pretrained(mp("Qwen3-TTS-12Hz-1.7B-Base"), device_map="cuda:0", dtype=torch.bfloat16)
        prompt = m.create_voice_clone_prompt(ref_audio=ref, ref_text="\n\n".join(l["text"] for l in REF_TEXT_LINES))
        for l in LINES:
            for k in range(3):
                torch.manual_seed(2000 + k)
                w, sr = m.generate_voice_clone(text=l["text"], language="English", voice_clone_prompt=prompt, do_sample=True)
                p = out / f"line_{l['id']}_{k}.wav"; sf.write(p, w[0], sr)
                h = tr(w[0], sr); scores.append({"file": p.name, "id": l["id"], "wer": round(wer(l["text"], h), 3),
                                                 "sec": round(len(w[0]) / sr, 2), "asr": h})
                print(scores[-1], flush=True)
    Path(out, f"{cmd}_scores.json").write_text(json.dumps(scores, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "remote":
        args = " ".join(sys.argv[2:])
        remote = (f"cd /tmp && VO_FILE={os.environ.get('VO_FILE', '')} CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 "
                  f"PYTHONPATH=/cwStorage/home/jijh/venvs/film2_qwen_tts_overlay /tmp/jijh-djev/env234/bin/python {HERE} {args}")
        sys.exit(subprocess.call(["ssh", "-o", "RemoteCommand=none", "-o", "RequestTTY=no", "-o", "BatchMode=yes",
                                  "-o", "ClearAllForwardings=yes", "gpucw1", remote], stdin=subprocess.DEVNULL))
    worker(*sys.argv[1:])
