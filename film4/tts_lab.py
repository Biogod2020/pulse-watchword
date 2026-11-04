"""Narration bake-off for the Pulse film (runs on gpucw1; launch with `tts_lab.py remote <engine> ...` from headcw).

Paragraph-level generation (several script lines in ONE pass, so intonation flows across sentences) vs. the old
per-line clone. Engines:
  qdesign  Qwen3-TTS VoiceDesign, voice described in words          (qwen stack)
  qclone   Qwen3-TTS Base, cloned from REF (+ REF_TEXT)              (qwen stack)
  qline    Qwen3-TTS VoiceDesign, one line per pass + a per-line delivery note (qwen stack)
  mgen     MOSS-VoiceGenerator, voice described in words             (moss stack)
  mclone   MOSS-TTS v1.5, cloned from REF                            (moss stack)
usage (on gpucw1):  tts_lab.py <engine> OUT PARAS.json SEEDS [GPU]
PARAS.json = [{"id": "P1", "lines": ["M1", ...], "text": "..."}]; output OUT/<engine>_<para>_s<seed>.wav
"""
import json, os, shlex, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve()
VOICE = ("A warm, natural female narrator in her early thirties with a soft American accent. Close-mic, relaxed and "
         "conversational, like a thoughtful founder explaining something she is proud of to a friend. Varied, lively "
         "intonation with a gentle smile, clear and not breathy; sentences end naturally, not falling flat. Unhurried but not slow. "
         "Not a radio announcer, not a robotic assistant, not salesy.")
ANCHOR = "/cwStorage/home/jijh/pulse_data/film4/tts_lab/anchor.wav"
REF = "/cwStorage/home/jijh/pulse_data/film4/tts_lab/ref_warmF.wav"
REF_TEXT = "/cwStorage/home/jijh/pulse_data/film4/tts_lab/ref_warmF.txt"


def qwen(engine, out, paras, seeds):
    import torch, soundfile as sf
    from qwen_tts import Qwen3TTSModel
    base = "/tmp/jijh-tts/models/"
    if engine == "qclone":
        m = Qwen3TTSModel.from_pretrained(base + "Qwen3-TTS-12Hz-1.7B-Base", device_map="cuda:0", dtype=torch.bfloat16)
        prompt = m.create_voice_clone_prompt(ref_audio=REF, ref_text=open(REF_TEXT).read().strip())
    else:
        m = Qwen3TTSModel.from_pretrained(base + "Qwen3-TTS-12Hz-1.7B-VoiceDesign", device_map="cuda:0", dtype=torch.bfloat16)
    for p in paras:
        units = [(p["id"], p["text"], p.get("note", ""))] if engine != "qline" else [(f"{p['id']}{k}", t, n) for k, (t, n) in
                                                                                     enumerate(zip(p["line_texts"], p["notes"]))]
        for s in seeds:
            for uid, text, note in units:
                torch.manual_seed(s)
                if engine == "qclone":
                    w, sr = m.generate_voice_clone(text=text, language="English", voice_clone_prompt=prompt, do_sample=True)
                else:
                    ins = VOICE + (" Delivery: " + note if note else "")
                    w, sr = m.generate_voice_design(text=text, language="English", instruct=ins, do_sample=True, temperature=0.8)
                f = out / f"{engine}_{uid}_s{s}.wav"; sf.write(f, w[0], sr); print("wrote", f, len(w[0]) / sr, flush=True)


def moss(engine, out, paras, seeds):
    import torch, soundfile as sf
    from transformers import AutoModel, AutoProcessor
    torch.backends.cuda.enable_cudnn_sdp(False)
    M = "/tmp/jijh-tts/moss"; name = "MOSS-VoiceGenerator" if engine == "mgen" else "MOSS-TTS-v1.5"
    proc = AutoProcessor.from_pretrained(f"{M}/{name}", trust_remote_code=True, codec_path=f"{M}/MOSS-Audio-Tokenizer")
    proc.audio_tokenizer = proc.audio_tokenizer.to("cuda")
    model = AutoModel.from_pretrained(f"{M}/{name}", trust_remote_code=True, attn_implementation="sdpa", dtype=torch.bfloat16).to("cuda").eval()
    ref = os.environ.get("REF_WAV", REF); ins = os.environ.get("MOSS_INS", VOICE)   # overrides for the one-pass film read
    if engine == "mclone":
        wav, sr = sf.read(ref, dtype="float32"); wav = torch.from_numpy(wav.T if wav.ndim > 1 else wav[None, :])
        ref_codes = proc.encode_audios_from_wav([wav], sr)[0]
    for p in paras:
        for s in seeds:
            torch.manual_seed(s)
            msg = (proc.build_user_message(text=p["text"], instruction=ins) if engine == "mgen" else
                   proc.build_user_message(text=p["text"], reference=[ref_codes], language="English"))
            batch = proc([[msg]], mode="generation")
            with torch.no_grad():
                o = model.generate(input_ids=batch["input_ids"].to("cuda"), attention_mask=batch["attention_mask"].to("cuda"), max_new_tokens=4000)
            a = proc.decode(o)[0].audio_codes_list[0]; a = a.unsqueeze(0) if a.ndim == 1 else a
            f = out / f"{engine}_{p['id']}_s{s}.wav"; sf.write(f, a.detach().cpu().float().numpy().T, proc.model_config.sampling_rate)
            print("wrote", f, flush=True)


def score(out, paras):
    """SenseVoice ASR word error rate of every take in OUT against its paragraph (or line) text."""
    import re, glob, soundfile as sf, scipy.signal as ss, sherpa_onnx
    sv = "/tmp/jijh-tts/models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
    r = sherpa_onnx.OfflineRecognizer.from_sense_voice(model=f"{sv}/model.int8.onnx", tokens=f"{sv}/tokens.txt", num_threads=8,
                                                       language="en", use_itn=True)
    W = lambda s: re.findall(r"[a-z0-9']+", s.lower())
    def wer(a, b):
        a, b = W(a), W(b); d = list(range(len(b) + 1))
        for i in range(1, len(a) + 1):
            p, d[0] = d[0], i
            for j in range(1, len(b) + 1):
                p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (a[i - 1] != b[j - 1]))
        return d[len(b)] / max(1, len(a))
    texts = {}
    for p in paras:
        texts[p["id"]] = p["text"]
        for k, t in enumerate(p.get("line_texts", [])):
            texts[f"{p['id']}{k}"] = t
    res = []
    for f in sorted(glob.glob(f"{out}/*.wav")):
        uid = Path(f).stem.split("_")[1]
        if uid not in texts:
            continue
        w, sr = sf.read(f, always_2d=True); x = ss.resample_poly(w.mean(1), 16000, sr).astype("float32")
        s = r.create_stream(); s.accept_waveform(16000, x); r.decode_stream(s); h = s.result.text
        res.append({"file": Path(f).name, "sec": round(len(x) / 16000, 2), "wer": round(wer(texts[uid], h), 3), "asr": h})
        print(res[-1], flush=True)
    Path(out, "asr_scores.json").write_text(json.dumps(res, indent=1))


def prod(out, lines, seeds):
    """Final takes: VoiceDesign per line with its delivery note, several seeds; each take scored by ASR word error
    rate and by speaker-embedding cosine similarity to ANCHOR (the bake-off winner), written to OUT/manifest_<ids>.json."""
    import re, numpy as np, torch, soundfile as sf, scipy.signal as ss, sherpa_onnx
    from qwen_tts import Qwen3TTSModel
    base = "/tmp/jijh-tts/models/"
    m = Qwen3TTSModel.from_pretrained(base + "Qwen3-TTS-12Hz-1.7B-VoiceDesign", device_map="cuda:0", dtype=torch.bfloat16)
    enc = Qwen3TTSModel.from_pretrained(base + "Qwen3-TTS-12Hz-1.7B-Base", device_map="cuda:0", dtype=torch.bfloat16)
    sv = "/tmp/jijh-tts/models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
    asr = sherpa_onnx.OfflineRecognizer.from_sense_voice(model=f"{sv}/model.int8.onnx", tokens=f"{sv}/tokens.txt", num_threads=8,
                                                         language="en", use_itn=True)
    W = lambda x: re.findall(r"[a-z0-9']+", x.lower())
    def wer(a, b):
        a, b = W(a), W(b); d = list(range(len(b) + 1))
        for i in range(1, len(a) + 1):
            p, d[0] = d[0], i
            for j in range(1, len(b) + 1):
                p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (a[i - 1] != b[j - 1]))
        return d[len(b)] / max(1, len(a))
    def emb(w, sr):
        w = ss.resample_poly(w, 24000, sr) if sr != 24000 else w
        e = enc.model.extract_speaker_embedding(audio=np.asarray(w, np.float32), sr=24000).float().flatten(); return e / e.norm()
    aw, asr_ = sf.read(ANCHOR); anchor = emb(aw if aw.ndim == 1 else aw.mean(1), asr_)
    res = []
    for ln in lines:
        for s in seeds:
            torch.manual_seed(s)
            w, sr = m.generate_voice_design(text=ln["text"], language="English", instruct=VOICE + " Delivery: " + ln["note"],
                                            do_sample=True, temperature=0.8)
            w = w[0]; f = out / f"{ln['id']}_s{s}.wav"; sf.write(f, w, sr)
            st = asr.create_stream(); st.accept_waveform(16000, ss.resample_poly(w, 16000, sr).astype("float32")); asr.decode_stream(st)
            r = {"id": ln["id"], "file": f.name, "sec": round(len(w) / sr, 2), "wer": round(wer(ln["text"], st.result.text), 3),
                 "sim": round(float(emb(w, sr) @ anchor), 4), "asr": st.result.text}
            res.append(r); print(r, flush=True)
    Path(out, f"manifest_{lines[0]['id']}_{lines[-1]['id']}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "remote":
        engine, out, paras, seeds = sys.argv[2:6]; gpu = sys.argv[6] if len(sys.argv) > 6 else "1"
        pp = ("/cwStorage/home/jijh/venvs/moss_overlay:/cwStorage/home/jijh/scratch/film2_tts/ov_qwen" if engine.startswith("m")
              else "/cwStorage/home/jijh/venvs/film2_qwen_tts_overlay")
        envs = " ".join(f"{k}={shlex.quote(os.environ[k])}" for k in ("REF_WAV", "MOSS_INS") if k in os.environ)
        cmd = (f"cd /tmp && {envs} CUDA_VISIBLE_DEVICES={gpu} HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONPATH={pp} "
               f"/tmp/jijh-djev/env234/bin/python {HERE} {engine} {out} {paras} {seeds}")
        sys.exit(subprocess.call(["ssh", "-o", "RemoteCommand=none", "-o", "RequestTTY=no", "-o", "BatchMode=yes",
                                  "-o", "ClearAllForwardings=yes", "gpucw1", cmd], stdin=subprocess.DEVNULL))
    engine, out, paras, seeds = sys.argv[1:5]
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    paras = json.load(open(paras)); seeds = [int(x) for x in seeds.split(",")]
    if engine == "score":
        score(out, paras)
    elif engine == "prod":
        prod(out, paras, seeds)
    else:
        (qwen if engine.startswith("q") else moss)(engine, out, paras, seeds)

