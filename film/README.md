# Pulse demo film (English, 92.8 s)

Output (not in git): `film/out/pulse_demo_final.mp4` (1920x1080, 24 fps, -14 LUFS) and `film/out/pulse_demo_720p.mp4`.
Working files: `/cwStorage/home/jijh/pulse_data/film/` (keyframes, H3 clips, UI takes, audio, reviews).

## What is real and what is generated
- **Generated:** the webcam feeds (Maya, Miso the cat, Leo) and the cinematic shots: GPT Image keyframes
  (`prompts/`, `bible.json`) animated with MiniMax H3 on gpucw1 (`h3/`).
- **Real:** every UI shot is a screen recording of Pulse Live running live on those feeds (`record_pulse.py`, film
  instance on :18910). Watch questions, readings, second looks, cards, reels and the keys answer are the app's own
  output. Waiting time is cut for length. UI shots carry a "LIVE APP · camera feed generated" tag; the end card says so.
- Numbers in the explainer come from the recorded session (`ui/drink.json`): 46 clips read, 2 second looks, 2 moments.

## Pipeline
1. `bible.json` -> character sheets + room plate (GPT Image) -> keyframes by editing the plate (`prompts/kf_*.txt`).
2. H3 (fast ComfyUI on gpucw1): 10 s webcam shots with the larryvrh turbo v4 LoRA at 1024x576, first/last frames;
   5 s cinematic shots with the official turbo 8-step LoRA at 1344x768 (`h3/jobs_*.json`).
3. `record_pulse.py main drink walkin`: fake camera = H3 clips (y4m), 2560x1440 capture (CSS layout 1536x864),
   CDP screencast ~30 fps, beat log + element boxes, moment media saved before "Forget everything". Retries a take
   when the detection does not happen (see limits).
4. `motion.html` + `render_motion.py`: deterministic title, explainer, share, memory and end scenes (Inter).
5. `tts_en.py`: Qwen3-TTS VoiceDesign (3 voices, one-pass script) -> blind Gemini pick -> per-line clones.
   Music: ACE-Step 1.5 XL cues (`music/*.txt`), blind Gemini picks (`blind_pick.py`).
6. `cut.py`: virtual camera over the UI takes (crop keyframes from recorded beats), tags, crossfades, VO placement
   (multi-sentence lines split at the pause nearest the sentence boundary), ducking, sfx, the clips' own sound;
   final loudness -14 LUFS.
7. Reviews: Gemini 4 personas (creative director, ordinary viewer, hackathon judge/investor, QA) on the storyboard
   and on 4 cuts. Final round: director 9.2, viewer 9.5, judge 9.5, QA "ready to publish" (round 1: 8.2 / 7 / 8.5).

## Product fixes the film exposed (all in live.py / web/live.html)
Unmirrored 16:9 stage + "Right now" sparklines; English-only labels in English mode; bundled Inter + emoji-safe
labels; per-connection clock offset (a reloaded page used to stall judging); 30 s cooldown per moment; re-arm after a
rejected candidate (a "reaching for the glass" candidate used to block the real sip); "Taking a second look…" pill;
reading shown before the second look, second look no longer blocks later clips; moment sent before its reel renders;
keys answer located from the evidence frame + evidence replay on the stage; "Forget everything" keeps the watches.

## Limits (be honest when presenting)
- Detection on this footage is not always reliable: the cat watch needed 1-5 attempts per take, because the fast
  model's answer depends on how the drafted question is worded and on the other questions asked with it.
  Candidate idea: generate candidates from a short plain phrasing as well, and let the second look keep precision.
- MiniMax H3 Community License restricts public use in the US / EU / UK / South Korea; check it before posting the
  film publicly from those regions.
