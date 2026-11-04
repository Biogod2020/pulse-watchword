# Pulse — tell your camera what to watch for

**Meta Global AI Developer Hackathon · track: Build for Your World**

**One line:** Type a sentence like "tell me when the cat walks across my laptop". Pulse watches every second with
**Meta Muse Glimmer** running on our own GPU, but only thinks hard when something looks likely. When the event
happens, it saves a shareable moment, and you can ask it later "where did I leave my keys?".

Demo video (2:43): `<YouTube link>` · Repository: https://github.com/Biogod2020/pulse-watchword

---

## The problem
Home and desk cameras recognise only a few fixed labels (person, package, pet). A vision-language model understands
any request, but running a big one on every second of video, from every camera, all day, is slow and expensive.
So in practice nobody can say "tell me when the kids get into the snack cupboard" or "tell me when the cat goes for the cake".

## What Pulse does
1. **Plain words in.** Muse Glimmer turns your sentence into one precise yes/no question, with the near-misses it
   should ignore ("…walking across the laptop, not just sitting near it").
2. **Quick look, every second.** Muse Glimmer at *reasoning strength low* sees 4 frames covering the last
   2 seconds and answers each watch with one token. We read **P(yes) from the token log-probabilities**, so every
   watch has a smooth score instead of a hard label.
3. **Second look, only on candidates.** When the score crosses the candidate line, the *same* model is asked again
   at *reasoning strength medium*: it reasons over the frames before it answers. Only confirmed events become moments.
4. **Moments and memory.** A confirmed event becomes a card and a 9:16 reel. Frames stay in a 3-minute in-memory
   ring and are deleted from disk when they leave it. "Where did I leave my keys?" is a tool-calling agent run by
   Muse Glimmer: it scans back through memory (`find_object`) and answers with the frame as evidence.

Everything runs on **one model, Muse Glimmer 30B (open weights), on a single H100, with no cloud API**. Open weights are
what make the design work: we serve the model ourselves, so the every-second quick look costs no API fees. We pre-fill
the answer and read token log-probabilities, and we pick the reasoning strength per call (low for the quick look,
medium for the second look).

## How we use Meta's models
| Role | Model | How |
|---|---|---|
| Fast judge (every second) | Muse Glimmer 30B, BF16, vLLM 0.32 | 4 frames + question; the assistant turn is pre-filled with `Answer:` so the model answers the user directly; `max_tokens=1`, `top_logprobs=20`; P(yes) = mass on yes / (yes + no) |
| Second look (candidates only) | Muse Glimmer, `Reasoning strength: medium` | the model thinks in its reasoning channel (vLLM `muse_glimmer` reasoning parser), then answers yes/no |
| Brain | Muse Glimmer, `Reasoning strength: low` | writes the watch question; answers questions with tools (`find_object`, `describe_recent`, `recent_moments`) through vLLM's `muse_glimmer` tool parser |
| Optional | Muse Spark via the Meta Model API | `--brain muse --verifier muse` is implemented (OpenAI-compatible `https://api.meta.ai/v1`), **but not demonstrated**: we did not have an API key during the build |

Serving: `tools/serve_glimmer.sh` · launcher: `start_muse.sh` · judge: `ChatYesNoDecider` in `pulse/providers.py`.

## Results (our own measurements)
**Held-out test on real first-person video.** EPIC-KITCHENS-100 `P12_101`: 34 minutes from a head-mounted camera,
with human action labels. Every second, the judge reads a 2-second clip (4 frames) and answers 6 activity questions.
Alert thresholds were chosen on minutes 0–10 only and then frozen. The table scores minutes 10–34. We ran the same
procedure on the open model we started with (the baseline).

| Activity | Muse Glimmer: alerts right (n) | Baseline: alerts right | Chance | Event ROC-AUC (Glimmer / baseline) |
|---|---|---|---|---|
| Washing at the sink | **78%** (23) | 54% | 7% | 0.97 / 0.96 |
| Stirring a pot | **74%** (43) | 18% | 36% | 0.89 / 0.89 |
| Pouring | **73%** (11) | 35% | 9% | 0.92 / 0.79 |
| Opening a cupboard or jar | **67%** (6) | 24% | 10% | 0.81 / 0.80 |
| Putting something down | **42%** (41) | 20% | 11% | 0.87 / 0.71 |
| Turning a tap or knob | 50% (34) | **63%** | 9% | 0.90 / 0.94 |

"Chance" is the share of randomly placed alerts that would land on the action. Throughput: 2,044 clips × 6 questions
in 495 s on one H100 (≈4× real time; questions run in parallel and share the image prefix). A quick-look read takes
about 0.2 s. A second look takes 4–14 s.

**Live demo on stock footage** (the takes in the video; each is a real run of `start_muse.sh`):

| Take | What happened |
|---|---|
| "tell me when the cat walks across my laptop" | quick look peaked at 0.33, second look confirmed |
| "tell me when the cat is on the dinner table" | 0.35, confirmed |
| "tell me every time I drink coffee" | 0.54, confirmed (on the first attempt the second look rejected the candidate) |
| "tell me if the cat goes for the cake" | 0.16, confirmed |
| "where did I leave my keys?" | Glimmer called `find_object`, scanned 21 frames, answered correctly in 7.3 s |

**Discrimination check** (`tools/glimmer_scan.py`, 6 clips, every second). We also asked each clip a question meant
for a different clip. Those mismatched questions never scored above 0.08.

Details and scripts: `docs/validation_muse_glimmer.md`, `eval_glimmer_windows.py`, `eval_heldout.py`.

## Next: Ray-Ban Meta glasses (roadmap, not built)
A head-mounted camera is the natural home for Pulse. Meta's Wearables Device Access Toolkit streams the glasses'
camera to a phone app; Pulse would read that stream, chime in the open-ear speakers, and save the moment to the phone.
The demo video shows this as a **labelled concept**. The footage is a real head-mounted camera (EPIC-KITCHENS), and
the alert is Muse Glimmer's real output on that footage. The glasses integration itself is not built.

## What was built during the hackathon (Oct 27 – Nov 4, 2026)
Everything in this repository was written during the hackathon (see the git history, Oct 31 – Nov 4):
- the Pulse runtime: plain-language watches, the two-speed cascade, triggers, the memory ring, and look-back recall;
- the Muse Glimmer integration: the log-probability yes/no judge, the reasoning-strength second look, the
  tool-calling brain, and the vLLM serving script;
- Pulse Live (browser webcam app, moment cards and reels);
- the evaluations (`eval_glimmer_windows.py`, `eval_heldout.py`, `tools/glimmer_scan.py`) and the demo-film tooling.

**Not ours / pre-existing:** Muse Glimmer weights (Meta); vLLM; the demo-video narration voice (AI-generated with MOSS-TTS v1.5, open weights); stock footage (Pexels, Mixkit, see
`film2/CREDITS.md`); EPIC-KITCHENS-100 (CC BY-NC 4.0). Before switching to Muse Glimmer we prototyped with
DiffusionGemma 26B-A4B (Google, open weights) through a pre-existing structured-output server ("djev", not in this
repo; we added video input to it during the hackathon). It remains as a
selectable baseline (`--judge djev`) and appears only as the comparison row in our results.

## Limits (honest)
- **Glimmer is cautious.** It misses small, quick actions: a short wave, or a hand briefly at the mouth. On these
  clips the quick look peaked at 0.33–0.38, and the second look said no.
- **Turning a tap or knob** is worse than with our baseline (50% vs 63% of alerts right).
- **The second look takes 4–14 s**, so alerts arrive a few seconds after the event.
- **Wording matters.** The same request, phrased differently, can be missed. The candidate line (0.15) was set by hand
  on a handful of clips.
- **Small evidence base.** One 34-minute video plus stock clips. "Opening" has only 6 alerts.
- **Not demonstrated:** Muse Spark through the Meta Model API (no key) and the glasses integration.

## Run it
```bash
# GPU node (1x H100 80 GB): serve Muse Glimmer
bash tools/serve_glimmer.sh 0 18850
# app host: tunnel + Pulse Live on Muse Glimmer only
./start_muse.sh --port 18900
# then open http://localhost:18900 (browsers allow the camera on localhost)
python -m pytest -q tests
python eval_heldout.py runs/video_windows_full_c2_s1.json runs/glimmer_windows_full_c2_s1.json   # rescoring needs no GPU: the per-clip scores are in runs/
# python eval_glimmer_windows.py   # regenerates the Glimmer scores (needs EPIC-KITCHENS P12_101 frames and the Glimmer server)
```

## Team
`<names>`
