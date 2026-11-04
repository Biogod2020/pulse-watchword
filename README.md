# Pulse — tell your camera what to watch for

**Meta Global AI Developer Hackathon (Oct 27 – Nov 4, 2026) · track: Build for Your World**

Type a sentence like *"tell me when the cat walks across my laptop"*. Pulse watches every second with **Meta Muse
Glimmer** (open weights, one GPU, no cloud API). It thinks hard only when something looks likely, then saves the
moment as a shareable card. Later you can ask *"where did I leave my keys?"*.

- **Demo video (2:43):** `<YouTube link>`
- **Submission write-up** (what it does, how Meta models are used, results, limits, what was built during the
  hackathon): [SUBMISSION.md](SUBMISSION.md)
- **Measurements:** [docs/validation_muse_glimmer.md](docs/validation_muse_glimmer.md)

## How it works
| Step | Model call |
|---|---|
| Plain words → one precise yes/no question | Muse Glimmer, reasoning strength *low* |
| Every second: 4 frames from the last 2 s → P(yes) from the first answer token's log-probabilities | Muse Glimmer, assistant turn pre-filled, `max_tokens=1` |
| Candidate (≥ 0.15) → second look | Muse Glimmer, reasoning strength *medium* |
| "Where did I leave my keys?" | Muse Glimmer tool calls (`find_object`, `describe_recent`) over a 3-minute in-memory ring |

## Quick start
```bash
pip install -r requirements.txt                 # app host; ffmpeg on PATH (or PULSE_FFMPEG)
# GPU node (1x 80 GB): serve Muse Glimmer with vLLM >= 0.32
bash tools/serve_glimmer.sh 0 18850            # MODEL=/path/to/Muse-Glimmer-30B if downloaded locally
# app host (same machine, or set GPU=user@gpu-host to tunnel port 18850)
./start_muse.sh --port 18900
# open http://localhost:18900 and allow the camera
python -m pytest -q tests
```

## Repository map
| Path | What |
|---|---|
| `live.py`, `web/live.html` | Pulse Live: FastAPI + WebSocket server and the browser app |
| `pulse/providers.py` | model adapters. `ChatYesNoDecider` turns any OpenAI-compatible vision chat into a yes/no probability reader (Muse Glimmer). Muse Spark via the Meta Model API is implemented but not demonstrated |
| `pulse/stream.py`, `pulse/triggers.py`, `pulse/recall.py`, `pulse/cards.py` | frame ring, triggers, look-back recall, moment cards and reels |
| `tools/serve_glimmer.sh`, `start_muse.sh` | serve Muse Glimmer; run Pulse on Muse Glimmer only |
| `eval_glimmer_windows.py`, `eval_heldout.py`, `tools/glimmer_scan.py` | held-out evaluation on EPIC-KITCHENS-100 P12_101; stock-clip scan |
| `film4/` | the submission film: motion scenes, cut script, narration pipeline (`docs/audio_narration.md`) |
| `film/`, `film2/`, `film3/` | earlier demo films |

Paths to datasets, footage and model weights in the evaluation and film scripts point at our own machines.
Adjust them before rerunning. License: Apache-2.0 (see `LICENSE`). Muse Glimmer weights are Meta's (Apache 2.0).
Stock footage and EPIC-KITCHENS are not included: see `film2/CREDITS.md` and the end card of the film.

## Development history (earlier path, still selectable as a baseline)

Pulse is a general runtime for assistants that live alongside you. Speech and first-person camera frames
arrive as events. On every event a cheap decision model answers a handful of narrow yes/no trigger
questions over working memory (and the frame itself), and only the capabilities whose trigger fires wake
up: tool-calling agents, vision reads, memory writes. Some trigger questions are generated from working
memory itself (one per open task), so what the assistant watches for changes with what it is tracking.

Memory has two layers:
- Working memory: current facts (tasks, where items were last seen, counts), each tied to the input
  revision it rests on, so a corrected transcript invalidates exactly what was built on the old wording.
- Long-term memory: every utterance and kept frame embedded with EmbeddingGemma 2 into one text/image
  space, so "where did I put my glasses?" can find a frame nobody described. Frames and utterances are
  ranked separately because text-text similarity runs higher than text-image.

## Pulse Live (browser app, your webcam)
Tell your camera, in plain words, what to watch for. When it happens, Pulse saves the moment as a
shareable card (1080x1350) and a 9:16 reel, and you can ask "where did I leave my keys?" afterwards.

    ./start_live.sh                      # baseline judge (djev :18831), brain (:8317), verifier key
    # then open http://localhost:18900 (browsers allow the camera on localhost)

How it decides: every second the browser's last 2 s of frames (4 fps) go to djev as one video clip, and
every watch is a yes/no question. A clip at 0.5 or above is a candidate. A gpt-6-luna Decisions call on 3
frames confirms it (yes at 0.2 or above) before a moment is saved. Without a verifier
(`--verifier none`), a moment needs two agreeing clips at 0.8.

Privacy: frames live in a 3-minute in-memory ring and are deleted from disk when they leave it. Only a
keyframe every 30 s and moment evidence are kept. "Forget everything" deletes the session.

Known limits: the verifier evidence is a small sample (one kitchen recording plus webcam tests), and the
built-in gestures were tuned on little data. Verifier calls are billed to the OpenAI key (capped in
`~/.config/pulse/luna_spend.json`).

## Packs
- `packs/daily.py`: commitments, where things were left, tasks seen being done, questions (demo:
  `scenarios/daily/`, synthetic speech + GPT Image first-person frames).
- `packs/surgical_count.py`: an operating-room sponge-count example (record keeping).

## Providers (pluggable)
- Decision / vision: djev (DiffusionGemma structured reads with per-option probabilities, image input and
  captions; a pre-existing server reached through a new client) or any OpenAI-compatible model.
- Agent: any OpenAI-compatible tool-calling model (a local DeepSeek proxy stands in during development;
  Meta Muse Spark is the target).
- Embeddings: EmbeddingGemma 2 via `tools/embed_server.py`.

## Run
    python -m pytest -q tests
    python tools/embed_server.py --port 18820         # on a CPU or GPU node
    python run_daily.py --djev-url http://127.0.0.1:18811 --embed-url http://127.0.0.1:18820
    python eval_triggers.py                            # trigger-question design comparison
