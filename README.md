# Pulse: an intuition heartbeat for always-on agents

Built for the Meta Global AI Developer Hackathon (Oct 27 – Nov 4, 2026). Everything in this repository
was written during the hackathon.

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

    # on headcw
    ./start_live.sh                      # checks judge tunnels :18811/:18831, brain :8317, verifier key
    # on your own computer
    ssh -L 18900:127.0.0.1:18900 headcw
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
