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
