# Pulse: an intuition heartbeat for always-on agents

Built for the Meta Global AI Developer Hackathon (Oct 27 – Nov 4, 2026). Everything in this
repository was written during the hackathon.

On every event (a transcript line, an ASR revision, a camera frame) a cheap decision model answers a
handful of multiple-choice trigger questions over versioned working memory, and only the capabilities
whose trigger fires wake up: tool-calling agents, vision recounts, writers. Deterministic rules run as a
backstop. Every fact carries the input revision it rests on, so a corrected transcript invalidates exactly
what was built on the old wording.

First capability pack: operating-room sponge count reconciliation (record keeping, not clinical advice).
Scenario data is synthetic.

## Providers (pluggable)
- Decision: any OpenAI-compatible model (label only), or djev (DiffusionGemma structured read with
  per-option probabilities; a pre-existing server reached through a new client). Meta Muse Spark Open is the
  target decider.
- Agent: any OpenAI-compatible tool-calling model. Meta Muse Spark is the target; a local DeepSeek proxy is
  the temporary stand-in until API credits arrive.
- Vision: fixture labels now; SAM 3.1 next.

## Run
    python -m pytest -q tests
    python run_scenario.py --decider deepseek
    python run_scenario.py --decider djev --djev-url http://127.0.0.1:18811
