#!/usr/bin/env bash
# Pulse Live on Meta Muse Glimmer only (open weights, one GPU, no cloud API):
# fast judge (low reasoning, yes/no log-probabilities), second look (medium reasoning) and brain (watch questions,
# tool-calling answers) all use the vLLM server started by tools/serve_glimmer.sh on the GPU node.
#   ./start_muse.sh [--port 18900] [--data DIR]
set -u
cd "$(dirname "$0")"
PY=${PY:-python3}
GPU=${GPU:-}   # user@gpu-host: if set, a missing model port is tunnelled from there
up() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
if ! up 18850 && [ -n "$GPU" ]; then
  echo "port 18850 is down; opening a tunnel to $GPU"
  ssh -F /dev/null -f -N -o BatchMode=yes -o ConnectTimeout=8 -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 -L 127.0.0.1:18850:127.0.0.1:18850 "$GPU" || true
  sleep 1
fi
up 18850 && echo "ok   Muse Glimmer :18850" || { echo "FAIL Muse Glimmer :18850 (run tools/serve_glimmer.sh on the GPU node)"; exit 1; }
exec "$PY" live.py --judge glimmer --brain glimmer --verifier glimmer "$@"
