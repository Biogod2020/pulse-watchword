#!/usr/bin/env bash
# Serve Meta Muse Glimmer 30B (open weights, Apache 2.0) with vLLM (>= 0.32, which ships the muse_glimmer model,
# tool parser and reasoning parser) for Pulse's judge, second look and brain. One 80 GB GPU (BF16 ~60 GB).
# usage: bash tools/serve_glimmer.sh [gpu=0] [port=18850] [util=0.88]
#   MODEL=/path/to/Muse-Glimmer-30B   (default: the Hugging Face id; tools/dl_ranges.sh fetches it in parallel)
#   VLLM=/path/to/vllm                 (default: vllm on PATH)
set -euo pipefail
GPU=${1:-0}; PORT=${2:-18850}; UTIL=${3:-0.88}
VLLM=${VLLM:-vllm}
MODEL=${MODEL:-meta-models/Muse-Glimmer-30B}
export CUDA_VISIBLE_DEVICES=$GPU VLLM_USE_FLASHINFER_SAMPLER=0 OMP_NUM_THREADS=8
exec "$VLLM" serve "$MODEL" --served-model-name muse-glimmer \
  --gpu-memory-utilization "$UTIL" --max-model-len 16384 --max-num-seqs 32 \
  --limit-mm-per-prompt '{"image": 4}' --max-logprobs 20 \
  --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser muse_glimmer \
  --reasoning-parser muse_glimmer --host 127.0.0.1 --port "$PORT"
