#!/usr/bin/env bash
# Parallel ranged download of Hugging Face files (single streams can be slow). HF_BASE defaults to huggingface.co.
# usage: dl_ranges.sh <repo> <outdir> <parts> <file> [file ...]
set -u
REPO=$1; OUT=$2; N=$3; shift 3; mkdir -p "$OUT"
for F in "$@"; do
  URL="${HF_BASE:-https://huggingface.co}/$REPO/resolve/main/$F"
  SIZE=$(curl -sIL "$URL" | awk 'tolower($1)=="content-length:"{s=$2} END{print s+0}' | tr -d '\r')
  if [ "$SIZE" -lt 10000000 ]; then curl -sL -o "$OUT/$F" "$URL"; echo "small $F"; continue; fi
  P=$(( (SIZE + N - 1) / N )); pids=""
  for i in $(seq 0 $((N-1))); do
    a=$((i*P)); b=$(( (i+1)*P - 1 )); [ $b -ge $SIZE ] && b=$((SIZE-1))
    ( for t in 1 2 3 4 5; do curl -sL -r $a-$b -o "$OUT/$F.part$i" "$URL" && [ $(stat -c %s "$OUT/$F.part$i") -eq $((b-a+1)) ] && break; sleep 3; done ) & pids="$pids $!"
  done
  wait $pids
  cat $(for i in $(seq 0 $((N-1))); do echo "$OUT/$F.part$i"; done) > "$OUT/$F" && rm -f "$OUT/$F".part*
  echo "done $F $(stat -c %s "$OUT/$F") / $SIZE"
done
