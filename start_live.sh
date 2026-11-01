#!/usr/bin/env bash
# Start Pulse Live on headcw: check the model endpoints, open the GPU tunnels if they are down, then serve.
# On your own computer:  ssh -L 18900:127.0.0.1:18900 headcw   then open http://localhost:18900
set -u
cd "$(dirname "$0")"
PY=${PY:-../medcopilot/.venv/bin/python}
GPU=${GPU:-jijh@192.168.17.113}
up() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }

for port in 18831; do
  if ! up "$port"; then
    echo "port $port is down; opening a tunnel to $GPU"
    ssh -F /dev/null -f -N -o BatchMode=yes -o ConnectTimeout=8 -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=30 -L "127.0.0.1:$port:127.0.0.1:$port" "$GPU" || true
    sleep 1
  fi
  up "$port" && echo "ok   judge :$port" || echo "FAIL judge :$port (start djev on the GPU node; see README)"
done
up 8317 && echo "ok   brain :8317 (CLIProxyAPI)" || echo "FAIL brain :8317 (plain-language watches will not work)"
[ -s ~/.config/pulse/openai_key ] && echo "ok   verifier key" || echo "warn no verifier key; run with --verifier none"

exec "$PY" live.py "$@"
