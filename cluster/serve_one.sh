#!/usr/bin/env bash
# One Ollama server on GPU 0, for the M2 jobs (one model at a time). Source it after
# cluster/job_setup.sh:
#   MODELS_DIR=<folder> PARALLEL=8 SERVER_LOG=logs/x.log source cluster/serve_one.sh
# Afterwards OLLAMA_HOST and SERVER_URL point at it and stop_server stops it. OLLAMA_DEBUG=2
# (trace) makes the server log every rendered prompt, which `distill check-template` reads; it
# also logs every token it encodes or decodes (about 18 MB a request), and those lines are
# dropped before they reach the log.
port=$(python -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
export OLLAMA_HOST="127.0.0.1:$port"
SERVER_URL="http://$OLLAMA_HOST"
CUDA_VISIBLE_DEVICES=0 OLLAMA_NUM_PARALLEL=${PARALLEL:-8} OLLAMA_MAX_LOADED_MODELS=1 \
    OLLAMA_KEEP_ALIVE=-1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_CONTEXT_LENGTH=2048 OLLAMA_NOPRUNE=1 \
    OLLAMA_MODELS="$MODELS_DIR" OLLAMA_DEBUG=${OLLAMA_DEBUG:-0} \
    ollama serve > >(grep --line-buffered -avE 'msg=(de|en)coded ' > "$SERVER_LOG") 2>&1 &
SERVER_PID=$!  # ollama's, not the filter's
stop_server() { kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null || true; }
for _ in $(seq 1 120); do
    if curl -sf "$SERVER_URL/api/tags" > /dev/null; then break; fi
    sleep 1
done
if ! curl -sf "$SERVER_URL/api/tags" > /dev/null; then
    echo "ollama did not start within 120 s" >&2
    return 1
fi
echo "ollama at $SERVER_URL (models in $MODELS_DIR, $(ollama --version | tail -1))"
