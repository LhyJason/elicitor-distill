#!/usr/bin/env bash
#SBATCH --job-name=distill-model
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.log
# M2 (SPEC §5, §6, §7.1): one student or zero-shot baseline, from the base weights to an
# evaluated Ollama model, in one job of at most 1 h on one GPU. Submit from the repo root:
#   cluster/submit.sh cluster/train.sbatch configs/update_1.7b.toml   # an SFT student
#   cluster/submit.sh cluster/train.sbatch Qwen3-1.7B                 # a zero-shot baseline
# Steps: `distill train` (students only) → `distill export --create` into the agent group's
# writable model folder → a 20-sample smoke test (one worker, no cache, 0 parse failures) →
# `distill check-template` on the prompts the server logged → `distill eval-offline` on the
# validation set (both roles for a baseline) into results/m2/<name>/<role>/.
set -eo pipefail
source cluster/job_setup.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p logs results/m2
for var in AGENT_OLLAMA_MODELS HF_DIR LLAMA_CPP; do
    if [[ -z "${!var}" ]]; then echo "set $var in cluster/local.env" >&2; exit 2; fi
done

if [[ "$1" == *.toml ]]; then
    read -r name role base < <(python - "$1" <<'PY'
import sys, tomllib
with open(sys.argv[1], "rb") as f:
    c = tomllib.load(f)
print(c["name"], c["role"], c["base"].split("/")[-1])
PY
)
    start=$(date +%s)
    distill train "$1" --base-dir "$HF_DIR"
    echo "train: $(( $(date +%s) - start )) s"
    adapter=(--adapter "models/$name/adapter")
    roles=("$role")
else
    base=$1
    name="$(echo "$base" | tr '[:upper:]' '[:lower:]')-zs"  # e.g. qwen3-1.7b-zs
    adapter=()
    roles=(update judge)  # a baseline is one model for both roles
fi
model="elicitor-$name"
log="logs/ollama-${SLURM_JOB_ID:-local}.log"
MODELS_DIR="$AGENT_OLLAMA_MODELS" PARALLEL=8 SERVER_LOG="$log" OLLAMA_DEBUG=2 \
    source cluster/serve_one.sh
trap stop_server EXIT

start=$(date +%s)
distill export "$model" --base "$HF_DIR/$base" "${adapter[@]}" --out "models/$name" \
    --llama-cpp "$LLAMA_CPP" --create
echo "export: $(( $(date +%s) - start )) s"

status=0
for role in "${roles[@]}"; do
    before=$(wc -l < "$log")
    distill eval-offline --model "$model" --role "$role" --base-url "$SERVER_URL" \
        --limit 20 --workers 1 --strict --no-unconstrained || status=1
    tail -n +"$((before + 1))" "$log" > "logs/prompts-${SLURM_JOB_ID:-local}-$role.log"
    distill check-template --log "logs/prompts-${SLURM_JOB_ID:-local}-$role.log" \
        --role "$role" --limit 20 || status=1
    distill eval-offline --model "$model" --role "$role" --base-url "$SERVER_URL" \
        --workers 8 --out "results/m2/$name/$role" || status=1
done
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
exit "$status"
