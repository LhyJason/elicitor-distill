#!/usr/bin/env bash
#SBATCH --job-name=distill-teacher
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.log
# M2 ceiling (SPEC §7.1): the teachers' agreement with themselves on the validation set, at
# request seed 1 against their stored seed-0 replies. One job of at most 2 h on one GPU, the
# teachers served from the shared model folder one after the other. Submit from the repo root:
#   cluster/submit.sh cluster/teacher.sbatch
set -eo pipefail
source cluster/job_setup.sh
mkdir -p logs results/m2
MODELS_DIR="${USER_OLLAMA_MODELS:-$OLLAMA_MODELS}" PARALLEL=8 SERVER_LOG="logs/ollama-${SLURM_JOB_ID:-local}.log" \
    source cluster/serve_one.sh
trap stop_server EXIT

status=0
distill eval-offline --model qwen3:32b --role update --seed 1 --base-url "$SERVER_URL" \
    --workers 8 --no-unconstrained --out results/m2/teacher-qwen3-32b/update || status=1
distill eval-offline --model llama3.3:latest --role judge --seed 1 --think none \
    --base-url "$SERVER_URL" --workers 8 --no-unconstrained \
    --out results/m2/teacher-llama3.3/judge || status=1
exit "$status"
