#!/usr/bin/env bash
#SBATCH --job-name=distill-teacher
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.log
# M2 ceiling (SPEC §7.1): the teachers' agreement with themselves on the validation set, at
# request seed 1 against their stored seed-0 replies. One job of at most 2 h on one GPU, the
# teachers served from the shared model folder one after the other. Submit from the repo root,
# with the roles to run (both by default):
#   cluster/submit.sh cluster/teacher.sbatch [update] [judge]
# The 70B judge fits on one A40 only with 4 parallel slots, as in the collection (USER_PARALLEL);
# with 8 its cache spills to the CPU and a request takes minutes.
set -eo pipefail
source cluster/job_setup.sh
mkdir -p logs results/m2
roles=("$@")
if [[ ${#roles[@]} -eq 0 ]]; then roles=(update judge); fi
job="${SLURM_JOB_ID:-local}"
models="${USER_OLLAMA_MODELS:-$OLLAMA_MODELS}"

status=0
for role in "${roles[@]}"; do
    case $role in
        update) MODELS_DIR="$models" PARALLEL=8 SERVER_LOG="logs/ollama-$job-update.log" \
                    source cluster/serve_one.sh
                trap stop_server EXIT
                distill eval-offline --model qwen3:32b --role update --seed 1 \
                    --base-url "$SERVER_URL" --workers 8 --no-unconstrained \
                    --out results/m2/teacher-qwen3-32b/update || status=1 ;;
        judge)  MODELS_DIR="$models" PARALLEL=4 SERVER_LOG="logs/ollama-$job-judge.log" \
                    source cluster/serve_one.sh
                trap stop_server EXIT
                distill eval-offline --model llama3.3:latest --role judge --seed 1 --think none \
                    --base-url "$SERVER_URL" --workers 4 --no-unconstrained \
                    --out results/m2/teacher-llama3.3/judge || status=1 ;;
        *) echo "unknown role $role" >&2; exit 2 ;;
    esac
    stop_server
done
exit "$status"
