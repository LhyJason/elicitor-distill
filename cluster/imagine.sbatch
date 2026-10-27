#!/usr/bin/env bash
#SBATCH --job-name=distill-imagine
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.log
# M5 (SPEC §7.5): the teacher's `imagine` records for every (question, value) pair, then the
# imagine SFT sets beside the frozen M1 sets (their manifest entries are kept). At most 1 h on 2
# GPUs, both serving the agent model qwen3:32b: the prompt has no persona, so the simulator is
# not needed. A job that stops early is resubmitted with the same command and resumes through
# the reply cache. Submit from the repo root:
#   cluster/submit.sh cluster/imagine.sbatch
set -eo pipefail
export CONFIG="${1:-experiments/imagine.toml}"
export REQUIRE_GPU=1 AGENT_GPUS=2 USER_GPUS=0
source cluster/job_setup.sh
# The teacher sits in the shared model folder, not in the agent group's folder of students (found
# 10-27: job 64856349 stopped after 4 s, Ollama answering 404 for qwen3:32b).
export AGENT_OLLAMA_MODELS="${USER_OLLAMA_MODELS:-$OLLAMA_MODELS}"
source cluster/serve_models.sh
trap stop_models EXIT
# No user-group call is made, but Elicitor's client wants a server for both groups.
export ELICITOR_SERVERS_USER=$ELICITOR_SERVERS_AGENT
raw="data/raw/$(basename "$CONFIG" .toml)"
sha256sum data/travel/schema.toml data/travel/prompts/imagine.txt

status=0
start=$(date +%s)
distill imagine-collect "$CONFIG" --workers 16 --out "$raw" || status=$?
echo "imagine records: $(( $(date +%s) - start )) s (exit $status)"
distill build-sft --raw "$raw" --out data/sft --roles imagine || status=$?
exit "$status"
