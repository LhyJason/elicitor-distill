#!/usr/bin/env bash
#SBATCH --job-name=distill-collect
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.log
# M1 collection (SPEC §4, §9): the training personas' backstories (only if not written yet), the
# update records, the judge records, then the SFT sets, with the health of each step. At most
# 6 h on 4 GPUs. A job that stops early is resubmitted with the same command and resumes through
# the reply cache. Run the smoke file first; its request rates decide whether the full file fits
# (SPEC §12 lists the cuts). Submit from the repo root:
#   cluster/submit.sh cluster/collect.sbatch experiments/collect_smoke.toml
#   cluster/submit.sh cluster/collect.sbatch [experiments/collect.toml]
# The GPU split between the model groups is set at submission, e.g. AGENT_GPUS=1 USER_GPUS=3
# when the simulator (user group) is the bottleneck.
set -eo pipefail
export CONFIG="${1:-experiments/collect.toml}"
export REQUIRE_GPU=1
source cluster/job_setup.sh
source cluster/serve_models.sh
trap stop_models EXIT
name=$(basename "$CONFIG" .toml)
raw="data/raw/$name"
sha256sum data/personas_train.jsonl data/travel/*.jsonl data/travel/prompts/*.txt

if [[ ! -f data/backstories.jsonl ]]; then
    start=$(date +%s)
    distill make-backstories "$CONFIG" --workers 16 --out "$raw" || echo "some backstories stay flagged"
    echo "backstories: $(( $(date +%s) - start )) s"
fi
sha256sum data/backstories.jsonl

status=0
start=$(date +%s)
distill collect "$CONFIG" --workers 32 --out "$raw" || status=$?
echo "update records: $(( $(date +%s) - start )) s (exit $status)"

start=$(date +%s)
distill judge-collect "$CONFIG" --workers 32 --out "$raw" || status=$?
echo "judge records: $(( $(date +%s) - start )) s (exit $status)"

# The SFT sets of a smoke run stay beside its raw records; the full run's go to data/sft/.
if [[ "$name" == *smoke* ]]; then sft="$raw/sft"; else sft="data/sft"; fi
distill build-sft --raw "$raw" --out "$sft" || status=$?
exit "$status"
