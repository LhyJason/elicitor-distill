#!/usr/bin/env bash
#SBATCH --job-name=distill-e2e
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.log
# M3 (SPEC §7, §9, §10): the end-to-end job, at most 4 h on 4 GPUs, after D1-D6 were frozen.
# GPU 0 serves the seven M2 models from the agent group's folder (AGENT_GPUS=1), GPUs 1-3 the
# simulator llama3.3 (USER_GPUS=3), since only the simulator's answers to questions the teacher
# never asked are new. Elicitor's run folders (ELICITOR_RESULTS, default ~/elicitor/results:
# m5_full, m5_smoke_calibration, m7_judge) are read, never written; their reply caches are copied.
# Steps:
#   1. `distill build-test`: the update and judge test sets (the first read of eval data);
#   2. per update model, in parallel: `elicitor calibrate` on the dev personas (m5_smoke's file,
#      the model as agent) → `distill m3-file` with the measured rho → `elicitor run` of the two
#      arms (m5_full's settings) → `elicitor health`; beside them the test-set offline table;
#   3. `distill compare` against m5_full, `distill judge-eval` (D5), the test table;
#   4. `distill bench` on GPU 0 alone: the SFT students, then qwen3:32b (D6);
#   5. the tables to results_summary/m3/.
# A job that stops early is resubmitted with the same command; every step resumes or is skipped
# once its output exists. Submit from the repo root:
#   cluster/submit.sh cluster/e2e.sbatch
set -eo pipefail
source cluster/job_setup.sh
src="${ELICITOR_RESULTS:-$HOME/elicitor/results}"
out=results/m3
for f in "$src/m5_full/run_state.json" "$src/m5_smoke_calibration/llm_cache.sqlite" \
         "$src/m7_judge/units.jsonl"; do
    if [[ ! -f "$f" ]]; then echo "missing $f (set ELICITOR_RESULTS)" >&2; exit 2; fi
done
if [[ -z "$AGENT_OLLAMA_MODELS" ]]; then echo "set AGENT_OLLAMA_MODELS" >&2; exit 2; fi
mkdir -p logs "$out" experiments/m3 results_summary/m3
UPDATE=(upd-0.6b-sft upd-1.7b-sft upd-4b-sft qwen3-0.6b-zs qwen3-1.7b-zs qwen3-4b-zs)
JUDGE=(judge-1.7b-sft qwen3-0.6b-zs qwen3-1.7b-zs qwen3-4b-zs)  # the SFT judge first (D5)
start=$(date +%s)
since() { echo "$1: $(( $(date +%s) - start )) s since the start"; }

if [[ ! -f "$out/test/judge_test.jsonl.gz" ]]; then
    distill build-test --m5 "$src/m5_full" --judge "$src/m7_judge" --out "$out/test" --allow-eval
fi
sha256sum "$out"/test/*.jsonl.gz

export CONFIG=experiments/m3_e2e.toml AGENT_GPUS=${AGENT_GPUS:-1} USER_GPUS=${USER_GPUS:-3}
export AGENT_MODEL_OVERRIDE=elicitor-upd-1.7b-sft AGENT_MAX_LOADED=7 REQUIRE_GPU=1
source cluster/serve_models.sh
trap stop_models EXIT
for name in "${UPDATE[@]}" judge-1.7b-sft; do
    for url in "${agent_urls[@]}"; do load_model "$url" "elicitor-$name"; done
done
for url in "${agent_urls[@]}"; do
    loaded=$(OLLAMA_HOST="${url#http://}" ollama ps); echo "$loaded"
    if [[ $(echo "$loaded" | tail -n +2 | grep -c "100% GPU") != 7 ]]; then
        echo "not all seven models sit fully on the agent GPU" >&2; exit 1
    fi
done
agent_url=${agent_urls[0]}
since "servers"

chain() {  # one update model: calibrate, its experiment file, the run, its health
    local name=$1 cal="$out/$1/calibration" run="$out/$1/run"
    mkdir -p "$cal" "$run"
    [[ -f "$cal/llm_cache.sqlite" ]] || cp "$src/m5_smoke_calibration/llm_cache.sqlite" "$cal/"
    distill m3-file experiments/m3_calibrate.toml --model "elicitor-$name" \
        --name "calibrate_$name" --out "experiments/m3/calibrate_$name.toml"
    if [[ ! -f "$cal/calibration.json" ]]; then
        elicitor calibrate "experiments/m3/calibrate_$name.toml" --out "$cal" --workers 8
    fi
    distill m3-file experiments/m3_e2e.toml --model "elicitor-$name" --name "e2e_$name" \
        --calibration "$cal/calibration.json" --out "experiments/m3/e2e_$name.toml"
    [[ -f "$run/llm_cache.sqlite" ]] || cp "$src/m5_full/llm_cache.sqlite" "$run/"
    elicitor run "experiments/m3/e2e_$name.toml" --out "$run" --workers 16
    elicitor health "$run"
}

offline() {  # $1 name, $2 role: the test-set offline table of one model
    local dir="$out/$1/$2_test"
    [[ -f "$dir/summary.json" ]] && return 0
    distill eval-offline --model "elicitor-$1" --role "$2" --data "$out/test/$2_test.jsonl.gz" \
        --allow-eval --base-url "$agent_url" --workers 8 --no-unconstrained --out "$dir"
}

pids=()
for name in "${UPDATE[@]}"; do
    chain "$name" > "logs/e2e-${SLURM_JOB_ID:-local}-$name.log" 2>&1 & pids+=($!)
done
{
    for name in "${UPDATE[@]}"; do offline "$name" update || echo "offline $name update failed"; done
    for name in "${JUDGE[@]}"; do offline "$name" judge || echo "offline $name judge failed"; done
} > "logs/e2e-${SLURM_JOB_ID:-local}-offline.log" 2>&1 & pids+=($!)
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
since "runs and offline tables (exit $status)"
for name in "${UPDATE[@]}"; do
    echo "== $name"; tail -n 25 "logs/e2e-${SLURM_JOB_ID:-local}-$name.log"
done
grep -aE "health|failed|^\{" "logs/e2e-${SLURM_JOB_ID:-local}-offline.log" || true

runs=(); for name in "${UPDATE[@]}"; do runs+=("$out/$name/run"); done
judged=(); for name in "${JUDGE[@]}"; do judged+=("$out/$name/judge_test"); done
distill compare "${runs[@]}" --teacher "$src/m5_full" --out "$out/compare" || status=1
distill judge-eval "${judged[@]}" --out "$out/judge_test.md" || status=1
distill eval-table "$out" --out "$out/test_table.md" || status=1

stop_models
trap - EXIT
mkdir -p "$out/bench"
for model in elicitor-upd-0.6b-sft elicitor-upd-1.7b-sft elicitor-upd-4b-sft qwen3:32b; do
    file="$out/bench/${model//:/-}.json"
    [[ -f "$file" ]] && continue
    if [[ "$model" == qwen3:32b ]]; then models="${USER_OLLAMA_MODELS:-$OLLAMA_MODELS}"
    else models="$AGENT_OLLAMA_MODELS"; fi
    MODELS_DIR="$models" PARALLEL=8 SERVER_LOG="logs/ollama-${SLURM_JOB_ID:-local}-bench.log" \
        source cluster/serve_one.sh
    distill bench --model "$model" --base-url "$SERVER_URL" --out "$file" || status=1
    stop_server
done
since "bench"

python - "$out" <<'PY' > results_summary/m3/calibration.md
import json, pathlib, sys
print("| model | open | choice | yesno |\n|---|---:|---:|---:|")
for f in sorted(pathlib.Path(sys.argv[1]).glob("*/calibration/calibration.json")):
    rho = json.loads(f.read_text())["rho"]["llm"]
    cells = " | ".join(f"{rho[k]['rho']:.3f} [{rho[k]['ci_low']:.3f}, {rho[k]['ci_high']:.3f}]"
                       for k in ("open", "choice", "yesno"))
    print(f"| {f.parts[-3]} | {cells} |")
PY
cp "$out"/compare/{results.md,comparisons.csv,questions.csv,curves.csv} results_summary/m3/ || status=1
cp "$out"/compare/curve_*.png "$out"/{judge_test.md,test_table.md} results_summary/m3/ || status=1
cp "$out"/bench/*.json results_summary/m3/ || status=1
since "done"
exit "$status"
