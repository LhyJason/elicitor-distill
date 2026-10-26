#!/usr/bin/env bash
#SBATCH --job-name=distill-e2e
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.log
# M3 (SPEC §7, §9, §10): the end-to-end job, at most 4 h on 4 GPUs, after D1-D6 were frozen.
# GPU 0 serves the M2 models from the agent group's folder (AGENT_GPUS=1), one at a time: the
# cluster's GPUs are in exclusive-process mode and Ollama runs each model in a process of its
# own, so a second model cannot load beside the first (found 10-26: "CUDA-capable device(s) is/are
# busy or unavailable"); Ollama swaps them on request. GPUs 1-3 serve the simulator llama3.3
# (USER_GPUS=3), since only the simulator's answers to questions the teacher never asked are new. Elicitor's run folders (ELICITOR_RESULTS, default ~/elicitor/results:
# m5_full, m5_smoke_calibration, m7_judge) are read, never written; their reply caches are copied.
# Steps:
#   1. `distill build-test`: the update and judge test sets (the first read of eval data);
#   2. per model, one after another (the SFT students first): `elicitor calibrate` on the dev
#      personas (m5_smoke's file, the model as agent) → `distill m3-file` with the measured rho →
#      `elicitor run` of the two arms (m5_full's settings) → `elicitor health`, and beside the
#      run the model's test-set offline tables;
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
UPDATE=(upd-1.7b-sft upd-4b-sft upd-0.6b-sft qwen3-4b-zs qwen3-1.7b-zs qwen3-0.6b-zs)
JUDGE=(judge-1.7b-sft qwen3-4b-zs qwen3-1.7b-zs qwen3-0.6b-zs)  # the SFT judge first (D5)
start=$(date +%s)
since() { echo "$1: $(( $(date +%s) - start )) s since the start"; }

if [[ ! -f "$out/test/judge_test.jsonl.gz" ]]; then
    distill build-test --m5 "$src/m5_full" --judge "$src/m7_judge" --out "$out/test" --allow-eval
fi
sha256sum "$out"/test/*.jsonl.gz

export CONFIG=experiments/m3_e2e.toml AGENT_GPUS=${AGENT_GPUS:-1} USER_GPUS=${USER_GPUS:-3}
export AGENT_MODEL_OVERRIDE=elicitor-${UPDATE[0]} REQUIRE_GPU=1
source cluster/serve_models.sh
trap stop_models EXIT
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

# One model at a time on the agent GPU; its run and its offline tables share the loaded model.
# Each step runs in the background so that `set -e` holds inside it (it is off in `a || b`).
status=0
log="logs/e2e-${SLURM_JOB_ID:-local}"
for name in "${UPDATE[@]}" judge-1.7b-sft; do
    pids=()
    if [[ "$name" != judge-* ]]; then
        chain "$name" > "$log-$name.log" 2>&1 & pids+=($!)
        offline "$name" update > "$log-$name-offline.log" 2>&1 & pids+=($!)
    fi
    if [[ " ${JUDGE[*]} " == *" $name "* ]]; then
        offline "$name" judge > "$log-$name-judge.log" 2>&1 & pids+=($!)
    fi
    for pid in "${pids[@]}"; do wait "$pid" || status=1; done
    since "$name (exit status so far $status)"
    tail -n 25 "$log-$name.log" 2>/dev/null || true
    grep -ahE "health|^\{" "$log-$name"-*.log || true
done

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
