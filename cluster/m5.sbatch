#!/usr/bin/env bash
#SBATCH --job-name=distill-m5
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.log
# M5 (SPEC §7.5, Q16): `eig_mc` on a student, at most 1.5 h on 2 GPUs, after E1-E5 were frozen
# and `agent-1.7b-sft` was trained (cluster/train.sbatch configs/agent_1.7b.toml). GPU 0 serves
# the 1.7B models one at a time (exclusive-process GPUs, as in M3), GPU 1 the simulator, which is
# only called for questions no teacher run asked. Elicitor's run folders (ELICITOR_RESULTS,
# default ~/elicitor/results: m5_full, m5_smoke_calibration) and results/m3 are read, never
# written; their reply caches are copied.
# Steps:
#   1. R1, agent-1.7b-sft: `elicitor calibrate` on the dev personas → `distill m3-file` with its
#      rho → `elicitor run` of eig_mc and eig at K = 5 → `elicitor health`; beside the run its
#      update test table (E4);
#   2. the offline imagine table: teacher, agent-1.7b-sft, zero-shot 1.7B and upd-1.7b-sft
#      imagine, upd-1.7b-sft reads (E5);
#   3. R2, upd-1.7b-sft with its M3 calibration and M3's reply cache, under a deadline that
#      leaves time for step 4 (a run cut short leaves E3 untested);
#   4. `distill m5-report`, the tables to results_summary/m5/.
# A job that stops early may be resubmitted within the budget (SPEC §7.5); every step resumes
# or is skipped once its output exists. Submit from the repo root:
#   cluster/submit.sh cluster/m5.sbatch
set -eo pipefail
source cluster/job_setup.sh
src="${ELICITOR_RESULTS:-$HOME/elicitor/results}"
m3=results/m3 out=results/m5
AGENT=agent-1.7b-sft UPDATE=upd-1.7b-sft ZS=qwen3-1.7b-zs
for f in "$src/m5_full/run_state.json" "$src/m5_smoke_calibration/llm_cache.sqlite" \
         "$m3/$UPDATE/calibration/calibration.json" "$m3/$UPDATE/run/llm_cache.sqlite" \
         "$m3/$UPDATE/update_test/summary.json" "$m3/test/update_test.jsonl.gz" \
         data/sft/imagine_val.jsonl.gz; do
    if [[ ! -f "$f" ]]; then echo "missing $f (set ELICITOR_RESULTS)" >&2; exit 2; fi
done
if [[ -z "$AGENT_OLLAMA_MODELS" ]]; then echo "set AGENT_OLLAMA_MODELS" >&2; exit 2; fi
mkdir -p logs "$out" experiments/m5 results_summary/m5
start=$(date +%s)
deadline=$((start + 82 * 60))  # R2 stops here; the reports take a few minutes
since() { echo "$1: $(( $(date +%s) - start )) s since the start"; }
log="logs/m5-${SLURM_JOB_ID:-local}"

export CONFIG=experiments/m5_e2e.toml AGENT_GPUS=1 USER_GPUS=1
export AGENT_MODEL_OVERRIDE=elicitor-$AGENT REQUIRE_GPU=1
source cluster/serve_models.sh
trap stop_models EXIT
agent_url=${agent_urls[0]}
since "servers"

run() {  # $1 model, $2 calibration.json, $3 reply cache to start from, $4 time limit (0: none)
    local run="$out/$1/run"
    mkdir -p "$run"
    distill m3-file experiments/m5_e2e.toml --model "elicitor-$1" --name "m5_$1" \
        --calibration "$2" --out "experiments/m5/e2e_$1.toml"
    [[ -f "$run/llm_cache.sqlite" ]] || cp "$3" "$run/"
    timeout "$4" elicitor run "experiments/m5/e2e_$1.toml" --out "$run" --workers 16
    elicitor health "$run"
}

imagined() {  # $1 imaginer (a model, or teacher: the stored replies); upd-1.7b-sft reads
    local dir="$out/imagine/$1" who=$1
    [[ -f "$dir/summary.json" ]] && return 0
    [[ "$1" == teacher ]] || who="elicitor-$1"
    distill imagine-eval --imaginer "$who" --reader "elicitor-$UPDATE" --base-url "$agent_url" \
        --workers 8 --out "$dir"
}

status=0
cal="$out/$AGENT/calibration"
mkdir -p "$cal"
[[ -f "$cal/llm_cache.sqlite" ]] || cp "$src/m5_smoke_calibration/llm_cache.sqlite" "$cal/"
distill m3-file experiments/m3_calibrate.toml --model "elicitor-$AGENT" \
    --name "calibrate_$AGENT" --out "experiments/m5/calibrate_$AGENT.toml"
if [[ ! -f "$cal/calibration.json" ]]; then
    elicitor calibrate "experiments/m5/calibrate_$AGENT.toml" --out "$cal" --workers 8
fi
since "R1 calibration"
# Each step runs in the background so that `set -e` holds inside it (it is off in `a || b`).
run "$AGENT" "$cal/calibration.json" "$src/m5_full/llm_cache.sqlite" 0 > "$log-r1.log" 2>&1 &
r1=$!
test_dir="$out/$AGENT/update_test"
if [[ ! -f "$test_dir/summary.json" ]]; then
    distill eval-offline --model "elicitor-$AGENT" --role update --allow-eval \
        --data "$m3/test/update_test.jsonl.gz" --base-url "$agent_url" --workers 8 \
        --no-unconstrained --out "$test_dir" > "$log-test.log" 2>&1 || status=1
fi
wait "$r1" || status=1
since "R1 (exit status so far $status)"
tail -n 25 "$log-r1.log"

# One imaginer at a time, then the reader: one model swap on GPU 0 per step.
for who in teacher "$AGENT" "$ZS" "$UPDATE"; do imagined "$who" || status=1; done
since "imagine table"

left=$(( deadline - $(date +%s) ))
if (( left > 300 )); then
    run "$UPDATE" "$m3/$UPDATE/calibration/calibration.json" "$m3/$UPDATE/run/llm_cache.sqlite" \
        "$left" > "$log-r2.log" 2>&1 || echo "R2 did not finish (exit $?): E3 untested"
else
    echo "no time left for R2: E3 untested"
fi
since "R2"
tail -n 25 "$log-r2.log" 2>/dev/null || true

distill m5-report --teacher "$src/m5_full" --r1 "$out/$AGENT/run" --r2 "$out/$UPDATE/run" \
    --imagine-student "$out/imagine/$AGENT" --imagine-teacher "$out/imagine/teacher" \
    --agent-test "$test_dir/summary.json" --update-test "$m3/$UPDATE/update_test/summary.json" \
    --out "$out/report" || status=1

python - "$out/imagine" <<'PY' > results_summary/m5/imagine_table.md
import json, pathlib, sys
print("| imaginer | n | failed | empty | mass on value | open | choice | yesno | leaking "
      "| pairs differ | words |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for name in ("teacher", "agent-1.7b-sft", "qwen3-1.7b-zs", "upd-1.7b-sft"):
    f = pathlib.Path(sys.argv[1]) / name / "summary.json"
    if not f.exists():
        continue
    t = json.loads(f.read_text())["tables"]
    o, k = t["overall"], t["kind"]
    print(f"| {name} | {o['n']} | {o['failed']} | {o['empty']} | {o['mass_on_value']:.4f} | "
          + " | ".join(f"{k[x]['mass_on_value']:.4f}" for x in ("open", "choice", "yesno"))
          + f" | {o['leaking']:.4f} | {o['pairs_differ']:.4f} | {o['words']} |")
PY
cp "$out"/report/{verdicts.md,contrasts.csv} results_summary/m5/ || status=1
cp "$out"/report/compare/{results.md,comparisons.csv,questions.csv,curves.csv} \
    "$out"/report/compare/curve_*.png results_summary/m5/ || status=1
cp "$cal/calibration.json" results_summary/m5/calibration_agent.json || status=1
cat results_summary/m5/verdicts.md results_summary/m5/imagine_table.md || true
since "done"
exit "$status"
