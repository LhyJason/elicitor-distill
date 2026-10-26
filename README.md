# Elicitor-Distill

Small open models (Qwen3 0.6B, 1.7B and 4B with LoRA) trained to replace the 32B belief
updater and the 70B judge of [Elicitor](https://github.com/LhyJason/elicitor), an LLM agent
that chooses clarifying questions by expected information gain. The students are checked the
way the originals were: against the teacher's outputs, against the measured answer reliability,
and end to end in Elicitor's own experiment, with paired tests on 48 held-out simulated users.

Work in progress. The design and every decision are in [docs/SPEC.md](docs/SPEC.md).

## Development

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
ruff check . && ruff format --check . && pytest
distill make-personas --check    # the frozen training personas match the generator
distill collect experiments/collect_mock.toml --out results/collect_mock   # mock teachers
```

## Teacher data

`experiments/collect.toml` drives the collection on a GPU cluster (`cluster/collect.sbatch`,
Ollama servers as in Elicitor):

```bash
distill make-backstories experiments/collect.toml   # once, then frozen: data/backstories.jsonl
distill collect experiments/collect.toml --out data/raw/collect         # update records
distill judge-collect experiments/collect.toml --out data/raw/collect   # judge records
distill build-sft --raw data/raw/collect                                # data/sft/*.jsonl.gz
```

`data/travel/` is a verbatim copy of Elicitor's domain files at the pinned commit (checked
against `CHECKSUMS.sha256`); `data/personas_train.jsonl` holds this project's 160 training
personas. Elicitor's 48 evaluation personas are the test set and are never read before the
final evaluation.

## Students

`pip install -e ".[train]"` adds the training stack. One config per student in `configs/`;
`cluster/train.sbatch` runs the whole chain for one model on one GPU:

```bash
distill train configs/update_1.7b.toml --base-dir ~/hf   # LoRA SFT, best epoch on val
distill export elicitor-upd-1.7b-sft --base ~/hf/Qwen3-1.7B \
    --adapter models/upd-1.7b-sft/adapter --llama-cpp ~/llama.cpp --create   # GGUF + Ollama
distill eval-offline --model elicitor-upd-1.7b-sft --out results/m2/upd-1.7b-sft/update
distill eval-table results/m2                            # the validation table
distill bench --model elicitor-upd-1.7b-sft              # requests per second
```

`distill eval-offline --provider mock --model mock` runs the evaluation without a server.

## End to end (M3)

The hypotheses were frozen in the SPEC (§7.4) before any of this ran. `cluster/e2e.sbatch`
runs the whole evaluation in one job; Elicitor's run folders are read, never written:

```bash
distill build-test --m5 ~/elicitor/results/m5_full --judge ~/elicitor/results/m7_judge     --allow-eval                                          # the test sets: eval personas
distill m3-file experiments/m3_e2e.toml --model elicitor-upd-1.7b-sft --name e2e_upd-1.7b-sft     --calibration results/m3/upd-1.7b-sft/calibration/calibration.json     --out experiments/m3/e2e_upd-1.7b-sft.toml            # the model's own rho for eig
elicitor run experiments/m3/e2e_upd-1.7b-sft.toml --out results/m3/upd-1.7b-sft/run
distill compare results/m3/*/run --teacher ~/elicitor/results/m5_full --out results/m3/compare
distill judge-eval results/m3/judge-1.7b-sft/judge_test   # D5
```

## License

MIT, see [LICENSE](LICENSE). Model weights are not released (see the license note in the SPEC).
