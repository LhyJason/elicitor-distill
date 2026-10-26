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

## License

MIT, see [LICENSE](LICENSE). Model weights are not released (see the license note in the SPEC).
