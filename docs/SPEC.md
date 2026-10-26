# Elicitor-Distill — design spec (v1, 2026-10-25)

Design decisions are marked **Decision Qn** with the options considered and the choice; §0 indexes
them. The code is written from this file. A changed decision is recorded here before the code
changes. Milestone rows (§10) record what each milestone actually found.

---

## 0. Decision index

| Q | Topic | Choice | Status |
|---|---|---|---|
| Q1 | Project name | `elicitor-distill` (package `elicitor_distill`, CLI `distill`) | settled |
| Q2 | Relation to Elicitor | pip dependency pinned to one commit; domain data copied with checksums | settled |
| Q3 | What is distilled | the `update` role (main line) and the `judge` role (second line) | settled |
| Q4 | Student models | Qwen3-0.6B, 1.7B and 4B, thinking off | settled |
| Q5 | Distillation method | sequence-level: LoRA SFT on the teacher's JSON replies | settled |
| Q6 | Training data | 160 new training personas; Elicitor's 12 dev personas for validation; its 48 eval personas are the test set | settled |
| Q7 | Question sampling | 45 calibration-style questions per persona × 2 answer seeds | settled |
| Q8 | Training data in the repo | yes, gzipped JSONL | settled |
| Q9 | Serving the students | merged LoRA → GGUF → Ollama, same provider as the teacher | settled |
| Q10 | Quantisation | Q8_0 on the cluster; Q4_K_M for the 4B on a laptop; zero-shot baselines converted the same way | settled |
| Q11 | End-to-end arms | `sel=eig,upd=llm,phr=tpl` and `sel=random,upd=llm,phr=tpl`, K = 3 / 5 / 8, 5 replicates | settled |
| Q12 | Non-inferiority margin | −0.02 mass on truth and −0.03 MAP accuracy at K = 5 (one-sided 95 % bootstrap CI) | settled |
| Q13 | Hypotheses | D1–D6, frozen in this file before the end-to-end job | open until M3 |
| Q14 | Weights and licenses | code MIT; weights not released (Llama 3.3 license naming clause) | settled |

**Also fixed:** Python 3.12. No paid LLM APIs. English README with results. Sole author.

---

## 1. Goal and scope

**Pitch.** [Elicitor](https://github.com/LhyJason/elicitor) is an LLM agent that learns what a user
wants by asking a few well-chosen questions. Two of its roles run on large open models: the
*belief updater* (qwen3 32B scores how well a free-text answer fits each value of a preference
slot) and the *judge* (llama3.3 70B scores a recommended trip against the user's preferences).
This project trains students of 0.6B to 4B parameters to replace them and checks the students
the way Elicitor checked the originals: against the teacher, against the measured answer
reliability, and end to end on held-out simulated users with paired tests.

**Research question.** Can a LoRA student of at most 4B parameters, trained only on the teacher's
outputs, be *non-inferior* to the teacher on Elicitor's end-to-end metrics while raising
throughput by an order of magnitude? How much of the gap is closed by model size and how much by
fine-tuning?

**Deliverables.** Installable package with a CLI (`make-personas`, `collect`, `judge-collect`,
`build-sft`, `train`, `export`, `eval-offline`, `bench`, `compare`, `judge-eval`); the training
and validation sets committed to the repo; results tables and figures; a README; the Elicitor demo
running with a student on a laptop GPU.

**Non-goals.** Logit distillation, RL or preference optimisation, training the user simulator,
releasing weights, changing Elicitor, a second domain, human studies.

**Size limits.** The project stays small on purpose:
- Collection: one cluster job of at most 6 h on at most 4 GPUs. Training: one job of at most 1 h
  on one GPU per (role, size). End-to-end evaluation: one job of at most 4 h on 4 GPUs. At most
  40 GPU-hours in all. No limit is extended; §12 lists what is cut instead.
- Fewer than 2,500 lines of code outside tests.

**Elicitor as a dependency.** `pyproject.toml` pins Elicitor to commit
`41358e44bb27644d3cf4f4fc2188115a8d7e0c6a`. Its domain files (`data/travel/`: schema, the 60
personas and their backstories, the catalog, every prompt) are copied into this repo verbatim,
because Elicitor keeps them outside its package; `data/travel/CHECKSUMS.sha256` records their
hashes and a test checks the copy. Elicitor's full run (`m5_full`) has a fingerprint over these
bytes, so the copy must not drift.

> **Decision Q2 — relation to Elicitor.** A: pip dependency pinned to a commit, domain data
> copied. B: git submodule. C: copy the source. *Chosen A.* Every function this project needs
> (`build_roles`, `LLMClient`, `candidates`, `template_question`, `profile_lines`,
> `item_lines`, `analysis.stats`, `calibration_questions`) is importable. If an interface is
> missing, Elicitor gets a patch release and the pin moves; nothing of it is copied here.

## 2. What is distilled

> **Decision Q3.** The `update` role is the agent's main cost (one call per turn) and has an
> end-to-end metric. The `judge` role has Elicitor's M7 framework, where the true utility of
> every judged item is known, so a small judge can be checked against the truth, not only against
> the teacher. Not distilled: `simulate` (it is the environment, and the 70B model's diversity is
> part of the experiment design), `phrase` (no measurable effect in Elicitor's ablation),
> `imagine` (used only by `eig_mc`).

Facts about the two roles, from Elicitor's code at the pinned commit:

- `update` (agent group, qwen3 32B, temperature 0, 300 max tokens). Input: the slot
  description, its values with labels, the question as asked, the answer, and a note for
  categorical slots about "no preference". **No dialogue history.** Output: `evidence` (a
  quote, at most 300 characters) and an integer score 0–10 per value. The likelihood
  `((s + 0.5) / 10.5)` enters Bayes' rule in code. A student only has to reproduce the scores;
  the belief update does not change.
- `judge` (user group, llama3.3 70B, temperature 0, 150 max tokens). Input: the persona's
  preference profile and the item as an attribute list. Output: `reason` and a score 0–10.

Teachers are served exactly as in Elicitor (Ollama, JSON schema as `format`, `think` off for
qwen3, `num_ctx` 2048), so the training data is what the production prompt produces.

## 3. Students

> **Decision Q4.** `Qwen/Qwen3-0.6B`, `Qwen3-1.7B` and `Qwen3-4B`, the original hybrid-thinking
> models (the `-2507` instruct variants do not exist at 0.6B and 1.7B, and the three sizes must
> be one generation). Thinking is off in training (`enable_thinking=False` in the chat template)
> and in serving (`think: false`, as Elicitor already sends for the agent group). Three sizes
> give a size ladder; the zero-shot versions of the same three models, converted and served the
> same way, are the baselines, so LoRA is the only difference.

> **Decision Q5 — method.** A: sequence-level distillation (the teacher's reply is the label;
> plain SFT with the loss on the assistant turn only). B: logit distillation (needs the 32B
> teacher's forward pass in a framework other than Ollama; bf16 does not fit one 48 GB GPU).
> C: add preference optimisation against the ground truth. *Chosen A*: small data, a clear
> target, and the project's point is the verification, not the training trick. B and C are
> possible extensions (§10, M5).

## 4. Data

> **Decision Q6 — splits.** `train`: 160 personas drawn for this project (ids `t001`–`t160`,
> 40 per answer style, stratified over every slot's values like Elicitor's own, seed 20261025)
> with the rule that no training persona has exactly the ground truth of one of Elicitor's 60;
> a draw with a collision is replaced by the next seed, and the seed used is recorded in
> `data/personas_train.meta.json`. Backstories are written by the collection job (M1) with the
> simulator model, as Elicitor's `make-backstories` does, and then frozen. `dev`: Elicitor's 12
> development personas, for validation and model selection. `eval`: Elicitor's 48 evaluation
> personas, the **test set**: no command reads them without `--allow-eval`, which only the M3
> evaluation commands pass. A test asserts the guard.
>
> *M0 (10-25):* 160 personas drawn with seed 20261025, no redraw needed, no truth collision,
> 40 per style, every slot's values within one of each other.

> **Decision Q7 — question sampling.** Per persona the questions of Elicitor's `calibrate`: for
> every slot the open question, two choice questions and two yes/no questions, 45 in all, with
> template wording; each answered twice by the simulator with different seeds. Train: 160 × 90 =
> 14,400 (question, answer) pairs; validation: 12 × 90 = 1,080. This type mix is more balanced
> than the episodes' (which are mostly open questions), so the student sees enough choice and
> yes/no answers. LLM-phrased questions are not sampled: phrasing had no effect in Elicitor.
> Cost: about 29,000 teacher requests (simulate + update), fully parallel; the smoke run decides
> whether the job fits 6 h.

**Collection (`distill collect`).** Uses Elicitor's `LLMClient` and role registry with the
`[llm]` table of `m5_full.toml`. For every persona: questions → template text → `simulate`
(seed derived from the collection seed, persona, slot, question and answer index) → `update`
(seed 0, as in production). One record per pair: persona, split, slot, question, text, answer,
leaked slots, the teacher's scores and evidence, the teacher model, and **the prompt messages
the teacher received**, so the SFT prompt is byte-identical to the production prompt. Resumable
through Elicitor's reply cache; health checked with Elicitor's role statistics (any failed call
or a parse-failure rate ≥ 1 % fails the job). Leaking answers are kept: the teacher sees them in
production too.

**SFT format (`distill build-sft`).** One chat sample per line: the system and user messages as
rendered, and the assistant message = the teacher's reply **re-serialised through the role's
pydantic model** (compact JSON, keys in schema order), so the student learns one canonical
format. Records of failed teacher calls are dropped; "every value 5, no evidence" records are
kept (that is the behaviour on an uninformative answer). `meta` holds persona, slot, question
kind and style for the per-group tables.

**Judge data (`distill judge-collect`).** Units are (training persona, catalog item). Per
persona 25 items stratified over five utility bins (< 0.5, 0.5–0.6, 0.6–0.7, 0.7–0.8, ≥ 0.8;
a short bin is filled from its neighbours), so high-utility items are represented; 4,000 judge
calls (list presentation, user judge). Test set: the 256 units of Elicitor's `m7_judge` (user
judge, list), whose utility and teacher score already exist; no new calls.

> **Decision Q8.** The SFT sets (`data/sft/*.jsonl.gz`, synthetic, a few MB) are committed, so
> training is reproducible without a cluster. Raw records stay out of the repo.

Hygiene: persona ids `t…`; no account names, paths or hostnames in any data file.

## 5. Training

`transformers` + `peft` + `trl` (`SFTTrainer`) + `datasets` + `accelerate`; versions pinned in
M2. One config per (role, size) in `configs/`.

| Setting | Value |
|---|---|
| Base | `Qwen/Qwen3-{0.6B,1.7B,4B}`, bf16 |
| LoRA | r 16, alpha 32, dropout 0.05, on q/k/v/o/gate/up/down |
| Sequence length | 1,024 (update samples are 300–450 tokens; longer ones dropped and counted) |
| Loss | assistant turn only |
| Learning rate | 2e-4 (0.6B, 1.7B), 1e-4 (4B); cosine, 3 % warm-up |
| Batch | effective 32 |
| Epochs | 3 (update), 5 (judge); the offline metrics of §7.1 on the validation set after each epoch pick the checkpoint |
| Other | gradient checkpointing, seed 0 |

No hyper-parameter search, no full fine-tuning, one GPU. Weights are downloaded on a login
node and loaded offline on the compute node.

## 6. Export and serving

> **Decision Q9.** A: merge the LoRA, convert with llama.cpp's `convert_hf_to_gguf.py`, create
> an Ollama model, and use Elicitor's existing `ollama` provider unchanged. B: vLLM with LoRA
> through the `openai_compatible` provider. *Chosen A*: Elicitor's routed provider serves one
> provider kind for both model groups, and A makes the laptop demo a config change. The one
> cluster change: the agent group's Ollama servers read a writable model folder (the students),
> the user group keeps the shared read-only folder (llama3.3).

> **Decision Q10.** Q8_0 for every student and baseline on the cluster; the 4B also as Q4_K_M
> for the laptop. Zero-shot baselines go through the same HF → GGUF → Ollama path as the
> students, never through Ollama's library tags.

`distill export` runs the chain and a smoke test: 20 validation samples through Elicitor's
`update` role with zero parse failures. The Modelfile adds no system prompt or template of its
own. Students are served with the JSON schema as `format`, like the teacher; the validity of
unconstrained output is reported separately (§7.1).

## 7. Evaluation

### 7.1 Offline agreement with the teacher (`distill eval-offline`)

Validation set during M2; in M3 the **test set** = the de-duplicated `update` records of
Elicitor's eval personas in `m5_full` (rebuilt from each turn's question text, answer and
`parsed` scores). Per sample, student scores against teacher scores:

- **argmax agreement** (primary; ties count 1/j as in Elicitor's calibration);
- score MAE on the 0–10 scale; Spearman between the two score vectors;
- KL(teacher ‖ student) between the normalised likelihoods, the quantity that enters Bayes' rule;
- JSON validity with and without schema-constrained decoding;
- all of it by question kind and by answer style;
- ceiling: the teacher's agreement with itself at a different seed.

### 7.2 Answer reliability against the truth

`elicitor calibrate` with the student as the agent model, on the dev personas (where it is
meant to run): ρ for open, choice and yes/no questions with bootstrap CIs. Teacher: 0.99 / 0.97 /
0.85; rule updater: 0.79 / 0.83 / 0.64. The student's ρ table is what its `eig` selector uses.

### 7.3 End to end (`distill compare`, `distill bench`)

> **Decision Q11.** Each model gets a run folder with `m5_full.toml`'s settings, arms
> `sel=eig,upd=llm,phr=tpl` and `sel=random,upd=llm,phr=tpl` (the main system, and the selector
> that overtook it at K = 8), K = 3 / 5 / 8, 5 replicates, 48 eval personas, with the `[llm]`
> cache pointing at a **copy** of `m5_full`'s reply cache. Because Elicitor uses common random
> numbers and the simulator's cache key does not contain the agent model, a question the
> teacher asked gets the same answer for free; only `update` calls are new. With template
> wording these two selectors call no agent role but `update`, so the student is the only
> change. `eig_mc` is excluded: it calls `imagine`, which no student learned. The teacher's
> episodes are read from `m5_full`, not re-run.

`distill compare` pairs a student run with the teacher run by persona (replicate means first):
bootstrap CIs (10,000 resamples, seed 0) of the mean difference, paired Wilcoxon, Holm per metric
over every (model, arm, budget); metrics mass on truth, MAP accuracy, log loss, regret, hit@1,
violations; tables, markdown and per-turn curves with the teacher overlaid. It also reports how
often the student's `eig` asked the teacher's question sequence.

> **Decision Q12 — non-inferiority.** At K = 5 the teacher's `eig` + `llm` has 0.513 mass on
> truth, and the Holm-significant effects in Elicitor's run are of order 0.02–0.03. A student is
> non-inferior when the one-sided 95 % bootstrap lower bound of (student − teacher) exceeds
> −0.02 on mass on truth and −0.03 on MAP accuracy.

`distill bench`: 200 validation `update` calls with 16 workers on one GPU, requests per second,
p50 / p95 latency, GPU memory; the teacher under the same load.

### 7.4 Hypotheses (Decision Q13; frozen before the end-to-end job)

| | Hypothesis | Verdict rule |
|---|---|---|
| D1 | Zero-shot 4B is worse than the teacher: argmax agreement < 0.90 and mass on truth at K = 5 below −0.02 | both |
| D2 | SFT 4B is non-inferior (Q12) on the `eig` arm at K = 5 | CI bounds |
| D3 | SFT 1.7B is non-inferior likewise | CI bounds |
| D4 | After SFT, agreement is monotone 0.6B < 1.7B < 4B, and the 1.7B–4B gap shrinks to less than half of the zero-shot gap | point estimates |
| D5 | The SFT 1.7B judge reaches Spearman(score, utility) with a 95 % lower bound ≥ 0.80 on the 256 test units (teacher 0.871 [0.827, 0.906]) | bootstrap |
| D6 | SFT 1.7B Q8_0 serves `update` at ≥ 5 × the teacher's requests per second on one GPU | point estimate |

Predictions, written down to be compared with: D1 holds; D2 holds; D3 holds with the lower
bound close to the margin; D4 holds on both parts; D5 holds; D6 holds (8–15 ×). If D1 fails,
the finding is that this role never needed a 32B model, which is reported as such.

## 8. Repo layout and tooling

```text
src/elicitor_distill/
  data.py        domain copy and checksums        personas.py   make-personas
  splits.py      split guard                      collect.py    teacher data (M1)
  judge_collect.py (M1)   sft_data.py (M1)        train.py export.py eval_offline.py bench.py (M2)
  compare.py judge_eval.py (M3)                   cli.py
configs/       training configs (M2)       experiments/  collection, calibration, end-to-end files (M1, M3)
cluster/       Slurm scripts (M1)          data/         travel/ (copy), personas_train.jsonl, sft/
tests/         split guard, data, mock collection and SFT building, statistics on known data
docs/SPEC.md  docs/figures/  results_summary/
```

`src/` layout, pyproject + setuptools, ruff, pytest, GitHub Actions on Linux and Windows. CI
runs the tests, `make-personas --check`, and (from M1) collection and SFT building on Elicitor's
mock provider; training is only exercised up to the trainer's construction (`--dry-run`), since
CI downloads no weights.

## 9. Infrastructure

- Cluster: Slurm, A40 48 GB GPUs; one Ollama server per GPU as in Elicitor (two for the agent
  group, two for the user group). Site settings in an untracked `cluster/local.env`.
- Jobs: `collect.sbatch` (4 GPUs, 6 h) → `train.sbatch` × 6 (1 GPU, 1 h each) → export (CPU) →
  `e2e.sbatch` (4 GPUs, 4 h: calibrate, the runs, bench). Each ends with a health check.
- Elicitor's run folders (`m5_full`, `m5_smoke`, `m7_judge`) are read, never written; the reply
  cache is copied before use.
- Laptop (6 GB GPU): the 1.7B Q8_0 or 4B Q4_K_M student runs under Ollama for
  `elicitor demo --updater llm`.

## 10. Milestones

| M | Content | Done when |
|---|---|---|
| M0 | repo, pyproject with the Elicitor pin, CI, domain copy with checksums, `make-personas`, split guard, this spec | CI green; `personas_train.jsonl` frozen; the guard has a test. **Done 10-25:** 13 tests; 160 personas, seed 20261025, no redraw |
| M1 | `collect`, `judge-collect`, `build-sft` (runnable on the mock provider), cluster scripts; the collection job | mock collection in CI; the job finishes within 6 h with 0 failed calls and parse failures < 1 % per role; `data/sft/*.jsonl.gz` frozen with line counts and hashes recorded here |
| M2 | `train`, `export`, `eval-offline`, `bench`; three `update` students and three zero-shot baselines as Ollama models; the 1.7B judge student | six models answer 20 validation samples with 0 parse failures; the validation table; SFT 4B agreement above zero-shot 4B (else the format or template is debugged before M3) |
| M3 | D1–D6 frozen → `calibrate` × 6 → end-to-end runs × 6 → `compare` → `bench` → test-set offline table (the first read of eval data) → judge test | job within 4 h; every run complete (10 cells) and healthy; `results_summary/` committed; D1–D6 each marked as found |
| M4 | README with results, laptop demo transcript, fresh-clone test, repo made public | README numbers match `results_summary/`; hygiene check clean |
| M5 | optional, at most one: pairwise judge task; logit distillation; distilling `imagine` so `eig_mc` can run on a student | one job of at most 2 h, hypotheses written here first |

## 11. Hygiene and licenses

- Sole author. No account names, usernames or home paths in the repo.
- The eval split is never read before M3 (`--allow-eval`).
- Code: MIT. Student bases: Qwen3, Apache-2.0. The training inputs (simulated answers) and the
  judge labels are outputs of Llama 3.3, whose community license requires a model trained on
  its outputs and *distributed* to carry "Llama" at the start of its name and a "Built with
  Llama" notice. **Decision Q14:** version 1 distributes no weights, only code, data and results;
  the README says so. If weights are released later, they are named and labelled as the license
  requires.

## 12. Risks and what is cut

| Risk | Response |
|---|---|
| Collection job over 6 h | in order: one answer seed instead of two; 100 training personas; the judge collection moved to its own job |
| Zero-shot students produce invalid JSON | schema-constrained decoding keeps the pipeline running; unconstrained validity is a reported metric |
| A student's ρ ranks question types differently, so `eig` asks other questions | a finding, not a bug; the question-sequence agreement is reported |
| GGUF conversion of a merged Qwen3 model fails | the export chain is proven on the 0.6B first; fallback: Ollama importing the safetensors folder directly |
| 4B training over 1 h | smaller per-device batch, 2 epochs; else the 4B stays zero-shot only, noted here |
| Few de-duplicated eval `update` records (< 2,000) | the test-set offline table is reported with its wide CIs; the end-to-end layer carries the conclusion |
| Elicitor lacks an interface | patch release of Elicitor, pin moved; no code copied |
