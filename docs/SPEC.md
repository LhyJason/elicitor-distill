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
| Q7 | Question sampling | 45 calibration-style questions per persona × 1 answer seed (cut from 2 after the smoke run) | settled |
| Q8 | Training data in the repo | yes, gzipped JSONL | settled |
| Q9 | Serving the students | merged LoRA → GGUF → Ollama, same provider as the teacher | settled |
| Q10 | Quantisation | Q8_0 on the cluster; Q4_K_M for the 4B on a laptop; zero-shot baselines converted the same way | settled |
| Q11 | End-to-end arms | `sel=eig,upd=llm,phr=tpl` and `sel=random,upd=llm,phr=tpl`, K = 3 / 5 / 8, 5 replicates | settled |
| Q12 | Non-inferiority margin | −0.02 mass on truth and −0.03 MAP accuracy at K = 5 (one-sided 95 % bootstrap CI) | settled |
| Q13 | Hypotheses | D1–D6, frozen 10-26 before the end-to-end job (§7.4) | settled |
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
>
> *M1 (10-25), first cut of §12:* **one answer seed**. The smoke run (8 personas, 2 / 2 GPUs)
> drew 720 pairs in 1,385 s: the simulator (70B, 54 s per attempt under load) is the bottleneck,
> the updater (3.7 s) is not. Two seeds would take about 10 h with the judge collection, and
> still about 6.8 h with the split moved to 1 / 3. One seed with the 1 / 3 split: train
> 160 × 45 = 7,200 pairs, validation 12 × 45 = 540, about 4.1 h. The second seed would have
> added about 87 % new pairs (10–13 % of the smoke pairs repeated an answer).

**Collection file.** `experiments/collect.toml` drives every collection command: the splits
(`train` and `dev`; the file loader rejects `eval`, and the split guard stands behind it), the
collection seed 20261026, one answer seed (Q7), the judge bins, and an `[llm]` table identical to
`m5_full.toml`'s. `collect_smoke.toml` is the same file cut to the first 4 personas of each
split; `collect_mock.toml` runs the dev personas on Elicitor's mock provider in CI.

**Backstories (`distill make-backstories`).** The simulator's prompt includes the persona's
backstory, so training personas get one before any answer is drawn: written once by the
simulator model with Elicitor's own procedure (`elicitor.backstories`: style and a seeded age,
never the ground truth; a backstory that names a slot value is redrawn), for all 160 training
personas, into `data/backstories.jsonl`, where Elicitor's persona loader merges them into
`personas_train.jsonl`. The file is then frozen (the command refuses to overwrite it without
`--force`), and `collect` refuses to run while a persona lacks one.

**Collection (`distill collect`).** Uses Elicitor's `LLMClient` and role registry with the
`[llm]` table of `m5_full.toml`. For every persona: questions → template text → `simulate`
(through Elicitor's `LLMSimulator`; seed derived from the collection seed, persona, question and
answer index) → `update` (seed 0, as in production). The `update` context is built as Elicitor's
LLM updater builds it; a test checks that both produce the same cache key. One record per pair:
persona, split, style, slot, question, text, answer, leaked slots, a status (`ok`, `failed`, or
`empty_answer` when the simulator's reply never validated: the updater makes no call on an empty
answer), the teacher's scores and evidence, the teacher model, and **the prompt messages the
teacher received**, so the SFT prompt is byte-identical to the production prompt. Records go to
`data/raw/<file name>/update_<split>.jsonl` with `update_stats.json` (counts, LLM statistics,
health problems, the collection file's text). Resumable through Elicitor's reply cache in the
same folder; health checked with Elicitor's role statistics (any failed call or transport error,
or a parse-failure rate ≥ 1 %, fails the step). Leaking answers are kept: the teacher sees them
in production too.

**SFT format (`distill build-sft`).** One chat sample per line: the system and user messages as
rendered, and the assistant message = the teacher's reply **re-serialised through the role's
pydantic model** (compact JSON, keys in schema order), so the student learns one canonical
format. Records of failed teacher calls and empty answers are dropped; "every value 5, no
evidence" records are kept (that is the behaviour on an uninformative answer). A prompt that
occurs twice in a split (both answer seeds drew the same answer) is kept once: the teacher's
reply to it is one cached reply, and a duplicate would only weight it. Training personas give
the `train` set, dev personas the `val` set. `meta` holds persona, style, slot, question kind and
answer index (update) or item, utility and bin (judge) for the per-group tables. The gzip files
are byte-reproducible (empty gzip header); `data/sft/MANIFEST.json` records rows, drops and the
SHA-256 of each file, compressed and not.

**Judge data (`distill judge-collect`).** Units are (persona, catalog item). Per persona 25
items stratified over five utility bins (< 0.5, 0.5–0.6, 0.6–0.7, 0.7–0.8, ≥ 0.8), drawn with a
generator seeded by persona, so high-utility items are represented. Of the 172 training and dev
personas 121 have fewer than five items at ≥ 0.8; a short bin is filled from its neighbours, the
nearer first and the higher of two equally near ones first. Context and prompt are those of
Elicitor's main judge (user group, list presentation: profile lines and item lines). 4,000 judge
calls for the training personas and 300 for the dev personas (the `val` set). Test set: the 256
units of Elicitor's `m7_judge` (user judge, list), whose utility and teacher score already
exist; no new calls.

> **Decision Q8.** The SFT sets (`data/sft/*.jsonl.gz`, synthetic, a few MB) are committed, so
> training is reproducible without a cluster. Raw records stay out of the repo.

Hygiene: persona ids `t…`; no account names, paths or hostnames in any data file.

## 5. Training

`transformers` (`Trainer`) + `peft` + `accelerate`; versions pinned in M2. One config per
(role, size) in `configs/`: `update_{0.6b,1.7b,4b}.toml` and `judge_1.7b.toml`.

*M2 (10-26):* `trl` and `datasets` are not used. `distill train` tokenises the samples itself:
the prompt is rendered with the base model's chat template and `enable_thinking=False`, and every
rendering must equal the prompt the exported model is served (§6), else training stops; the
completion is the rest of the full conversation's rendering (the teacher's JSON and the
end-of-turn token). Prompt and completion are tokenised separately, as at serving time, and the
loss is on the completion. A checkpoint is saved per epoch; each is scored on the validation set
by greedy, unconstrained generation (argmax agreement for `update`, score MAE for `judge`; ties
go to the earlier epoch), and the best adapter is kept. Versions (`train` extra): torch 2.14.1 (the
default CUDA 13 build; the cluster's A40 driver is 595.71), transformers 5.20.2, peft 0.21.2,
accelerate 1.15.0, safetensors 0.8.0, sentencepiece 0.2.1 (imported by llama.cpp's Qwen vocabulary
code); llama.cpp at `d854d98ba` (tag b11889) with its `gguf-py` 0.19.0. The tiny-model test
passes with them on a cluster GPU node (10-26). The data loader runs in the training process:
worker processes forked from it can deadlock.

| Setting | Value |
|---|---|
| Base | `Qwen/Qwen3-{0.6B,1.7B,4B}`, bf16 |
| LoRA | r 16, alpha 32, dropout 0.05, on q/k/v/o/gate/up/down |
| Sequence length | 1,024 (update samples are 300–450 tokens; longer ones dropped and counted) |
| Loss | assistant turn only |
| Learning rate | 2e-4 (0.6B, 1.7B), 1e-4 (4B); cosine, 3 % warm-up |
| Batch | effective 32 |
| Epochs | 3 (update; 2 for the 4B, §12), 5 (judge); the offline metrics of §7.1 on the validation set after each epoch pick the checkpoint |
| Other | seed 0; gradient checkpointing on the 0.6B only (the larger models' activations fit in the A40's 48 GB, and without it a step is about a third cheaper); validation generation in batches of 128 for the larger models |

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

`distill export` runs the chain; the job then runs the smoke test: 20 validation samples
through Elicitor's role (one worker, no cache) with zero parse failures. The Modelfile adds no
system prompt. Students are served with the JSON schema as `format`, like the teacher; the
validity of unconstrained output is reported separately (§7.1).

*M2 (10-26), the chat template:* the Modelfile **sets the template explicitly** instead of
leaving Ollama to pick one from the GGUF: a system and a user message rendered exactly as
Qwen3's own template renders them with `enable_thinking=False` (an empty think block opens the
reply), for every student and baseline. The teacher's Ollama template differs (it appends
`/no_think` to the user turn when `think` is off), but a student's prompt only has to match its
own training (§5), and the baselines get Qwen3's native format. Greedy decoding is kept free of
a repeat penalty (`repeat_penalty 1`). `distill check-template` closes the loop: the job's
server logs every rendered prompt (`OLLAMA_DEBUG=2`), and the smoke test's 20 must equal the
training rendering. Ollama 0.24 logs the prompt only at trace level (`msg="completion request"`;
the debug line has its length), along with a line per token encoded or decoded, about 18 MB a
request (found 10-26: 4 GB in the first minutes of a full evaluation). Those lines are dropped
before the log is written, and only the smoke test and the check run on the tracing server; the
evaluation restarts it quietly.

## 7. Evaluation

### 7.1 Offline agreement with the teacher (`distill eval-offline`)

Validation set during M2; in M3 the **test set** = the de-duplicated `update` records of
Elicitor's eval personas in `m5_full` (rebuilt from each turn's question text, answer and
`parsed` scores). Per sample, student scores against teacher scores:

- **argmax agreement** (primary): |S ∩ T| / |S ∪ T| for the student's and the teacher's sets
  of top values, so 1/j when the student ties j values one of which is the teacher's top (as in
  Elicitor's calibration) and 1 for the same tie;
- score MAE on the 0–10 scale; Spearman between the two score vectors;
- KL(teacher ‖ student) between the normalised likelihoods, the quantity that enters Bayes' rule;
- JSON validity with and without schema-constrained decoding;
- all of it by question kind and by answer style;
- ceiling: the teacher's agreement with itself at a different seed.

The model gets exactly the stored messages through Elicitor's client and role settings (a
pass-through prompt; seed 0; three attempts; the schema as `format`). A failed call counts as
equal scores, which is what Elicitor's updater does with one (the belief does not move). Judge
samples: absolute score error, exact and within-one agreement, and over the samples Spearman of
the student's score with the teacher's and with the true utility (by utility bin and style).

*M2 validation table (10-26).* `update_val` (516) and `judge_val` (300), seed 0, Q8_0 through
Ollama 0.24 with the schema as `format`, no failed call anywhere; "unconstrained" is the share
of samples that validate without the schema (three attempts). Ceiling rows are the teachers at
request seed 1 against their stored seed-0 replies (the 70B judge served with 4 parallel slots,
§9).

| `update` | agreement | MAE | Spearman | KL | unconstrained |
|---|---:|---:|---:|---:|---:|
| zero-shot 0.6B | 0.482 | 3.551 | 0.457 | 0.534 | 0 |
| zero-shot 1.7B | 0.832 | 1.877 | 0.728 | 0.346 | 0 |
| zero-shot 4B | 0.923 | 1.486 | 0.808 | 0.262 | 0 |
| SFT 0.6B | 0.967 | 0.464 | 0.945 | 0.090 | 0.996 |
| SFT 1.7B | 0.972 | 0.361 | 0.952 | 0.075 | 1 |
| SFT 4B | 0.974 | 0.326 | 0.959 | 0.060 | 0.988 |
| ceiling, qwen3:32b | 0.998 | 0.045 | 0.997 | 0.006 | – |

| `judge` | MAE | exact | within 1 | Spearman teacher | Spearman utility | unconstrained |
|---|---:|---:|---:|---:|---:|---:|
| zero-shot 0.6B | 5.943 | 0 | 0.017 | 0.309 | 0.266 | 0.753 |
| zero-shot 1.7B | 2.643 | 0.067 | 0.257 | 0.477 | 0.401 | 0.730 |
| zero-shot 4B | 1.760 | 0.117 | 0.410 | 0.791 | 0.699 | 0.950 |
| SFT 1.7B | 0.387 | 0.690 | 0.923 | 0.915 | 0.729 | 1 |
| ceiling, llama3.3 70B | 0.093 | 0.930 | 0.977 | 0.979 | 0.764 | – |

The teacher's own Spearman with utility on these 300 is 0.756 (0.871 on the 256 test units of
D5). Read as a preview only, since §7.4 is judged on the test set in M3: SFT agreement is
monotone in size, with the 1.7B–4B gap (0.002) far below the zero-shot one (0.091); zero-shot 4B
is already above D1's 0.90 on validation. No zero-shot base produced a single valid `update`
reply without the schema (1,548 attempts each for the 4B, about 55 tokens a reply, every one
rejected). Code fences are already tolerated by Elicitor's parser, so the replies break the
strict model itself (exactly the slot's value names as keys, no extra keys); the raw texts are
not kept.

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

**Frozen 10-26**, after M2 and before any test data is read. The hypotheses, thresholds and
predictions above are those of v1, unchanged: the validation preview (§7.1) already puts zero-shot
4B at 0.923 agreement, above D1's 0.90, and moving the threshold after seeing that would fit the
hypothesis to the data. What each verdict reads:

- *Agreement* (D1, D4) is the argmax agreement of the test-set offline table (§7.1): the
  de-duplicated `update` records of the 48 eval personas in `m5_full`, every arm with the LLM
  updater, served at seed 0 with the schema as `format`.
- *Mass on truth at K = 5* (D1) is the paired mean difference model − teacher on the `eig` arm
  (`sel=eig,upd=llm,phr=tpl`), per persona over the 5 replicates; D1's second part holds when
  this point estimate is below −0.02.
- *Non-inferiority* (D2, D3) is Q12 on the `eig` arm at K = 5: the 5th percentile of 10,000
  bootstrap means (seed 0) of the per-persona differences is above −0.02 for mass on truth and
  above −0.03 for MAP accuracy; both must hold. The other arm, budgets and metrics are reported,
  with Holm per metric over every (model, arm, budget), but decide nothing.
- *D4* holds when the SFT agreements are strictly increasing in size and
  agreement(SFT 4B) − agreement(SFT 1.7B) < ½ (agreement(zero-shot 4B) − agreement(zero-shot 1.7B)).
- *D5*: Spearman of the SFT 1.7B judge's score with the true utility, pooled over the 256 units of
  `m7_judge` (user judge, list presentation), and its CI by resampling personas (10,000 resamples,
  seed 0), the method that gave the teacher's [0.827, 0.906]; it holds when the 2.5th
  percentile is ≥ 0.80.
- *D6*: `distill bench` on one A40 each, 200 validation samples, 16 workers, SFT 1.7B Q8_0
  against qwen3:32b; it holds when the student's requests per second are ≥ 5 × the teacher's.

*M3 job (code 10-26).* `cluster/e2e.sbatch`: GPU 0 serves the M2 models one at a time, the
SFT students first, GPUs 1–3 the simulator. The cluster's GPUs are in exclusive-process mode and
Ollama runs every model in a process of its own, so a second model cannot load beside the first
(found 10-26 by the first submission, which stopped after 46 s). In `m5_full` the simulator made
6,000 new requests and the updater 4,553 (the rest were cache hits), so a student run pays its
own `update` calls and only the answers to questions the teacher's runs never asked. Per model:
`elicitor calibrate` with the m5_smoke file and a copy of its calibration cache (only `update` is
new), `distill m3-file` with the rounded rho, `elicitor run` with a copy of `m5_full`'s cache,
`elicitor health`, and beside the run the model's test-set offline tables, without the
unconstrained pass (reported on validation, §7.1). Then `compare`, `judge-eval`, and `bench` on
GPU 0 alone, one model at a time.

*M3 verdicts (10-27).* Job 64845112; the tables are in `results_summary/m3/`. Test sets:
`update_test` 4,086 samples, `judge_test` 256. Differences are model − teacher on the `eig` arm at
K = 5, with the 95 % CI and the one-sided lower bound.

| | Found | Numbers |
|---|---|---|
| D1 | **does not hold** as frozen (one part of two) | agreement 0.956, not < 0.90; mass on truth −0.124 [−0.135, −0.113], below −0.02 |
| D2 | **holds** | mass on truth +0.0061 (lower −0.0005); MAP accuracy −0.0005 (lower −0.0037) |
| D3 | **holds**, far from the margin | mass on truth +0.0095 (lower +0.0054, Holm p 0.0096: above the teacher); MAP accuracy −0.0028 (lower −0.0069) |
| D4 | **holds** | SFT 0.976 < 0.980 < 0.987; 1.7B–4B gap 0.007 against ½ × 0.035 for zero-shot |
| D5 | **does not hold**, narrowly | Spearman with utility 0.832 [0.790, 0.868]; lower bound 0.790 < 0.80 (teacher 0.871 [0.827, 0.906], zero-shot 4B 0.758) |
| D6 | **holds** (9.8 ×, inside the predicted 8–15 ×) | SFT 1.7B 8.61 requests/s against qwen3:32b 0.875; p50 1.73 s against 16.5 s; 3,977 against 23,611 MiB |

Against the predictions: D1 and D5 missed, and D3 was not close to the margin. D1's fallback
("this role never needed a 32B model") does not apply: the end-to-end half holds at six times the
margin. Zero-shot 4B ranks the right value first about as often as the teacher (MAP accuracy
−0.001) but spreads its scores (test KL 0.246 against 0.056 for SFT 4B), so the posterior
concentrates more slowly. Argmax agreement does not see that, and KL is the quantity that enters
Bayes' rule (§7.1). Every SFT student, the 0.6B included (−0.0017, lower −0.0065), is non-inferior
on both metrics; every zero-shot base loses 0.10–0.23 mass on truth at K = 5. The `eig` arm asked
the teacher's exact question sequence in every episode for every model but zero-shot 0.6B, whose
measured ρ (0.37 / 0.51 / 0.52) is far from the others (`calibration.md`). Outside that
model's `eig` arm the questions are the teacher's, so the gaps come from the updates alone. The SFT judge has 0.905 Spearman with the teacher's
scores and is 0.04 below the teacher on utility; the bar it misses is the frozen 0.80 lower bound.

## 8. Repo layout and tooling

```text
src/elicitor_distill/
  data.py        domain copy and checksums        personas.py   make-personas
  splits.py      split guard                      collect.py    teacher data (M1)
  judge_collect.py (M1)   sft_data.py (M1)        train.py export.py eval_offline.py bench.py (M2)
  m3.py compare.py judge_eval.py (M3)             cli.py
configs/       training configs (M2)       experiments/  collection, calibration, end-to-end files (M1, M3)
cluster/       Slurm scripts (M1)          data/         travel/ (copy), personas_train.jsonl, sft/
tests/         split guard, data, mock collection and SFT building, statistics on known data
docs/SPEC.md  docs/figures/  results_summary/
```

`src/` layout, pyproject + setuptools, ruff, pytest. The checks run locally before each commit:
ruff, the tests, `make-personas --check`, and (from M1) collection and SFT building on Elicitor's
mock provider (the CLI on the dev personas; the whole chain, backstories included, in the tests
on a copy of `data/`); training is only exercised up to the trainer's construction (`--dry-run`),
since the checks download no weights. The GitHub Actions workflow (Linux and Windows) ran these
until 10-26 and was then removed.

## 9. Infrastructure

- Cluster: Slurm, A40 48 GB GPUs; one Ollama server per GPU as in Elicitor (two for the agent
  group, two for the user group). Site settings in an untracked `cluster/local.env`.
- Model folders (Decision Q9): `cluster/serve_models.sh` is Elicitor's script with one change,
  a model folder per group (`AGENT_OLLAMA_MODELS`, `USER_OLLAMA_MODELS`, both defaulting to
  `OLLAMA_MODELS`). The collection serves both teachers from the shared folder.
- Collection: `collect.sbatch` runs `make-backstories` (if the file is missing), `collect`,
  `judge-collect` and `build-sft`. About 20,000 requests: 7,740 `simulate` and up to 7,740
  `update`, 4,300 `judge` (the 160 backstories are written once and committed). The smoke file
  runs first; its seconds per attempt per role decide the GPU split between the groups
  (`AGENT_GPUS` / `USER_GPUS`, 2 / 2 by default; the user group carries the 70B model) and
  whether the full file fits 6 h, else §12's cuts apply. The full job runs with 1 / 3 (Q7).
- Jobs: `collect.sbatch` (4 GPUs, 6 h) → `train.sbatch` × 7 (1 GPU, 1 h each: the three
  `update` students, the judge student, and the three zero-shot bases, each from weights to an
  Ollama model in the agent group's folder, smoke-tested, template-checked and evaluated on the
  validation set; a base serves both roles) and `teacher.sbatch` (1 GPU, 2 h: the teachers at
  seed 1, the ceiling) → `e2e.sbatch` (4 GPUs, 4 h: calibrate, the runs, bench). Each ends with
  a health check. In `teacher.sbatch` each teacher gets its own server; the 70B judge fits on
  one A40 only with 4 parallel slots, as in the collection (found 10-26: with 8 its cache spilled
  to the CPU and requests timed out). Its 300 judge calls took 1,104 s.
- Elicitor's run folders (`m5_full`, `m5_smoke`, `m7_judge`) are read, never written; the reply
  cache is copied before use.
- Laptop (6 GB GPU): the 1.7B Q8_0 or 4B Q4_K_M student runs under Ollama for
  `elicitor demo --updater llm`.

## 10. Milestones

| M | Content | Done when |
|---|---|---|
| M0 | repo, pyproject with the Elicitor pin, CI, domain copy with checksums, `make-personas`, split guard, this spec | CI green; `personas_train.jsonl` frozen; the guard has a test. **Done 10-25:** 13 tests; 160 personas, seed 20261025, no redraw |
| M1 | `collect`, `judge-collect`, `build-sft` (runnable on the mock provider), cluster scripts; the collection job | mock collection in CI; the job finishes within 6 h with 0 failed calls and parse failures < 1 % per role; `data/sft/*.jsonl.gz` frozen with line counts and hashes recorded here. **Code 10-25:** the four commands, the cluster scripts and the collection files; 26 tests, the mock chain in CI. **Smoke 10-25** (8 personas, 2 / 2 GPUs, 38 min): health OK, 0 failed calls, 0 parse failures in every role; 160 backstories (166 calls, 510 s) frozen in `data/backstories.jsonl` (SHA-256 `ddc1c8a3…92fa4e2`); seconds per attempt: update 3.7, simulate 54.3, judge 45.6. Full job: one answer seed, 1 / 3 GPUs (Q7). **Done 10-26** (4 h 21 min): health OK in both steps, 0 failed calls, 0 parse failures, 0 transport errors; 7,740 `update` (1,680 cache hits) and 7,740 `simulate` in 2 h 52 min, 4,300 `judge` in 1 h 29 min; seconds per attempt update 10.6, simulate 34.3, judge 39.5; leaking answers 11 % (train) and 9 % (dev). SFT sets frozen (`data/sft/MANIFEST.json` has the full hashes): `update_train` 5,704 rows from 7,200 records (1,496 duplicate prompts, mostly short answers that several personas gave alike), SHA-256 `9951b9abe9a9…`; `update_val` 516 from 540, `760d47af2f57…`; `judge_train` 4,000, `0f5fe1891629…`; `judge_val` 300, `233df3710610…`. Judge bins (train): 784 / 989 / 984 / 854 / 389 |
| M2 | `train`, `export`, `eval-offline`, `bench`; three `update` students and three zero-shot baselines as Ollama models; the 1.7B judge student | six models answer 20 validation samples with 0 parse failures; the validation table; SFT 4B agreement above zero-shot 4B (else the format or template is debugged before M3). **Code 10-26:** the four commands plus `eval-table` and `check-template`; four training configs; `train.sbatch`, `teacher.sbatch`; the training chain tested on a tiny random model (CPU; run where the `train` extra is installed, not in CI). **Done 10-26:** seven models created (three `update` students, the judge student, three zero-shot bases), each answering its 20 smoke samples with 0 parse failures and passing `check-template`; the validation table is in §7.1; SFT 4B agreement 0.974 against zero-shot 4B 0.923. Training on one A40 (best epoch by validation agreement, or judge MAE): 0.6B 1,339 s (3 epochs, best 2), 1.7B 1,549 s (3, best 2), 4B 2,116 s (2, best 2; about 100 s of selection per epoch), judge 1.7B 2,207 s (5, best 4). Found on the way: llama.cpp's conversion needs sentencepiece (§5); Ollama 0.24 logs prompts only at trace level (§6); the 70B judge ceiling needs 4 parallel slots on one A40 (§9) |
| M3 | D1–D6 frozen → `calibrate` × 6 → end-to-end runs × 6 → `compare` → `bench` → test-set offline table (the first read of eval data) → judge test | job within 4 h; every run complete (10 cells) and healthy; `results_summary/` committed; D1–D6 each marked as found. **Frozen 10-26** (§7.4). **Code 10-26:** `build-test` (the test sets from `m5_full` and `m7_judge`, behind `--allow-eval`; `eval-offline` and `bench` refuse a test file without it), `m3-file` (one model's file from a template, with its measured rho), `compare`, `judge-eval`; the templates `experiments/m3_calibrate.toml` (m5_smoke's file) and `m3_e2e.toml` (m5_full's, cut to the two arms); `cluster/e2e.sbatch`; 46 tests (6 new: the chain on Elicitor's mock provider, and the one-sided bound on known data). Code size 2,028 lines without blank, comment and docstring lines (2,709 physical). **Done 10-27** (job 64845112, 2 h 28 min on 4 A40s): all six runs 10/10 cells (1,440 episodes each), health OK, 0 failed calls and 0 parse failures in every role; per run 7,680 `update` calls (1,379 new; 2,421 for zero-shot 0.6B) and 7,680 `simulate` (all from `m5_full`'s cache, but 665 for zero-shot 0.6B); test sets `update_test` 4,086 samples and `judge_test` 256, read once; `results_summary/m3/` and `experiments/m3/` committed. D2, D3, D4, D6 hold; D1 (agreement 0.956) and D5 (lower bound 0.790) do not (§7.4). Two submissions failed first: exclusive-process GPUs (46 s; one model at a time on GPU 0) and `--gpus=4` split 3 + 1 across two nodes (83 s; `--nodes=1`) |
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
| Zero-shot students produce invalid JSON | schema-constrained decoding keeps the pipeline running; unconstrained validity is a reported metric. *Found 10-26:* without the schema no zero-shot base wrote one valid `update` reply, and 73–95 % of judge replies were valid (§7.1); with it, 0 failed calls |
| A student's ρ ranks question types differently, so `eig` asks other questions | a finding, not a bug; the question-sequence agreement is reported |
| GGUF conversion of a merged Qwen3 model fails | the export chain is proven on the 0.6B first; fallback: Ollama importing the safetensors folder directly |
| 4B training over 1 h | smaller per-device batch, 2 epochs; else the 4B stays zero-shot only, noted here. *Applied 10-26:* the 1.7B trains at 2.9 s a step without gradient checkpointing, so the 4B's three epochs would take about an hour on their own; it trains for 2 (35 min at 5–7 s a step; training, export, checks and evaluation fit in the 1 h job) |
| Few de-duplicated eval `update` records (< 2,000) | the test-set offline table is reported with its wide CIs; the end-to-end layer carries the conclusion |
| Elicitor lacks an interface | patch release of Elicitor, pin moved; no code copied |
