import json
import math
from pathlib import Path

import pytest
from elicitor.llm import LLMClient
from elicitor.llm.providers import Completion
from elicitor.schema import load_schema

from elicitor_distill.bench import bench
from elicitor_distill.cli import main
from elicitor_distill.eval_offline import (
    FLAT,
    argmax_agreement,
    evaluate_one,
    likelihood_kl,
    make_eval_client,
    passthrough_roles,
    rank_correlation,
    sample_context,
    summarize,
    teacher_values,
    update_metrics,
)
from elicitor_distill.export import (
    GENERATION_PREFIX,
    OLLAMA_TEMPLATE,
    check_prompts,
    modelfile,
    ollama_prompts,
    serving_prompt,
)
from elicitor_distill.sft_data import read_sft
from elicitor_distill.train import (
    IGNORE,
    encode,
    encode_all,
    load_train_config,
    render,
    selection_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
SFT = ROOT / "data" / "sft"


@pytest.fixture(scope="module")
def schema(data_dir):
    return load_schema(data_dir / "travel" / "schema.toml")


@pytest.fixture(scope="module")
def update_val():
    return read_sft(SFT / "update_val.jsonl.gz")


@pytest.fixture(scope="module")
def judge_val():
    return read_sft(SFT / "judge_val.jsonl.gz")


# ---------------------------------------------------------------- metrics


def test_argmax_agreement_counts_ties():
    assert argmax_agreement([9, 1, 1], [8, 2, 2]) == 1
    assert argmax_agreement([9, 1, 1], [1, 9, 1]) == 0
    assert argmax_agreement([9, 1, 1], [9, 9, 1]) == 0.5  # the student ties 2 values: 1/j
    assert argmax_agreement([5, 5, 5], [5, 5, 5]) == 1  # the same tie is full agreement


def test_kl_and_rank_correlation():
    assert likelihood_kl([9, 1, 5], [9, 1, 5]) == pytest.approx(0)
    assert likelihood_kl([10, 0, 0], [0, 10, 0]) > 1
    assert math.isnan(rank_correlation([5, 5, 5], [1, 2, 3]))  # a constant vector
    assert rank_correlation([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1)


# ---------------------------------------------------------------- evaluation


def test_passthrough_prompt_is_the_stored_prompt(data_dir, schema, update_val):
    roles = passthrough_roles(data_dir / "travel" / "prompts")
    sample = update_val[0]
    rendered = roles["update"].prompt.render(sample_context("update", sample, schema))
    assert rendered == sample["messages"][:2]
    assert roles["judge"].max_tokens == 150 and roles["update"].max_tokens == 300


class Garbage:
    name = "garbage"

    def complete(self, request):
        return Completion(text="not json")


def test_failed_update_counts_as_equal_scores(data_dir, schema, update_val):
    client = LLMClient(passthrough_roles(data_dir / "travel" / "prompts"), Garbage(),
                       {"agent": "x", "user": "x"})  # fmt: skip
    sample = update_val[0]
    row = evaluate_one(client, "update", schema, 0, 0, sample)
    teacher = teacher_values("update", sample, schema)
    assert row["status"] == "failed" and row["student"] is None
    flat = update_metrics(teacher, [FLAT] * len(teacher))
    assert {k: row[k] for k in flat} == pytest.approx(flat, nan_ok=True)
    assert client.stats["update"].failed_calls == 1
    assert summarize("update", [row])["failed"] == 1


def test_eval_offline_on_the_mock_is_a_perfect_student(tmp_path):
    out = tmp_path / "m2"
    for role in ("update", "judge"):
        args = ["eval-offline", "--provider", "mock", "--model", "mock", "--role", role,
                "--limit", "25", "--workers", "4", "--strict",
                "--out", str(out / role)]  # fmt: skip
        assert main(args) == 0
    update = json.loads((out / "update" / "summary.json").read_text(encoding="utf-8"))
    overall = update["tables"]["overall"]
    assert overall["n"] == 25 and overall["agreement"] == 1 and overall["mae"] == 0
    assert overall["kl"] == 0 and overall["unconstrained_valid"] == 1
    assert set(update["tables"]["kind"]) <= {"open", "choice", "yesno"}
    judge = json.loads((out / "judge" / "summary.json").read_text(encoding="utf-8"))
    assert judge["tables"]["overall"]["exact"] == 1
    assert (out / "update" / "table.md").read_text(encoding="utf-8").startswith("### mock")
    assert len((out / "judge" / "rows.jsonl").read_text(encoding="utf-8").splitlines()) == 25
    table = tmp_path / "table.md"
    assert main(["eval-table", str(out), "--out", str(table)]) == 0
    text = table.read_text(encoding="utf-8")
    assert "### update" in text and "### judge" in text and "| mock | 0 |" in text


def test_bench_on_the_mock(tmp_path, data_dir, schema, update_val):
    client = make_eval_client(data_dir / "travel" / "prompts", "mock", provider="mock")
    result = bench(client, update_val[:10], schema, workers=4)
    assert result["requests"] == 10 and result["failed_calls"] == 0
    assert result["requests_per_s"] > 0
    cached = make_eval_client(data_dir / "travel" / "prompts", "mock", provider="mock",
                              cache=tmp_path / "c.sqlite")  # fmt: skip
    with pytest.raises(ValueError, match="without a reply cache"):
        bench(cached, update_val[:2], schema)


# ---------------------------------------------------------------- serving


QWEN3_PROMPT = (
    "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU $x<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


def test_serving_prompt_is_qwen3_without_thinking():
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "U $x"}]
    assert serving_prompt(messages) == QWEN3_PROMPT
    assert OLLAMA_TEMPLATE.endswith(GENERATION_PREFIX.removeprefix("<|im_start|>assistant\n"))


def test_modelfile(tmp_path):
    text = modelfile(tmp_path / "m.gguf")
    assert text.startswith(f"FROM {(tmp_path / 'm.gguf').resolve().as_posix()}\n")
    assert f'TEMPLATE """{OLLAMA_TEMPLATE}"""' in text
    assert 'PARAMETER stop "<|im_end|>"' in text and "PARAMETER repeat_penalty 1" in text
    assert "SYSTEM" not in text


def _go_quote(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def test_check_prompts_reads_the_debug_log(update_val):
    samples = update_val[:3]
    lines = []
    for s in samples:  # Ollama 0.24: the length at debug level, the prompt at trace level
        prompt = serving_prompt(s["messages"][:2])
        lines += [
            'time=2026-10-26T10:00:00Z level=DEBUG source=server.go:1580 msg="completion request" '
            f'images=0 prompt={len(prompt)} format="{{}}"',
            'time=2026-10-26T10:00:00Z level=TRACE source=server.go:1581 msg="completion request" '
            f"prompt={_go_quote(prompt)}",
        ]
    log = "\n".join(["level=INFO msg=starting", *lines])
    assert ollama_prompts(log) == [serving_prompt(s["messages"][:2]) for s in samples]
    assert check_prompts(ollama_prompts(log), samples) == []
    bad = log.replace("/think", "/no_think").replace("<|im_start|>assistant", "<|im_start|>asst")
    problems = check_prompts(ollama_prompts(bad), samples)
    assert len(problems) == 3 and "differs at character" in problems[0]
    assert "no prompts" in check_prompts([], samples)[0]


# ---------------------------------------------------------------- training


def test_training_configs():
    configs = {p.name: load_train_config(p) for p in (ROOT / "configs").glob("*.toml")}
    assert set(configs) == {"update_0.6b.toml", "update_1.7b.toml", "update_4b.toml",
                            "judge_1.7b.toml"}  # fmt: skip
    for c in configs.values():
        assert c.train.exists() and c.val.exists()
        assert c.per_device_batch * c.grad_accum == 32
        assert c.lora_r == 16 and c.max_length == 1024
    assert configs["update_4b.toml"].learning_rate == 1e-4
    assert configs["judge_1.7b.toml"].epochs == 5
    assert configs["judge_1.7b.toml"].max_new_tokens == 150
    assert configs["update_1.7b.toml"].base_path("/hf") == Path("/hf/Qwen3-1.7B")


def test_unknown_training_key_is_an_error(tmp_path):
    path = tmp_path / "x.toml"
    path.write_text('name="x"\nrole="update"\nbase="b"\ntrain="t"\nval="v"\nlr=1\n')
    with pytest.raises(ValueError, match="unknown keys"):
        load_train_config(path)


class StubTokenizer:
    """Qwen3's chat template for system/user/assistant without thinking, characters as tokens."""

    def __init__(self, generation_prefix: str = GENERATION_PREFIX):
        self.prefix = generation_prefix

    def apply_chat_template(self, messages, tokenize, add_generation_prompt=False, **kwargs):
        assert kwargs == {"enable_thinking": False} and not tokenize
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
                       for m in messages if m["role"] != "assistant")  # fmt: skip
        if messages[-1]["role"] == "assistant":
            return text + self.prefix + messages[-1]["content"] + "<|im_end|>\n"
        return text + (self.prefix if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens):
        return {"input_ids": [ord(c) for c in text]}


def test_encode_masks_the_prompt(update_val):
    tokenizer, sample = StubTokenizer(), update_val[0]
    prompt, completion = render(tokenizer, sample["messages"])
    assert prompt == serving_prompt(sample["messages"][:2])
    assert completion == sample["messages"][2]["content"] + "<|im_end|>\n"
    f = encode(tokenizer, sample, 10_000)
    assert len(f["input_ids"]) == len(f["labels"]) == len(prompt) + len(completion)
    assert f["labels"][: len(prompt)] == [IGNORE] * len(prompt)
    assert f["labels"][len(prompt) :] == [ord(c) for c in completion]
    kept, counts = encode_all(tokenizer, update_val[:5], len(prompt) + len(completion))
    assert counts["samples"] == 5 and counts["kept"] + counts["too_long"] == 5
    assert counts["max_tokens"] <= len(prompt) + len(completion)


def test_render_refuses_a_template_that_differs_from_serving(update_val):
    tokenizer = StubTokenizer(generation_prefix="<|im_start|>assistant\n")  # thinking left on
    with pytest.raises(ValueError, match="serving prompt"):
        render(tokenizer, update_val[0]["messages"])


def test_selection_metrics(schema, update_val, judge_val):
    texts = [s["messages"][-1]["content"] for s in update_val[:20]]
    assert selection_metrics("update", update_val[:20], texts, schema)["score"] == 1
    bad = selection_metrics("update", update_val[:20], ["{}"] * 20, schema)
    assert bad["valid"] == 0 and bad["score"] < 1
    texts = [s["messages"][-1]["content"] for s in judge_val[:20]]
    assert selection_metrics("judge", judge_val[:20], texts, schema) == {
        "score": 0, "mae": 0, "valid": 1}  # fmt: skip
