"""The training chain on a tiny random Qwen3 model and a character tokenizer, on the CPU.

Needs the `train` extra (skipped without it); downloads nothing.
"""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from elicitor.jsonl import dumps_jsonl  # noqa: E402
from elicitor.schema import load_schema  # noqa: E402

from elicitor_distill.cli import main  # noqa: E402
from elicitor_distill.export import merge_adapter  # noqa: E402
from elicitor_distill.sft_data import gzip_bytes, read_sft  # noqa: E402
from elicitor_distill.train import load_train_config, train  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SPECIAL = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>"]
# Qwen3's template for system, user and assistant turns without thinking (the tool and reasoning
# branches left out), written the way Qwen writes it.
TEMPLATE = (
    r"{%- for m in messages %}"
    r"{%- if m.role == 'assistant' %}"
    r"{{- '<|im_start|>assistant\n<think>\n\n</think>\n\n' + m.content + '<|im_end|>\n' }}"
    r"{%- else %}{{- '<|im_start|>' + m.role + '\n' + m.content + '<|im_end|>\n' }}{%- endif %}"
    r"{%- endfor %}"
    r"{%- if add_generation_prompt %}"
    r"{{- '<|im_start|>assistant\n' }}"
    r"{%- if enable_thinking is defined and enable_thinking is false %}"
    r"{{- '<think>\n\n</think>\n\n' }}{%- endif %}"
    r"{%- endif %}"
)


def tiny_model(folder: Path, texts: list[str]) -> Path:
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers

    chars = sorted({c for t in texts for c in t} | set('0123456789{}":,'))
    vocab = {tok: i for i, tok in enumerate(SPECIAL + chars)}
    core = Tokenizer(models.WordLevel(vocab, unk_token="<|endoftext|>"))
    core.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    core.decoder = decoders.Fuse()
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=core, eos_token="<|im_end|>", pad_token="<|endoftext|>",
        additional_special_tokens=SPECIAL[3:], chat_template=TEMPLATE,
    )  # fmt: skip
    config = transformers.Qwen3Config(
        vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=4096,
        tie_word_embeddings=True, eos_token_id=vocab["<|im_end|>"],
        pad_token_id=vocab["<|endoftext|>"],
    )  # fmt: skip
    torch.manual_seed(0)
    transformers.AutoModelForCausalLM.from_config(config).save_pretrained(folder)
    tok.save_pretrained(folder)
    return folder


def test_train_select_and_merge(tmp_path, data_dir):
    samples = read_sft(ROOT / "data" / "sft" / "update_val.jsonl.gz")[:12]
    for name, part in (("train", samples[:8]), ("val", samples[8:])):
        (tmp_path / f"{name}.jsonl.gz").write_bytes(gzip_bytes(dumps_jsonl(part).encode()))
    texts = [m["content"] for s in samples for m in s["messages"]]
    tiny_model(tmp_path / "hf" / "Tiny", texts)
    config_path = tmp_path / "tiny.toml"
    config_path.write_text(
        'name = "tiny"\nrole = "update"\nbase = "test/Tiny"\ntrain = "train.jsonl.gz"\n'
        'val = "val.jsonl.gz"\nmax_length = 4096\nepochs = 2\nper_device_batch = 2\n'
        "grad_accum = 2\nmax_new_tokens = 8\neval_batch = 4\n",
        encoding="utf-8",
    )
    config = load_train_config(config_path)
    schema = load_schema(data_dir / "travel" / "schema.toml")

    out = tmp_path / "models" / "tiny"
    assert main(["train", str(config_path), "--base-dir", str(tmp_path / "hf"),
                 "--out", str(out), "--dry-run"]) == 0  # fmt: skip
    assert not (out / "adapter").exists()

    selection = train(config, schema, out, tmp_path / "hf", progress=lambda s: None)
    assert [e["epoch"] for e in selection["epochs"]] == [1, 2]
    assert selection["data"]["kept"] == 8 and selection["optimizer_steps"] == 4
    assert (out / "adapter" / "adapter_model.safetensors").exists()
    assert not (out / "checkpoints").exists()
    saved = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    assert saved["best_epoch"] == selection["best_epoch"]

    merged = merge_adapter(tmp_path / "hf" / "Tiny", out / "adapter", tmp_path / "merged")
    assert (merged / "model.safetensors").exists() and (merged / "tokenizer.json").exists()
