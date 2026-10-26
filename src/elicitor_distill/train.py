"""LoRA SFT of a student on the teacher's replies (SPEC §5): `distill train`.

Prompts are rendered with the base model's chat template and `enable_thinking=False`, and every
rendering is checked against `serving_prompt`, the text Ollama will send the exported model, so
the training prompt is the serving prompt. The completion is the rest of the full conversation's
rendering (the teacher's JSON and the end-of-turn token); prompt and completion are tokenised
separately, as at serving time, and the loss is on the completion only. Samples longer than
`max_length` tokens are dropped and counted.

A checkpoint is saved after every epoch. Each is then scored on the validation set by greedy
generation with the offline metrics of SPEC §7.1 (argmax agreement for `update`, score MAE for
`judge`), and the best adapter is copied to `<out>/adapter`. `selection.json` records the scores.

The training stack (`torch`, `transformers`, `peft`) is the `train` extra and is imported only
inside the functions that need it. `--dry-run` stops once the trainer is built.
"""

from __future__ import annotations

import json
import math
import shutil
import time
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from elicitor.schema import Schema

from elicitor_distill.eval_offline import (
    FLAT,
    ROLES,
    parse_values,
    score_mae,
    teacher_values,
    update_metrics,
)
from elicitor_distill.export import serving_prompt
from elicitor_distill.sft_data import read_sft

IGNORE = -100  # label of a token the loss skips
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
_KEYS = {"name", "role", "base", "train", "val", "max_length", "epochs", "learning_rate",
         "per_device_batch", "grad_accum", "warmup", "seed", "max_new_tokens", "eval_batch",
         "lora"}  # fmt: skip
_LORA_KEYS = {"r", "alpha", "dropout", "targets"}

Progress = Callable[[str], None]


@dataclass(frozen=True)
class TrainConfig:
    name: str  # also the folder under models/ and, prefixed with "elicitor-", the Ollama model
    role: str
    base: str  # Hugging Face id, e.g. "Qwen/Qwen3-1.7B"
    train: Path
    val: Path
    max_length: int = 1024
    epochs: int = 3
    learning_rate: float = 2e-4
    per_device_batch: int = 8
    grad_accum: int = 4
    warmup: float = 0.03  # share of the optimiser steps
    seed: int = 0
    max_new_tokens: int = 300  # the role's max_tokens in Elicitor
    eval_batch: int = 32
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: tuple[str, ...] = LORA_TARGETS
    text: str = field(default="", compare=False, repr=False)

    def base_path(self, base_dir: str | Path | None) -> Path | str:
        """The base weights: `<base_dir>/<model name>` when a folder is given (offline nodes),
        else the Hugging Face id."""
        return Path(base_dir) / self.base.split("/")[-1] if base_dir else self.base


def load_train_config(path: str | Path) -> TrainConfig:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    raw = tomllib.loads(text)
    unknown = set(raw) - _KEYS
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    lora = dict(raw.pop("lora", {}))
    unknown = set(lora) - _LORA_KEYS
    if unknown:
        raise ValueError(f"{path}: unknown [lora] keys {sorted(unknown)}")
    if raw["role"] not in ROLES:
        raise ValueError(f"{path}: role must be one of {ROLES}")
    for key in ("train", "val"):
        raw[key] = (path.parent / raw[key]).resolve()
    for key in ("r", "alpha", "dropout"):
        if key in lora:
            raw[f"lora_{key}"] = lora[key]
    if "targets" in lora:
        raw["lora_targets"] = tuple(lora["targets"])
    return TrainConfig(**raw, text=text)


# ---------------------------------------------------------------- data


def render(tokenizer: Any, messages: list[dict[str, str]]) -> tuple[str, str]:
    """(prompt, completion) text of a chat sample; raises if the prompt is not the one the
    exported model will be served."""
    prompt = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    full = tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False)
    if prompt != serving_prompt(messages[:-1]):
        raise ValueError(
            "the chat template does not render the serving prompt:\n"
            f"{prompt!r}\nexpected\n{serving_prompt(messages[:-1])!r}"
        )
    if not full.startswith(prompt):
        raise ValueError("the rendered conversation does not start with the rendered prompt")
    return prompt, full[len(prompt) :]


def encode(tokenizer: Any, sample: dict[str, Any], max_length: int) -> dict[str, Any] | None:
    """Token ids and labels (the prompt masked); None when longer than `max_length`."""
    prompt, completion = render(tokenizer, sample["messages"])
    p = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    c = tokenizer(completion, add_special_tokens=False)["input_ids"]
    if len(p) + len(c) > max_length:
        return None
    return {"input_ids": p + c, "attention_mask": [1] * (len(p) + len(c)),
            "labels": [IGNORE] * len(p) + c}  # fmt: skip


def encode_all(
    tokenizer: Any, samples: list[dict[str, Any]], max_length: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    features = [encode(tokenizer, s, max_length) for s in samples]
    kept = [f for f in features if f is not None]
    lengths = [len(f["input_ids"]) for f in kept]
    return kept, {
        "samples": len(samples),
        "kept": len(kept),
        "too_long": len(samples) - len(kept),
        "max_tokens": max(lengths, default=0),
        "mean_tokens": round(sum(lengths) / len(lengths)) if lengths else 0,
    }


# ---------------------------------------------------------------- checkpoint selection


def selection_metrics(
    role: str, samples: list[dict[str, Any]], texts: list[str], schema: Schema
) -> dict[str, Any]:
    """Offline metrics of unconstrained generations; `score` is what the selection maximises."""
    rows = []
    for sample, text in zip(samples, texts, strict=True):
        teacher = teacher_values(role, sample, schema)
        student = parse_values(role, text, sample, schema)
        rows.append((teacher, student))
    valid = sum(s is not None for _, s in rows) / len(rows)
    if role == "update":
        metrics = [update_metrics(t, s if s is not None else [FLAT] * len(t)) for t, s in rows]
        agreement = sum(m["agreement"] for m in metrics) / len(metrics)
        mae = sum(m["mae"] for m in metrics) / len(metrics)
        return {"score": round(agreement, 4), "agreement": round(agreement, 4),
                "mae": round(mae, 4), "valid": round(valid, 4)}  # fmt: skip
    ok = [(t, s) for t, s in rows if s is not None]
    mae = score_mae([t for t, _ in ok], [s for _, s in ok]) if ok else float(10)
    return {"score": round(-mae, 4), "mae": round(mae, 4), "valid": round(valid, 4)}


def generate(
    model: Any, tokenizer: Any, prompts: list[str], max_new_tokens: int, batch: int
) -> list[str]:
    """Greedy completions, batched by prompt length, returned in prompt order."""
    import torch

    tokenizer.padding_side = "left"
    stop = [tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out: list[str] = [""] * len(prompts)
    model.eval()
    for start in range(0, len(order), batch):
        idx = order[start : start + batch]
        enc = tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True,
                        add_special_tokens=False).to(model.device)  # fmt: skip
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 eos_token_id=stop, pad_token_id=tokenizer.pad_token_id,
                                 use_cache=True)  # fmt: skip
        new = gen[:, enc["input_ids"].shape[1] :]
        for i, ids in zip(idx, new, strict=True):
            out[i] = tokenizer.decode(ids, skip_special_tokens=True)
    tokenizer.padding_side = "right"
    return out


def epoch_checkpoints(folder: Path) -> list[Path]:
    """`checkpoint-<step>` folders in step order: one per epoch."""
    found = [p for p in folder.glob("checkpoint-*") if p.name.split("-")[-1].isdigit()]
    return sorted(found, key=lambda p: int(p.name.split("-")[-1]))


# ---------------------------------------------------------------- training


def train(
    config: TrainConfig,
    schema: Schema,
    out_dir: str | Path,
    base_dir: str | Path | None = None,
    dry_run: bool = False,
    progress: Progress = print,
) -> dict[str, Any]:
    """Train, select the best epoch, write `<out>/adapter` and `<out>/selection.json`."""
    import torch
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    out = Path(out_dir)
    base = config.base_path(base_dir)
    tokenizer = AutoTokenizer.from_pretrained(base)
    tokenizer.padding_side = "right"
    train_samples, val_samples = read_sft(config.train), read_sft(config.val)
    features, counts = encode_all(tokenizer, train_samples, config.max_length)
    progress(f"train: {counts}")

    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    model = get_peft_model(
        model,
        LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.lora_targets),
            task_type="CAUSAL_LM",
        ),  # fmt: skip
    )
    trainable, total = model.get_nb_trainable_parameters()
    progress(f"LoRA: {trainable:,} trainable of {total:,} parameters")

    per_step = config.per_device_batch * config.grad_accum
    steps = math.ceil(len(features) / per_step) * config.epochs
    args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.per_device_batch,
        gradient_accumulation_steps=config.grad_accum,
        learning_rate=config.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=math.ceil(config.warmup * steps),
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        train_sampling_strategy="group_by_length",  # less padding; the order is still seeded
        logging_steps=10,
        save_strategy="epoch",
        save_only_model=True,
        seed=config.seed,
        data_seed=config.seed,
        report_to="none",
    )
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=IGNORE,
                                      pad_to_multiple_of=8)  # fmt: skip
    trainer = Trainer(model=model, args=args, train_dataset=features, data_collator=collator,
                      processing_class=tokenizer)  # fmt: skip
    plan = {"name": config.name, "role": config.role, "base": str(base), "data": counts,
            "optimizer_steps": steps, "effective_batch": per_step,
            "trainable_parameters": trainable}  # fmt: skip
    if dry_run:
        progress("dry run: trainer built, not training")
        return {**plan, "dry_run": True}

    start = time.perf_counter()
    result = trainer.train()
    train_seconds = time.perf_counter() - start
    progress(f"trained in {train_seconds:.0f} s, loss {result.training_loss:.4f}")

    prompts = [render(tokenizer, s["messages"])[0] for s in val_samples]
    epochs = []
    for epoch, ckpt in enumerate(epoch_checkpoints(out / "checkpoints"), start=1):
        set_peft_model_state_dict(model, load_file(ckpt / "adapter_model.safetensors"))
        t0 = time.perf_counter()
        texts = generate(model, tokenizer, prompts, config.max_new_tokens, config.eval_batch)
        metrics = selection_metrics(config.role, val_samples, texts, schema)
        epochs.append({"epoch": epoch, "checkpoint": ckpt.name, **metrics,
                       "seconds": round(time.perf_counter() - t0, 1)})  # fmt: skip
        progress(f"epoch {epoch}: {metrics}")
    best = max(epochs, key=lambda e: (e["score"], -e["epoch"]))  # ties go to the earlier epoch
    adapter = out / "adapter"
    if adapter.exists():
        shutil.rmtree(adapter)
    shutil.copytree(out / "checkpoints" / best["checkpoint"], adapter)
    selection = {
        **plan,
        "train_seconds": round(train_seconds, 1),
        "training_loss": round(result.training_loss, 4),
        "log": trainer.state.log_history,
        "epochs": epochs,
        "best_epoch": best["epoch"],
        "config": config.text,
    }
    (out / "selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    shutil.rmtree(out / "checkpoints")
    return selection
