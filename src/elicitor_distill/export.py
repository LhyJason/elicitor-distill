"""From a trained adapter (or a base model) to an Ollama model (SPEC §6): `distill export`.

1. Merge the LoRA adapter into the base weights (skipped for a zero-shot baseline).
2. Convert the merged folder to GGUF with llama.cpp's `convert_hf_to_gguf.py` (Q8_0).
3. Write a Modelfile and `ollama create` the model.

The Modelfile sets the chat template explicitly rather than leaving Ollama to guess one from the
GGUF: it renders a system and a user message exactly as Qwen3's own template does with
`enable_thinking=False` (`serving_prompt`), which is also how `distill train` renders the
training prompts (it checks every sample). The teacher's Ollama template differs (it appends
"/no_think" to the user turn), but the student's prompt only has to match its own training.
`check-template` closes the loop on the cluster: it compares the prompts Ollama logged at debug
level with `serving_prompt` of the same samples.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

MODELS_DIR = "models"
OUTTYPE = "q8_0"  # Decision Q10
GENERATION_PREFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"

# Ollama (Go text/template) version of `serving_prompt`; the system message arrives as .System.
OLLAMA_TEMPLATE = """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}
{{- range .Messages }}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ .Content }}<|im_end|>
{{ end }}
{{- end }}<|im_start|>assistant
<think>

</think>

"""
STOP = ("<|im_end|>", "<|im_start|>")


def serving_prompt(messages: list[dict[str, str]]) -> str:
    """The text a student sees for a system and a user message: Qwen3's chat template with
    `add_generation_prompt=True, enable_thinking=False`."""
    parts = []
    for m in messages:
        if m["role"] in ("system", "user"):
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
    return "".join(parts) + GENERATION_PREFIX


def modelfile(gguf: Path) -> str:
    stops = "".join(f'PARAMETER stop "{s}"\n' for s in STOP)
    # Greedy decoding at temperature 0 must not be bent by a repeat penalty (JSON repeats).
    return (
        f"FROM {gguf.resolve().as_posix()}\n"
        f'TEMPLATE """{OLLAMA_TEMPLATE}"""\n'
        f"{stops}"
        "PARAMETER repeat_penalty 1\n"
    )


def merge_adapter(base: Path, adapter: Path, out: Path) -> Path:
    """The adapter merged into the bf16 base weights, saved with the tokenizer."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(base).save_pretrained(out)
    return out


def convert_gguf(hf_dir: Path, gguf: Path, llama_cpp: Path, outtype: str = OUTTYPE) -> Path:
    script = llama_cpp / "convert_hf_to_gguf.py"
    if not script.exists():
        raise FileNotFoundError(f"{script} not found (clone llama.cpp, SPEC §9)")
    gguf.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(script), str(hf_dir), "--outfile", str(gguf), "--outtype", outtype]
    subprocess.run(cmd, check=True)
    return gguf


def ollama_create(name: str, path: Path) -> None:
    """`ollama create` against the server in OLLAMA_HOST."""
    if shutil.which("ollama") is None:
        raise FileNotFoundError("ollama is not on PATH")
    subprocess.run(["ollama", "create", name, "-f", str(path)], check=True)


def export(
    name: str,
    base: Path,
    adapter: Path | None,
    out: Path,
    llama_cpp: Path,
    outtype: str = OUTTYPE,
    create: bool = False,
) -> dict[str, Any]:
    """Merge (if there is an adapter), convert, write the Modelfile, optionally create the model.
    Returns the paths; `export.json` in `out` records them."""
    out.mkdir(parents=True, exist_ok=True)
    hf_dir = merge_adapter(base, adapter, out / "merged") if adapter else base
    gguf = convert_gguf(hf_dir, out / f"{name}-{outtype}.gguf", llama_cpp, outtype)
    path = out / "Modelfile"
    path.write_text(modelfile(gguf), encoding="utf-8", newline="\n")
    if create:
        ollama_create(name, path)
    if adapter and hf_dir != base:
        shutil.rmtree(hf_dir)  # the merged weights are rebuilt from the adapter on demand
    info = {
        "name": name,
        "base": base.as_posix(),
        "adapter": adapter.as_posix() if adapter else None,
        "gguf": gguf.as_posix(),
        "gguf_bytes": gguf.stat().st_size,
        "outtype": outtype,
        "created": create,
    }
    (out / "export.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


# ---------------------------------------------------------------- template check

# Ollama 0.24 logs the rendered prompt only at trace level (`OLLAMA_DEBUG=2`), as
# `msg="completion request" prompt="..."`; its debug line carries the length (`prompt=1757`),
# which the quote requirement skips. Older servers logged `msg="chat request"` at debug level.
_PROMPT = re.compile(r'msg="(?:chat|completion) request".*?\sprompt=("(?:[^"\\]|\\.)*")')


def ollama_prompts(log_text: str) -> list[str]:
    """The rendered prompts an Ollama server logged (`OLLAMA_DEBUG=2`), in order. The log
    quotes them Go-style, which JSON reads for the escapes that occur in prompts."""
    return [json.loads(m.group(1)) for m in _PROMPT.finditer(log_text)]


def check_prompts(logged: list[str], samples: list[dict[str, Any]]) -> list[str]:
    """Problems found comparing logged prompts with `serving_prompt` of the samples, which were
    sent in this order (one worker, no cache); empty when every prompt matches."""
    if not logged:
        return ["no prompts in the log (start the server with OLLAMA_DEBUG=2)"]
    if len(logged) < len(samples):
        return [f"{len(logged)} prompts logged for {len(samples)} samples"]
    problems = []
    for i, (got, sample) in enumerate(zip(logged, samples, strict=False)):
        want = serving_prompt(sample["messages"][:2])
        if got != want:
            at = next((j for j, (a, b) in enumerate(zip(got, want, strict=False)) if a != b),
                      min(len(got), len(want)))  # fmt: skip
            window = slice(max(0, at - 20), at + 40)
            problems.append(
                f"sample {i}: differs at character {at}: "
                f"logged {got[window]!r}, expected {want[window]!r}"
            )
    return problems
