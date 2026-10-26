"""The three persona splits and the guard that keeps the test split out of reach (SPEC §4).

- `train`: this project's own personas (`data/personas_train.jsonl`), used for teacher data.
- `dev`: Elicitor's 12 development personas, used for validation and model selection.
- `eval`: Elicitor's 48 evaluation personas. They are the test set. Nothing may load them
  without `allow_eval=True`, which only the final evaluation commands pass (M3).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from elicitor.personas import DEV, EVAL, Persona, load_personas

TRAIN = "train"
SPLITS = (TRAIN, DEV, EVAL)
TRAIN_FILE = "personas_train.jsonl"


class EvalSplitError(RuntimeError):
    """A command tried to read evaluation personas without saying so."""


def guard(personas: Iterable[Persona], allow_eval: bool = False) -> list[Persona]:
    """The personas as a list; raises if any is in the eval split and that was not allowed."""
    out = list(personas)
    if not allow_eval:
        hidden = [p.id for p in out if p.split == EVAL]
        if hidden:
            raise EvalSplitError(
                f"{len(hidden)} evaluation personas requested (first: {hidden[0]}); the eval "
                "split is the test set and needs allow_eval=True (`--allow-eval`)"
            )
    return out


def load_split(
    data_dir: str | Path, split: str, domain: str = "travel", allow_eval: bool = False
) -> list[Persona]:
    """Personas of one split, with backstories where the file exists."""
    data_dir = Path(data_dir)
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; choose from {SPLITS}")
    if split == TRAIN:
        personas = load_personas(data_dir / TRAIN_FILE)
    else:
        personas = load_personas(data_dir / domain / "personas.jsonl", split=split)
    return guard(personas, allow_eval)
