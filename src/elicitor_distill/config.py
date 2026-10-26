"""Collection files (SPEC §4): TOML read into a dataclass. Unknown keys are errors.

One file drives `make-backstories`, `collect` and `judge-collect`. Its `[llm]` table is passed to
Elicitor's `build_client` unchanged, so the teachers are served exactly as in Elicitor's full run.
Data paths are relative to the file, as in Elicitor's experiment files.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from elicitor.personas import DEV

from elicitor_distill.data import DOMAIN
from elicitor_distill.splits import TRAIN

COLLECT_SPLITS = (TRAIN, DEV)  # never the eval split: it is the test set
_KEYS = {"name", "domain", "data_dir", "seed", "splits", "answer_seeds", "max_personas",
         "judge", "llm"}  # fmt: skip
_JUDGE_KEYS = {"edges", "per_bin"}


@dataclass(frozen=True)
class CollectConfig:
    name: str
    domain: str
    data_dir: Path
    seed: int
    splits: tuple[str, ...] = COLLECT_SPLITS
    answer_seeds: int = 2  # answers per question (Decision Q7)
    max_personas: int | None = None  # first n personas of each split (smoke and mock files)
    judge_edges: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8)  # utility bin edges (SPEC §4)
    judge_per_bin: int = 5
    llm: dict[str, Any] = field(default_factory=dict)
    text: str = field(default="", compare=False, repr=False)  # the TOML text, for provenance

    @property
    def domain_dir(self) -> Path:
        return self.data_dir / self.domain

    @property
    def schema_path(self) -> Path:
        return self.domain_dir / "schema.toml"

    @property
    def catalog_path(self) -> Path:
        return self.domain_dir / "catalog.jsonl"

    @property
    def prompts_dir(self) -> Path:
        return self.domain_dir / "prompts"


def load_collect_config(path: str | Path) -> CollectConfig:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    raw = tomllib.loads(text)
    unknown = set(raw) - _KEYS
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    judge = dict(raw.get("judge", {}))
    unknown = set(judge) - _JUDGE_KEYS
    if unknown:
        raise ValueError(f"{path}: unknown [judge] keys {sorted(unknown)}")
    splits = tuple(raw.get("splits", COLLECT_SPLITS))
    bad = set(splits) - set(COLLECT_SPLITS)
    if bad:
        raise ValueError(f"{path}: splits must be among {COLLECT_SPLITS}, got {sorted(bad)}")
    edges = tuple(float(e) for e in judge.get("edges", (0.5, 0.6, 0.7, 0.8)))
    if list(edges) != sorted(edges):
        raise ValueError(f"{path}: [judge] edges must be increasing")
    max_personas = raw.get("max_personas")
    return CollectConfig(
        name=raw["name"],
        domain=raw.get("domain", DOMAIN),
        data_dir=(path.parent / raw.get("data_dir", "../data")).resolve(),
        seed=int(raw["seed"]),
        splits=splits,
        answer_seeds=int(raw.get("answer_seeds", 2)),
        max_personas=None if max_personas is None else int(max_personas),
        judge_edges=edges,
        judge_per_bin=int(judge.get("per_bin", 5)),
        llm=dict(raw.get("llm", {})),
        text=text,
    )
