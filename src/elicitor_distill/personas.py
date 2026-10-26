"""Training personas (SPEC §4, Decision Q6): `distill make-personas`.

Sampled like Elicitor's personas (every value of a slot about equally often, answer styles
balanced) with a seed of this project's own, all in the `train` split and with ids `t001`…, so
they never collide with Elicitor's `p001`–`p060`. No training persona may have exactly the
ground truth of one of Elicitor's dev or eval personas: if the draw contains such a collision
the whole set is drawn again with the next seed, and the seed actually used is recorded in a
sidecar file beside the persona file. Backstories are written later by the collection job (M1)
and stay empty here, as in Elicitor's `make-data`.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from elicitor.jsonl import dumps_jsonl
from elicitor.personas import STYLES, Persona, load_personas, stratified
from elicitor.schema import Schema, load_schema

from elicitor_distill.data import DOMAIN, sha256_file
from elicitor_distill.splits import TRAIN, TRAIN_FILE

N_TRAIN = 160
TRAIN_SEED = 20261025
META_FILE = "personas_train.meta.json"
_MAX_REDRAWS = 50


def truth_key(truth: dict[str, str], schema: Schema) -> tuple[str, ...]:
    return tuple(truth[s.name] for s in schema.slots)


def draw(schema: Schema, n: int, seed: int) -> list[Persona]:
    """One stratified draw of n training personas."""
    if n % len(STYLES):
        raise ValueError(f"n must be a multiple of {len(STYLES)}")
    rng = random.Random(seed)
    columns = {s.name: stratified(s.value_names, n, rng) for s in schema.slots}
    styles = stratified(STYLES, n, rng)
    return [
        Persona(
            id=f"t{i + 1:03d}",
            split=TRAIN,
            style=styles[i],
            truth={name: col[i] for name, col in columns.items()},
        )
        for i in range(n)
    ]


@dataclass(frozen=True)
class Draw:
    personas: list[Persona]
    seed_requested: int
    seed_used: int
    redraws: int


def generate_train_personas(
    schema: Schema, excluded: list[Persona], n: int = N_TRAIN, seed: int = TRAIN_SEED
) -> Draw:
    """n personas whose ground truth differs from every excluded persona's."""
    taken = {truth_key(p.truth, schema) for p in excluded}
    for redraws in range(_MAX_REDRAWS):
        personas = draw(schema, n, seed + redraws)
        keys = [truth_key(p.truth, schema) for p in personas]
        if not (taken & set(keys)) and len(set(keys)) == len(keys):
            return Draw(personas, seed, seed + redraws, redraws)
    raise RuntimeError(f"no collision-free draw within {_MAX_REDRAWS} seeds")


def expected_files(
    data_dir: str | Path, n: int = N_TRAIN, seed: int = TRAIN_SEED
) -> dict[Path, str]:
    """{path: content} of the persona file and its sidecar, as the generator makes them now."""
    data_dir = Path(data_dir)
    domain_dir = data_dir / DOMAIN
    schema = load_schema(domain_dir / "schema.toml")
    source = domain_dir / "personas.jsonl"
    excluded = load_personas(source)
    result = generate_train_personas(schema, excluded, n, seed)
    meta = {
        "n": n,
        "seed_requested": result.seed_requested,
        "seed_used": result.seed_used,
        "redraws": result.redraws,
        "excluded_personas": len(excluded),
        "excluded_from_sha256": sha256_file(source),
        "per_style": n // len(STYLES),
    }
    return {
        data_dir / TRAIN_FILE: dumps_jsonl(p.to_dict() for p in result.personas),
        data_dir / META_FILE: json.dumps(meta, indent=2) + "\n",
    }


def write_train_personas(
    data_dir: str | Path, n: int = N_TRAIN, seed: int = TRAIN_SEED
) -> list[Path]:
    files = expected_files(data_dir, n, seed)
    for path, text in files.items():
        path.write_text(text, encoding="utf-8", newline="\n")
    return list(files)


def check_train_personas(
    data_dir: str | Path, n: int = N_TRAIN, seed: int = TRAIN_SEED
) -> list[Path]:
    """Paths whose content differs from what the generator produces now."""
    return [
        path
        for path, text in expected_files(data_dir, n, seed).items()
        if not path.exists() or path.read_text(encoding="utf-8") != text
    ]
