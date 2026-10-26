import json
from collections import Counter

from elicitor.personas import STYLES, load_personas
from elicitor.schema import load_schema

from elicitor_distill.data import DOMAIN, verify_domain_data
from elicitor_distill.personas import (
    META_FILE,
    N_TRAIN,
    TRAIN_SEED,
    check_train_personas,
    draw,
    generate_train_personas,
    truth_key,
)
from elicitor_distill.splits import TRAIN, TRAIN_FILE


def test_domain_data_matches_pinned_checksums(data_dir):
    assert verify_domain_data(data_dir / DOMAIN) == []


def test_domain_data_is_elicitors_split(data_dir):
    personas = load_personas(data_dir / DOMAIN / "personas.jsonl")
    assert Counter(p.split for p in personas) == {"dev": 12, "eval": 48}
    assert all(p.backstory for p in personas)  # backstories.jsonl merged in


def test_committed_training_personas_are_current(data_dir):
    assert check_train_personas(data_dir) == []


def test_training_personas_shape(data_dir):
    schema = load_schema(data_dir / DOMAIN / "schema.toml")
    personas = load_personas(data_dir / TRAIN_FILE)
    assert len(personas) == N_TRAIN
    assert [p.id for p in personas] == [f"t{i:03d}" for i in range(1, N_TRAIN + 1)]
    assert {p.split for p in personas} == {TRAIN}
    assert Counter(p.style for p in personas) == dict.fromkeys(STYLES, N_TRAIN // len(STYLES))
    assert all(p.backstory == "" for p in personas)
    for slot in schema.slots:
        counts = Counter(p.truth[slot.name] for p in personas)
        assert set(counts) == set(slot.value_names)
        assert max(counts.values()) - min(counts.values()) <= 1


def test_no_truth_collision_with_elicitor_personas(data_dir):
    schema = load_schema(data_dir / DOMAIN / "schema.toml")
    theirs = {
        truth_key(p.truth, schema) for p in load_personas(data_dir / DOMAIN / "personas.jsonl")
    }
    ours = [truth_key(p.truth, schema) for p in load_personas(data_dir / TRAIN_FILE)]
    assert not theirs & set(ours)
    assert len(set(ours)) == len(ours)
    meta = json.loads((data_dir / META_FILE).read_text(encoding="utf-8"))
    assert meta["seed_requested"] == TRAIN_SEED
    assert meta["seed_used"] == TRAIN_SEED + meta["redraws"]


def test_draw_is_deterministic_and_redraws_on_collision(data_dir):
    schema = load_schema(data_dir / DOMAIN / "schema.toml")
    a = draw(schema, 8, 1)
    b = draw(schema, 8, 1)
    assert [p.to_dict() for p in a] == [p.to_dict() for p in b]
    # Excluding one of the drawn personas forces at least one redraw.
    result = generate_train_personas(schema, [a[0]], n=8, seed=1)
    assert result.redraws >= 1
    assert result.seed_used == 1 + result.redraws
    assert truth_key(a[0].truth, schema) not in {
        truth_key(p.truth, schema) for p in result.personas
    }
