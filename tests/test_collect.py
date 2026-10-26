import json
import shutil
from collections import Counter
from pathlib import Path

import pytest
from elicitor.belief import Belief
from elicitor.catalog import load_catalog
from elicitor.jsonl import read_jsonl
from elicitor.llm.roles import update_model
from elicitor.questions import Question
from elicitor.schema import load_schema
from elicitor.updaters.llm import LLMUpdater

from elicitor_distill.cli import main
from elicitor_distill.collect import (
    CACHE,
    OK,
    MissingBackstoriesError,
    backstories_path,
    collect,
    make_client,
    make_train_backstories,
    update_context,
)
from elicitor_distill.config import load_collect_config
from elicitor_distill.judge_collect import judge_collect, judge_items, utility_bin
from elicitor_distill.sft_data import MANIFEST, build_sft, read_sft
from elicitor_distill.splits import EvalSplitError, load_split

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"

MOCK_FILE = """
name = "mock_chain"
domain = "travel"
data_dir = "../data"
seed = 7
splits = ["train", "dev"]
answer_seeds = 2
max_personas = 2

[llm]
provider = "mock"
"""


@pytest.fixture
def workspace(tmp_path, data_dir):
    """A copy of data/ and a mock collection file beside it, as in the repo, without the frozen
    training backstories (the tests write their own; Elicitor's in travel/ stay)."""

    def ignore(folder, names):
        top = Path(folder) == Path(data_dir)
        return {n for n in names if n in ("raw", "sft") or (top and n == "backstories.jsonl")}

    shutil.copytree(data_dir, tmp_path / "data", ignore=ignore)
    (tmp_path / "experiments").mkdir()
    path = tmp_path / "experiments" / "mock.toml"
    path.write_text(MOCK_FILE, encoding="utf-8")
    return tmp_path, load_collect_config(path)


# ---------------------------------------------------------------- collection files


@pytest.mark.parametrize("name", ["collect.toml", "collect_smoke.toml", "collect_mock.toml"])
def test_collection_files_load(name):
    config = load_collect_config(EXPERIMENTS / name)
    assert config.data_dir == (ROOT / "data").resolve()
    assert "eval" not in config.splits


def test_full_and_smoke_files_differ_only_in_name_and_size():
    full = load_collect_config(EXPERIMENTS / "collect.toml")
    smoke = load_collect_config(EXPERIMENTS / "collect_smoke.toml")
    assert full.max_personas is None and smoke.max_personas == 4
    assert (full.seed, full.splits, full.answer_seeds, full.llm) == (
        smoke.seed, smoke.splits, smoke.answer_seeds, smoke.llm
    )  # fmt: skip
    assert full.llm["models"] == {"agent": "qwen3:32b", "user": "llama3.3:latest"}
    assert full.llm["think"] == {"agent": False} and full.llm["num_ctx"] == 2048


def test_collection_file_refuses_the_eval_split(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text(MOCK_FILE.replace('["train", "dev"]', '["dev", "eval"]'), encoding="utf-8")
    with pytest.raises(ValueError, match="splits"):
        load_collect_config(path)
    path.write_text("budgets = [3]\n" + MOCK_FILE, encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_collect_config(path)


def test_eval_personas_stay_guarded(data_dir):
    with pytest.raises(EvalSplitError):
        load_split(data_dir, "eval")


# ---------------------------------------------------------------- the prompt is production's


def test_update_context_hits_the_llm_updaters_cache_entry(workspace):
    root, config = workspace
    schema = load_schema(config.schema_path)
    client = make_client(config, root / "raw")
    try:
        for slot_name, question in [
            ("climate", Question("climate")),  # categorical: with the no-preference note
            ("budget", Question("budget", "yesno", ("low",))),  # ordinal: without it
        ]:
            slot = schema.slot(slot_name)
            text, answer = "Any thoughts?", "Somewhere warm, and cheap please."
            LLMUpdater(client).update(Belief.prior(schema), question, text, answer)
            with client.track() as tally:
                client.call("update", update_context(slot, question, text, answer), seed=0)
            assert tally["update"].cache_hits == 1 and tally["update"].attempts == 0
    finally:
        client.close()


# ---------------------------------------------------------------- the mock chain


def test_mock_chain_backstories_collect_judge_sft(workspace):
    root, config = workspace
    raw = root / "data" / "raw"

    with pytest.raises(MissingBackstoriesError):
        collect(config, raw)

    flagged, llm = make_train_backstories(config, raw, workers=4)
    assert flagged == [] and llm["roles"]["backstory"]["calls"] == 160
    stories = read_jsonl(backstories_path(config))
    assert len(stories) == 160 and all(s["backstory"] for s in stories)
    assert all(p.backstory for p in load_split(root / "data", "train"))

    result = collect(config, raw, workers=4)
    assert result["problems"] == []
    for split, n in (("train", 2), ("dev", 2)):
        rows = read_jsonl(raw / f"update_{split}.jsonl")
        assert len(rows) == n * 45 * 2 == result["splits"][split]["records"]
        assert {r["split"] for r in rows} == {split}
        assert Counter(r["question"]["kind"] for r in rows) == {
            "open": n * 9 * 2, "choice": n * 18 * 2, "yesno": n * 18 * 2
        }  # fmt: skip
        assert all(r["status"] == OK and r["teacher"] == "mock" for r in rows)
        roles = [[m["role"] for m in r["messages"]] for r in rows]
        assert all(rs == ["system", "user"] for rs in roles)
        assert all(r["answer"] in r["messages"][1]["content"] for r in rows)

    # A rerun reads everything from the reply cache and writes the same records.
    before = (raw / "update_train.jsonl").read_bytes()
    again = collect(config, raw, workers=4)
    assert (raw / "update_train.jsonl").read_bytes() == before
    assert all(s["attempts"] == 0 for s in again["llm"]["roles"].values())
    assert (raw / CACHE).exists()

    judged = judge_collect(config, raw, workers=4)
    assert judged["problems"] == []
    rows = read_jsonl(raw / "judge_train.jsonl")
    assert len(rows) == 2 * 25
    assert all(r["status"] == OK and 0 <= r["score"] <= 10 for r in rows)
    assert all("- climate: " in r["messages"][1]["content"] for r in rows)

    schema = load_schema(config.schema_path)
    sft = root / "data" / "sft"
    manifest = build_sft(raw, sft, schema)
    assert set(manifest["files"]) == {
        "update_train.jsonl.gz", "update_val.jsonl.gz", "judge_train.jsonl.gz",
        "judge_val.jsonl.gz",
    }  # fmt: skip
    assert manifest["teachers"] == {"update": ["mock"], "judge": ["mock"]}
    info = manifest["files"]["update_train.jsonl.gz"]
    assert info["rows"] + info["dropped"]["duplicate"] == info["records"] == 180

    samples = read_sft(sft / "update_train.jsonl.gz")
    assert len(samples) == info["rows"]
    prompts = [json.dumps(s["messages"][:2]) for s in samples]
    assert len(set(prompts)) == len(prompts)
    sample = samples[0]
    assert [m["role"] for m in sample["messages"]] == ["system", "user", "assistant"]
    slot = schema.slot(sample["meta"]["slot"])
    reply = json.loads(sample["messages"][2]["content"])
    assert list(reply) == ["evidence", "scores"]
    assert list(reply["scores"]) == list(slot.value_names)  # canonical key order
    update_model(slot).model_validate(reply)
    assert " " not in sample["messages"][2]["content"].split('"scores"')[1]  # compact JSON

    judge_sample = read_sft(sft / "judge_val.jsonl.gz")[0]
    assert list(json.loads(judge_sample["messages"][2]["content"])) == ["reason", "score"]

    # Rebuilding gives the same bytes (no timestamp in the gzip header).
    first = {name: (sft / name).read_bytes() for name in manifest["files"]}
    assert build_sft(raw, sft, schema) == manifest
    assert {name: (sft / name).read_bytes() for name in manifest["files"]} == first
    assert json.loads((sft / MANIFEST).read_text(encoding="utf-8")) == manifest


def test_build_sft_drops_failed_and_empty_records(tmp_path, data_dir):
    schema = load_schema(data_dir / "travel" / "schema.toml")
    base = {"persona": "t001", "style": "direct", "slot": "pace", "answer_index": 0,
            "question": {"slot": "pace", "kind": "open"}, "teacher": "mock"}  # fmt: skip
    ok = {**base, "status": "ok", "evidence": "slow", "messages": [
        {"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        "scores": {"relaxed": 9, "balanced": 3, "packed": 1, "no_preference": 2}}  # fmt: skip
    rows = [
        ok,
        {**ok, "answer_index": 1},  # the same prompt again
        {**base, "status": "failed", "messages": ok["messages"]},
        {**base, "status": "empty_answer"},
    ]
    for split in ("train", "dev"):
        (tmp_path / f"update_{split}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )
    manifest = build_sft(tmp_path, tmp_path / "sft", schema, roles=("update",))
    info = manifest["files"]["update_train.jsonl.gz"]
    assert info["rows"] == 1
    assert info["dropped"] == {"failed": 1, "empty_answer": 1, "duplicate": 1}
    with pytest.raises(FileNotFoundError):
        build_sft(tmp_path, tmp_path / "sft", schema)  # no judge records


# ---------------------------------------------------------------- judge units


def test_judge_items_are_stratified_and_filled_from_neighbours(data_dir):
    schema = load_schema(data_dir / "travel" / "schema.toml")
    catalog = load_catalog(data_dir / "travel" / "catalog.jsonl")
    edges = (0.5, 0.6, 0.7, 0.8)
    from elicitor.recommender import utility

    short_top = 0
    for persona in load_split(data_dir, "train")[:40]:
        items = judge_items(schema, catalog, persona, edges, 5, seed=1)
        assert len(items) == len(set(items)) == 25
        assert items == judge_items(schema, catalog, persona, edges, 5, seed=1)
        bins = [utility_bin(utility(schema, catalog[i].attrs, persona.truth), edges) for i in items]
        available = Counter(
            utility_bin(utility(schema, item.attrs, persona.truth), edges) for item in catalog
        )
        got = Counter(bins)
        for b in range(5):
            assert got[b] >= min(5, available[b])  # every bin gives what it has, up to 5
        if available[4] < 5 and all(available[b] >= 5 for b in range(4)):
            short_top += 1
            assert got[4] == available[4]
            # The gap is filled from the nearest bin first: bin 3 until it runs out, then 2...
            assert got[3] == min(available[3], 10 - available[4])
            if got[3] < 10 - available[4]:
                assert got[2] > 5
            else:
                assert got[2] == got[1] == got[0] == 5
    assert short_top > 0  # the case occurs in the real data


def test_utility_edges_belong_to_the_bin_above():
    edges = (0.5, 0.6, 0.7, 0.8)
    assert utility_bin(0.49, edges) == 0
    assert utility_bin(0.5, edges) == 1
    assert utility_bin(0.7 - 1e-12, edges) == 3  # float error at an edge
    assert utility_bin(1.0, edges) == 4


# ---------------------------------------------------------------- CLI


def test_cli_collection_commands_on_the_mock_file(tmp_path, capsys):
    out = tmp_path / "raw"
    mock = str(EXPERIMENTS / "collect_mock.toml")
    assert main(["collect", mock, "--out", str(out), "--workers", "2"]) == 0
    assert main(["judge-collect", mock, "--out", str(out), "--workers", "2"]) == 0
    text = capsys.readouterr().out
    assert text.count("health: OK") == 2
    assert len(read_jsonl(out / "update_dev.jsonl")) == 4 * 90
    assert len(read_jsonl(out / "judge_dev.jsonl")) == 4 * 25
    stats = json.loads((out / "update_stats.json").read_text(encoding="utf-8"))
    assert stats["problems"] == [] and stats["config"].startswith("# Mock collection")
    # build-sft needs the training records too.
    assert main(["build-sft", "--raw", str(out), "--out", str(tmp_path / "sft")]) == 2


def test_cli_collect_without_backstories_stops(workspace, capsys):
    root, _ = workspace
    path = root / "experiments" / "mock.toml"
    assert main(["collect", str(path), "--out", str(root / "raw")]) == 2
    assert "make-backstories" in capsys.readouterr().err
    assert main(["make-backstories", str(path), "--out", str(root / "raw")]) == 0
    assert main(["make-backstories", str(path), "--out", str(root / "raw")]) == 2  # frozen
