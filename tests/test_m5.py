import json
import shutil
from pathlib import Path

import pytest
from elicitor.belief import Belief
from elicitor.experiments.config import load_config
from elicitor.experiments.runner import run_experiment
from elicitor.jsonl import read_jsonl
from elicitor.questions import Question
from elicitor.rng import derive_seed
from elicitor.schema import load_schema
from elicitor.selectors.eig_mc import MCEIGSelector

from elicitor_distill.cli import main
from elicitor_distill.collect import make_client
from elicitor_distill.config import load_collect_config
from elicitor_distill.imagine import (
    collect_imagine,
    evaluate_imagine,
    imagine_context,
    imagine_pairs,
    pair_means,
)
from elicitor_distill.m3 import write_m3_file
from elicitor_distill.m5 import EIG, MC
from elicitor_distill.sft_data import MANIFEST, build_sft, read_sft
from elicitor_distill.train import load_train_config

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"


def _mock_collect_file(tmp: Path, train: int = 2, val: int = 1) -> Path:
    path = tmp / "imagine_mock.toml"
    path.write_text(
        f'name = "imagine_mock"\ndomain = "travel"\ndata_dir = "{(ROOT / "data").as_posix()}"\n'
        f"seed = 3\n\n[imagine]\ntrain_samples = {train}\nval_samples = {val}\n\n"
        '[llm]\nprovider = "mock"\n',
        encoding="utf-8",
    )
    return path


def test_imagine_file_is_the_teacher_of_m5_full():
    config = load_collect_config(EXPERIMENTS / "imagine.toml")
    collect = load_collect_config(EXPERIMENTS / "collect.toml")
    assert (config.imagine_train, config.imagine_val, config.seed) == (8, 2, 20261027)
    assert config.llm == collect.llm


def test_there_are_465_question_value_pairs(data_dir):
    pairs = imagine_pairs(load_schema(data_dir / "travel" / "schema.toml"))
    assert len(pairs) == 465 and len({(s.name, qi) for s, qi, _, _ in pairs}) == 105


def test_imagine_context_hits_eig_mcs_cache_entry(tmp_path, data_dir):
    config = load_collect_config(_mock_collect_file(tmp_path))
    schema = load_schema(config.schema_path)
    client = make_client(config, tmp_path)
    try:
        for question, value in [(Question("climate"), "tropical"),
                                (Question("budget", "yesno", ("low",)), "high")]:  # fmt: skip
            MCEIGSelector(client, seed=8001).imagine(Belief.prior(schema), question, value, 2)
            key = json.dumps(question.to_dict(), sort_keys=True)
            seed = derive_seed(8001, "imagine", key, value, 2) % 2**31
            context = imagine_context(schema.slot(question.slot), question, value)
            with client.track() as tally:
                client.call("imagine", context, seed=seed)
            assert tally["imagine"].cache_hits == 1 and tally["imagine"].attempts == 0
    finally:
        client.close()


@pytest.fixture(scope="module")
def imagine_sets(tmp_path_factory, data_dir):
    """Mock records, built into an SFT folder that already holds the frozen M1 manifest."""
    tmp = tmp_path_factory.mktemp("m5")
    config = load_collect_config(_mock_collect_file(tmp))
    result = collect_imagine(config, tmp / "raw", workers=4)
    sft = tmp / "sft"
    sft.mkdir()
    shutil.copy(data_dir / "sft" / MANIFEST, sft / MANIFEST)
    manifest = build_sft(tmp / "raw", sft, load_schema(config.schema_path), ("imagine",))
    return tmp, result, manifest


def test_mock_collection_keeps_repeated_prompts_and_the_frozen_manifest(imagine_sets, data_dir):
    tmp, result, manifest = imagine_sets
    assert result["problems"] == [] and result["splits"]["train"]["ok"] == 930
    frozen = json.loads((data_dir / "sft" / MANIFEST).read_text(encoding="utf-8"))
    for name, entry in frozen["files"].items():
        assert manifest["files"][name] == entry
    assert manifest["files"]["imagine_train.jsonl.gz"]["rows"] == 930  # 2 per pair, none dropped
    assert manifest["files"]["imagine_val.jsonl.gz"]["rows"] == 465
    train = read_sft(tmp / "sft" / "imagine_train.jsonl.gz")
    prompts = {json.dumps(s["messages"][:2]) for s in train}
    assert len(prompts) == 465  # each prompt twice, at two seeds
    assert len({s["meta"]["seed"] for s in train}) == 930
    raw = read_jsonl(tmp / "raw" / "imagine_train.jsonl")
    assert json.loads(train[0]["messages"][-1]["content"]) == {"answer": raw[0]["answer"]}


def test_imagine_eval_reads_the_stored_and_the_served_answers(imagine_sets, data_dir):
    tmp, _, _ = imagine_sets
    schema = load_schema(data_dir / "travel" / "schema.toml")
    val = read_sft(tmp / "sft" / "imagine_val.jsonl.gz")
    config = load_collect_config(_mock_collect_file(tmp))
    client = make_client(config, tmp / "eval")
    try:
        stored = evaluate_imagine(val, schema, None, client, workers=4)
        served = evaluate_imagine(val, schema, client, client, workers=4)
    finally:
        client.close()
    assert stored["problems"] == [] and set(stored["llm"]) == {"read"}
    # The mock teacher's answers are the mock imaginer's at the same seeds.
    assert [r["answer"] for r in stored["rows"]] == [r["answer"] for r in served["rows"]]
    assert stored["tables"] == served["tables"]
    overall = stored["tables"]["overall"]
    assert overall["n"] == 465 and 0.25 < overall["mass_on_value"] <= 1
    assert overall["pairs_differ"] == 0  # one sample per pair
    assert set(stored["tables"]["kind"]) == {"open", "choice", "yesno"}
    assert len(pair_means(stored["rows"])) == 465


def test_imagine_eval_cli_on_the_mock_provider(imagine_sets, tmp_path):
    tmp, _, _ = imagine_sets
    args = ["imagine-eval", "--provider", "mock", "--reader", "mock", "--workers", "2",
            "--data", str(tmp / "sft" / "imagine_val.jsonl.gz"), "--limit", "20"]  # fmt: skip
    assert main([*args, "--imaginer", "teacher", "--out", str(tmp_path / "t")]) == 0
    assert main([*args, "--imaginer", "mock", "--strict", "--out", str(tmp_path / "s")]) == 0
    summary = json.loads((tmp_path / "s" / "summary.json").read_text(encoding="utf-8"))
    assert summary["imaginer"] == "mock" and summary["tables"]["overall"]["n"] == 20


def test_agent_config_trains_on_both_roles():
    config = load_train_config(ROOT / "configs" / "agent_1.7b.toml")
    update = load_train_config(ROOT / "configs" / "update_1.7b.toml")
    assert [p.name for p in config.train] == ["update_train.jsonl.gz", "imagine_train.jsonl.gz"]
    assert config.val == update.val and config.role == "update"
    for key in ("base", "epochs", "learning_rate", "per_device_batch", "grad_accum", "lora_r",
                "max_new_tokens", "gradient_checkpointing"):  # fmt: skip
        assert getattr(config, key) == getattr(update, key)


def test_m5_template_is_m5_full_cut_to_two_arms_at_k5(tmp_path):
    config = load_config(EXPERIMENTS / "m5_e2e.toml")
    assert [a.name for a in config.arms] == [EIG, MC] and list(config.budgets) == [5]
    assert (config.seed, config.replicates, config.split) == (8000, 5, "eval")
    assert config.selector_params["eig_mc"] == {
        "rho": {"open": 0.99, "choice": 0.97, "yesno": 0.85},
        "shortlist": 5,
        "samples": 3,
    }
    calibration = tmp_path / "calibration.json"
    rho = {"open": {"rho": 0.981}, "choice": {"rho": 0.962}, "yesno": {"rho": 0.84}}
    calibration.write_text(json.dumps({"rho": {"llm": rho}}))
    out = tmp_path / "e2e.toml"
    write_m3_file(EXPERIMENTS / "m5_e2e.toml", out, "e2e", "elicitor-agent-1.7b-sft", calibration)
    params = load_config(out).selector_params
    assert (
        params["eig"]["rho"]
        == params["eig_mc"]["rho"]
        == {"open": 0.98, "choice": 0.96, "yesno": 0.84}
    )
    assert params["eig_mc"]["shortlist"] == 5


def test_m5_report_on_mock_runs(tmp_path, imagine_sets):
    dirs = {}
    for name, model in (("teacher", "qwen3:32b"), ("r1", "elicitor-agent-1.7b-sft")):
        path = tmp_path / "experiments" / f"{name}.toml"
        write_m3_file(EXPERIMENTS / "m5_e2e.toml", path, name, model, None)
        text = path.read_text(encoding="utf-8").replace('provider = "ollama"', 'provider = "mock"')
        path.write_text(text.replace("replicates = 5", "replicates = 2"), encoding="utf-8")
        dirs[name] = tmp_path / "results" / name
        run_experiment(load_config(path), dirs[name], max_personas=3)
    imagine = imagine_sets[0] / "sft" / "imagine_val.jsonl.gz"
    for name in ("teacher", "mock"):
        assert (
            main(
                [
                    "imagine-eval",
                    "--provider",
                    "mock",
                    "--reader",
                    "mock",
                    "--imaginer",
                    name,
                    "--data",
                    str(imagine),
                    "--out",
                    str(tmp_path / f"imagine_{name}"),
                ]
            )
            == 0
        )
    for name, agreement in (("agent", 0.978), ("update", 0.980)):
        (tmp_path / f"{name}.json").write_text(
            json.dumps({"tables": {"overall": {"agreement": agreement}}})
        )
    assert main(["m5-report", "--teacher", str(dirs["teacher"]), "--r1", str(dirs["r1"]),
                 "--r2", str(tmp_path / "missing"),
                 "--imagine-student", str(tmp_path / "imagine_mock"),
                 "--imagine-teacher", str(tmp_path / "imagine_teacher"),
                 "--agent-test", str(tmp_path / "agent.json"),
                 "--update-test", str(tmp_path / "update.json"),
                 "--out", str(tmp_path / "m5")]) == 0  # fmt: skip
    text = (tmp_path / "m5" / "verdicts.md").read_text(encoding="utf-8")
    # Identical runs on the mock: E1 and E4 hold, E5 is exactly 0, R2 never ran.
    assert "| E1 | **holds** |" in text and "| E4 | **holds** |" in text
    assert "| E3 | not tested | R2 not finished |" in text
    assert "| E5 | **holds** | +0.0000, lower +0.0000 over 465 pairs |" in text
    assert (tmp_path / "m5" / "compare" / "curve_mass_on_truth.png").exists()
    contrasts = (tmp_path / "m5" / "contrasts.csv").read_text(encoding="utf-8")
    assert contrasts.count("\nE1,") == 6 and "\nE3," not in contrasts
