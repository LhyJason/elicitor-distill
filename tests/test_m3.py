import json
from pathlib import Path

import numpy as np
import pytest
from elicitor.experiments.config import load_config
from elicitor.experiments.runner import run_experiment
from elicitor.jsonl import write_jsonl
from elicitor.personas import load_personas
from elicitor.schema import load_schema

from elicitor_distill.cli import main
from elicitor_distill.compare import (
    MARGINS,
    METRICS,
    NI_ARM,
    NI_BUDGET,
    compare,
    comparison_rows,
    one_sided_lower,
)
from elicitor_distill.eval_offline import teacher_values
from elicitor_distill.m3 import build_test, measured_rho, write_m3_file
from elicitor_distill.sft_data import read_sft
from elicitor_distill.splits import EvalSplitError

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "experiments"


def _mock_file(tmp: Path, name: str, model: str) -> Path:
    """The end-to-end template for one model, on Elicitor's mock provider."""
    out = tmp / "experiments" / f"{name}.toml"
    write_m3_file(TEMPLATES / "m3_e2e.toml", out, name, model, None)
    text = out.read_text(encoding="utf-8").replace('provider = "ollama"', 'provider = "mock"')
    out.write_text(text, encoding="utf-8")
    return out


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """A 'teacher' and a 'student' run of the two arms on 2 eval personas, mock provider."""
    tmp = tmp_path_factory.mktemp("m3")
    dirs = {}
    for name, model in (("teacher", "qwen3:32b"), ("student", "elicitor-upd-1.7b-sft")):
        dirs[name] = tmp / "results" / name
        run_experiment(load_config(_mock_file(tmp, name, model)), dirs[name], max_personas=2)
    return tmp, dirs


def test_m3_file_sets_model_rho_and_data_folder(tmp_path):
    calibration = tmp_path / "calibration.json"
    rho = {"open": {"rho": 0.9879}, "choice": {"rho": 0.9739}, "yesno": {"rho": 0.8472}}
    calibration.write_text(json.dumps({"rho": {"rule": {}, "llm": rho}}))
    assert measured_rho(calibration) == {"open": 0.99, "choice": 0.97, "yesno": 0.85}
    out = tmp_path / "m3" / "e2e_x.toml"
    info = write_m3_file(TEMPLATES / "m3_e2e.toml", out, "e2e_x", "elicitor-x", calibration)
    config = load_config(out)
    assert info["rho"] == {"open": 0.99, "choice": 0.97, "yesno": 0.85}
    assert config.llm["models"] == {"agent": "elicitor-x", "user": "llama3.3:latest"}
    assert config.data_dir == (ROOT / "data").resolve()
    assert config.selector_params["eig"]["rho"] == info["rho"]
    assert [a.name for a in config.arms] == [
        "sel=eig,upd=llm,phr=tpl",
        "sel=random,upd=llm,phr=tpl",
    ]


def test_calibrate_template_is_m5_smoke_but_the_agent():
    config = load_config(TEMPLATES / "m3_calibrate.toml")
    assert (config.seed, config.split, config.simulator) == (7000, "dev", {"kind": "llm"})
    assert config.llm["models"] == {"agent": "qwen3:32b", "user": "llama3.3:latest"}


def test_build_test_from_a_run_and_a_judge_folder(runs, data_dir):
    tmp, dirs = runs
    schema = load_schema(data_dir / "travel" / "schema.toml")
    personas = load_personas(data_dir / "travel" / "personas.jsonl", split="eval")[:2]
    judge = tmp / "m7_judge"
    units = [{"persona": p.id, "item": item} for p in personas for item in ("i001", "i002")]
    write_jsonl(judge / "units.jsonl", units)
    scores = [{"kind": "score", "judge": "user", "presentation": "list", "failed": False,
               "score": 3 + i, "reason": "ok", **u} for i, u in enumerate(units)]  # fmt: skip
    other = {**scores[0], "presentation": "plan_user", "score": 9}  # not the teacher's condition
    write_jsonl(judge / "judgments.jsonl", [*scores, other])

    with pytest.raises(EvalSplitError):
        build_test(dirs["teacher"], judge, data_dir, schema, tmp / "test")
    counts = build_test(dirs["teacher"], judge, data_dir, schema, tmp / "test", allow_eval=True)
    assert counts["judge"] == 4 and counts["update"] > 0
    update = read_sft(tmp / "test" / "update_test.jsonl.gz")
    prompts = [json.dumps(s["messages"][:2]) for s in update]
    assert len(set(prompts)) == len(prompts)  # de-duplicated
    assert {s["meta"]["split"] for s in update} == {"eval"}
    episodes = [json.loads(line) for line in (dirs["teacher"] / "episodes.jsonl").open()]
    turn = episodes[0]["turns"][0]
    first = next(s for s in update if turn["answer"] in s["messages"][1]["content"])
    scores_back = teacher_values("update", first, schema)
    assert scores_back == list(turn["parsed"]["scores"].values())
    judge_set = read_sft(tmp / "test" / "judge_test.jsonl.gz")
    assert [teacher_values("judge", s, schema) for s in judge_set] == [3, 4, 5, 6]


def test_eval_offline_and_judge_eval_need_allow_eval(runs, tmp_path):
    tmp, _ = runs
    test = tmp / "test"
    if not (test / "judge_test.jsonl.gz").exists():
        pytest.skip("needs test_build_test_from_a_run_and_a_judge_folder")
    args = ["eval-offline", "--provider", "mock", "--model", "mock", "--role", "judge", "--data",
            str(test / "judge_test.jsonl.gz"), "--no-unconstrained", "--workers", "1"]  # fmt: skip
    with pytest.raises(EvalSplitError):
        main(args)
    assert main([*args, "--allow-eval", "--out", str(tmp_path / "judge")]) == 0
    assert main(["judge-eval", str(tmp_path / "judge"), "--out", str(tmp_path / "d5.md")]) == 0
    text = (tmp_path / "d5.md").read_text(encoding="utf-8")
    assert "| teacher |" in text and "| mock |" in text


def test_identical_student_is_non_inferior_with_the_same_questions(runs):
    tmp, dirs = runs
    result = compare(dirs["teacher"], [dirs["student"]], tmp / "compare", resamples=200)
    ni = [r for r in result["rows"] if (r["arm"], r["budget"]) == (NI_ARM, NI_BUDGET)]
    assert {r["metric"] for r in ni} >= set(MARGINS)
    assert all(r["diff"] == 0 and r["p_holm"] == 1 for r in ni)
    assert all(r["non_inferior"] for r in ni if r["margin"] is not None)
    assert all(q["same_questions"] == 1 for q in result["questions"])
    for name in ("comparisons.csv", "curves.csv", "questions.csv", "results.md",
                 "curve_mass_on_truth.png"):  # fmt: skip
        assert (tmp / "compare" / name).exists()


def _episode(persona: str, replicate: int, mass: float) -> dict:
    metrics = {**dict.fromkeys(METRICS, 0.5), "mass_on_truth": mass}
    return {"persona": persona, "arm": NI_ARM, "budget": NI_BUDGET, "replicate": replicate,
            "final_metrics": metrics, "initial_metrics": metrics, "turns": [], "n_turns": 0,
            "stop_reason": "budget"}  # fmt: skip


def test_non_inferiority_reads_the_one_sided_bound():
    teacher = [_episode(f"p{i}", r, 0.5 + 0.01 * i) for i in range(12) for r in range(2)]
    worse = [_episode(f"p{i}", r, 0.45 + 0.01 * i) for i in range(12) for r in range(2)]
    near = [
        _episode(f"p{i}", r, 0.5 + 0.01 * i - 0.01 * (i % 2)) for i in range(12) for r in range(2)
    ]
    rows = comparison_rows(teacher, {"worse": worse, "near": near}, resamples=500)
    mass = {r["model"]: r for r in rows if r["metric"] == "mass_on_truth"}
    assert mass["worse"]["diff"] == pytest.approx(-0.05) and not mass["worse"]["non_inferior"]
    assert mass["near"]["diff"] == pytest.approx(-0.005) and mass["near"]["non_inferior"]
    assert mass["near"]["lower_95"] < mass["near"]["diff"]
    diffs = np.array([-0.01, 0.0] * 6)
    assert one_sided_lower(diffs, 500) == pytest.approx(np.percentile(
        diffs[np.random.default_rng(0).integers(0, 12, (500, 12))].mean(axis=1), 5))  # fmt: skip
