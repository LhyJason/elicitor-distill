"""The M3 inputs (SPEC §7.1, §7.3): test sets from Elicitor's runs, and the per-model files.

`distill build-test` reads Elicitor's run folders, never this project's data: the `update` test
set is every turn of `m5_full` read by the LLM updater (all arms, replicates and budgets), as an
SFT sample with the messages the teacher received, de-duplicated by prompt as in `build-sft`;
the `judge` test set is the 256 units of `m7_judge` with the user judge's list-presentation
score. Both are eval personas, so the command needs `--allow-eval`, and every sample's meta
says `split: eval`, which `eval-offline` refuses to read without the flag.

`distill m3-file` writes one model's experiment file from a template in `experiments/`: the
template's name, agent model and data folder replaced, and, with `--calibration`, every
selector's rho replaced by the LLM updater's rho the model measured on the dev personas
(rounded to two decimals, as the teacher's 0.99 / 0.97 / 0.85 were). The written file is read
back with Elicitor's loader before it is kept.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from elicitor.catalog import load_catalog
from elicitor.experiments.config import Arm, load_config
from elicitor.experiments.store import RunStore, cell_id
from elicitor.jsonl import dumps_jsonl, read_jsonl
from elicitor.judge import item_lines
from elicitor.llm.roles import JudgeReply, build_roles
from elicitor.personas import EVAL
from elicitor.questions import Question
from elicitor.recommender import utility
from elicitor.schema import Schema
from elicitor.simulator.llm import profile_lines

from elicitor_distill.collect import update_context
from elicitor_distill.data import DOMAIN
from elicitor_distill.judge_collect import utility_bin
from elicitor_distill.sft_data import gzip_bytes, update_reply
from elicitor_distill.splits import load_split

TEST_DIR = "results/m3/test"
JUDGE_EDGES = (0.5, 0.6, 0.7, 0.8)  # the bins of the judge training data (experiments/collect)
JUDGE_CONDITION = ("user", "list")  # Elicitor's main judge, the teacher of the judge role


def _sample(prompt: list[dict[str, str]], reply: str, meta: dict[str, Any]) -> dict[str, Any]:
    return {"messages": [*prompt, {"role": "assistant", "content": reply}], "meta": meta}


def update_test(
    run_dir: str | Path, schema: Schema, prompts_dir: Path, styles: dict[str, str]
) -> list[dict[str, Any]]:
    store = RunStore(run_dir)
    state = store.read_state()
    if state is None or state["split"] != EVAL:
        raise ValueError(f"{run_dir} is not a run on the eval personas")
    render = build_roles(prompts_dir)["update"].prompt.render
    samples, seen = [], set()
    for arm in (a for a in state["arms"] if Arm.parse(a).updater == "llm"):
        for r in range(state["replicates"]):
            if not store.is_done(cell_id(arm, r)):
                raise ValueError(f"{run_dir}: cell {cell_id(arm, r)} is not finished")
            for episode in store.read_cell(cell_id(arm, r)):
                for turn in episode["turns"]:
                    parsed = turn["parsed"]
                    if "scores" not in parsed:  # an empty answer or a failed call
                        continue
                    q = turn["question"]
                    question = Question(q["slot"], q["kind"], tuple(q.get("values", ())))
                    slot = schema.slot(question.slot)
                    prompt = render(update_context(slot, question, turn["text"], turn["answer"]))
                    key = json.dumps(prompt, ensure_ascii=False)
                    if key in seen:
                        continue
                    seen.add(key)
                    reply = update_reply({"slot": slot.name, **parsed}, schema)
                    meta = {"persona": episode["persona"], "style": styles[episode["persona"]],
                            "slot": slot.name, "kind": question.kind, "split": EVAL}  # fmt: skip
                    samples.append(_sample(prompt, reply, meta))
    return samples


def judge_test(
    judge_dir: str | Path, schema: Schema, domain_dir: Path, personas: dict[str, Any]
) -> list[dict[str, Any]]:
    judge_dir = Path(judge_dir)
    catalog = {item.id: item for item in load_catalog(domain_dir / "catalog.jsonl")}
    render = build_roles(domain_dir / "prompts")["judge"].prompt.render
    scores = {
        (j["persona"], j["item"]): j
        for j in read_jsonl(judge_dir / "judgments.jsonl")
        if j["kind"] == "score" and (j["judge"], j["presentation"]) == JUDGE_CONDITION
    }
    samples = []
    for unit in read_jsonl(judge_dir / "units.jsonl"):
        judged = scores.get((unit["persona"], unit["item"]))
        if judged is None or judged["failed"]:
            continue
        persona, item = personas[unit["persona"]], catalog[unit["item"]]
        u = utility(schema, item.attrs, persona.truth)
        context = {"profile": profile_lines(schema, persona),
                   "item": item_lines(schema, item.attrs)}  # fmt: skip
        reply = JudgeReply(reason=judged["reason"], score=judged["score"]).model_dump_json()
        meta = {"persona": persona.id, "style": persona.style, "item": item.id,
                "utility": round(u, 6), "bin": utility_bin(u, JUDGE_EDGES),
                "split": EVAL}  # fmt: skip
        samples.append(_sample(render(context), reply, meta))
    return samples


def build_test(
    m5_dir: str | Path, judge_dir: str | Path, data_dir: str | Path, schema: Schema,
    out_dir: str | Path, allow_eval: bool = False,
) -> dict[str, int]:  # fmt: skip
    """Write `update_test.jsonl.gz` and `judge_test.jsonl.gz`; returns their row counts."""
    domain_dir = Path(data_dir) / DOMAIN
    personas = {p.id: p for p in load_split(data_dir, EVAL, allow_eval=allow_eval)}
    styles = {pid: p.style for pid, p in personas.items()}
    sets = {
        "update": update_test(m5_dir, schema, domain_dir / "prompts", styles),
        "judge": judge_test(judge_dir, schema, domain_dir, personas),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for role, samples in sets.items():
        (out / f"{role}_test.jsonl.gz").write_bytes(gzip_bytes(dumps_jsonl(samples).encode()))
    return {role: len(samples) for role, samples in sets.items()}


def measured_rho(calibration: str | Path) -> dict[str, float]:
    result = json.loads(Path(calibration).read_text(encoding="utf-8"))
    return {kind: round(r["rho"], 2) for kind, r in result["rho"]["llm"].items()}


def _set(text: str, key: str, value: str) -> str:
    """Replace every `key = ...` line's value; the template must have one."""
    text, n = re.subn(rf"(?m)^{re.escape(key)} = .*$", lambda _: f"{key} = {value}", text)
    if not n:
        raise ValueError(f"the template has no `{key} = ...` line")
    return text


def write_m3_file(
    template: str | Path, out: str | Path, name: str, model: str, calibration: str | Path | None
) -> dict[str, Any]:
    template, out = Path(template), Path(out)
    text = template.read_text(encoding="utf-8")
    data_dir = load_config(template).data_dir  # resolved against the template's folder
    out.parent.mkdir(parents=True, exist_ok=True)
    relative = Path(os.path.relpath(data_dir, out.parent.resolve())).as_posix()
    text = _set(_set(_set(text, "name", json.dumps(name)), "agent", json.dumps(model)),
                "data_dir", json.dumps(relative))  # fmt: skip
    rho = measured_rho(calibration) if calibration else None
    if rho:
        text = _set(text, "rho", "{ " + ", ".join(f"{k} = {v}" for k, v in rho.items()) + " }")
    header = f"# Written by `distill m3-file` from {template.name}; model {model}.\n"
    out.write_text(header + text, encoding="utf-8", newline="\n")
    config = load_config(out)  # the file must load, with the values asked for
    assert config.llm["models"]["agent"] == model and config.name == name
    assert rho is None or all(p.get("rho") == rho for p in config.selector_params.values())
    return {"file": out.as_posix(), "model": model, "rho": rho}
