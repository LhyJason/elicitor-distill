"""The `imagine` role (SPEC §7.5, Decision Q15): `distill imagine-collect` and `imagine-eval`.

`eig_mc` asks the agent model to imagine the answer of a user whose preference on the asked slot
is a given value. Its prompt holds no persona and no belief, so its inputs are the 465 pairs of a
candidate question and a value of its slot. The collection asks the teacher for `imagine_train`
samples of every pair (the `train` set) and `imagine_val` more (`dev`, which `build-sft` turns
into `imagine_val`), each at a seed of its own, with the context built exactly as Elicitor's
`eig_mc` builds it (a test checks the cache entry). Outputs: `imagine_<split>.jsonl` and
`imagine_stats.json`.

The offline evaluation lets an imaginer (a served model, or the teacher's stored replies) answer
every sample at its stored seed, then a fixed reader reads each answer with the `update` role.
Per answer the *mass on the imagined value* is the reader's normalised likelihood of that value,
the posterior from a uniform belief; an empty or failed answer, which `eig_mc` hands the updater
as no answer, leaves the uniform 1/n.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from elicitor.experiments.health import role_problems
from elicitor.experiments.store import write_text_atomic
from elicitor.jsonl import write_jsonl
from elicitor.llm import LLMClient, LLMParseError
from elicitor.llm.roles import SimulateReply, update_scores
from elicitor.metrics import leaked_slots
from elicitor.personas import DEV
from elicitor.phrasing import template_question
from elicitor.questions import Question, candidates
from elicitor.rng import derive_seed
from elicitor.schema import Schema, Slot, load_schema
from elicitor.updaters.llm import likelihood

from elicitor_distill.collect import (
    EMPTY,
    FAILED,
    OK,
    RAW_DIR,
    UPDATE_SEED,
    Progress,
    make_client,
    run_tracked,
    update_context,
    write_stats,
)
from elicitor_distill.config import CollectConfig
from elicitor_distill.splits import TRAIN

TEACHER = "teacher"  # the imaginer whose answers are the stored replies


def imagine_context(slot: Slot, question: Question, value: str) -> dict[str, Any]:
    """The `imagine` role's context, as Elicitor's `eig_mc` builds it."""
    return {
        "slot": slot,
        "q": question,
        "value": value,
        "topic": slot.description.rstrip("."),
        "preference": slot.value(value).label,
        "question": template_question(slot, question),
    }


def imagine_pairs(schema: Schema) -> list[tuple[Slot, int, Question, str]]:
    """(slot, index of the question among the slot's candidates, question, value)."""
    return [(slot, qi, q, v) for slot in schema.slots
            for qi, q in enumerate(candidates(slot)) for v in slot.value_names]  # fmt: skip


def sample_question(schema: Schema, meta: dict[str, Any]) -> tuple[Slot, Question]:
    slot = schema.slot(meta["slot"])
    return slot, candidates(slot)[meta["qi"]]


def imagine_one(
    client: LLMClient, schema: Schema, seed: int, split: str,
    slot: Slot, qi: int, question: Question, value: str, j: int,
) -> dict[str, Any]:  # fmt: skip
    key = json.dumps(question.to_dict(), sort_keys=True)
    request_seed = derive_seed(seed, "distill-imagine", split, key, value, j) % 2**31
    context = imagine_context(slot, question, value)
    row: dict[str, Any] = {
        "split": split, "slot": slot.name, "qi": qi, "kind": question.kind,
        "question": question.to_dict(), "value": value, "sample": j, "seed": request_seed,
        "teacher": client.models[client.roles["imagine"].group],
        "messages": client.roles["imagine"].prompt.render(context),
    }  # fmt: skip
    try:
        reply = client.call("imagine", context, seed=request_seed)
    except LLMParseError:
        return {**row, "status": FAILED, "answer": ""}
    answer = reply.answer.strip()  # type: ignore[attr-defined]
    return {**row, "status": OK if answer else EMPTY, "answer": answer,
            "leaks": leaked_slots(schema, slot.name, answer)}  # fmt: skip


def collect_imagine(
    config: CollectConfig,
    out_dir: str | Path = RAW_DIR,
    workers: int = 1,
    progress: Progress | None = None,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    schema = load_schema(config.schema_path)
    samples = {TRAIN: config.imagine_train, DEV: config.imagine_val}
    tasks = [(split, *pair, j) for split, n in samples.items()
             for pair in imagine_pairs(schema) for j in range(n)]  # fmt: skip
    client = make_client(config, out)
    start = time.perf_counter()
    try:
        rows, llm = run_tracked(client, lambda t: imagine_one(client, schema, config.seed, *t),
                                tasks, workers, progress, label="imagine records")  # fmt: skip
    finally:
        client.close()
    counts = {}
    for split in samples:
        mine = [r for r in rows if r["split"] == split]
        write_jsonl(out / f"imagine_{split}.jsonl", mine)
        counts[split] = {"records": len(mine),
                         **{s: sum(r["status"] == s for r in mine) for s in (OK, FAILED, EMPTY)},
                         "leaking_answers": sum(bool(r.get("leaks")) for r in mine)}  # fmt: skip
    return write_stats(out / "imagine_stats.json", config,
                       {"seconds": round(time.perf_counter() - start, 1), "splits": counts,
                        "llm": llm})  # fmt: skip


# ---------------------------------------------------------------- offline evaluation


def stored_answer(sample: dict[str, Any]) -> str:
    return SimulateReply.model_validate_json(sample["messages"][-1]["content"]).answer.strip()


def imagine_answer(client: LLMClient, schema: Schema, sample: dict[str, Any]) -> dict[str, Any]:
    slot, question = sample_question(schema, sample["meta"])
    context = imagine_context(slot, question, sample["meta"]["value"])
    try:
        reply = client.call("imagine", context, seed=sample["meta"]["seed"])
    except LLMParseError:
        return {"status": FAILED, "answer": ""}
    answer = reply.answer.strip()  # type: ignore[attr-defined]
    return {"status": OK if answer else EMPTY, "answer": answer}


def read_answer(
    client: LLMClient, schema: Schema, sample: dict[str, Any], answer: str
) -> dict[str, Any]:
    """The reader's mass on the imagined value (uniform when there is nothing to read)."""
    slot, question = sample_question(schema, sample["meta"])
    n = len(slot.value_names)
    if not answer:
        return {"mass": 1 / n, "read": "none"}
    context = update_context(slot, question, template_question(slot, question), answer)
    try:
        reply = client.call("update", context, seed=UPDATE_SEED)
    except LLMParseError:
        return {"mass": 1 / n, "read": FAILED}
    lk = likelihood(update_scores(reply, slot))
    return {"mass": lk[slot.value_names.index(sample["meta"]["value"])] / sum(lk), "read": OK}


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_pair: dict[tuple, set[str]] = defaultdict(set)
    for r in rows:
        by_pair[(r["slot"], r["qi"], r["value"])].add(r["answer"])
    answered = [r for r in rows if r["answer"]]
    return {
        "n": len(rows),
        "failed": sum(r["status"] == FAILED for r in rows),
        "empty": sum(r["status"] == EMPTY for r in rows),
        "mass_on_value": round(sum(r["mass"] for r in rows) / len(rows), 4),
        "leaking": round(sum(bool(r["leaks"]) for r in rows) / len(rows), 4),
        "pairs_differ": round(sum(len(a) > 1 for a in by_pair.values()) / len(by_pair), 4),
        "words": round(sum(len(r["answer"].split()) for r in answered) / max(1, len(answered)), 1),
    }


def evaluate_imagine(
    samples: list[dict[str, Any]],
    schema: Schema,
    imaginer: LLMClient | None,
    reader: LLMClient,
    workers: int = 8,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Answers first (all of them, so a server holding one model swaps once), then reads."""
    llm: dict[str, Any] = {}
    if imaginer is None:
        answers = [{"status": OK, "answer": stored_answer(s)} for s in samples]
    else:
        answers, llm["imagine"] = run_tracked(
            imaginer, lambda s: imagine_answer(imaginer, schema, s), samples, workers, progress,
            label="imagined")  # fmt: skip
    reads, llm["read"] = run_tracked(
        reader, lambda t: read_answer(reader, schema, *t),
        [(s, a["answer"]) for s, a in zip(samples, answers, strict=True)], workers, progress,
        label="read")  # fmt: skip
    rows = [{**{k: s["meta"][k] for k in ("slot", "qi", "kind", "value", "sample")}, **a, **r,
             "leaks": leaked_slots(schema, s["meta"]["slot"], a["answer"])}
            for s, a, r in zip(samples, answers, reads, strict=True)]  # fmt: skip
    by_kind = {
        k: _summary([r for r in rows if r["kind"] == k]) for k in sorted({r["kind"] for r in rows})
    }
    tables = {"overall": _summary(rows), "kind": by_kind}
    problems = [p for stats in llm.values() if stats for p in role_problems(stats["roles"])]
    return {"rows": rows, "tables": tables, "llm": llm, "problems": problems}


def write_imagine_report(out_dir: str | Path, result: dict[str, Any], provenance: dict) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "rows.jsonl", result["rows"])
    summary = {**provenance, **{k: v for k, v in result.items() if k != "rows"}}
    write_text_atomic(out / "summary.json", json.dumps(summary, indent=2) + "\n")


def pair_means(rows: list[dict[str, Any]]) -> dict[tuple, float]:
    """Mass on the imagined value, averaged per (slot, question, value) pair."""
    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        groups[(r["slot"], r["qi"], r["value"])].append(r["mass"])
    return {k: sum(v) / len(v) for k, v in groups.items()}
