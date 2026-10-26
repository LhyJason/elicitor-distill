"""Teacher data for the `update` role (SPEC §4, Decision Q7): `distill collect`.

For every persona of the collected splits: Elicitor's calibration questions (per slot the open
question, two choice questions and two yes/no questions) in template wording, each answered
`answer_seeds` times by Elicitor's LLM simulator, and every answer read by the teacher through
the `update` role with seed 0, as in production. The prompt context is built exactly as
Elicitor's LLM updater builds it (a test checks that both hit the same cache entry), and the
rendered messages are stored with each record, so the SFT prompt is the production prompt.

Training personas need backstories first (`distill make-backstories`): the simulator puts them in
its prompt, as it does for Elicitor's personas. They are written once by the simulator model with
Elicitor's own procedure, into `data/backstories.jsonl`, where Elicitor's persona loader merges
them into `personas_train.jsonl`, and then frozen.

Outputs in the output folder: `update_<split>.jsonl` (one record per question and answer seed)
and `update_stats.json`. Replies are cached in `llm_cache.sqlite`, so an interrupted job resumes
for free. The health bar is Elicitor's: no failed call, no transport error, parse failures below
1 % in every role.
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from elicitor import __version__ as elicitor_version
from elicitor.backstories import write_backstories
from elicitor.calibration import calibration_questions
from elicitor.experiments.health import role_problems
from elicitor.experiments.store import sum_stats, write_text_atomic
from elicitor.jsonl import write_jsonl
from elicitor.llm import LLMClient, LLMParseError, build_client
from elicitor.llm.roles import update_scores, value_lines
from elicitor.metrics import leaked_slots
from elicitor.personas import BACKSTORIES, Persona
from elicitor.phrasing import template_question
from elicitor.questions import Question
from elicitor.rng import derive_rng
from elicitor.schema import CATEGORICAL, Schema, Slot, load_schema
from elicitor.simulator.llm import LLMSimulator, load_styles
from elicitor.updaters.llm import _NO_PREFERENCE_NOTE

from elicitor_distill import __version__
from elicitor_distill.config import CollectConfig
from elicitor_distill.data import ELICITOR_COMMIT
from elicitor_distill.splits import TRAIN, load_split

RAW_DIR = "data/raw"
CACHE = "llm_cache.sqlite"
UPDATE_SEED = 0  # the seed Elicitor's LLM updater sends
OK, FAILED, EMPTY = "ok", "failed", "empty_answer"  # record status
_PROGRESS_STEPS = 20

Progress = Callable[[str], None]


class MissingBackstoriesError(RuntimeError):
    """Training personas have no backstories yet."""


# ---------------------------------------------------------------- shared helpers


def make_client(config: CollectConfig, out_dir: str | Path) -> LLMClient:
    return build_client(config.llm, config.prompts_dir, Path(out_dir) / CACHE)


def split_personas(config: CollectConfig, split: str) -> list[Persona]:
    """The split's personas (the guard refuses the eval split), cut to `max_personas`."""
    personas = load_split(config.data_dir, split, config.domain)
    return personas[: config.max_personas] if config.max_personas else personas


def run_tracked(
    client: LLMClient,
    fn: Callable[[Any], dict[str, Any]],
    tasks: Sequence[Any],
    workers: int,
    progress: Progress | None = None,
    label: str = "tasks",
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """fn over the tasks in a thread pool, in task order; each task's LLM calls are counted in a
    tally of its own and the tallies summed. Prints a line every 5 % when `progress` is given."""
    stats: list[dict[str, Any] | None] = []
    lock = threading.Lock()
    done = 0
    step = max(1, len(tasks) // _PROGRESS_STEPS)
    start = time.perf_counter()

    def run(task: Any) -> dict[str, Any]:
        nonlocal done
        with client.track() as tally:
            row = fn(task)
        with lock:
            stats.append(client.stats_dict(tally))
            done += 1
            if progress and (done % step == 0 or done == len(tasks)):
                seconds = time.perf_counter() - start
                progress(f"{label}: {done}/{len(tasks)} in {seconds:.0f} s")
        return row

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        rows = list(pool.map(run, tasks))
    return rows, sum_stats(stats)


def write_stats(path: Path, config: CollectConfig, body: dict[str, Any]) -> dict[str, Any]:
    """Provenance and counts as JSON; returns the body with the health problems added."""
    llm = body.get("llm")
    problems = role_problems(llm["roles"]) if llm else []
    out = {
        "name": config.name,
        "version": __version__,
        "elicitor": {"version": elicitor_version, "commit": ELICITOR_COMMIT},
        **body,
        "problems": problems,
        "config": config.text,
    }
    write_text_atomic(path, json.dumps(out, indent=2) + "\n")
    return out


# ---------------------------------------------------------------- backstories


def backstories_path(config: CollectConfig) -> Path:
    """Beside `personas_train.jsonl`, where Elicitor's persona loader looks for them."""
    return config.data_dir / BACKSTORIES


def make_train_backstories(
    config: CollectConfig,
    out_dir: str | Path = RAW_DIR,
    workers: int = 1,
    progress: Progress | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Backstories for every training persona (not cut to `max_personas`: the file is frozen).
    Returns the personas still flagged after the last draw, and the LLM statistics."""
    personas = load_split(config.data_dir, TRAIN, config.domain)
    client = make_client(config, out_dir)
    try:
        stories = write_backstories(
            client,
            load_schema(config.schema_path),
            personas,
            load_styles(config.prompts_dir / "styles.toml"),
            backstories_path(config),
            workers=workers,
            progress=progress,
        )
        llm = client.stats_dict()
    finally:
        client.close()
    return [s.persona for s in stories if s.flagged], llm


# ---------------------------------------------------------------- update records


def update_context(slot: Slot, question: Question, text: str, answer: str) -> dict[str, Any]:
    """The `update` role's context, as Elicitor's LLM updater builds it."""
    return {
        "slot": slot,
        "q": question,  # for the mock only; the prompt sees the question as asked
        "slot_description": slot.description,
        "values": value_lines(slot),
        "question": text,
        "answer": answer,
        "no_preference_note": _NO_PREFERENCE_NOTE if slot.kind == CATEGORICAL else "",
    }


def collection_tasks(
    schema: Schema, personas: list[Persona], seed: int, answer_seeds: int
) -> list[tuple[Persona, Question, int]]:
    return [
        (persona, question, a)
        for persona in personas
        for question in calibration_questions(schema, persona, seed)
        for a in range(answer_seeds)
    ]


def collect_one(
    client: LLMClient,
    simulator: LLMSimulator,
    schema: Schema,
    seed: int,
    persona: Persona,
    question: Question,
    answer_index: int,
) -> dict[str, Any]:
    """Ask, answer, read: one record."""
    slot = schema.slot(question.slot)
    text = template_question(slot, question)
    rng = derive_rng(seed, persona.id, "collect", json.dumps(question.to_dict()), answer_index)
    answer = simulator.respond(question, text, persona, rng)
    row: dict[str, Any] = {
        "persona": persona.id,
        "split": persona.split,
        "style": persona.style,
        "slot": slot.name,
        "question": question.to_dict(),
        "answer_index": answer_index,
        "text": text,
        "answer": answer,
        "leaks": leaked_slots(schema, slot.name, answer),
    }
    if not answer.strip():  # the LLM updater skips an empty answer without a call
        return {**row, "status": EMPTY}
    context = update_context(slot, question, text, answer)
    messages = client.roles["update"].prompt.render(context)
    teacher = client.models[client.roles["update"].group]
    row.update(teacher=teacher, messages=messages)
    try:
        reply = client.call("update", context, seed=UPDATE_SEED)
    except LLMParseError:
        return {**row, "status": FAILED}
    scores = update_scores(reply, slot)
    return {
        **row,
        "status": OK,
        "scores": dict(zip(slot.value_names, scores, strict=True)),
        "evidence": reply.evidence,  # type: ignore[attr-defined]
    }


def split_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    status = Counter(r["status"] for r in rows)
    return {
        "personas": len({r["persona"] for r in rows}),
        "records": len(rows),
        **{s: status.get(s, 0) for s in (OK, FAILED, EMPTY)},
        "leaking_answers": sum(bool(r["leaks"]) for r in rows),
    }


def collect(
    config: CollectConfig,
    out_dir: str | Path = RAW_DIR,
    workers: int = 1,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Collect every split of the file; writes the records and `update_stats.json`."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    schema = load_schema(config.schema_path)
    personas = {split: split_personas(config, split) for split in config.splits}
    missing = [p.id for ps in personas.values() for p in ps if not p.backstory]
    if missing:
        raise MissingBackstoriesError(
            f"{len(missing)} personas have no backstory (first: {missing[0]}); run "
            "`distill make-backstories` with this file first"
        )

    client = make_client(config, out)
    simulator = LLMSimulator(schema, client, load_styles(config.prompts_dir / "styles.toml"))
    start = time.perf_counter()
    try:
        tasks = [
            task
            for split in config.splits
            for task in collection_tasks(schema, personas[split], config.seed, config.answer_seeds)
        ]
        rows, llm = run_tracked(
            client,
            lambda t: collect_one(client, simulator, schema, config.seed, *t),
            tasks,
            workers,
            progress,
            label="update records",
        )
    finally:
        client.close()

    counts = {}
    for split in config.splits:
        mine = [r for r in rows if r["split"] == split]
        write_jsonl(out / f"update_{split}.jsonl", mine)
        counts[split] = split_counts(mine)
    return write_stats(
        out / "update_stats.json",
        config,
        {"seconds": round(time.perf_counter() - start, 1), "splits": counts, "llm": llm},
    )
