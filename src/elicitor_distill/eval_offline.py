"""Offline agreement with the teacher (SPEC §7.1): `distill eval-offline`.

Every SFT sample is sent to a served model with exactly the messages the teacher received: the
roles here are Elicitor's `update` and `judge` roles (sampling settings, output models) with a
pass-through prompt that sends the stored system and user text unchanged. Calls go through
Elicitor's `LLMClient` (seed 0, the role's retries, the JSON schema as `format`), so the model
sees what Elicitor would send it. Unconstrained validity is a second pass without `format`, one
attempt per sample.

Per `update` sample, student scores against teacher scores:
- argmax agreement: |S ∩ T| / |S ∪ T| for the student's and the teacher's sets of top values
  (1 for the same set; 1/j when the student ties j values, one of them the teacher's top, as in
  Elicitor's calibration);
- score MAE on the 0–10 scale and the Spearman correlation of the two score vectors (undefined,
  and left out, when either vector is constant);
- KL(teacher ‖ student) between the normalised likelihoods, the quantity Bayes' rule uses.
A failed call counts as equal scores, which is what Elicitor's updater does with it (the belief
is left unchanged). Per `judge` sample: absolute score error and exact agreement; over samples
the Spearman correlations of the student's score with the teacher's and with the true utility.
Everything is also given per question kind and answer style (update) or utility bin and style
(judge).
"""

from __future__ import annotations

import dataclasses
import json
import math
import string
import time
import warnings
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
from elicitor.experiments.health import role_problems
from elicitor.experiments.store import write_text_atomic
from elicitor.jsonl import write_jsonl
from elicitor.llm import LLMClient, LLMParseError
from elicitor.llm.cache import ReplyCache
from elicitor.llm.client import parse_reply
from elicitor.llm.providers import OllamaProvider, Provider, make_provider
from elicitor.llm.roles import AGENT, JudgeReply, Prompt, Role, build_roles, update_model
from elicitor.llm.roles import update_scores as reply_scores
from elicitor.schema import Schema
from elicitor.updaters.llm import likelihood
from scipy.stats import spearmanr

from elicitor_distill.collect import Progress, run_tracked

ROLES = ("update", "judge")
EVAL_SEED = 0  # the seed Elicitor sends for both roles
FLAT = 5  # a failed update leaves the belief as equal scores would
PASSTHROUGH = Prompt(system=string.Template("$system"), user=string.Template("$user"))
_GROUPS = {"update": ("kind", "style"), "judge": ("bin", "style")}


# ---------------------------------------------------------------- serving


class UnconstrainedOllama(OllamaProvider):
    """Ollama without the JSON schema as `format`: does the model write valid JSON by itself?"""

    def payload(self, request):  # type: ignore[no-untyped-def]
        payload = super().payload(request)
        del payload["format"]
        return payload


def _echo(ctx: dict[str, Any], rng: Any) -> Any:
    """Mock reply: the teacher's own (a perfect student), for tests on the mock provider."""
    return ctx["output"].model_validate_json(ctx["reference"])


def passthrough_roles(prompts_dir: str | Path) -> dict[str, Role]:
    """Elicitor's `update` and `judge` roles with the stored messages as the prompt. Both are
    served by the agent model group: a student (or a baseline) is the agent model here."""
    roles = build_roles(prompts_dir)
    return {
        name: dataclasses.replace(
            roles[name],
            prompt=PASSTHROUGH,
            output=lambda ctx: ctx["output"],
            mock=_echo,
            group=AGENT,
        )
        for name in ROLES
    }


def make_eval_client(
    prompts_dir: str | Path,
    model: str,
    provider: str = "ollama",
    base_url: str = "http://127.0.0.1:11434",
    num_ctx: int = 2048,
    constrained: bool = True,
    cache: str | Path | None = None,
    max_attempts: int = 3,
    timeout: float = 600.0,
    think: bool | None = False,
    roles: dict[str, Role] | None = None,
) -> LLMClient:
    """A client for one served model, with Elicitor's `[llm]` settings (`think` off, `num_ctx`
    2048) unless told otherwise; `think=None` sends no `think` field (models without a thinking
    mode, like the judge teacher). The roles are the pass-through ones unless given."""
    server: Provider
    if provider == "mock":
        server = make_provider("mock")
    elif provider == "ollama":
        kind = OllamaProvider if constrained else UnconstrainedOllama
        server = kind(base_url=base_url, timeout=timeout, num_ctx=num_ctx, think=think)
    else:
        raise ValueError(f"unknown provider {provider!r}")
    return LLMClient(
        roles=roles or passthrough_roles(prompts_dir),
        provider=server,
        models={"agent": model, "user": model},
        cache=None if cache is None else ReplyCache(cache),
        max_attempts=max_attempts,
    )


def output_model(role: str, sample: dict[str, Any], schema: Schema) -> Any:
    return update_model(schema.slot(sample["meta"]["slot"])) if role == "update" else JudgeReply


def sample_context(role: str, sample: dict[str, Any], schema: Schema) -> dict[str, Any]:
    system, user, assistant = sample["messages"]
    return {
        "system": system["content"],
        "user": user["content"],
        "reference": assistant["content"],
        "output": output_model(role, sample, schema),
    }


def reply_values(role: str, reply: Any, sample: dict[str, Any], schema: Schema) -> Any:
    """Scores in the slot's value order (update) or the score (judge)."""
    if role == "update":
        return reply_scores(reply, schema.slot(sample["meta"]["slot"]))
    return reply.score


def teacher_values(role: str, sample: dict[str, Any], schema: Schema) -> Any:
    model = output_model(role, sample, schema)
    reply = model.model_validate_json(sample["messages"][-1]["content"])
    return reply_values(role, reply, sample, schema)


def parse_values(role: str, text: str, sample: dict[str, Any], schema: Schema) -> Any:
    """A raw completion's scores or score, as Elicitor's client parses it; None if invalid."""
    reply = parse_reply(output_model(role, sample, schema), text)
    return None if reply is None else reply_values(role, reply, sample, schema)


# ---------------------------------------------------------------- metrics


def _top(scores: list[int]) -> set[int]:
    best = max(scores)
    return {i for i, s in enumerate(scores) if s == best}


def argmax_agreement(teacher: list[int], student: list[int]) -> float:
    t, s = _top(teacher), _top(student)
    return len(t & s) / len(t | s)


def score_mae(teacher: list[int], student: list[int]) -> float:
    return float(np.mean(np.abs(np.subtract(teacher, student, dtype=float))))


def rank_correlation(a: Iterable[float], b: Iterable[float]) -> float:
    """Spearman's rho; nan when either side is constant (or there are fewer than 3 points)."""
    a, b = np.asarray(list(a), dtype=float), np.asarray(list(b), dtype=float)
    if a.size < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(spearmanr(a, b).statistic)


def likelihood_kl(teacher: list[int], student: list[int]) -> float:
    p = np.asarray(likelihood(teacher))
    q = np.asarray(likelihood(student))
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def update_metrics(teacher: list[int], student: list[int]) -> dict[str, float]:
    return {
        "agreement": argmax_agreement(teacher, student),
        "mae": score_mae(teacher, student),
        "spearman": rank_correlation(teacher, student),
        "kl": likelihood_kl(teacher, student),
    }


def _mean(values: Iterable[float]) -> float | None:
    """Mean of the defined values (None when there are none)."""
    kept = [v for v in values if v is not None and not math.isnan(v)]
    return round(float(np.mean(kept)), 4) if kept else None


def _round(value: float) -> float | None:
    return None if math.isnan(value) else round(value, 4)


def summarize(role: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate metrics of a list of per-sample rows (see `evaluate`)."""
    out: dict[str, Any] = {
        "n": len(rows),
        "failed": sum(r["status"] != "ok" for r in rows),
    }
    checked = [r for r in rows if r.get("unconstrained_valid") is not None]
    if checked:
        out["unconstrained_valid"] = _mean(float(r["unconstrained_valid"]) for r in checked)
    if role == "update":
        for key in ("agreement", "mae", "spearman", "kl"):
            out[key] = _mean(r[key] for r in rows)
        out["spearman_defined"] = sum(not math.isnan(r["spearman"]) for r in rows)
        return out
    ok = [r for r in rows if r["status"] == "ok"]
    out["mae"] = _mean(abs(r["student"] - r["teacher"]) for r in ok)
    out["exact"] = _mean(float(r["student"] == r["teacher"]) for r in ok)
    out["within_1"] = _mean(float(abs(r["student"] - r["teacher"]) <= 1) for r in ok)
    out["spearman_teacher"] = _round(
        rank_correlation([r["student"] for r in ok], [r["teacher"] for r in ok])
    )
    out["spearman_utility"] = _round(
        rank_correlation([r["student"] for r in ok], [r["meta"]["utility"] for r in ok])
    )
    out["teacher_spearman_utility"] = _round(
        rank_correlation([r["teacher"] for r in ok], [r["meta"]["utility"] for r in ok])
    )
    return out


def summary_tables(role: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Overall and per group (question kind or utility bin, and answer style)."""
    tables: dict[str, Any] = {"overall": summarize(role, rows)}
    for key in _GROUPS[role]:
        values = sorted({r["meta"][key] for r in rows}, key=str)
        tables[key] = {
            str(v): summarize(role, [r for r in rows if r["meta"][key] == v]) for v in values
        }
    return tables


# ---------------------------------------------------------------- evaluation


def evaluate_one(
    client: LLMClient, role: str, schema: Schema, seed: int, index: int, sample: dict[str, Any]
) -> dict[str, Any]:
    teacher = teacher_values(role, sample, schema)
    row: dict[str, Any] = {"index": index, "meta": sample["meta"], "teacher": teacher}
    try:
        reply = client.call(role, sample_context(role, sample, schema), seed=seed)
    except LLMParseError:
        row.update(status="failed", student=None)
    else:
        row.update(status="ok", student=reply_values(role, reply, sample, schema))
    if role == "update":
        student = row["student"] if row["student"] is not None else [FLAT] * len(teacher)
        row.update(update_metrics(teacher, student))
    return row


def unconstrained_one(
    client: LLMClient, role: str, schema: Schema, seed: int, sample: dict[str, Any]
) -> dict[str, Any]:
    try:
        client.call(role, sample_context(role, sample, schema), seed=seed)
    except LLMParseError:
        return {"valid": False}
    return {"valid": True}


def evaluate(
    samples: list[dict[str, Any]],
    role: str,
    schema: Schema,
    make_client: Callable[[bool], LLMClient],
    workers: int = 8,
    seed: int = EVAL_SEED,
    unconstrained: bool = True,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Per-sample rows, the summary tables and the LLM statistics. `make_client(constrained)`
    builds the client of each pass."""
    start = time.perf_counter()
    client = make_client(True)
    try:
        tasks = list(enumerate(samples))
        rows, llm = run_tracked(
            client,
            lambda t: evaluate_one(client, role, schema, seed, *t),
            tasks,
            workers,
            progress,
            label=f"{role} samples",
        )
    finally:
        client.close()
    result: dict[str, Any] = {"llm": llm}
    if unconstrained:
        free = make_client(False)
        try:
            valid, free_llm = run_tracked(
                free,
                lambda s: unconstrained_one(free, role, schema, seed, s),
                samples,
                workers,
                progress,
                label=f"{role} unconstrained",
            )
        finally:
            free.close()
        for row, v in zip(rows, valid, strict=True):
            row["unconstrained_valid"] = v["valid"]
        result["llm_unconstrained"] = free_llm
    result.update(
        seconds=round(time.perf_counter() - start, 1),
        tables=summary_tables(role, rows),
        problems=role_problems(llm["roles"]) if llm else [],
        rows=rows,
    )
    return result


def strict_problems(llm: dict[str, Any] | None) -> list[str]:
    """The smoke bar (SPEC §6): not a single parse failure or failed call."""
    problems = []
    for name, s in (llm or {}).get("roles", {}).items():
        if s.get("parse_failures") or s.get("failed_calls") or s.get("transport_errors"):
            problems.append(
                f"{name}: {s.get('parse_failures', 0)} parse failures, "
                f"{s.get('failed_calls', 0)} failed calls, "
                f"{s.get('transport_errors', 0)} transport errors"
            )
    return problems


# ---------------------------------------------------------------- reports

_COLUMNS = {
    "update": ("n", "failed", "agreement", "mae", "spearman", "kl", "unconstrained_valid"),
    "judge": ("n", "failed", "mae", "exact", "within_1", "spearman_teacher", "spearman_utility",
              "unconstrained_valid"),
}  # fmt: skip


def _cell(value: Any) -> str:
    if value is None:
        return "–"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def markdown_table(role: str, tables: dict[str, Any], title: str) -> str:
    columns = _COLUMNS[role]
    lines = [f"### {title}", "", "| group | " + " | ".join(columns) + " |",
             "|---|" + "---:|" * len(columns)]  # fmt: skip
    groups = [("all", tables["overall"])]
    for key in _GROUPS[role]:
        groups += [(f"{key} {name}", t) for name, t in tables[key].items()]
    for name, t in groups:
        lines.append(f"| {name} | " + " | ".join(_cell(t.get(c)) for c in columns) + " |")
    return "\n".join(lines) + "\n"


def write_report(out_dir: str | Path, result: dict[str, Any], provenance: dict[str, Any]) -> None:
    """`rows.jsonl`, `summary.json` (everything but the rows) and `table.md`."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "rows.jsonl", result["rows"])
    summary = {**provenance, **{k: v for k, v in result.items() if k != "rows"}}
    write_text_atomic(out / "summary.json", json.dumps(summary, indent=2) + "\n")
    title = f"{provenance['model']} — {provenance['role']} ({provenance['data']})"
    write_text_atomic(out / "table.md", markdown_table(provenance["role"], result["tables"], title))


def collect_summaries(root: str | Path) -> list[dict[str, Any]]:
    """Every `summary.json` under a folder, sorted by role and model."""
    found = [json.loads(p.read_text(encoding="utf-8")) for p in Path(root).rglob("summary.json")]
    return sorted(found, key=lambda s: (s["role"], s["model"], s.get("seed", 0)))


def overview_table(summaries: list[dict[str, Any]]) -> str:
    """One markdown table per role, a row per evaluated model (the M2 validation table)."""
    parts = []
    for role in ROLES:
        mine = [s for s in summaries if s["role"] == role]
        if not mine:
            continue
        columns = _COLUMNS[role]
        parts += [f"### {role}", "", "| model | seed | " + " | ".join(columns) + " |",
                  "|---|---:|" + "---:|" * len(columns)]  # fmt: skip
        for s in mine:
            t = s["tables"]["overall"]
            cells = " | ".join(_cell(t.get(c)) for c in columns)
            parts.append(f"| {s['model']} | {s.get('seed', 0)} | {cells} |")
        parts.append("")
    return "\n".join(parts)
