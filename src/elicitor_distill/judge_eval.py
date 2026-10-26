"""The judge on the test set (SPEC §7.4, D5): `distill judge-eval`.

Reads the `eval-offline` folders of the judge role on the 256 units of `m7_judge` and gives, per
model and for the teacher (whose scores every folder holds), the Spearman correlation of the
score with the true utility, pooled over units, with Elicitor's CI by resampling personas
(`elicitor.analysis.judge_report.spearman_ci`, the method of the teacher's [0.827, 0.906]),
beside the agreement with the teacher. D5 holds when the SFT student's lower bound is ≥ 0.80.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from elicitor.analysis.judge_report import spearman_ci
from elicitor.analysis.stats import BOOTSTRAP_RESAMPLES
from elicitor.jsonl import read_jsonl

from elicitor_distill.m3 import JUDGE_CONDITION

D5_BOUND = 0.80


def _utility_ci(rows: list[dict[str, Any]], key: str, resamples: int) -> dict[str, Any]:
    units = [{"persona": r["meta"]["persona"], "item": r["meta"]["item"],
              "utility": r["meta"]["utility"]} for r in rows]  # fmt: skip
    scores = {(u["persona"], u["item"]): {JUDGE_CONDITION: float(r[key])}
              for u, r in zip(units, rows, strict=True) if r[key] is not None}  # fmt: skip
    run = SimpleNamespace(units=units, personas=sorted({u["persona"] for u in units}))
    return spearman_ci(run, scores, JUDGE_CONDITION, resamples)  # type: ignore[arg-type]


def judge_table(folders: list[str | Path], resamples: int = BOOTSTRAP_RESAMPLES) -> list[dict]:
    out = []
    for i, folder in enumerate(folders):
        summary = json.loads((Path(folder) / "summary.json").read_text(encoding="utf-8"))
        rows = read_jsonl(Path(folder) / "rows.jsonl")
        overall = summary["tables"]["overall"]
        ci = _utility_ci(rows, "student", resamples)
        out.append({"model": summary["model"], "n": overall["n"], "failed": overall["failed"],
                    "mae": overall["mae"], "exact": overall["exact"],
                    "spearman_teacher": overall["spearman_teacher"], **_prefixed(ci)})  # fmt: skip
        if i == 0:
            out.append({"model": "teacher", "n": overall["n"], "failed": 0,
                        **_prefixed(_utility_ci(rows, "teacher", resamples))})  # fmt: skip
    return out


def _prefixed(ci: dict[str, Any]) -> dict[str, Any]:
    return {"spearman_utility": ci["estimate"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"]}


def markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["| model | n | failed | MAE | exact | Spearman teacher | Spearman utility | 95 % CI |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]  # fmt: skip

    def cell(v: Any) -> str:
        return "–" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)

    for r in rows:
        lines.append(" | ".join([f"| {r['model']}", *(cell(r.get(k)) for k in (
            "n", "failed", "mae", "exact", "spearman_teacher", "spearman_utility")),
            f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}] |"]))  # fmt: skip
    return "\n".join(lines) + f"\n\nD5 bar: the SFT judge's CI lower bound ≥ {D5_BOUND}.\n"
