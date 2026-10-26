"""End to end against the teacher (SPEC §7.3, Q12, §7.4): `distill compare`.

Each student run is paired with the teacher's episodes in `m5_full` (read, never written) by
persona, the 5 replicates averaged first, with Elicitor's own paired statistics: the mean
difference student − teacher with a 95 % bootstrap CI, the paired Wilcoxon test, and Holm per
metric over every (model, arm, budget). Non-inferiority (Q12) reads the one-sided 95 % lower
bound, the 5th percentile of the same bootstrap means, against the margin of the metric.
Also: per-turn curves with the teacher overlaid, and how often a student's episode asked the
teacher's questions in the teacher's order.

Outputs: `comparisons.csv`, `curves.csv`, `questions.csv`, `results.md` and one figure per
curve metric (the `eig` arm at the largest budget).
"""

from __future__ import annotations

import csv
import io
import json
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from elicitor.analysis.report import LOWER_IS_BETTER, curve, episode_value, per_persona
from elicitor.analysis.stats import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, holm, paired_test
from elicitor.experiments.store import RunStore, cell_id, write_text_atomic

ARMS = ("sel=eig,upd=llm,phr=tpl", "sel=random,upd=llm,phr=tpl")  # Decision Q11
METRICS = ("mass_on_truth", "map_accuracy", "log_loss", "regret", "hit@1", "violations")
CURVE_METRICS = ("mass_on_truth", "map_accuracy", "regret")
MARGINS = {"mass_on_truth": -0.02, "map_accuracy": -0.03}  # Decision Q12
NI_ARM, NI_BUDGET = ARMS[0], 5
TEACHER = "teacher"


def read_run(run_dir: str | Path, arms: tuple[str, ...] = ARMS) -> tuple[dict, list[dict]]:
    """The run's state and the episodes of the given arms; every one of their cells must be done."""
    store = RunStore(run_dir)
    state = store.read_state()
    if state is None:
        raise ValueError(f"{run_dir} is not a run folder")
    cells = [cell_id(a, r) for a in state["arms"] if a in arms for r in range(state["replicates"])]
    missing = [c for c in cells if not store.is_done(c)]
    if missing or not cells:
        raise ValueError(f"{run_dir}: {len(missing)} of {len(cells)} cells not finished")
    return state, [e for c in cells for e in store.read_cell(c)]


def agent_model(state: dict[str, Any]) -> str:
    return tomllib.loads(state["config"])["llm"]["models"]["agent"]


def one_sided_lower(diffs: np.ndarray, resamples: int = BOOTSTRAP_RESAMPLES) -> float:
    """5th percentile of the bootstrap means: the one-sided 95 % lower bound (Q12)."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, diffs.size, size=(resamples, diffs.size))
    return float(np.percentile(diffs[idx].mean(axis=1), 5))


def _questions(episodes: list[dict]) -> dict[tuple, str]:
    return {
        (e["arm"], e["budget"], e["persona"], e["replicate"]):
            json.dumps([t["question"] for t in e["turns"]], sort_keys=True)
        for e in episodes
    }  # fmt: skip


def comparison_rows(
    teacher: list[dict], runs: dict[str, list[dict]], resamples: int
) -> list[dict[str, Any]]:
    rows = []
    for metric in METRICS:
        higher = metric not in LOWER_IS_BETTER
        base = per_persona(teacher, lambda e, m=metric: episode_value(e, m))
        family = []
        for model, episodes in runs.items():
            values = per_persona(episodes, lambda e, m=metric: episode_value(e, m))
            for key in sorted(k for k in values if k in base):
                personas = sorted(set(values[key]) & set(base[key]))
                a = np.array([values[key][p] for p in personas], dtype=float)
                b = np.array([base[key][p] for p in personas], dtype=float)
                r = paired_test(a, b, higher, resamples)
                lower = one_sided_lower(np.round(a - b, 10), resamples)
                margin = MARGINS.get(metric)
                family.append(
                    {"metric": metric, "model": model, "arm": key[0], "budget": key[1],
                     "n": r.n, "mean_model": r.mean_a, "mean_teacher": r.mean_b, "diff": r.diff,
                     "ci_low": r.ci_low, "ci_high": r.ci_high, "lower_95": lower,
                     "margin": margin, "non_inferior": None if margin is None else lower > margin,
                     "wins": r.wins, "ties": r.ties, "losses": r.losses, "p": r.p,
                     "better": "higher" if higher else "lower"}
                )  # fmt: skip
        for row, p_holm in zip(family, holm([row["p"] for row in family]), strict=True):
            row["p_holm"] = p_holm
        rows += family
    return rows


def curve_rows(runs: dict[str, list[dict]]) -> list[dict[str, Any]]:
    rows = []
    for model, episodes in runs.items():
        for metric in CURVE_METRICS:
            values = per_persona(episodes, lambda e, m=metric: curve(e, m))
            for (arm, budget), by_p in values.items():
                matrix = np.array([by_p[p] for p in sorted(by_p)], dtype=float)
                se = matrix.std(axis=0, ddof=1) / np.sqrt(len(matrix))
                rows += [{"model": model, "metric": metric, "arm": arm, "budget": budget,
                          "turn": t, "mean": matrix[:, t].mean(), "se": se[t]}
                         for t in range(matrix.shape[1])]  # fmt: skip
    return rows


def question_rows(teacher: list[dict], runs: dict[str, list[dict]]) -> list[dict[str, Any]]:
    """Share of a model's episodes whose question sequence equals the teacher's."""
    base = _questions(teacher)
    rows = []
    for model, episodes in runs.items():
        same: dict[tuple, list[bool]] = defaultdict(list)
        for key, seq in _questions(episodes).items():
            same[key[:2]].append(base.get(key) == seq)
        rows += [{"model": model, "arm": arm, "budget": budget, "episodes": len(v),
                  "same_questions": sum(v) / len(v)}
                 for (arm, budget), v in sorted(same.items())]  # fmt: skip
    return rows


def _csv(rows: list[dict[str, Any]]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(
        {k: f"{v:.6g}" if isinstance(v, float) else v for k, v in row.items()} for row in rows
    )
    return buf.getvalue()


def markdown(rows: list[dict], questions: list[dict]) -> str:
    lines = [f"### Non-inferiority (Q12): `{NI_ARM}` at K = {NI_BUDGET}", "",
             "| model | metric | model − teacher | 95 % CI | one-sided lower | margin | verdict |",
             "|---|---|---:|---|---:|---:|---|"]  # fmt: skip
    for r in rows:
        if r["arm"] == NI_ARM and r["budget"] == NI_BUDGET and r["margin"] is not None:
            verdict = "non-inferior" if r["non_inferior"] else "not shown"
            lines.append(f"| {r['model']} | {r['metric']} | {r['diff']:+.4f} | "
                         f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | {r['lower_95']:+.4f} | "
                         f"{r['margin']} | {verdict} |")  # fmt: skip
    lines += ["", "### Every comparison (Holm per metric over model × arm × budget)", "",
              "| metric | model | arm | K | model | teacher | diff | 95 % CI | p (Holm) |",
              "|---|---|---|---:|---:|---:|---:|---|---:|"]  # fmt: skip
    for r in rows:
        lines.append(f"| {r['metric']} | {r['model']} | {r['arm']} | {r['budget']} | "
                     f"{r['mean_model']:.4f} | {r['mean_teacher']:.4f} | {r['diff']:+.4f} | "
                     f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | {r['p_holm']:.3g} |"
                     )  # fmt: skip
    lines += ["", "### Episodes that asked the teacher's questions in the teacher's order", "",
              "| model | arm | K | episodes | same |", "|---|---|---:|---:|---:|"]  # fmt: skip
    lines += [f"| {q['model']} | {q['arm']} | {q['budget']} | {q['episodes']} | "
              f"{q['same_questions']:.3f} |" for q in questions]  # fmt: skip
    return "\n".join(lines) + "\n"


def plot(curves: list[dict], metric: str, budget: int, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    for model in dict.fromkeys(r["model"] for r in curves):
        pts = [r for r in curves if (r["model"], r["metric"], r["arm"], r["budget"])
               == (model, metric, NI_ARM, budget)]  # fmt: skip
        style = {"color": "black", "linewidth": 2.5} if model == TEACHER else {"linewidth": 1.5}
        ax.plot([r["turn"] for r in pts], [r["mean"] for r in pts], marker="o", markersize=3,
                label=model, **style)  # fmt: skip
    ax.set(xlabel="turn", ylabel=metric, title=f"{NI_ARM}, K = {budget}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def compare(
    teacher_dir: str | Path, run_dirs: list[str | Path], out_dir: str | Path,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:  # fmt: skip
    state, teacher = read_run(teacher_dir)
    runs: dict[str, list[dict]] = {}
    for d in run_dirs:
        s, episodes = read_run(d)
        if (s["seed"], s["budgets"], s["personas"]) != (state["seed"], state["budgets"],
                                                       state["personas"]):  # fmt: skip
            raise ValueError(f"{d} does not pair with {teacher_dir}: seed, budgets or personas")
        runs[agent_model(s)] = episodes
    rows = comparison_rows(teacher, runs, resamples)
    curves = curve_rows({TEACHER: teacher, **runs})
    questions = question_rows(teacher, runs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, table in (("comparisons", rows), ("curves", curves), ("questions", questions)):
        write_text_atomic(out / f"{name}.csv", _csv(table))
    write_text_atomic(out / "results.md", markdown(rows, questions))
    for metric in CURVE_METRICS:
        plot(curves, metric, max(state["budgets"]), out / f"curve_{metric}.png")
    return {"rows": rows, "questions": questions, "models": list(runs)}
