"""The M5 verdicts (SPEC §7.5, Decision Q16): `distill m5-report`.

Four paired contrasts at K = 5, each per persona with the replicates averaged first, by Elicitor's
paired statistics, with Holm per metric over the four: E1 R1 `eig_mc` − teacher `eig_mc`; E2 R1
`eig_mc` − R1 `eig`; E3 R1 `eig_mc` − R2 `eig_mc`; E4 R1 `eig` − teacher `eig`. E5 reads the
offline `imagine` rows: the student's and the teacher's mass on the imagined value, averaged per
(question, value) pair, and the one-sided lower bound of their mean difference. A run that did
not finish (R2 under its deadline) leaves its contrast out and its hypothesis untested.

Outputs: `contrasts.csv`, `verdicts.md`, and `distill compare`'s tables and curves of the runs
against the teacher in `compare/`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from elicitor.analysis.report import LOWER_IS_BETTER, episode_value, per_persona
from elicitor.analysis.stats import BOOTSTRAP_RESAMPLES, holm, paired_test
from elicitor.experiments.store import write_text_atomic
from elicitor.jsonl import read_jsonl

from elicitor_distill.compare import MARGINS, METRICS, _csv, compare, one_sided_lower, read_run
from elicitor_distill.imagine import pair_means

MC, EIG = "sel=eig_mc,upd=llm,phr=tpl", "sel=eig,upd=llm,phr=tpl"
ARMS = (MC, EIG)
BUDGET = 5
FORGETTING = 0.005  # E4: update test agreement may drop this much below upd-1.7b-sft's
E5_MARGIN = -0.03


def contrast(name: str, a: list[dict], arm_a: str, b: list[dict], arm_b: str, metric: str,
             resamples: int) -> dict[str, Any]:  # fmt: skip
    higher = metric not in LOWER_IS_BETTER
    va = per_persona(a, lambda e: episode_value(e, metric))[(arm_a, BUDGET)]
    vb = per_persona(b, lambda e: episode_value(e, metric))[(arm_b, BUDGET)]
    personas = sorted(set(va) & set(vb))
    x = np.array([va[p] for p in personas], dtype=float)
    y = np.array([vb[p] for p in personas], dtype=float)
    r = paired_test(x, y, higher, resamples)
    return {"contrast": name, "metric": metric, "n": r.n, "mean_a": r.mean_a, "mean_b": r.mean_b,
            "diff": r.diff, "ci_low": r.ci_low, "ci_high": r.ci_high,
            "lower_95": one_sided_lower(np.round(x - y, 10), resamples), "p": r.p}  # fmt: skip


def _try_run(run_dir: str | Path | None) -> list[dict] | None:
    if run_dir is None:
        return None
    try:
        return read_run(run_dir, ARMS)[1]
    except ValueError:  # not finished
        return None


def agreement(summary: str | Path) -> float:
    return json.loads(Path(summary).read_text(encoding="utf-8"))["tables"]["overall"]["agreement"]


def m5_report(
    teacher_dir: str | Path, r1_dir: str | Path, r2_dir: str | Path | None,
    imagine_student: str | Path, imagine_teacher: str | Path,
    agent_test: str | Path, update_test: str | Path, out_dir: str | Path,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:  # fmt: skip
    teacher = read_run(teacher_dir, ARMS)[1]
    r1 = read_run(r1_dir, ARMS)[1]
    r2 = _try_run(r2_dir)
    out = Path(out_dir)
    compare(teacher_dir, [r1_dir] + ([r2_dir] if r2 else []), out / "compare", resamples,
            arms=ARMS, ni_arms=ARMS)  # fmt: skip
    pairs = [("E1", r1, MC, teacher, MC), ("E2", r1, MC, r1, EIG), ("E4", r1, EIG, teacher, EIG)]
    if r2:
        pairs.append(("E3", r1, MC, r2, MC))
    rows = []
    for metric in METRICS:
        family = [contrast(n, a, aa, b, ab, metric, resamples) for n, a, aa, b, ab in pairs]
        for row, p_holm in zip(family, holm([r["p"] for r in family]), strict=True):
            row["p_holm"] = p_holm
        rows += family
    get = {(r["contrast"], r["metric"]): r for r in rows}

    def ni(name: str) -> bool:
        return all(get[(name, m)]["lower_95"] > margin for m, margin in MARGINS.items())

    student = pair_means(read_jsonl(Path(imagine_student) / "rows.jsonl"))
    base = pair_means(read_jsonl(Path(imagine_teacher) / "rows.jsonl"))
    diffs = np.array([student[k] - base[k] for k in sorted(base)], dtype=float)
    e5 = {"diff": float(diffs.mean()), "lower_95": one_sided_lower(diffs, resamples),
          "pairs": len(diffs)}  # fmt: skip
    agent_agr, update_agr = agreement(agent_test), agreement(update_test)
    mass = {n: get[(n, "mass_on_truth")] for n, *_ in pairs}
    verdicts = {
        "E1": (ni("E1"), "lower {:+.4f} mass, {:+.4f} MAP".format(
            get[("E1", "mass_on_truth")]["lower_95"], get[("E1", "map_accuracy")]["lower_95"])),
        "E2": (mass["E2"]["ci_low"] > 0, _ci(mass["E2"])),
        "E3": (mass["E3"]["ci_low"] > 0, _ci(mass["E3"])) if r2 else (None, "R2 not finished"),
        "E4": (agent_agr >= update_agr - FORGETTING and ni("E4"),
               f"agreement {agent_agr:.3f} against {update_agr:.3f}; eig lower "
               f"{get[('E4', 'mass_on_truth')]['lower_95']:+.4f} mass, "
               f"{get[('E4', 'map_accuracy')]['lower_95']:+.4f} MAP"),
        "E5": (e5["lower_95"] > E5_MARGIN,
               f"{e5['diff']:+.4f}, lower {e5['lower_95']:+.4f} over {e5['pairs']} pairs"),
    }  # fmt: skip
    write_text_atomic(out / "contrasts.csv", _csv(rows))
    write_text_atomic(out / "verdicts.md", markdown(verdicts, rows))
    return {"verdicts": verdicts, "rows": rows, "e5": e5}


def _ci(r: dict[str, Any]) -> str:
    return f"{r['diff']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"


def markdown(verdicts: dict[str, tuple], rows: list[dict[str, Any]]) -> str:
    word = {True: "**holds**", False: "**does not hold**", None: "not tested"}
    lines = ["### M5 hypotheses (Q16)", "", "| | Found | Numbers |", "|---|---|---|"]
    lines += [f"| {k} | {word[v]} | {text} |" for k, (v, text) in verdicts.items()]
    lines += ["", "### Contrasts at K = 5 (Holm per metric over the contrasts)", "",
              "| contrast | metric | a | b | a − b | 95 % CI | one-sided lower | p (Holm) |",
              "|---|---|---:|---:|---:|---|---:|---:|"]  # fmt: skip
    lines += [f"| {r['contrast']} | {r['metric']} | {r['mean_a']:.4f} | {r['mean_b']:.4f} | "
              f"{_ci(r).replace(' [', ' | [')} | {r['lower_95']:+.4f} | {r['p_holm']:.3g} |"
              for r in rows]  # fmt: skip
    return "\n".join(lines) + "\n"
