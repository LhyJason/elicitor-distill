"""`distill`: one argparse CLI (SPEC §8). Run it from the repository root."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from elicitor_distill import __version__
from elicitor_distill.bench import BENCH_N, BENCH_WORKERS
from elicitor_distill.collect import RAW_DIR
from elicitor_distill.data import DOMAIN, verify_domain_data
from elicitor_distill.eval_offline import ROLES as EVAL_ROLES
from elicitor_distill.export import MODELS_DIR, OUTTYPE
from elicitor_distill.m3 import TEST_DIR
from elicitor_distill.personas import (
    N_TRAIN,
    TRAIN_SEED,
    check_train_personas,
    write_train_personas,
)
from elicitor_distill.sft_data import ROLES, SFT_DIR
from elicitor_distill.splits import EVAL, EvalSplitError

DEFAULT_URL = "http://127.0.0.1:11434"


def _make_personas(args: argparse.Namespace) -> int:
    bad = verify_domain_data(f"{args.data_dir}/{DOMAIN}")
    if bad:
        print(f"domain data differs from the pinned Elicitor commit: {', '.join(bad)}")
        return 1
    if args.check:
        stale = check_train_personas(args.data_dir, args.n, args.seed)
        if stale:
            print("stale: " + ", ".join(p.as_posix() for p in stale))
            return 1
        print("training personas are up to date")
        return 0
    for path in write_train_personas(args.data_dir, args.n, args.seed):
        print(f"wrote {path.as_posix()}")
    return 0


def _report(result: dict[str, Any]) -> int:
    """Print the counts and the LLM health of a collection step; 1 when the health bar fails."""
    from elicitor.experiments.health import format_roles

    for split, counts in result["splits"].items():
        print(f"{split}: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    print(f"{result['seconds']:.0f} s")
    if result.get("llm"):
        print(format_roles(result["llm"]["roles"]))
    if result["problems"]:
        print("health: PROBLEMS")
        for problem in result["problems"]:
            print(f"  - {problem}")
        return 1
    print("health: OK")
    return 0


def _make_backstories(args: argparse.Namespace) -> int:
    from elicitor.experiments.health import format_roles

    from elicitor_distill.collect import backstories_path, make_train_backstories
    from elicitor_distill.config import load_collect_config

    config = load_collect_config(args.config)
    path = backstories_path(config)
    if path.exists() and not args.force:
        print(f"error: {path.as_posix()} exists and is frozen; pass --force", file=sys.stderr)
        return 2
    flagged, llm = make_train_backstories(config, args.out, args.workers, progress=print)
    print(format_roles(llm["roles"]))
    return 1 if flagged else 0


def _collect(args: argparse.Namespace) -> int:
    from elicitor_distill.collect import MissingBackstoriesError, collect
    from elicitor_distill.config import load_collect_config

    try:
        result = collect(load_collect_config(args.config), args.out, args.workers, print)
    except MissingBackstoriesError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return _report(result)


def _judge_collect(args: argparse.Namespace) -> int:
    from elicitor_distill.config import load_collect_config
    from elicitor_distill.judge_collect import judge_collect

    return _report(judge_collect(load_collect_config(args.config), args.out, args.workers, print))


def _build_sft(args: argparse.Namespace) -> int:
    from elicitor.schema import load_schema

    from elicitor_distill.sft_data import build_sft

    schema = load_schema(f"{args.data_dir}/{DOMAIN}/schema.toml")
    try:
        manifest = build_sft(args.raw, args.out, schema, tuple(args.roles))
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for name, f in manifest["files"].items():
        print(f"{args.out}/{name}: {f['rows']} rows from {f['records']} records, "
              f"dropped {f['dropped']}, sha256 {f['sha256'][:12]}")  # fmt: skip
    print(f"teachers: {manifest['teachers']}")
    return 0


def _schema(data_dir: str) -> Any:
    from elicitor.schema import load_schema

    return load_schema(f"{data_dir}/{DOMAIN}/schema.toml")


def _val_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    from elicitor_distill.sft_data import read_sft

    path = args.data or f"{SFT_DIR}/{args.role}_val.jsonl.gz"
    samples = read_sft(path)
    if not getattr(args, "allow_eval", False) and any(
        s["meta"].get("split") == EVAL for s in samples
    ):
        raise EvalSplitError(f"{path} holds eval personas (the test set); pass --allow-eval")
    return samples[: args.limit] if args.limit else samples


def _train(args: argparse.Namespace) -> int:
    from elicitor_distill.train import load_train_config, train

    config = load_train_config(args.config)
    out = args.out or f"{MODELS_DIR}/{config.name}"
    result = train(config, _schema(args.data_dir), out, args.base_dir, args.dry_run)
    if args.dry_run:
        print(json.dumps(result, indent=2))
    else:
        print(f"best epoch {result['best_epoch']}: adapter in {out}/adapter")
    return 0


def _export(args: argparse.Namespace) -> int:
    from elicitor_distill.export import export

    out = Path(args.out or f"{MODELS_DIR}/{args.name.removeprefix('elicitor-')}")
    adapter = Path(args.adapter) if args.adapter else None
    info = export(args.name, Path(args.base), adapter, out, Path(args.llama_cpp), args.outtype,
                  args.create)  # fmt: skip
    print(json.dumps(info, indent=2))
    return 0


def _eval_client(args: argparse.Namespace, constrained: bool, cache: str | None = None) -> Any:
    from elicitor_distill.eval_offline import make_eval_client

    return make_eval_client(f"{args.data_dir}/{DOMAIN}/prompts", args.model, args.provider,
                            args.base_url, args.num_ctx, constrained, cache,
                            think=None if args.think == "none" else False)  # fmt: skip


def _eval_offline(args: argparse.Namespace) -> int:
    from elicitor.experiments.health import format_roles

    from elicitor_distill.eval_offline import evaluate, strict_problems, write_report

    samples = _val_samples(args)
    result = evaluate(samples, args.role, _schema(args.data_dir),
                      lambda constrained: _eval_client(args, constrained),
                      args.workers, args.seed, not args.no_unconstrained, print)  # fmt: skip
    data = args.data or f"{SFT_DIR}/{args.role}_val.jsonl.gz"
    provenance = {"model": args.model, "role": args.role, "data": Path(data).name,
                  "limit": args.limit, "seed": args.seed}  # fmt: skip
    if args.out:
        write_report(args.out, result, provenance)
    print(format_roles(result["llm"]["roles"]))
    print(json.dumps(result["tables"]["overall"]))
    problems = result["problems"] + (strict_problems(result["llm"]) if args.strict else [])
    if problems:
        print("health: PROBLEMS")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("health: OK")
    return 0


def _eval_table(args: argparse.Namespace) -> int:
    from elicitor_distill.eval_offline import collect_summaries, overview_table

    text = overview_table(collect_summaries(args.root))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8", newline="\n")
    print(text)
    return 0


def _check_template(args: argparse.Namespace) -> int:
    from elicitor_distill.export import check_prompts, ollama_prompts

    logged = ollama_prompts(Path(args.log).read_text(encoding="utf-8", errors="replace"))
    problems = check_prompts(logged, _val_samples(args))
    for problem in problems:
        print(problem)
    if not problems:
        print(f"template OK: {len(_val_samples(args))} logged prompts match the serving prompt")
    return 1 if problems else 0


def _bench(args: argparse.Namespace) -> int:
    from elicitor_distill.bench import bench

    client = _eval_client(args, constrained=True)
    try:
        result = bench(client, _val_samples(args), _schema(args.data_dir), args.role, args.workers)
    finally:
        client.close()
    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def _build_test(args: argparse.Namespace) -> int:
    from elicitor_distill.m3 import build_test

    counts = build_test(args.m5, args.judge, args.data_dir, _schema(args.data_dir), args.out,
                        allow_eval=args.allow_eval)  # fmt: skip
    print(", ".join(f"{role}_test: {n} samples" for role, n in counts.items()))
    return 0


def _m3_file(args: argparse.Namespace) -> int:
    from elicitor_distill.m3 import write_m3_file

    print(json.dumps(write_m3_file(args.template, args.out, args.name, args.model,
                                   args.calibration)))  # fmt: skip
    return 0


def _compare(args: argparse.Namespace) -> int:
    from elicitor_distill.compare import compare

    result = compare(args.teacher, args.runs, args.out)
    print(Path(args.out, "results.md").read_text(encoding="utf-8").split("\n\n### Every")[0])
    return 0 if result["rows"] else 1


def _judge_eval(args: argparse.Namespace) -> int:
    from elicitor_distill.judge_eval import judge_table, markdown

    text = markdown(judge_table(args.folders))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8", newline="\n")
    print(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="distill", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("make-personas", help="write (or check) the frozen training personas")
    p.add_argument("--data-dir", default="data", help="folder holding <domain>/ (default: data)")
    p.add_argument("--n", type=int, default=N_TRAIN)
    p.add_argument("--seed", type=int, default=TRAIN_SEED)
    p.add_argument("--check", action="store_true", help="only compare with the committed files")
    p.set_defaults(func=_make_personas)

    def collection(name: str, help: str, func: Any, workers: int) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help)
        p.add_argument("config", help="collection TOML file (e.g. experiments/collect.toml)")
        p.add_argument("--workers", type=int, default=workers, help="threads sending requests")
        p.add_argument(
            "--out", default=RAW_DIR, help=f"raw records and reply cache (default: {RAW_DIR})"
        )
        p.set_defaults(func=func)
        return p

    p = collection("make-backstories", "write the training personas' frozen backstories",
                   _make_backstories, 16)  # fmt: skip
    p.add_argument("--force", action="store_true", help="overwrite an existing backstories file")
    collection("collect", "collect the update teacher's records", _collect, 32)
    collection("judge-collect", "collect the judge teacher's records", _judge_collect, 32)

    p = sub.add_parser("build-sft", help="raw records to the gzipped SFT sets and their manifest")
    p.add_argument("--raw", default=RAW_DIR, help=f"raw records (default: {RAW_DIR})")
    p.add_argument("--out", default=SFT_DIR, help=f"output folder (default: {SFT_DIR})")
    p.add_argument("--data-dir", default="data", help="folder holding <domain>/ (default: data)")
    p.add_argument("--roles", nargs="+", choices=ROLES, default=list(ROLES))
    p.set_defaults(func=_build_sft)

    p = sub.add_parser("train", help="LoRA SFT of a student; picks the best epoch on val")
    p.add_argument("config", help="training TOML file (e.g. configs/update_1.7b.toml)")
    p.add_argument("--out", help=f"output folder (default: {MODELS_DIR}/<name>)")
    p.add_argument("--base-dir", help="folder holding the base weights (offline nodes)")
    p.add_argument("--data-dir", default="data", help="folder holding <domain>/ (default: data)")
    p.add_argument("--dry-run", action="store_true", help="stop once the trainer is built")
    p.set_defaults(func=_train)

    p = sub.add_parser("export", help="adapter or base model to GGUF and an Ollama model")
    p.add_argument("name", help="Ollama model name, e.g. elicitor-upd-1.7b-sft")
    p.add_argument("--base", required=True, help="folder of the base weights")
    p.add_argument("--adapter", help="LoRA adapter folder (none: a zero-shot baseline)")
    p.add_argument("--out", help=f"output folder (default: {MODELS_DIR}/<name without prefix>)")
    p.add_argument("--llama-cpp", default="llama.cpp", help="llama.cpp checkout")
    p.add_argument("--outtype", default=OUTTYPE)
    p.add_argument("--create", action="store_true", help="also `ollama create` the model")
    p.set_defaults(func=_export)

    def served(name: str, help: str, func: Any, workers: int) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help)
        p.add_argument("--model", required=True, help="served model name")
        p.add_argument("--role", choices=EVAL_ROLES, default="update")
        p.add_argument("--data", help="SFT file (default: data/sft/<role>_val.jsonl.gz)")
        p.add_argument("--limit", type=int, help="only the first n samples")
        p.add_argument("--provider", choices=("ollama", "mock"), default="ollama")
        p.add_argument("--base-url", default=os.environ.get("ELICITOR_BASE_URL", DEFAULT_URL))
        p.add_argument("--num-ctx", type=int, default=2048)
        p.add_argument(
            "--think",
            choices=("false", "none"),
            default="false",
            help="send think=false (qwen3) or no think field (llama3.3)",
        )
        p.add_argument("--workers", type=int, default=workers)
        p.add_argument("--data-dir", default="data", help="folder holding <domain>/")
        p.add_argument("--out", help="output folder (eval-offline) or JSON file (bench)")
        p.add_argument("--allow-eval", action="store_true", help="read a test set (M3 only)")
        p.set_defaults(func=func)
        return p

    p = served("eval-offline", "agreement of a served model with the teacher", _eval_offline, 8)
    p.add_argument("--seed", type=int, default=0, help="request seed (1: teacher ceiling)")
    p.add_argument("--no-unconstrained", action="store_true", help="skip the pass without schema")
    p.add_argument("--strict", action="store_true", help="fail on any parse failure (smoke)")
    p = served("bench", "requests per second of a served model", _bench, BENCH_WORKERS)
    p.set_defaults(limit=BENCH_N)

    p = sub.add_parser("eval-table", help="one table of every eval-offline summary in a folder")
    p.add_argument("root", help="folder searched for summary.json files")
    p.add_argument("--out", help="markdown file")
    p.set_defaults(func=_eval_table)

    p = sub.add_parser("check-template", help="compare Ollama's logged prompts with training's")
    p.add_argument("--log", required=True, help="Ollama server log written with OLLAMA_DEBUG=2")
    p.add_argument("--role", choices=EVAL_ROLES, default="update")
    p.add_argument("--data", help="SFT file the logged requests came from")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=_check_template)

    p = sub.add_parser("build-test", help="the M3 test sets from Elicitor's m5_full and m7_judge")
    p.add_argument("--m5", required=True, help="Elicitor's m5_full run folder (read only)")
    p.add_argument("--judge", required=True, help="Elicitor's m7_judge folder (read only)")
    p.add_argument("--out", default=TEST_DIR, help=f"output folder (default: {TEST_DIR})")
    p.add_argument("--data-dir", default="data", help="folder holding <domain>/ (default: data)")
    p.add_argument("--allow-eval", action="store_true", help="required: reads the test set")
    p.set_defaults(func=_build_test)

    p = sub.add_parser("m3-file", help="one model's calibration or end-to-end experiment file")
    p.add_argument("template", help="experiments/m3_calibrate.toml or experiments/m3_e2e.toml")
    p.add_argument("--model", required=True, help="the served agent model")
    p.add_argument("--name", required=True, help="experiment name (the run folder's)")
    p.add_argument("--out", required=True, help="file to write")
    p.add_argument("--calibration", help="calibration.json whose LLM rho the selectors use")
    p.set_defaults(func=_m3_file)

    p = sub.add_parser("compare", help="student runs against the teacher's m5_full, paired")
    p.add_argument("runs", nargs="+", help="student run folders")
    p.add_argument("--teacher", required=True, help="Elicitor's m5_full run folder (read only)")
    p.add_argument("--out", required=True, help="table and figure folder")
    p.set_defaults(func=_compare)

    p = sub.add_parser("judge-eval", help="D5: judge Spearman with utility on the test set")
    p.add_argument("folders", nargs="+", help="eval-offline folders of the judge test set")
    p.add_argument("--out", help="markdown file")
    p.set_defaults(func=_judge_eval)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
