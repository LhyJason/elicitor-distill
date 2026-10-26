"""`distill`: one argparse CLI (SPEC §8). Run it from the repository root."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Any

from elicitor_distill import __version__
from elicitor_distill.collect import RAW_DIR
from elicitor_distill.data import DOMAIN, verify_domain_data
from elicitor_distill.personas import (
    N_TRAIN,
    TRAIN_SEED,
    check_train_personas,
    write_train_personas,
)
from elicitor_distill.sft_data import ROLES, SFT_DIR


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
