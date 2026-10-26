"""`distill`: one argparse CLI (SPEC §8). Run it from the repository root."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from elicitor_distill import __version__
from elicitor_distill.data import DOMAIN, verify_domain_data
from elicitor_distill.personas import (
    N_TRAIN,
    TRAIN_SEED,
    check_train_personas,
    write_train_personas,
)


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
