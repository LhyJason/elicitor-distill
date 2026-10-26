"""The domain data this project builds on (SPEC §1).

`data/travel/` is a verbatim copy of Elicitor's domain files at the pinned commit: the schema,
the 60 dev/eval personas with their backstories, the catalog and every prompt. They are copied,
not read from the installed package, because Elicitor keeps them outside its package. The copy
is checked against the recorded SHA-256 sums so a drift from the pinned commit cannot go
unnoticed (the M5 run's fingerprint depends on these bytes).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

ELICITOR_COMMIT = "41358e44bb27644d3cf4f4fc2188115a8d7e0c6a"
DOMAIN = "travel"
CHECKSUMS = "CHECKSUMS.sha256"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_checksums(domain_dir: Path) -> dict[str, str]:
    """{relative path: sha256} from the `sha256sum` style file in the domain folder."""
    out: dict[str, str] = {}
    for line in (domain_dir / CHECKSUMS).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        out[name.lstrip("*")] = digest
    return out


def verify_domain_data(domain_dir: str | Path) -> list[str]:
    """Relative paths whose content differs from the recorded checksum (or that are missing)."""
    domain_dir = Path(domain_dir)
    bad = []
    for name, digest in read_checksums(domain_dir).items():
        path = domain_dir / name
        if not path.exists() or sha256_file(path) != digest:
            bad.append(name)
    return bad
