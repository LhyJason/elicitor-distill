"""Raw teacher records to chat SFT samples (SPEC §4): `distill build-sft`.

One sample per line: the system and user messages exactly as the teacher received them, and an
assistant message holding the teacher's reply re-serialised through the role's pydantic model
(compact JSON, keys in schema order), so the student learns one canonical format. Records whose
teacher call failed, and answers the simulator left empty (which the updater never reads), are
dropped; a prompt that occurs twice in a split (the same answer drawn by both answer seeds) is
kept once, since the teacher's reply to it is the same cached reply. Training personas become the
`train` set, Elicitor's dev personas the `val` set.

The files are gzipped JSONL written byte for byte reproducibly (no timestamp or file name in the
gzip header). `MANIFEST.json` records rows, drops and the SHA-256 of every file, compressed and
uncompressed.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from elicitor.jsonl import dumps_jsonl, read_jsonl
from elicitor.llm.roles import JudgeReply, update_model
from elicitor.personas import DEV
from elicitor.schema import Schema

from elicitor_distill.collect import EMPTY, FAILED, OK
from elicitor_distill.splits import TRAIN

SFT_DIR = "data/sft"
MANIFEST = "MANIFEST.json"
ROLES = ("update", "judge")
SETS = {TRAIN: "train", DEV: "val"}  # persona split -> SFT set


def update_reply(row: dict[str, Any], schema: Schema) -> str:
    model = update_model(schema.slot(row["slot"]))
    reply = model.model_validate({"evidence": row["evidence"], "scores": row["scores"]})
    return reply.model_dump_json(by_alias=True)


def judge_reply(row: dict[str, Any], schema: Schema) -> str:
    return JudgeReply(reason=row["reason"], score=row["score"]).model_dump_json()


def update_meta(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "persona": row["persona"],
        "style": row["style"],
        "slot": row["slot"],
        "kind": row["question"]["kind"],
        "answer_index": row["answer_index"],
    }


def judge_meta(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "persona": row["persona"],
        "style": row["style"],
        "item": row["item"],
        "utility": row["utility"],
        "bin": row["bin"],
    }


_BUILDERS: dict[str, tuple[Callable[..., str], Callable[[dict[str, Any]], dict[str, Any]]]] = {
    "update": (update_reply, update_meta),
    "judge": (judge_reply, judge_meta),
}


def sft_samples(
    role: str, rows: list[dict[str, Any]], schema: Schema
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Samples in record order, and how many records were dropped for which reason."""
    reply_of, meta_of = _BUILDERS[role]
    samples, seen = [], set()
    dropped = Counter({FAILED: 0, EMPTY: 0, "duplicate": 0})
    for row in rows:
        if row["status"] != OK:
            dropped[row["status"]] += 1
            continue
        prompt = row["messages"]
        key = json.dumps(prompt, ensure_ascii=False)
        if key in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(key)
        assistant = {"role": "assistant", "content": reply_of(row, schema)}
        samples.append({"messages": [*prompt, assistant], "meta": meta_of(row)})
    return samples, dict(dropped)


def gzip_bytes(data: bytes) -> bytes:
    """gzip with an empty header (mtime 0, no file name), so equal input gives equal bytes."""
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as f:
        f.write(data)
    return buffer.getvalue()


def read_sft(path: str | Path) -> list[dict[str, Any]]:
    text = gzip.decompress(Path(path).read_bytes()).decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def build_sft(
    raw_dir: str | Path, out_dir: str | Path, schema: Schema, roles: tuple[str, ...] = ROLES
) -> dict[str, Any]:
    """Write `<role>_<set>.jsonl.gz` for every role and set, and the manifest; returns it.
    Raises FileNotFoundError when a raw file is missing."""
    raw, out = Path(raw_dir), Path(out_dir)
    paths = {(role, split): raw / f"{role}_{split}.jsonl" for role in roles for split in SETS}
    missing = [p.as_posix() for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"raw records missing: {', '.join(missing)}")
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {}
    teachers: dict[str, list[str]] = {}
    for (role, split), path in paths.items():
        rows = read_jsonl(path)
        samples, dropped = sft_samples(role, rows, schema)
        text = dumps_jsonl(samples).encode("utf-8")
        packed = gzip_bytes(text)
        name = f"{role}_{SETS[split]}.jsonl.gz"
        (out / name).write_bytes(packed)
        files[name] = {
            "rows": len(samples),
            "records": len(rows),
            "personas": len({s["meta"]["persona"] for s in samples}),
            "dropped": dropped,
            "sha256": hashlib.sha256(packed).hexdigest(),
            "sha256_jsonl": hashlib.sha256(text).hexdigest(),
        }
        found = sorted({r["teacher"] for r in rows if "teacher" in r})
        teachers[role] = sorted({*teachers.get(role, []), *found})
    manifest = {"teachers": teachers, "files": files}
    (out / MANIFEST).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest
