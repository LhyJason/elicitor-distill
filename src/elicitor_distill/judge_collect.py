"""Teacher data for the `judge` role (SPEC §4): `distill judge-collect`.

A unit is a (persona, catalog item). Per persona the items are stratified over the true utility:
`per_bin` items from each bin between the `edges` (default < 0.5, 0.5–0.6, 0.6–0.7, 0.7–0.8,
≥ 0.8), drawn with a generator seeded by persona. Most personas have fewer than five items at
≥ 0.8, so a short bin is filled from its neighbours, the nearer first and the higher of two
equally near ones first. Each unit is shown to the main judge of Elicitor's judge study (the user
model group) in the list presentation, with the same context: the persona's profile lines and
the item's attribute lines.

Outputs in the output folder: `judge_<split>.jsonl` and `judge_stats.json`; replies are cached
with the update records' in `llm_cache.sqlite`.
"""

from __future__ import annotations

import bisect
import time
from collections import Counter
from pathlib import Path
from typing import Any

from elicitor.catalog import Item, load_catalog
from elicitor.jsonl import write_jsonl
from elicitor.judge import item_lines
from elicitor.llm import LLMClient, LLMParseError
from elicitor.personas import Persona
from elicitor.recommender import utility, violations
from elicitor.rng import derive_rng, derive_seed
from elicitor.schema import Schema, load_schema
from elicitor.simulator.llm import profile_lines

from elicitor_distill.collect import (
    FAILED,
    OK,
    RAW_DIR,
    Progress,
    make_client,
    run_tracked,
    split_personas,
    write_stats,
)
from elicitor_distill.config import CollectConfig

_TIE = 1e-9  # a utility this close to an edge belongs to the bin above it
_SEED_MOD = 2**31


def utility_bin(u: float, edges: tuple[float, ...]) -> int:
    return bisect.bisect_right(edges, u + _TIE)


def judge_items(
    schema: Schema,
    catalog: list[Item],
    persona: Persona,
    edges: tuple[float, ...],
    per_bin: int,
    seed: int,
) -> list[int]:
    """Catalog indices of the persona's units, in catalog order."""
    bins = [utility_bin(utility(schema, item.attrs, persona.truth), edges) for item in catalog]
    n_bins = len(edges) + 1
    rng = derive_rng(seed, "judge_items", persona.id)
    pools = []
    for b in range(n_bins):
        pool = [i for i, ib in enumerate(bins) if ib == b]
        rng.shuffle(pool)
        pools.append(pool)
    used = [min(per_bin, len(pool)) for pool in pools]
    for b in range(n_bins):
        short = per_bin - min(per_bin, len(pools[b]))
        for distance in range(1, n_bins):
            for nb in (b + distance, b - distance):
                if not (short and 0 <= nb < n_bins):
                    continue
                extra = min(short, len(pools[nb]) - used[nb])
                used[nb] += extra
                short -= extra
    return sorted(i for b in range(n_bins) for i in pools[b][: used[b]])


def judge_one(
    client: LLMClient,
    schema: Schema,
    catalog: list[Item],
    edges: tuple[float, ...],
    seed: int,
    persona: Persona,
    index: int,
) -> dict[str, Any]:
    item = catalog[index]
    u = utility(schema, item.attrs, persona.truth)
    context = {
        "profile": profile_lines(schema, persona),
        "item": item_lines(schema, item.attrs),
        "utility": u,  # for the mock only (not a string, so never in the prompt)
    }
    role = client.roles["judge"]
    row: dict[str, Any] = {
        "persona": persona.id,
        "split": persona.split,
        "style": persona.style,
        "item": item.id,
        "utility": round(u, 6),
        "bin": utility_bin(u, edges),
        "violations": violations(schema, item.attrs, persona.truth),
        "teacher": client.models[role.group],
        "messages": role.prompt.render(context),
    }
    try:
        reply = client.call(
            "judge", context, seed=derive_seed(seed, "judge", persona.id, item.id) % _SEED_MOD
        )
    except LLMParseError:
        return {**row, "status": FAILED}
    return {**row, "status": OK, "score": reply.score, "reason": reply.reason}  # type: ignore[attr-defined]


def judge_collect(
    config: CollectConfig,
    out_dir: str | Path = RAW_DIR,
    workers: int = 1,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Judge every split's units; writes the records and `judge_stats.json`."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    schema = load_schema(config.schema_path)
    catalog = load_catalog(config.catalog_path)
    edges = config.judge_edges
    tasks = [
        (persona, index)
        for split in config.splits
        for persona in split_personas(config, split)
        for index in judge_items(schema, catalog, persona, edges, config.judge_per_bin, config.seed)
    ]
    client = make_client(config, out)
    start = time.perf_counter()
    try:
        rows, llm = run_tracked(
            client,
            lambda t: judge_one(client, schema, catalog, edges, config.seed, *t),
            tasks,
            workers,
            progress,
            label="judge records",
        )
    finally:
        client.close()

    counts = {}
    for split in config.splits:
        mine = [r for r in rows if r["split"] == split]
        write_jsonl(out / f"judge_{split}.jsonl", mine)
        status = Counter(r["status"] for r in mine)
        counts[split] = {
            "personas": len({r["persona"] for r in mine}),
            "records": len(mine),
            **{s: status.get(s, 0) for s in (OK, FAILED)},
            "per_bin": dict(sorted(Counter(r["bin"] for r in mine).items())),
        }
    return write_stats(
        out / "judge_stats.json",
        config,
        {"seconds": round(time.perf_counter() - start, 1), "splits": counts, "llm": llm},
    )
