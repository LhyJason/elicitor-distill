"""Throughput of a served model on the `update` role (SPEC §7.3): `distill bench`.

A fixed load: the first `n` validation samples, sent by `workers` threads without a reply cache
to one server, after one warm-up call (which loads the model). Reported: requests per second,
completion tokens per second, p50 / p95 latency and the GPU memory in use at the end (from
`nvidia-smi`, when there is one). The teacher is measured the same way.
"""

from __future__ import annotations

import contextlib
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from elicitor.llm import LLMClient, LLMParseError
from elicitor.schema import Schema

from elicitor_distill.eval_offline import EVAL_SEED, sample_context

BENCH_N = 200
BENCH_WORKERS = 16


def gpu_memory_mib() -> list[int] | None:
    """Memory in use per GPU, or None without `nvidia-smi`."""
    if shutil.which("nvidia-smi") is None:
        return None
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        return None
    return [int(line) for line in out.stdout.split() if line.strip().isdigit()]


def bench(
    client: LLMClient,
    samples: list[dict[str, Any]],
    schema: Schema,
    role: str = "update",
    workers: int = BENCH_WORKERS,
) -> dict[str, Any]:
    if client.cache is not None:
        raise ValueError("bench needs a client without a reply cache")
    contexts = [sample_context(role, s, schema) for s in samples]
    client.call(role, contexts[0], seed=EVAL_SEED)  # warm-up: loads the model
    client.reset_stats()  # the pool's threads count into this client-wide tally

    def timed(context: dict[str, Any]) -> float:
        start = time.perf_counter()
        with contextlib.suppress(LLMParseError):  # counted as a failed call
            client.call(role, context, seed=EVAL_SEED)
        return time.perf_counter() - start

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        latencies = list(pool.map(timed, contexts))
    wall = time.perf_counter() - start
    stats = client.stats_dict()["roles"].get(role, {})
    return {
        "role": role,
        "model": client.models["agent"],
        "requests": len(contexts),
        "workers": workers,
        "seconds": round(wall, 2),
        "requests_per_s": round(len(contexts) / wall, 3),
        "completion_tokens_per_s": round(stats.get("completion_tokens", 0) / wall, 1),
        "latency_p50_s": round(float(np.percentile(latencies, 50)), 3),
        "latency_p95_s": round(float(np.percentile(latencies, 95)), 3),
        "parse_failures": stats.get("parse_failures", 0),
        "failed_calls": stats.get("failed_calls", 0),
        "gpu_memory_mib": gpu_memory_mib(),
    }
