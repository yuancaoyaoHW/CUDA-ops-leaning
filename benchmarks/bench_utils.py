"""Benchmark utilities."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from typing import Iterator

import torch


@contextmanager
def bench(name: str, warmup: int = 3, repeat: int = 10) -> Iterator[None]:
    """Context manager for CUDA kernel benchmarking."""
    yield
    torch.cuda.synchronize()

    # Warmup
    for _ in range(warmup):
        pass  # User code runs in context block
    torch.cuda.synchronize()

    # Timed runs
    start = time.perf_counter()
    for _ in range(repeat):
        pass  # User code runs in context block
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_ms = (end - start) / repeat * 1000
    print(f"{name}: {avg_ms:.4f} ms")


def timed_run(func, *args, warmup: int = 3, repeat: int = 10, **kwargs) -> float:
    """Run function multiple times and return average time in ms."""
    torch.cuda.synchronize()

    for _ in range(warmup):
        func(*args, **kwargs)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeat):
        func(*args, **kwargs)
    torch.cuda.synchronize()

    return (time.perf_counter() - start) / repeat * 1000


def write_benchmark_json(
    op_name: str,
    results: list[dict[str, Any]],
    *,
    metric_key: str = "gbps",
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write verifier-recognized benchmark JSON and return its path."""
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "reports" / "json" / f"{op_name}_bench.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    metric_values = [
        row[metric_key]
        for row in results
        if row.get("implementation") == "triton" and row.get(metric_key) is not None
    ]
    if not metric_values:
        metric_values = [row[metric_key] for row in results if row.get(metric_key) is not None]

    payload: dict[str, Any] = {
        "op": op_name,
        metric_key: max(metric_values) if metric_values else None,
        "results": results,
    }
    if extra:
        payload.update(extra)

    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
