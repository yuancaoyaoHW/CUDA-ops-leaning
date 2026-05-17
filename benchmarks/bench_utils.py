"""Benchmark utilities."""

from __future__ import annotations

import time
from contextlib import contextmanager
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