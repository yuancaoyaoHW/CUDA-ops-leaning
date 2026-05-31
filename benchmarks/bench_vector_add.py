"""Benchmark Triton vector add vs PyTorch."""

from __future__ import annotations

import time
from pathlib import Path

import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.bench_utils import write_benchmark_json
from kernels.triton.vector_add import triton_vector_add


def bench(func, args, warmup: int = 10, repeat: int = 100) -> float:
    """Benchmark a CUDA function."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        func(*args)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeat):
        func(*args)
    torch.cuda.synchronize()

    return (time.perf_counter() - start) / repeat * 1e3  # ms


def bandwidth_gbs(n_elements: int, dtype: torch.dtype, elapsed_ms: float) -> float:
    """Estimate effective memory bandwidth in GB/s.

    Vector add reads two inputs and writes one output, so we count 3 transfers.
    """
    bytes_moved = 3 * n_elements * torch.empty((), dtype=dtype).element_size()
    return bytes_moved / (elapsed_ms / 1e3) / 1e9


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    device_name = torch.cuda.get_device_name(0)
    print(f"device: {device_name}")
    print()

    results = []
    for n in [1024, 4096, 65536, 1_048_576, 16_777_216]:
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        y = torch.randn_like(x)

        # Warmup
        triton_vector_add(x, y)
        x + y
        torch.cuda.synchronize()

        # Benchmark
        t_triton = bench(triton_vector_add, (x, y))
        t_torch = bench(lambda a, b: a + b, (x, y))
        gbps_triton = bandwidth_gbs(n, x.dtype, t_triton)
        gbps_torch = bandwidth_gbs(n, x.dtype, t_torch)
        dtype = str(x.dtype)

        results.extend(
            [
                {
                    "shape": [n],
                    "dtype": dtype,
                    "implementation": "triton",
                    "ms": t_triton,
                    "gbps": gbps_triton,
                },
                {
                    "shape": [n],
                    "dtype": dtype,
                    "implementation": "torch",
                    "ms": t_torch,
                    "gbps": gbps_torch,
                },
            ]
        )

        print(
            f"n={n:>12,} | "
            f"Triton: {t_triton:7.3f} ms | {gbps_triton:7.2f} GB/s | "
            f"PyTorch: {t_torch:7.3f} ms | {gbps_torch:7.2f} GB/s | "
            f"ratio: {t_triton/t_torch:.2f}"
        )

    path = write_benchmark_json("vector_add", results, extra={"device": device_name})
    print()
    print(f"wrote JSON: {path}")


if __name__ == "__main__":
    main()
