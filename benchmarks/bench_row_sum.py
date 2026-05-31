"""Benchmark Triton row_sum vs PyTorch."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.bench_utils import write_benchmark_json
from kernels.triton.row_sum import triton_row_sum


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


def bandwidth_gbs(rows: int, cols: int, dtype: torch.dtype, elapsed_ms: float) -> float:
    """Estimate effective memory bandwidth in GB/s."""
    element_size = torch.empty((), dtype=dtype).element_size()
    bytes_moved = (rows * cols + rows) * element_size
    return bytes_moved / (elapsed_ms / 1e3) / 1e9


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    device_name = torch.cuda.get_device_name(0)
    print(f"device: {device_name}")
    print()

    shapes = [
        (128, 256),
        (128, 257),
        (512, 512),
        (1024, 1024),
        (2048, 1024),
        (4096, 2048),
    ]

    results = []
    for rows, cols in shapes:
        x = torch.randn((rows, cols), device="cuda", dtype=torch.float32)

        triton_row_sum(x)
        x.sum(dim=1)
        torch.cuda.synchronize()

        t_triton = bench(triton_row_sum, (x,))
        t_torch = bench(lambda a: a.sum(dim=1), (x,))
        gbps_triton = bandwidth_gbs(rows, cols, x.dtype, t_triton)
        gbps_torch = bandwidth_gbs(rows, cols, x.dtype, t_torch)
        dtype = str(x.dtype)

        results.extend(
            [
                {
                    "shape": [rows, cols],
                    "dtype": dtype,
                    "implementation": "triton",
                    "ms": t_triton,
                    "gbps": gbps_triton,
                },
                {
                    "shape": [rows, cols],
                    "dtype": dtype,
                    "implementation": "torch",
                    "ms": t_torch,
                    "gbps": gbps_torch,
                },
            ]
        )

        print(
            f"shape=({rows:>4}, {cols:>4}) | "
            f"Triton: {t_triton:7.3f} ms | {gbps_triton:7.2f} GB/s | "
            f"PyTorch: {t_torch:7.3f} ms | {gbps_torch:7.2f} GB/s | "
            f"ratio: {t_triton/t_torch:.2f}"
        )

    path = write_benchmark_json("row_sum", results, extra={"device": device_name})
    print()
    print(f"wrote JSON: {path}")


if __name__ == "__main__":
    main()
