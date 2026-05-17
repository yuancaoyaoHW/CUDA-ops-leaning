"""Benchmark Triton row_max vs PyTorch."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernels.triton.row_max import triton_row_max


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

    print(f"device: {torch.cuda.get_device_name(0)}")
    print()

    shapes = [
        (128, 256),
        (128, 257),
        (512, 512),
        (1024, 1024),
        (2048, 1024),
        (4096, 2048),
    ]

    for rows, cols in shapes:
        x = torch.randn((rows, cols), device="cuda", dtype=torch.float32)

        triton_row_max(x)
        x.max(dim=1).values
        torch.cuda.synchronize()

        t_triton = bench(triton_row_max, (x,))
        t_torch = bench(lambda a: a.max(dim=1).values, (x,))
        gbps_triton = bandwidth_gbs(rows, cols, x.dtype, t_triton)
        gbps_torch = bandwidth_gbs(rows, cols, x.dtype, t_torch)

        print(
            f"shape=({rows:>4}, {cols:>4}) | "
            f"Triton: {t_triton:7.3f} ms | {gbps_triton:7.2f} GB/s | "
            f"PyTorch: {t_torch:7.3f} ms | {gbps_torch:7.2f} GB/s | "
            f"ratio: {t_triton/t_torch:.2f}"
        )


if __name__ == "__main__":
    main()
