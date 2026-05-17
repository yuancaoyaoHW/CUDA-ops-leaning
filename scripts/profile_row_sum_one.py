#!/usr/bin/env python3
"""Run one Triton row_sum launch for Nsight Compute profiling."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernels.triton.row_sum import triton_row_sum


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    x = torch.empty((4096, 2048), device="cuda", dtype=torch.float32)

    # First launch lets Triton compile and warm the kernel.
    triton_row_sum(x)
    torch.cuda.synchronize()

    # Profile this launch with: ncu --launch-skip 1 --launch-count 1 ...
    output = triton_row_sum(x)
    torch.cuda.synchronize()

    print(f"row_sum output shape: {tuple(output.shape)}")


if __name__ == "__main__":
    main()
