#!/usr/bin/env python3
"""Verify Triton kernel compilation and execution."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x, y, z, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(x + offs, mask=mask, other=0.0)
    b = tl.load(y + offs, mask=mask, other=0.0)
    tl.store(z + offs, a + b, mask=mask)


def main():
    if not torch.cuda.is_available():
        raise SystemExit("ERROR: CUDA not available")

    print(f"torch: {torch.__version__}")
    print(f"triton: {triton.__version__}")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"capability: {torch.cuda.get_device_capability(0)}")

    n = 1_000_003
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    z = torch.empty_like(x)

    grid = (triton.cdiv(n, 1024),)
    add_kernel[grid](x, y, z, n, BLOCK=1024)

    torch.testing.assert_close(z, x + y)
    print("Triton vector add: OK")


if __name__ == "__main__":
    main()