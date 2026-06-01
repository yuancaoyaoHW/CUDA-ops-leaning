"""Triton kernel smoke test — must be a .py file (triton.jit needs inspect.getsource)."""
import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


n = 1024
x = torch.randn(n, device="cuda")
y = torch.randn(n, device="cuda")
out = torch.empty(n, device="cuda")
_add_kernel[(n // 256,)](x, y, out, n, BLOCK=256)
assert torch.allclose(out, x + y), "Triton kernel test failed!"
print("  ✓ Triton kernel smoke test passed")
