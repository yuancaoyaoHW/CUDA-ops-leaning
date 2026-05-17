from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit()
def row_max_kernel(x_ptr, output_ptr, cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * cols + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < cols

    values = tl.load(x_ptr + offsets, mask=mask, other=-float("inf"))
    row_max = tl.max(values, axis=0)

    tl.store(output_ptr + pid, row_max)


def triton_row_max(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 2:
        raise ValueError("x must be a 2D tensor")
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")

    rows, cols = x.shape
    output = torch.empty(rows, device=x.device, dtype=x.dtype)
    if rows == 0:
        return output
    if cols == 0:
        raise ValueError("x must have at least one column")

    BLOCK_SIZE = triton.next_power_of_2(cols)
    grid = lambda meta: (rows,)
    row_max_kernel[grid](x, output, cols, BLOCK_SIZE=BLOCK_SIZE)

    return output


if __name__ == "__main__":
    # 初始化一个随机矩阵 (例如 4 行, 100 列)
    torch.manual_seed(42)
    rows, cols = 4, 100
    x = torch.rand((rows, cols), device="cuda", dtype=torch.float32)

    # 调用 Triton 算子
    out_triton = triton_row_max(x)

    # 调用原生的 PyTorch 算子作为对照基准
    out_torch = torch.max(x, dim=1)

    print("Triton output:\n", out_triton)
    print("PyTorch output:\n", out_torch)

    # 验证最大误差
    max_diff = torch.max(torch.abs(out_triton - out_torch[0])).item()
    print(f"\nMax difference between Triton and PyTorch: {max_diff:.6e}")

    if max_diff < 1e-5:
        print("✅ 测试通过！")
    else:
        print("❌ 结果不一致！")
