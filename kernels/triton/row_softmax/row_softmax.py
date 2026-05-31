from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit()
def row_softmax_kernel(
    x_ptr,
    output_ptr,
    cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * cols + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < cols

    values = tl.load(x_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    row_max = tl.max(values, axis=0)
    numerator = tl.exp(values - row_max)
    denominator = tl.sum(numerator, axis=0)
    output = numerator / denominator

    tl.store(output_ptr + offsets, output, mask=mask)


def _check_row_softmax_input(x: torch.Tensor, *, require_cuda: bool) -> None:
    if x.dim() != 2:
        raise ValueError("x must be a 2D tensor")
    if require_cuda and not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if x.dtype not in (torch.float32, torch.float16):
        raise ValueError("x dtype must be float32 or float16")
    if x.shape[1] == 0:
        raise ValueError("x must have at least one column")


def pytorch_row_softmax(x: torch.Tensor) -> torch.Tensor:
    _check_row_softmax_input(x, require_cuda=False)

    row_max = x.max(dim=1, keepdim=True).values
    numerator = torch.exp(x.float() - row_max.float())
    denominator = numerator.sum(dim=1, keepdim=True)
    return (numerator / denominator).to(x.dtype)


def triton_row_softmax(x: torch.Tensor) -> torch.Tensor:
    _check_row_softmax_input(x, require_cuda=True)

    rows, cols = x.shape
    output = torch.empty_like(x)
    if rows == 0:
        return output

    BLOCK_SIZE = triton.next_power_of_2(cols)
    grid = lambda meta: (rows,)
    row_softmax_kernel[grid](x, output, cols, BLOCK_SIZE=BLOCK_SIZE)

    return output
