from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kernels.triton.row_softmax import pytorch_row_softmax
from kernels.triton.row_softmax import triton_row_softmax


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available"
)


def stable_reference(x: torch.Tensor) -> torch.Tensor:
    row_max = x.max(dim=1, keepdim=True).values
    numerator = torch.exp(x.float() - row_max.float())
    return (numerator / numerator.sum(dim=1, keepdim=True)).to(x.dtype)


@pytest.mark.parametrize("shape", [(128, 256), (128, 257), (1, 257), (513, 64)])
def test_pytorch_row_softmax_matches_torch_float32(shape: tuple[int, int]) -> None:
    torch.manual_seed(shape[0] * 1000 + shape[1])
    x = torch.randn(shape, device="cuda", dtype=torch.float32)

    actual = pytorch_row_softmax(x)
    expected = torch.softmax(x, dim=1)

    assert actual.shape == x.shape
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("shape", [(128, 256), (128, 257), (1, 257), (513, 64)])
def test_triton_row_softmax_matches_torch_float32(shape: tuple[int, int]) -> None:
    torch.manual_seed(shape[0] * 1000 + shape[1])
    x = torch.randn(shape, device="cuda", dtype=torch.float32)

    actual = triton_row_softmax(x)
    expected = torch.softmax(x, dim=1)

    assert actual.shape == x.shape
    assert actual.dtype == x.dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("shape", [(128, 256), (33, 1000)])
def test_triton_row_softmax_matches_torch_float16(shape: tuple[int, int]) -> None:
    torch.manual_seed(shape[0] * 1000 + shape[1])
    x = torch.randn(shape, device="cuda", dtype=torch.float16)

    actual = triton_row_softmax(x)
    expected = torch.softmax(x, dim=1)

    assert actual.shape == x.shape
    assert actual.dtype == x.dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)


def test_triton_row_softmax_handles_zero_rows() -> None:
    x = torch.empty((0, 256), device="cuda", dtype=torch.float32)

    actual = triton_row_softmax(x)

    assert actual.shape == x.shape
    torch.testing.assert_close(actual, torch.softmax(x, dim=1))


def test_triton_row_softmax_uses_negative_infinity_for_masked_values() -> None:
    x = -torch.arange(1, 1 + 3 * 257, device="cuda", dtype=torch.float32).reshape(3, 257)

    actual = triton_row_softmax(x)
    expected = stable_reference(x)

    assert torch.all(x < 0)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_triton_row_softmax_is_stable_for_large_logits() -> None:
    x = torch.tensor(
        [
            [1000.0, 1001.0, 1002.0, -1000.0, -1001.0],
            [-1000.0, -999.0, -998.0, -997.0, -996.0],
        ],
        device="cuda",
        dtype=torch.float32,
    )

    actual = triton_row_softmax(x)
    expected = stable_reference(x)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_triton_row_softmax_rejects_non_2d_input() -> None:
    x = torch.randn(16, device="cuda", dtype=torch.float32)

    with pytest.raises(ValueError, match="2D"):
        triton_row_softmax(x)


def test_triton_row_softmax_rejects_non_contiguous_input() -> None:
    x = torch.randn((8, 16), device="cuda", dtype=torch.float32).t()

    assert not x.is_contiguous()

    with pytest.raises(ValueError, match="contiguous"):
        triton_row_softmax(x)


def test_triton_row_softmax_rejects_zero_cols() -> None:
    x = torch.empty((8, 0), device="cuda", dtype=torch.float32)

    with pytest.raises(ValueError, match="at least one column"):
        triton_row_softmax(x)


def test_triton_row_softmax_rejects_unsupported_dtype() -> None:
    x = torch.ones((8, 16), device="cuda", dtype=torch.float64)

    with pytest.raises(ValueError, match="float32 or float16"):
        triton_row_softmax(x)
