from __future__ import annotations

import pytest
import torch

from kernels.triton.row_sum import triton_row_sum


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")


@pytest.mark.parametrize("shape", [(128, 256), (128, 257), (1, 257), (513, 64)])
def test_triton_row_sum_matches_torch_float32(shape: tuple[int, int]) -> None:
    torch.manual_seed(shape[0] * 1000 + shape[1])
    x = torch.randn(shape, device="cuda", dtype=torch.float32)

    actual = triton_row_sum(x)
    expected = x.sum(dim=1)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected)


def test_triton_row_sum_matches_torch_float16() -> None:
    torch.manual_seed(1024)
    x = torch.randn((128, 257), device="cuda", dtype=torch.float16)

    actual = triton_row_sum(x)
    expected = x.sum(dim=1)

    assert actual.dtype == x.dtype
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)


def test_triton_row_sum_handles_zero_rows() -> None:
    x = torch.empty((0, 256), device="cuda", dtype=torch.float32)

    actual = triton_row_sum(x)
    expected = x.sum(dim=1)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected)


def test_triton_row_sum_handles_zero_cols() -> None:
    x = torch.empty((8, 0), device="cuda", dtype=torch.float32)

    actual = triton_row_sum(x)
    expected = x.sum(dim=1)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected)


def test_triton_row_sum_rejects_non_2d_input() -> None:
    x = torch.randn(16, device="cuda", dtype=torch.float32)

    with pytest.raises(ValueError, match="2D"):
        triton_row_sum(x)


def test_triton_row_sum_rejects_non_contiguous_input() -> None:
    x = torch.randn((8, 16), device="cuda", dtype=torch.float32).t()

    assert not x.is_contiguous()

    with pytest.raises(ValueError, match="contiguous"):
        triton_row_sum(x)
