from __future__ import annotations

import pytest
import torch

from kernels.triton.axpy import triton_axpy


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")


@pytest.mark.parametrize("n_elements", [1024, 4097])
def test_triton_axpy_matches_torch(n_elements: int) -> None:
    torch.manual_seed(n_elements)
    x = torch.randn(n_elements, device="cuda", dtype=torch.float32)
    y = torch.randn(n_elements, device="cuda", dtype=torch.float32)
    alpha = 0.5

    actual = triton_axpy(x, y, alpha=alpha)
    expected = alpha * x + y

    torch.testing.assert_close(actual, expected)


def test_triton_axpy_rejects_shape_mismatch() -> None:
    x = torch.randn(16, device="cuda", dtype=torch.float32)
    y = torch.randn(17, device="cuda", dtype=torch.float32)

    with pytest.raises(ValueError, match="same shape"):
        triton_axpy(x, y)


def test_triton_axpy_rejects_non_contiguous_inputs() -> None:
    x = torch.randn(4, 8, device="cuda", dtype=torch.float32).t()
    y = torch.randn(4, 8, device="cuda", dtype=torch.float32).t()

    assert not x.is_contiguous()
    assert not y.is_contiguous()

    with pytest.raises(ValueError, match="contiguous"):
        triton_axpy(x, y)


def test_triton_axpy_handles_empty_tensor() -> None:
    x = torch.empty(0, device="cuda", dtype=torch.float32)
    y = torch.empty(0, device="cuda", dtype=torch.float32)

    actual = triton_axpy(x, y, alpha=0.5)
    expected = 0.5 * x + y

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected)


def test_triton_axpy_matches_torch_float16() -> None:
    torch.manual_seed(1024)
    x = torch.randn(1024, device="cuda", dtype=torch.float16)
    y = torch.randn(1024, device="cuda", dtype=torch.float16)
    alpha = 0.5

    actual = triton_axpy(x, y, alpha=alpha)
    expected = alpha * x + y

    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
