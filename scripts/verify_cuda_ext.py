#!/usr/bin/env python3
"""Verify PyTorch CUDA extension JIT compilation."""

from __future__ import annotations

import torch
from torch.utils.cpp_extension import load_inline

CPP_SRC = r'''
#include <torch/extension.h>

torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_cuda", &add_cuda, "add_cuda");
}
'''

CUDA_SRC = r'''
#include <torch/extension.h>

__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b) {
    auto c = torch::empty_like(a);
    int n = a.numel();
    int blocks = (n + 255) / 256;
    add_kernel<<<blocks, 256>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n);
    return c;
}
'''


def main():
    if not torch.cuda.is_available():
        raise SystemExit("ERROR: CUDA not available")

    print(f"torch: {torch.__version__}")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print("Compiling CUDA extension (first run may take 30-60s)...")

    mod = load_inline(
        name="mini_add_ext",
        cpp_sources=CPP_SRC,
        cuda_sources=CUDA_SRC,
        with_cuda=True,
        extra_cuda_cflags=["-O3"],
    )

    a = torch.randn(1_000_000, device="cuda", dtype=torch.float32)
    b = torch.randn_like(a)
    c = mod.add_cuda(a, b)

    torch.testing.assert_close(c, a + b)
    print("CUDA extension: OK")


if __name__ == "__main__":
    main()
