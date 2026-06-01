# Triton Kernels Lab

> 用 Triton 实现 LLM 推理核心算子，与 CUDA 版本进行性能和开发效率对比。

[![Triton](https://img.shields.io/badge/Triton-2.x-orange.svg)](https://triton-lang.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)

## Motivation

Triton 是 OpenAI 开发的 GPU 编程语言，以 block-level programming model 大幅降低 GPU kernel 开发门槛。越来越多的团队（Meta、NVIDIA、各大 LLM 公司）使用 Triton 替代手写 CUDA 进行快速原型开发。本项目用 Triton 重新实现 CUDA Ops Lab 中的核心算子，量化对比两者的性能差距和开发效率差异。

## Key Results

> ⚠️ 以下为目标值，需完成实现后填入实际数据

- 目标：Triton softmax 达到 CUDA 版本 80-100% 性能
- 目标：Triton GEMM 达到 CUDA 版本 70-90% 性能
- 目标：Triton 开发效率比 CUDA 高 3-5×（代码行数、开发时间）
- 目标：Triton fused_attention 正确实现 FlashAttention 算法

## Directory Structure

```
triton-kernels-lab/
├── kernels/
│   ├── vector_add.py              # 基础 Triton kernel
│   ├── matmul.py                  # Tiled GEMM with autotune
│   ├── softmax.py                 # Fused row-wise softmax
│   ├── layernorm.py               # Fused LayerNorm
│   ├── fused_attention.py         # FlashAttention in Triton
│   └── quantization.py            # INT8 quantize/dequantize
├── benchmarks/
│   ├── bench_all.py               # 统一 benchmark
│   ├── bench_vs_cuda.py           # Triton vs CUDA 对比
│   ├── bench_vs_pytorch.py        # Triton vs PyTorch 对比
│   └── plot_comparison.py         # 对比可视化
├── tests/
│   ├── test_vector_add.py
│   ├── test_matmul.py
│   ├── test_softmax.py
│   ├── test_layernorm.py
│   ├── test_attention.py
│   └── test_quantization.py
├── docs/
│   ├── triton_vs_cuda_report.md   # 对比分析报告
│   └── dev_efficiency_analysis.md # 开发效率分析
└── README.md
```

## Kernel 详细设计

### 01. Vector Add

```python
@triton.jit
def vector_add_kernel(x_ptr, y_ptr, out_ptr, n,
                      BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)
```

| 项目 | 内容 |
|------|------|
| **对比目标** | CUDA vector_add (grid-stride loop) |
| **Benchmark** | N = [1K, 100K, 1M, 10M, 100M] |
| **预期结论** | 性能接近 CUDA，代码量减少 60%+ |

### 02. MatMul

| 项目 | 内容 |
|------|------|
| **实现** | Block tiling + `@triton.autotune` 自动选择 tile size |
| **对比目标** | CUDA gemm_tiled / gemm_vectorized |
| **Benchmark** | M=N=K = [512, 1024, 2048, 4096] |
| **Autotune 配置** | BLOCK_M=[64,128], BLOCK_N=[64,128], BLOCK_K=[16,32,64] |
| **预期结论** | 达到 CUDA 手写版本 70-90%，开发时间 1/5 |

### 03. Softmax

| 项目 | 内容 |
|------|------|
| **实现** | 单 block 处理一行，online max + sum + normalize |
| **对比目标** | CUDA softmax_online |
| **Benchmark** | rows×cols = [1024×512, 1024×2048, 4096×4096] |
| **预期结论** | 性能接近甚至超过 CUDA 版本（Triton 编译器优化） |

### 04. LayerNorm

| 项目 | 内容 |
|------|------|
| **实现** | Fused mean + variance + normalize in single kernel |
| **对比目标** | CUDA layernorm + PyTorch `F.layer_norm` |
| **Benchmark** | batch×hidden = [32×4096, 128×4096] |
| **预期结论** | 性能接近 CUDA，比 PyTorch 快 10-20% |

### 05. Fused Attention

| 项目 | 内容 |
|------|------|
| **实现** | FlashAttention 算法：Q/K/V tiling + online softmax |
| **对比目标** | CUDA flash_attention_toy + `flash-attn` 官方库 |
| **Benchmark** | seq_len = [256, 512, 1024, 2048], heads=32, head_dim=128 |
| **预期结论** | 达到 CUDA 简化版 60-80%，远超 naive attention |

### 06. Quantization Kernel

| 项目 | 内容 |
|------|------|
| **实现** | FP16→INT8 per-tensor/per-channel quantize + dequantize |
| **对比目标** | CUDA quantize_int8 |
| **Benchmark** | tensor size = [1M, 10M, 100M] |
| **预期结论** | 性能接近 CUDA，展示 Triton 处理整数运算的能力 |

---

## Setup

```bash
# 安装依赖
pip install triton torch numpy pytest matplotlib

# 运行测试
pytest tests/ -v

# 运行 benchmark
python benchmarks/bench_all.py

# 运行 CUDA vs Triton 对比
python benchmarks/bench_vs_cuda.py --output results/comparison.json
```

## Correctness Test Design

```python
import torch
import pytest
from kernels.matmul import triton_matmul

@pytest.mark.parametrize("M,N,K", [(512,512,512), (1024,1024,1024), (2048,2048,2048)])
def test_matmul(M, N, K):
    A = torch.randn(M, K, device='cuda', dtype=torch.float16)
    B = torch.randn(K, N, device='cuda', dtype=torch.float16)
    result = triton_matmul(A, B)
    expected = torch.mm(A, B)
    assert torch.allclose(result, expected, atol=1e-2, rtol=1e-2)
```

## Benchmark Design

### 对比维度

| 维度 | 指标 |
|------|------|
| 性能 | latency (ms), throughput (TFLOPS/GB/s) |
| 开发效率 | 代码行数, 开发时间, 调试难度 |
| 可维护性 | 可读性, 修改成本 |
| 可移植性 | 跨 GPU 架构兼容性 |

### 对比表格模板

| Kernel | CUDA (μs) | Triton (μs) | Triton/CUDA | CUDA LoC | Triton LoC | LoC Ratio |
|--------|-----------|-------------|-------------|----------|------------|-----------|
| vector_add | — | — | — | ~50 | ~20 | 0.4× |
| matmul | — | — | — | ~200 | ~60 | 0.3× |
| softmax | — | — | — | ~100 | ~40 | 0.4× |
| layernorm | — | — | — | ~120 | ~50 | 0.4× |
| fused_attention | — | — | — | ~400 | ~100 | 0.25× |
| quantization | — | — | — | ~80 | ~30 | 0.4× |

## Profiling Method

```bash
# Triton 编译后的 PTX/SASS 分析
TRITON_PRINT_AUTOTUNING=1 python kernels/matmul.py

# 使用 Nsight Compute 分析 Triton 生成的 kernel
ncu --set full python -c "from kernels.matmul import triton_matmul; ..."
```

## Expected Metrics

| Kernel | 目标：Triton vs CUDA | 目标：Triton vs PyTorch |
|--------|---------------------|------------------------|
| vector_add | 95-100% | 100%+ |
| matmul | 70-90% | vs cuBLAS 50-70% |
| softmax | 80-100% | 100-120% |
| layernorm | 90-100% | 100-120% |
| fused_attention | 60-80% | 200-400% (vs naive) |
| quantization | 85-95% | N/A |

## Resume Bullet

*（完成后使用）*
- "Implemented 6 Triton kernels (GEMM, FlashAttention, LayerNorm) achieving 80-95% of hand-written CUDA performance with 3-5× less development time"
- "Conducted systematic Triton vs CUDA comparison demonstrating optimal use cases for each programming model"

## Interview Talking Points

### 核心问题

1. **"Triton 和 CUDA 怎么选？"**
   - Triton：快速原型、中等复杂度 kernel、需要 autotune
   - CUDA：极致性能、warp-level 控制、复杂 memory 模式

2. **"Triton 的局限性？"**
   - 不支持 warp-level primitives（`__shfl_down_sync`）
   - Block-level 抽象限制了某些优化（如 persistent kernel）
   - 编译时间较长

3. **"Autotune 怎么工作？"**
   - 枚举 tile size 配置，运行时选最快的
   - 类似 CUTLASS 的 tile 搜索，但自动化
