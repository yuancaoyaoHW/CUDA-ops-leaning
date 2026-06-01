# CUDA Ops Learning Lab

> 从零实现 LLM 推理核心算子，配套完整 benchmark 和 Nsight profiling 报告。

[![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Motivation

LLM 推理的核心性能瓶颈在 GPU kernel 层。本项目从零实现 8 个核心算子，覆盖从基础 memory-bound 操作到 compute-bound 矩阵运算，再到 LLM 特有的 attention 和 normalization kernel。每个算子都有从 naive 到 optimized 的完整优化路径，配套 Nsight Compute profiling 和 benchmark 数据。

## Key Results

> ⚠️ 以下为目标值，需完成实现后填入实际数据

- 目标：Tiled GEMM 达到 cuBLAS 60%+ 性能
- 目标：FlashAttention 实现 O(N) memory，比 naive 快 2-4×
- 目标：RMSNorm 性能接近 PyTorch（< 1.5× 差距）
- 目标：所有 memory-bound kernel 达到 80%+ bandwidth utilization

## Directory Structure

```
cuda-ops-lab/
├── kernels/
│   ├── 01_vector_add/
│   │   ├── vector_add.cu          # Grid-stride loop 实现
│   │   ├── vector_add_test.py     # PyTorch 对比测试
│   │   └── README.md              # 实现思路 + profiling 分析
│   ├── 02_reduction/
│   │   ├── reduce_sum_v1.cu       # Interleaved addressing
│   │   ├── reduce_sum_v2.cu       # Sequential addressing
│   │   ├── reduce_sum_v3.cu       # Warp shuffle
│   │   ├── reduce_max.cu          # Max reduction
│   │   ├── reduction_test.py
│   │   └── README.md
│   ├── 03_matmul/
│   │   ├── gemm_naive.cu          # 每线程一个输出元素
│   │   ├── gemm_tiled.cu          # Shared memory tiling
│   │   ├── gemm_vectorized.cu     # float4 vectorized load
│   │   ├── gemm_register_tile.cu  # Register blocking
│   │   ├── matmul_test.py
│   │   └── README.md
│   ├── 04_softmax/
│   │   ├── softmax_naive.cu       # 3-pass: max, sum, normalize
│   │   ├── softmax_online.cu      # Online safe softmax (1-pass)
│   │   ├── softmax_test.py
│   │   └── README.md
│   ├── 05_layernorm/
│   │   ├── layernorm.cu           # Welford online algorithm
│   │   ├── layernorm_test.py
│   │   └── README.md
│   ├── 06_rmsnorm/
│   │   ├── rmsnorm.cu             # Fused RMSNorm
│   │   ├── fused_add_rmsnorm.cu   # residual + RMSNorm fusion
│   │   ├── rmsnorm_test.py
│   │   └── README.md
│   ├── 07_rope/
│   │   ├── rope.cu               # Rotary Position Embedding
│   │   ├── rope_test.py
│   │   └── README.md
│   └── 08_flash_attention_toy/
│       ├── naive_attention.cu     # O(N²) memory baseline
│       ├── flash_attention.cu     # Tiled + online softmax
│       ├── attention_test.py
│       └── README.md
├── tests/
│   ├── conftest.py                # PyTorch CUDA fixtures
│   ├── test_all_kernels.py        # 统一正确性测试
│   └── run_tests.sh
├── benchmarks/
│   ├── bench_all.py               # 统一 benchmark runner
│   ├── bench_gemm.py              # GEMM 专项 benchmark
│   ├── bench_attention.py         # Attention 专项 benchmark
│   └── plot_results.py            # 可视化脚本
├── profiling/
│   ├── ncu_profile.sh             # Nsight Compute 自动化脚本
│   ├── nsys_profile.sh            # Nsight Systems 自动化脚本
│   └── reports/                   # Profiling 报告输出
├── include/
│   ├── cuda_utils.h               # 错误检查宏、计时工具
│   └── tensor.h                   # 简单 Tensor 封装
├── CMakeLists.txt
├── Makefile
├── setup.py                       # PyTorch extension build
└── README.md
```

## 算子详细设计

### 01. Vector Add（入门）

| 项目 | 内容 |
|------|------|
| **实现思路** | naive → grid-stride loop → vectorized (float4) |
| **正确性验证** | `torch.allclose(custom_result, a + b, atol=1e-5)` |
| **Benchmark 设计** | N = [1K, 10K, 100K, 1M, 10M, 100M]，测量 effective bandwidth |
| **Profiling 指标** | `dram__throughput.avg.pct_of_peak_sustained` |
| **性能目标** | 目标：>85% theoretical memory bandwidth |
| **面试要点** | Grid-stride loop 的优势、memory coalescing 原理、bandwidth 计算方法 |
| **简历 bullet** | *（完成后使用）* "Implemented memory-optimized CUDA vector operations achieving X% of peak bandwidth on [GPU]" |

### 02. Reduction（Sum, Max）

| 项目 | 内容 |
|------|------|
| **实现思路** | interleaved → sequential → warp shuffle → multi-block atomic |
| **正确性验证** | `torch.allclose(custom_sum, tensor.sum())` |
| **Benchmark 设计** | N = [1K, 10K, 100K, 1M, 10M]，对比 4 个版本 |
| **Profiling 指标** | warp divergence, memory throughput, occupancy |
| **性能目标** | 目标：>80% memory bandwidth，warp shuffle 版本最快 |
| **面试要点** | Reduction tree 原理、warp divergence 消除、`__shfl_down_sync` 用法 |
| **简历 bullet** | *（完成后使用）* "Optimized parallel reduction kernel eliminating warp divergence, achieving X% of peak memory bandwidth" |

### 03. MatMul（核心，重点投入）

| 项目 | 内容 |
|------|------|
| **实现思路** | naive → tiled (shared memory) → vectorized (float4) → register blocking |
| **正确性验证** | `torch.allclose(custom_mm, torch.mm(A, B), atol=1e-3)` |
| **Benchmark 设计** | M=N=K=[256, 512, 1024, 2048, 4096] + LLM 常见形状 |
| **Profiling 指标** | compute throughput, shared memory bank conflict, occupancy |
| **性能目标** | 目标：tiled 20-40% cuBLAS, register tile 60-80% cuBLAS |
| **面试要点** | Tiling 策略、tile size 选择约束、double buffering、CUTLASS 层级 |
| **简历 bullet** | *（完成后使用）* "Built tiled GEMM kernel achieving X% of cuBLAS performance through shared memory tiling and register blocking" |

### 04. Softmax（Row-wise, Online）

| 项目 | 内容 |
|------|------|
| **实现思路** | 3-pass (max→sum→normalize) → 2-pass → online 1-pass |
| **正确性验证** | `torch.allclose(custom_softmax, F.softmax(x, dim=-1), atol=1e-5)` |
| **Benchmark 设计** | rows×cols = [1024×128, 1024×512, 1024×2048, 4096×4096] |
| **Profiling 指标** | memory throughput, arithmetic intensity |
| **性能目标** | 目标：online 版本 < 2× PyTorch 差距 |
| **面试要点** | Numerical stability (减 max)、online softmax trick、FlashAttention 中的应用 |
| **简历 bullet** | *（完成后使用）* "Implemented numerically-stable online softmax kernel with single-pass algorithm used in FlashAttention" |

### 05. LayerNorm

| 项目 | 内容 |
|------|------|
| **实现思路** | two-pass (mean→variance→normalize) → Welford online algorithm |
| **正确性验证** | `torch.allclose(custom_ln, F.layer_norm(x, [H]), atol=1e-4)` |
| **Benchmark 设计** | batch×seq = [1×1024, 8×1024, 32×2048], hidden = [768, 4096] |
| **Profiling 指标** | memory bandwidth utilization |
| **性能目标** | 目标：< 1.5× PyTorch 差距 |
| **面试要点** | Welford algorithm 数值稳定性、与 RMSNorm 的区别 |
| **简历 bullet** | *（完成后使用）* "Implemented fused LayerNorm kernel using Welford's online algorithm for numerical stability" |

### 06. RMSNorm

| 项目 | 内容 |
|------|------|
| **实现思路** | naive (两次 global read) → fused (一次 read, reduction + normalize) |
| **正确性验证** | 对比 `x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)` |
| **Benchmark 设计** | 同 LayerNorm + fused vs unfused 对比 |
| **Profiling 指标** | memory bandwidth, kernel fusion 效果 |
| **性能目标** | 目标：fused 版本比 unfused 快 30-50% |
| **面试要点** | 为什么 LLM 用 RMSNorm 而非 LayerNorm、fusion 减少 memory round-trip |
| **简历 bullet** | *（完成后使用）* "Built fused RMSNorm kernel reducing global memory accesses by 50% through operator fusion" |

### 07. RoPE（Rotary Position Embedding）

| 项目 | 内容 |
|------|------|
| **实现思路** | 按 head_dim 对分，应用旋转矩阵 cos/sin |
| **正确性验证** | 对比 HuggingFace `apply_rotary_pos_emb` |
| **Benchmark 设计** | batch×seq×heads×head_dim 不同配置 |
| **Profiling 指标** | memory throughput (element-wise, memory-bound) |
| **性能目标** | 目标：正确性优先，性能接近 PyTorch |
| **面试要点** | RoPE 数学原理、为什么比 absolute PE 好、在 inference 中的计算 |
| **简历 bullet** | *（完成后使用）* "Implemented RoPE kernel for LLM inference with correct handling of interleaved/non-interleaved formats" |

### 08. Flash Attention Toy（简化版）

| 项目 | 内容 |
|------|------|
| **实现思路** | naive O(N²) memory → tiled + online softmax O(N) memory |
| **正确性验证** | `torch.allclose(custom_attn, F.scaled_dot_product_attention(...), atol=1e-3)` |
| **Benchmark 设计** | seq_len = [128, 256, 512, 1024, 2048, 4096], 固定 batch=4, heads=32, head_dim=128 |
| **Profiling 指标** | HBM read/write bytes, compute throughput |
| **性能目标** | 目标：比 naive 快 2-4×，memory 从 O(N²) 降到 O(N) |
| **面试要点** | Tiling 策略、online softmax 在 attention 中的应用、为什么减少 HBM 访问 |
| **简历 bullet** | *（完成后使用）* "Built simplified FlashAttention kernel reducing memory complexity from O(N²) to O(N) with X× speedup over naive implementation" |

---

## Setup

### 环境要求
- NVIDIA GPU (Compute Capability >= 7.0)
- CUDA Toolkit 12.x
- Python 3.10+, PyTorch 2.x (CUDA build)
- Nsight Compute, Nsight Systems

### 编译运行

```bash
# 克隆项目
git clone <repo-url> && cd cuda-ops-lab

# 编译所有 kernel
make all

# 或使用 CMake
mkdir build && cd build && cmake .. && make -j$(nproc)

# 安装 Python binding
pip install -e .

# 运行正确性测试
pytest tests/ -v

# 运行 benchmark
python benchmarks/bench_all.py --output results/

# 运行 profiling
bash profiling/ncu_profile.sh kernels/03_matmul/gemm_tiled
```

### 云 GPU 环境（无本地 GPU 时）

```bash
# AutoDL / Colab
pip install torch --index-url https://download.pytorch.org/whl/cu121
# 确认 CUDA 可用
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## Correctness Test Design

每个 kernel 的正确性测试遵循统一模式：

```python
import torch
import pytest
from cuda_ops_lab import vector_add, gemm_tiled, softmax_online

@pytest.mark.parametrize("N", [1024, 65536, 1048576])
def test_vector_add(N):
    a = torch.randn(N, device='cuda')
    b = torch.randn(N, device='cuda')
    result = vector_add(a, b)
    expected = a + b
    assert torch.allclose(result, expected, atol=1e-5, rtol=1e-5)

@pytest.mark.parametrize("size", [256, 512, 1024, 2048])
def test_gemm_tiled(size):
    A = torch.randn(size, size, device='cuda', dtype=torch.float16)
    B = torch.randn(size, size, device='cuda', dtype=torch.float16)
    result = gemm_tiled(A, B)
    expected = torch.mm(A, B)
    assert torch.allclose(result, expected, atol=1e-2, rtol=1e-2)
```

---

## Benchmark Design

```python
# benchmarks/bench_all.py
import torch
import numpy as np

def benchmark_kernel(fn, *args, warmup=100, repeat=1000):
    """统一 benchmark 函数，使用 CUDA events 计时"""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeat):
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return {
        "median_ms": np.median(times),
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
    }
```

---

## Profiling Method

### Nsight Compute（单 kernel 分析）

```bash
ncu --set full \
    --target-processes all \
    --export report_%k \
    ./build/kernel_binary
```

关键指标：
- `sm__throughput.avg.pct_of_peak_sustained` — SM 利用率
- `dram__throughput.avg.pct_of_peak_sustained` — Memory BW 利用率
- `sm__warps_active.avg.pct_of_peak_sustained` — Occupancy

### Nsight Systems（端到端 timeline）

```bash
nsys profile --trace=cuda,nvtx \
    --output timeline_report \
    python benchmarks/bench_all.py
```

---

## Interview Talking Points

### 2 分钟项目叙事

"我从零实现了一个 CUDA kernel library，覆盖 LLM 推理的 8 个核心算子。从最基础的 vector_add 理解 memory coalescing，到实现 tiled GEMM 理解 shared memory tiling 策略，再到实现简化版 FlashAttention 理解 online softmax 和 tiling 如何将 memory 从 O(N²) 降到 O(N)。每个 kernel 都有完整的 Nsight Compute profiling，我能准确说出每个 kernel 的瓶颈是 memory-bound 还是 compute-bound，以及对应的优化方向。"

### 深度追问准备

1. "你的 GEMM 达到 cuBLAS 多少？" → 具体百分比 + 瓶颈分析
2. "Tile size 怎么选？" → shared memory 容量约束 + occupancy tradeoff
3. "FlashAttention 为什么快？" → 减少 HBM 访问 + online softmax 数学推导
4. "RMSNorm fusion 省了什么？" → 减少一次 global memory round-trip
5. "怎么判断 kernel 是 memory-bound 还是 compute-bound？" → arithmetic intensity + roofline
