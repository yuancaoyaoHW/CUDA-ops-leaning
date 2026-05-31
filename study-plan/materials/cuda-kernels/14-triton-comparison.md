# Triton 对照实现总结

## 1. 学习目标

- 理解 Triton 编程模型与 CUDA 的核心差异
- 掌握 Triton 的 program/block 抽象与自动优化机制
- 能够将 CUDA kernel 思路转化为 Triton 实现
- 理解 Triton 的性能边界与适用场景
- 掌握 Triton 的 autotuning 机制

## 2. 前置知识

- CUDA 编程模型（thread/block/grid）
- GPU 内存层次
- 基本的 kernel 优化概念

## 3. 核心术语表

| 术语 | 英文全称 | 定义 |
|------|----------|------|
| Triton | Triton | OpenAI 开发的 GPU 编程语言，基于 Python |
| Program | Triton Program | Triton 的执行单位，对应 CUDA 的 block |
| program_id | Program ID | 当前 program 的索引，对应 blockIdx |
| tl.constexpr | Compile-time Constant | 编译时常量，用于 block size 等 |
| Autotuning | Autotuning | 自动搜索最优 kernel 配置 |
| Block Pointer | Block Pointer | Triton 2.0+ 的块级指针抽象 |
| num_warps | Number of Warps | 每个 program 使用的 warp 数 |
| num_stages | Number of Stages | 软件流水线的 stage 数（double buffering） |

## 4. CUDA vs Triton 对照表

| 概念 | CUDA | Triton |
|------|------|--------|
| 执行单位 | Thread | 无显式 thread（编译器管理） |
| 并行粒度 | Block (多 thread) | Program (一个 block) |
| 索引 | blockIdx, threadIdx | tl.program_id(axis) |
| 内存管理 | 手动 shared memory | 编译器自动管理 |
| 同步 | __syncthreads() | 隐式（编译器插入） |
| 向量化 | 手动 float4 | 自动（tl.load 向量化） |
| Bank conflict | 手动 padding/swizzle | 编译器自动避免 |
| Tensor Core | WMMA/MMA API | tl.dot（自动使用 TC） |
| Warp shuffle | __shfl_down_sync | tl.sum/tl.max（自动） |
| 调优 | 手动选参数 | @triton.autotune |

## 5. 各算子 Triton 实现对照

### 5.1 Vector Add

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)
```

### 5.2 Softmax

```python
@triton.jit
def softmax_kernel(input_ptr, output_ptr, n_cols, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(input_ptr + row * stride + offs, mask=mask, other=-float('inf'))
    x_max = tl.max(x, axis=0)
    exp_x = tl.exp(x - x_max)
    sum_exp = tl.sum(exp_x, axis=0)
    tl.store(output_ptr + row * stride + offs, exp_x / sum_exp, mask=mask)
```

### 5.3 GEMM (with autotuning)

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k))
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k) & (offs_n[None, :] < N))
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

### 5.4 RMSNorm

```python
@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, n_cols, eps, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * stride + offs, mask=mask).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask).to(tl.float32)
    rms = tl.sqrt(tl.sum(x * x, axis=0) / n_cols + eps)
    out = x / rms * w
    tl.store(out_ptr + row * stride + offs, out.to(tl.float16), mask=mask)
```

### 5.5 Fused Add + RMSNorm

```python
@triton.jit
def fused_add_rmsnorm_kernel(x_ptr, residual_ptr, w_ptr, out_ptr, n_cols, eps, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * stride + offs, mask=mask).to(tl.float32)
    res = tl.load(residual_ptr + row * stride + offs, mask=mask).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask).to(tl.float32)
    
    added = x + res
    tl.store(residual_ptr + row * stride + offs, added.to(tl.float16), mask=mask)  # update residual
    
    rms = tl.sqrt(tl.sum(added * added, axis=0) / n_cols + eps)
    out = added / rms * w
    tl.store(out_ptr + row * stride + offs, out.to(tl.float16), mask=mask)
```

## 6. Triton 性能对比

### 6.1 典型性能（vs cuBLAS/手写 CUDA）

| 算子 | Triton / cuBLAS | Triton / 手写 CUDA | 备注 |
|------|----------------|-------------------|------|
| GEMM (large) | 85-95% | 90-100% | Triton GEMM 接近 cuBLAS |
| GEMM (small M) | 70-85% | 80-90% | 小 M 时 Triton 调优空间有限 |
| Softmax | N/A | 95-105% | Triton 可能更快（自动优化） |
| RMSNorm | N/A | 90-100% | 接近手写 |
| FlashAttention | 80-90% | 85-95% | 复杂 kernel Triton 有差距 |
| Reduction | N/A | 85-95% | 简单 reduction Triton 表现好 |

### 6.2 Triton 的优势场景

1. **Fused kernels**：多个简单操作融合（如 add+norm+activation）
2. **快速原型**：比 CUDA 开发速度快 5-10x
3. **Memory-bound kernels**：编译器自动优化访存
4. **中等复杂度 kernels**：softmax、norm、elementwise

### 6.3 Triton 的劣势场景

1. **极致优化的 GEMM**：cuBLAS/CUTLASS 仍然更快（手动 warp 级优化）
2. **复杂的 warp-level 操作**：如 FlashAttention 的 warp specialization
3. **非规则并行模式**：如 sparse attention、dynamic shape
4. **需要精确控制 shared memory layout**：如避免特定 bank conflict pattern

## 7. Autotuning 机制

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],  # 当这些值变化时重新 tune
)
@triton.jit
def my_kernel(...):
    ...

# Autotuning 过程：
# 1. 第一次调用时，运行所有 configs
# 2. 测量每个 config 的执行时间
# 3. 选择最快的 config 缓存
# 4. 后续调用直接使用最优 config
```

## 8. 实验任务

1. 用 Triton 实现 vector_add，对比 PyTorch 和 CUDA 版本性能
2. 用 Triton 实现 softmax，测试不同 BLOCK_SIZE 的性能
3. 用 Triton 实现 GEMM with autotuning，对比 cuBLAS
4. 用 Triton 实现 fused add + RMSNorm，对比分开执行
5. 用 Triton 实现简化版 FlashAttention（无 causal mask）
6. 分析 Triton 生成的 PTX/SASS 代码，理解编译器优化

## 9. 习题 10 道

1. Triton 的 program 对应 CUDA 的什么概念？
2. `tl.program_id(0)` 对应 CUDA 的什么？
3. Triton 如何处理 shared memory？程序员需要手动管理吗？
4. `tl.dot(a, b)` 在硬件上使用什么指令？
5. Triton 的 `num_warps` 参数如何影响性能？
6. 为什么 Triton 的 BLOCK_SIZE 必须是 2 的幂？
7. Triton autotuning 的 `key` 参数是什么意思？
8. Triton 相比 CUDA 在哪些场景下性能更差？为什么？
9. 如何在 Triton 中实现 cross-program 的 reduction？
10. Triton 3.0 的 block pointer 抽象解决了什么问题？

## 10. 标准答案

1. Triton 的 program 对应 CUDA 的 thread block。一个 program 内部的并行由编译器管理。
2. 对应 `blockIdx.x`。
3. Triton 编译器自动管理 shared memory 的分配、加载和同步。程序员不需要手动声明或管理。
4. 使用 Tensor Core 的 MMA 指令（当输入满足对齐和类型要求时）。
5. num_warps 决定 block 内的并行度。更多 warp → 更高 occupancy 但可能更多 register pressure。需要根据 kernel 特性选择。
6. 因为 Triton 的向量操作基于 power-of-2 的 SIMD 宽度，编译器需要这个约束来生成高效代码。
7. `key` 指定哪些参数变化时需要重新 autotuning。例如 GEMM 的 M/N/K 变化时最优配置可能不同。
8. (a) 极致优化的大 GEMM（cuBLAS 有手动 warp-level 优化）；(b) 需要精确控制 warp 行为的 kernel（如 FA 的 warp specialization）；(c) 非 power-of-2 的问题规模。
9. 不能直接跨 program reduction。需要：每个 program 写 partial result 到 global memory → 第二个 kernel 做最终 reduction。
10. Block pointer 提供了更高层的内存访问抽象，让编译器更好地优化 memory access pattern（如自动 swizzle、prefetch），减少程序员手动计算指针的负担。
