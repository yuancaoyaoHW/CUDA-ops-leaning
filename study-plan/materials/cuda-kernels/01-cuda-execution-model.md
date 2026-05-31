# CUDA 执行模型与线程层次

## 1. 学习目标

- 理解 CUDA 编程模型中 thread、block、grid、warp、SM 的层次关系
- 掌握 kernel launch 配置参数的含义与选择依据
- 理解 warp 作为硬件调度基本单位的意义
- 能够根据问题规模设计合理的 grid/block 配置
- 理解 SM（Streaming Multiprocessor，流式多处理器）的资源限制对并行度的影响

## 2. 前置知识

- C/C++ 基础
- 并行计算基本概念（SIMD、SIMT）
- GPU 与 CPU 架构差异的直觉

## 3. 核心术语表

| 术语 | 英文全称 | 定义 |
|------|----------|------|
| Thread | Thread | GPU 上最小的执行单元，每个 thread 有独立的寄存器和 program counter |
| Block | Thread Block | 一组 thread 的集合，共享 shared memory，可通过 `__syncthreads()` 同步 |
| Grid | Grid | 一组 block 的集合，构成一次 kernel launch 的全部工作 |
| Warp | Warp | 32 个连续 thread 组成的硬件调度单位，以 SIMT 方式执行 |
| SM | Streaming Multiprocessor | GPU 的计算核心单元，包含多个 CUDA core、shared memory、register file |
| Kernel | Kernel | 在 GPU 上执行的函数，由 host 端 launch |
| SIMT | Single Instruction Multiple Threads | NVIDIA 的执行模型，一条指令驱动多个 thread |
| Occupancy | Occupancy | SM 上活跃 warp 数与最大可驻留 warp 数的比值 |
| Lane | Lane | warp 内每个 thread 的编号（0-31） |
| Block Scheduler | Block Scheduler | 硬件调度器，将 block 分配到 SM |

## 4. 动机

### 4.1 为什么需要理解执行模型？

CUDA kernel 的性能高度依赖于 thread 组织方式。错误的 block size 选择可能导致：
- SM 利用率不足（occupancy 过低）
- shared memory 或 register 溢出
- warp divergence 导致执行效率下降
- 无法隐藏 memory latency

### 4.2 面试高频问题

- "一个 warp 是什么？为什么是 32？"
- "block size 选 128 还是 256？依据是什么？"
- "一个 SM 最多能跑多少个 block？"
- "thread ID 如何映射到数据索引？"

## 5. 数学定义

### 5.1 线程索引计算

对于 1D grid + 1D block：
```
global_thread_id = blockIdx.x * blockDim.x + threadIdx.x
```

对于 2D grid + 2D block：
```
global_x = blockIdx.x * blockDim.x + threadIdx.x
global_y = blockIdx.y * blockDim.y + threadIdx.y
linear_id = global_y * gridDim.x * blockDim.x + global_x
```

### 5.2 Grid 配置计算

给定问题规模 N 和 block size B：
```
grid_size = ceil(N / B) = (N + B - 1) / B
```

### 5.3 Occupancy 计算

```
occupancy = active_warps_per_SM / max_warps_per_SM
```

其中 `active_warps_per_SM` 受以下因素限制：
- 每个 block 使用的 shared memory
- 每个 thread 使用的 register 数量
- 每个 SM 的最大 block 数
- 每个 SM 的最大 thread 数

## 6. 推导逻辑

### 6.1 从硬件到软件的映射

```
GPU
├── Device
│   ├── SM 0
│   │   ├── Warp Scheduler 0 → Warp 0 (thread 0-31)
│   │   ├── Warp Scheduler 1 → Warp 1 (thread 32-63)
│   │   ├── ...
│   │   ├── Shared Memory (configurable, e.g., 48KB/100KB)
│   │   └── Register File (e.g., 65536 registers)
│   ├── SM 1
│   │   └── ...
│   └── SM N-1
└── Global Memory (HBM, e.g., 80GB on A100)
```

### 6.2 Block 到 SM 的分配

1. Host 发起 kernel launch，指定 grid 和 block 配置
2. Block Scheduler 将 block 分配到有足够资源的 SM
3. 一个 SM 可以同时驻留多个 block（受资源限制）
4. Block 内的 thread 被划分为 warp（每 32 个一组）
5. Warp Scheduler 以 warp 为单位发射指令

### 6.3 Warp 执行模型

- 同一 warp 内的 32 个 thread 执行相同指令（SIMT）
- 如果 warp 内 thread 走不同分支（divergence），硬件串行执行各分支
- Warp 是 memory coalescing 的基本单位
- Warp 是调度的基本单位：当一个 warp stall（等待 memory），scheduler 切换到另一个 ready warp

### 6.4 为什么 warp size = 32？

这是 NVIDIA 硬件设计选择：
- 32 thread 共享一个 instruction fetch/decode 单元
- 32 对应一次 128-byte memory transaction（每 thread 4 bytes）
- 平衡了硬件复杂度和并行度

## 7. 算子流程

### 7.1 Kernel Launch 流程

```
Host                          Device
  │                              │
  ├─ cudaMalloc()               │
  ├─ cudaMemcpy(H2D)           │
  ├─ kernel<<<grid,block>>>()   │
  │                              ├─ Block Scheduler 分配 block
  │                              ├─ Warp Scheduler 调度 warp
  │                              ├─ 执行计算
  │                              ├─ 写回 global memory
  ├─ cudaDeviceSynchronize()    │
  ├─ cudaMemcpy(D2H)           │
  └─ cudaFree()                 │
```

### 7.2 Thread 生命周期

1. Block 被分配到 SM → thread 获得 register 和 shared memory 份额
2. Thread 执行 kernel 代码
3. 遇到 `__syncthreads()` 时等待同 block 内所有 thread
4. Kernel 执行完毕 → 资源释放 → SM 可接收新 block

## 8. PyTorch baseline

```python
import torch

# 向量加法 - PyTorch 实现
def vector_add_pytorch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b

# 使用
N = 1 << 20  # 1M elements
a = torch.randn(N, device='cuda')
b = torch.randn(N, device='cuda')
c = vector_add_pytorch(a, b)
```

## 9. CUDA 实现思路

### 9.1 向量加法 Kernel

```cuda
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Launch
int N = 1 << 20;
int block_size = 256;
int grid_size = (N + block_size - 1) / block_size;
vector_add<<<grid_size, block_size>>>(d_a, d_b, d_c, N);
```

### 9.2 Block Size 选择依据

| Block Size | 优点 | 缺点 |
|-----------|------|------|
| 32 | 最小 warp 对齐 | occupancy 可能低，block 数过多 |
| 64 | 2 warps，灵活 | 仍可能 occupancy 不足 |
| 128 | 4 warps，常用选择 | 平衡 |
| 256 | 8 warps，高 occupancy | register pressure 可能限制 |
| 512 | 16 warps | shared memory 可能不够分 |
| 1024 | 最大允许值 | 资源限制严重，很少使用 |

**经验法则**：大多数 kernel 使用 128 或 256。

### 9.3 多维 Block 示例

```cuda
// 2D block for matrix operations
dim3 block(16, 16);  // 256 threads per block
dim3 grid((width + 15) / 16, (height + 15) / 16);
matrix_kernel<<<grid, block>>>(...);
```

### 9.4 Thread 索引到数据映射

```cuda
// 1D
int tid = blockIdx.x * blockDim.x + threadIdx.x;

// 2D
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;

// 3D (batch + 2D)
int batch = blockIdx.z;
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

## 10. Triton 实现思路

```python
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(
    a_ptr, b_ptr, c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Triton 的 "program" 对应 CUDA 的 block
    pid = tl.program_id(axis=0)
    
    # 计算当前 block 处理的元素范围
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # 边界 mask
    mask = offsets < n_elements
    
    # 加载、计算、存储
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b
    tl.store(c_ptr + offsets, c, mask=mask)

# Launch
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
vector_add_kernel[grid](a, b, c, n_elements, BLOCK_SIZE=1024)
```

**Triton vs CUDA 对比**：
- Triton 的 `program_id` ≈ CUDA 的 `blockIdx`
- Triton 自动处理 warp 级别的细节
- Triton 的 BLOCK_SIZE 通常更大（1024-4096），因为它在 block 内做向量化
- Triton 不需要手动管理 shared memory（编译器自动决定）

## 11. Memory Access 分析

### 11.1 向量加法的访存模式

- 每个 thread 读 2 个 float（8 bytes），写 1 个 float（4 bytes）
- 总数据移动：12 bytes/thread × N threads = 12N bytes
- 计算量：1 FLOP/thread × N threads = N FLOPs
- Arithmetic Intensity = N / (12N) = 1/12 FLOP/byte → **memory-bound**

### 11.2 Coalescing 分析

连续 thread 访问连续地址 → 完美 coalesced：
```
Thread 0: a[0], b[0] → c[0]
Thread 1: a[1], b[1] → c[1]
...
Thread 31: a[31], b[31] → c[31]
```
一个 warp 的 32 个 float 读取合并为一次 128-byte transaction。

## 12. Parallelism 分析

### 12.1 向量加法

- 数据并行：每个元素独立，完美并行
- 无 thread 间依赖
- 无需同步
- Grid 级并行度 = N / block_size

### 12.2 一般 kernel 的并行度层次

| 层次 | 并行方式 | 粒度 |
|------|----------|------|
| Grid | 多 block 并行 | 粗粒度，跨 SM |
| Block | 多 warp 并行 | 中粒度，同 SM |
| Warp | SIMT 32-wide | 细粒度，lock-step |
| Instruction | ILP | 超细粒度，同 thread |

## 13. Compute-bound / Memory-bound 判断

### 13.1 Roofline 模型

```
Performance (FLOP/s) = min(Peak_Compute, Peak_BW × Arithmetic_Intensity)
```

对于 A100 SXM：
- Peak FP32 Compute: 19.5 TFLOPS
- Peak HBM BW: 2039 GB/s
- Ridge Point: 19.5T / 2039G ≈ 9.6 FLOP/byte

如果 AI < 9.6 → memory-bound
如果 AI > 9.6 → compute-bound

### 13.2 常见算子的 AI

| 算子 | Arithmetic Intensity | Bound |
|------|---------------------|-------|
| Vector Add | 1/12 | Memory |
| Reduction | 1/4 | Memory |
| Softmax | ~1/4 | Memory |
| RMSNorm | ~1/8 | Memory |
| GEMM (M=N=K=4096) | ~4096/3 ≈ 1365 | Compute |
| FlashAttention | ~seq_len/4 | Depends |

## 14. Profiling 指标

| 指标 | 工具 | 含义 |
|------|------|------|
| Achieved Occupancy | Nsight Compute | 实际活跃 warp 比例 |
| SM Throughput | Nsight Compute | SM 计算单元利用率 |
| Memory Throughput | Nsight Compute | 显存带宽利用率 |
| Warp Stall Reasons | Nsight Compute | warp 等待原因分布 |
| Kernel Duration | Nsight Systems | kernel 执行时间 |
| Grid Size | Nsight Systems | launch 的 block 数 |
| Block Size | Nsight Systems | 每 block 的 thread 数 |
| Register/Thread | Nsight Compute | 每 thread 使用的寄存器数 |
| Shared Memory/Block | Nsight Compute | 每 block 使用的 shared memory |

## 15. Benchmark 设计

### 15.1 实验变量

```python
# Block size sweep
block_sizes = [32, 64, 128, 256, 512, 1024]

# Problem size sweep
problem_sizes = [1<<16, 1<<18, 1<<20, 1<<22, 1<<24]

# 测量指标
metrics = ['kernel_time_ms', 'achieved_occupancy', 'memory_throughput_pct']
```

### 15.2 Benchmark 模板

```python
import torch
import time

def benchmark_kernel(kernel_fn, *args, warmup=10, repeat=100):
    # Warmup
    for _ in range(warmup):
        kernel_fn(*args)
    torch.cuda.synchronize()
    
    # Timing
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(repeat):
        kernel_fn(*args)
    end.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start.elapsed_time(end) / repeat
    return elapsed_ms
```

## 16. 常见错误

| 错误 | 现象 | 原因 | 修复 |
|------|------|------|------|
| 越界访问 | CUDA error / 结果错误 | 缺少边界检查 | 加 `if (idx < n)` |
| Block size > 1024 | Launch failure | 超过硬件限制 | 减小 block size |
| Grid size = 0 | 无输出 | N < block_size 时整除为 0 | 用 `(N+B-1)/B` |
| 忘记同步 | 结果不确定 | 读取未完成的计算 | 加 `cudaDeviceSynchronize()` |
| 2D 索引错误 | 结果错位 | row/col 计算反了 | 画图验证映射 |
| Shared memory 超限 | Launch failure | 申请超过 SM 容量 | 减小 tile size 或用 dynamic shared memory |

## 17. 实验任务

### 实验 1：Block Size Sweep
- 对 vector_add kernel，sweep block_size = {32, 64, 128, 256, 512, 1024}
- 固定 N = 1<<24
- 记录 kernel time 和 achieved occupancy
- 画图分析最优 block size

### 实验 2：Problem Size Scaling
- 固定 block_size = 256
- Sweep N = {1<<10, 1<<14, 1<<18, 1<<22, 1<<26}
- 记录 kernel time 和 effective bandwidth
- 分析 launch overhead 在小 N 时的占比

### 实验 3：2D Grid 配置
- 实现 matrix transpose kernel
- 比较 (16,16) vs (32,8) vs (8,32) block 配置
- 分析 coalescing 对性能的影响

### 实验 4：Occupancy 限制因素
- 写一个使用大量 register 的 kernel（手动展开循环）
- 用 `--ptxas-options=-v` 查看 register 使用量
- 观察 occupancy 变化

## 18. 习题 20 道

1. 一个 kernel 配置为 `<<<128, 256>>>`，总共启动多少个 thread？
2. 如果 N=1000，block_size=256，需要多少个 block？grid_size 应该设为多少？
3. 在一个 `<<<4, 128>>>` 的 kernel 中，`blockIdx.x=2, threadIdx.x=65` 的 global thread ID 是多少？
4. 一个 warp 包含多少个 thread？为什么？
5. 如果一个 block 有 256 个 thread，它包含多少个 warp？
6. SM 最多支持 2048 个 thread（如 A100），如果 block_size=256，一个 SM 最多驻留多少个 block？
7. 如果每个 thread 使用 64 个 register，SM 有 65536 个 register，一个 SM 最多驻留多少个 thread？多少个 warp？
8. Occupancy 为 50% 意味着什么？这一定是性能问题吗？
9. 为什么 block_size 通常选择 32 的倍数？
10. 如果 block_size=100（不是 32 的倍数），会发生什么？
11. `__syncthreads()` 的作用范围是什么？能跨 block 同步吗？
12. 为什么 CUDA 不支持跨 block 的全局同步（在同一 kernel 内）？
13. 一个 kernel 的 grid 有 10000 个 block，GPU 有 108 个 SM，这些 block 如何被执行？
14. 什么是 tail effect？如何缓解？
15. 解释 SIMT 和 SIMD 的区别。
16. 如果一个 warp 中有 16 个 thread 走 if 分支，16 个走 else 分支，执行效率是多少？
17. 为什么说 "warp 是调度的基本单位"？
18. 一个 SM 有 4 个 warp scheduler，这意味着什么？
19. Kernel launch 是异步的，这对 host 代码意味着什么？
20. 如何计算一个 kernel 的理论 occupancy？需要哪些信息？

## 19. 标准答案

1. 128 × 256 = 32,768 个 thread。
2. ceil(1000/256) = 4 个 block。grid_size = (1000 + 255) / 256 = 4。
3. global_id = 2 × 128 + 65 = 321。
4. 32 个 thread。这是 NVIDIA 硬件设计选择，对应 128-byte memory transaction 和指令发射宽度。
5. 256 / 32 = 8 个 warp。
6. 2048 / 256 = 8 个 block（假设不受其他资源限制）。
7. 65536 / 64 = 1024 个 thread = 32 个 warp。如果 max_warps_per_SM = 64，则 occupancy = 32/64 = 50%。
8. 意味着 SM 上只有一半的 warp slot 被占用。不一定是性能问题——如果 kernel 是 compute-bound 且 ILP 足够，50% occupancy 可能已经饱和计算单元。
9. 因为 warp size = 32。非 32 倍数会导致最后一个 warp 有 inactive thread，浪费硬件资源。
10. 硬件会分配 4 个 warp（128 thread），其中最后一个 warp 只有 4 个 active thread（100 - 96 = 4），28 个 thread 被 mask 掉但仍占用资源。
11. 作用范围是同一个 block 内的所有 thread。不能跨 block 同步。
12. 因为 block 的执行顺序不确定，且不是所有 block 同时驻留在 SM 上。如果允许全局同步，可能导致死锁（等待的 block 还没被调度）。
13. Block Scheduler 将 block 分批分配到 108 个 SM。每个 SM 执行完当前 block 后接收新 block，直到所有 10000 个 block 完成。类似线程池模型。
14. Tail effect：当大部分 SM 已完成工作，只剩少数 SM 还在执行最后几个 block，导致 GPU 利用率下降。缓解方法：增加 block 数量使其远大于 SM 数量，或使用 persistent kernel。
15. SIMD：一条指令操作一个向量寄存器，程序员显式使用向量类型。SIMT：每个 thread 有独立的 PC 和 register state，可以独立分支，但硬件以 warp 为单位发射指令。SIMT 对程序员更友好。
16. 50%。硬件先执行 if 分支（16 thread active），再执行 else 分支（另 16 thread active）。总时间 = 两个分支时间之和。
17. 因为 warp scheduler 以 warp 为粒度选择下一个要执行的指令。单个 thread 不能被独立调度。
18. 意味着每个时钟周期，SM 可以同时从 4 个不同的 warp 发射指令。这增加了指令级并行度和 latency hiding 能力。
19. Host 代码在 launch 后立即继续执行，不等待 kernel 完成。需要显式调用 `cudaDeviceSynchronize()` 或使用 CUDA event 来等待结果。
20. 需要：(a) 每 thread 的 register 数量，(b) 每 block 的 shared memory，(c) block size，(d) GPU 的 compute capability（决定 SM 资源上限）。用这些信息计算 SM 能驻留的最大 block/warp 数。

## 20. 复习卡片 30 张

1. **Q**: CUDA thread 层次从大到小？ **A**: Grid → Block → Warp → Thread
2. **Q**: Warp size？ **A**: 32 threads
3. **Q**: 计算 global thread ID (1D)？ **A**: `blockIdx.x * blockDim.x + threadIdx.x`
4. **Q**: Block size 最大值？ **A**: 1024 threads
5. **Q**: Grid size 计算公式？ **A**: `(N + block_size - 1) / block_size`
6. **Q**: SM 是什么？ **A**: Streaming Multiprocessor，GPU 的基本计算单元
7. **Q**: A100 有多少个 SM？ **A**: 108 个
8. **Q**: 一个 SM 最多驻留多少 thread (A100)？ **A**: 2048
9. **Q**: 一个 SM 最多驻留多少 block (A100)？ **A**: 32
10. **Q**: 一个 SM 最多驻留多少 warp (A100)？ **A**: 64
11. **Q**: Occupancy 定义？ **A**: active_warps / max_warps_per_SM
12. **Q**: `__syncthreads()` 同步范围？ **A**: 同一 block 内所有 thread
13. **Q**: 为什么不能跨 block 同步？ **A**: Block 执行顺序不确定，可能死锁
14. **Q**: Kernel launch 是同步还是异步？ **A**: 异步
15. **Q**: SIMT vs SIMD？ **A**: SIMT 每 thread 有独立 PC，可独立分支；SIMD 操作向量寄存器
16. **Q**: Warp divergence 是什么？ **A**: 同一 warp 内 thread 走不同分支，导致串行执行
17. **Q**: Block 如何被分配到 SM？ **A**: Block Scheduler 根据 SM 可用资源分配
18. **Q**: Register file 大小 (A100)？ **A**: 每 SM 65536 个 32-bit register
19. **Q**: Shared memory 最大值 (A100)？ **A**: 每 SM 最多 164KB（可配置）
20. **Q**: Tail effect 是什么？ **A**: 最后几个 block 执行时大部分 SM 空闲
21. **Q**: Block size 为什么选 32 的倍数？ **A**: 对齐 warp size，避免浪费
22. **Q**: 常用 block size？ **A**: 128 或 256
23. **Q**: A100 warp scheduler 数量？ **A**: 每 SM 4 个
24. **Q**: 什么限制 occupancy？ **A**: Register、shared memory、block 数上限
25. **Q**: Persistent kernel 是什么？ **A**: Block 数 = SM 数，用循环处理所有工作，避免 tail effect
26. **Q**: threadIdx 的维度？ **A**: 最多 3D (x, y, z)
27. **Q**: blockIdx 的维度？ **A**: 最多 3D (x, y, z)
28. **Q**: gridDim.x 最大值？ **A**: 2^31 - 1
29. **Q**: blockDim.x 最大值？ **A**: 1024（但 x*y*z ≤ 1024）
30. **Q**: 如何查看 kernel 的 register 使用量？ **A**: `nvcc --ptxas-options=-v` 或 Nsight Compute
