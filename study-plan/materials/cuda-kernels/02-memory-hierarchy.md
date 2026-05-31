# GPU 内存层次与访存优化

## 1. 学习目标

- 理解 GPU 内存层次结构：register → shared memory → L1/L2 cache → global memory (HBM)
- 掌握 global memory coalescing（合并访存）的条件与违反后果
- 理解 shared memory bank conflict（bank 冲突）的成因与解决方法
- 能够分析 kernel 的访存模式并判断是否高效
- 掌握 memory throughput 的计算方法

## 2. 前置知识

- CUDA 执行模型（thread/block/warp/SM）
- 基本的计算机体系结构知识（cache、DRAM）
- 位运算基础

## 3. 核心术语表

| 术语 | 英文全称 | 定义 |
|------|----------|------|
| Global Memory | Global Memory | GPU 主存（HBM），所有 thread 可访问，延迟高（~400 cycles） |
| Shared Memory | Shared Memory | SM 内的片上存储，同 block 内 thread 共享，延迟低（~20 cycles） |
| Register | Register | 每个 thread 私有的最快存储 |
| L1 Cache | L1 Cache | SM 级别的缓存，与 shared memory 共享物理空间 |
| L2 Cache | L2 Cache | 全局共享的缓存，位于 SM 和 HBM 之间 |
| HBM | High Bandwidth Memory | 高带宽显存，A100 为 HBM2e（2TB/s） |
| Coalescing | Memory Coalescing | 将 warp 内多个 thread 的访存请求合并为少量 memory transaction |
| Bank | Shared Memory Bank | shared memory 被划分为 32 个 bank，支持并行访问 |
| Bank Conflict | Bank Conflict | 同一 warp 内多个 thread 访问同一 bank 的不同地址 |
| Stride | Stride | 相邻 thread 访问地址之间的间隔 |
| Transaction | Memory Transaction | 一次 memory 访问操作，通常为 32B 或 128B |
| Sector | Cache Sector | L1/L2 cache 的最小访问单位（32B） |

## 4. 动机

### 4.1 为什么内存是瓶颈？

GPU 的计算能力远超内存带宽：
- A100: 19.5 TFLOPS (FP32) vs 2 TB/s HBM bandwidth
- 算术强度（Arithmetic Intensity）= FLOPs / Bytes
- 向量加法：2 FLOPs / 12 Bytes = 0.17 → 严重 memory-bound
- GEMM (大矩阵)：2MNK FLOPs / (M*K + K*N + M*N)*4 Bytes → compute-bound

### 4.2 内存层次带宽对比

| 层次 | 带宽 (A100) | 延迟 | 容量 |
|------|------------|------|------|
| Register | ~19 TB/s | 0 cycles | 256KB/SM |
| Shared Memory | ~19 TB/s | ~20 cycles | 164KB/SM (configurable) |
| L1 Cache | ~19 TB/s | ~30 cycles | 与 shared memory 共享 |
| L2 Cache | ~5 TB/s | ~200 cycles | 40MB |
| HBM (Global) | 2 TB/s | ~400 cycles | 80GB |

## 5. 数学定义

### 5.1 有效带宽计算

```
Effective Bandwidth (GB/s) = (Bytes_Read + Bytes_Written) / Time_seconds / 1e9
```

### 5.2 带宽利用率

```
Bandwidth Utilization = Effective_Bandwidth / Peak_Bandwidth
```

### 5.3 Coalescing 效率

```
Coalescing Efficiency = min_transactions_needed / actual_transactions
```

理想情况：warp 内 32 个 thread 访问连续 128 bytes → 1 个 128B transaction → 效率 100%

### 5.4 Bank Conflict 分析

Shared memory 有 32 个 bank，每个 bank 宽 4 bytes。地址到 bank 的映射：
```
bank_id = (address / 4) % 32
```

## 6. 推导逻辑

### 6.1 Global Memory Coalescing

**规则**：warp 内 32 个 thread 的访存请求被硬件合并为尽可能少的 memory transaction。

**理想模式**（完全合并）：
```
Thread 0 → addr + 0
Thread 1 → addr + 4
Thread 2 → addr + 8
...
Thread 31 → addr + 124
```
→ 1 个 128B transaction

**stride-2 模式**（部分合并）：
```
Thread 0 → addr + 0
Thread 1 → addr + 8
Thread 2 → addr + 16
...
Thread 31 → addr + 248
```
→ 2 个 128B transaction（浪费 50% 带宽）

**随机访问**（最差）：
```
Thread 0 → random_addr_0
Thread 1 → random_addr_1
...
```
→ 最多 32 个 32B transaction

### 6.2 Shared Memory Bank Conflict

**无冲突**：32 个 thread 访问 32 个不同 bank
```
Thread 0 → bank 0
Thread 1 → bank 1
...
Thread 31 → bank 31
```

**2-way bank conflict**：2 个 thread 访问同一 bank 的不同地址
```
Thread 0 → bank 0, word 0
Thread 16 → bank 0, word 32  // conflict!
```
→ 串行化为 2 次访问

**Broadcast**（无冲突）：多个 thread 访问同一 bank 的**同一地址**
→ 硬件广播，不算 conflict

### 6.3 矩阵转置中的 Bank Conflict

```cuda
// Naive: column-wise write causes 32-way bank conflict
__shared__ float tile[32][32];
tile[threadIdx.y][threadIdx.x] = input[...];  // OK, row-wise
output[...] = tile[threadIdx.x][threadIdx.y];  // column-wise → conflict!

// Fix: padding
__shared__ float tile[32][33];  // +1 padding
// Now bank_id = (col * 33 + row) % 32, avoids conflict
```

## 7. 算子流程

### 7.1 使用 Shared Memory 的典型流程

```
1. 从 global memory 加载数据到 shared memory（coalesced）
2. __syncthreads()
3. 从 shared memory 读取数据进行计算（可能非连续访问）
4. __syncthreads()（如果需要再次读写 shared memory）
5. 将结果写回 global memory（coalesced）
```

### 7.2 Tiled Matrix Multiply 示例

```cuda
__global__ void matmul_tiled(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;
    
    for (int t = 0; t < K; t += TILE) {
        // Load tile from global to shared (coalesced)
        As[threadIdx.y][threadIdx.x] = A[row * K + t + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];
        __syncthreads();
        
        // Compute using shared memory
        for (int k = 0; k < TILE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    
    C[row * N + col] = sum;
}
```

## 8. PyTorch baseline

```python
import torch

def matmul_pytorch(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """PyTorch baseline - 调用 cuBLAS"""
    return torch.mm(A, B)

# Benchmark
M, N, K = 1024, 1024, 1024
A = torch.randn(M, K, device='cuda')
B = torch.randn(K, N, device='cuda')

# Warmup
for _ in range(10):
    C = matmul_pytorch(A, B)
torch.cuda.synchronize()

# Timing
import time
start = time.perf_counter()
for _ in range(100):
    C = matmul_pytorch(A, B)
torch.cuda.synchronize()
elapsed = (time.perf_counter() - start) / 100
print(f"Time: {elapsed*1000:.3f} ms")
print(f"TFLOPS: {2*M*N*K / elapsed / 1e12:.2f}")
```

## 9. CUDA 实现思路

### 9.1 Coalesced vs Non-coalesced 对比实验

```cuda
// Coalesced: 连续 thread 访问连续地址
__global__ void copy_coalesced(float* dst, const float* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = src[idx];
}

// Non-coalesced: stride 访问
__global__ void copy_strided(float* dst, const float* src, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int src_idx = idx * stride;
    if (src_idx < n) dst[idx] = src[src_idx];
}
```

### 9.2 Shared Memory 使用模式

```cuda
// 动态 shared memory
extern __shared__ float smem[];

__global__ void reduce_shared(float* input, float* output, int n) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load to shared memory
    smem[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) output[blockIdx.x] = smem[0];
}
```

### 9.3 避免 Bank Conflict 的技巧

```cuda
// 技巧 1: Padding
__shared__ float tile[32][33];  // 33 instead of 32

// 技巧 2: Swizzle
int swizzled_col = threadIdx.x ^ threadIdx.y;
float val = tile[threadIdx.y][swizzled_col];

// 技巧 3: 使用 float4 向量化访问
__shared__ float4 tile_vec[32][8];  // 32 * 8 * 16B = 4KB
```

## 10. Triton 实现思路

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Triton 自动管理 shared memory
    # 用户只需指定 tile size，编译器处理 bank conflict
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Pointers
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

**Triton 优势**：自动处理 shared memory 分配、bank conflict avoidance、memory coalescing。

## 11. Memory Access 分析

### 11.1 访存模式分类

| 模式 | 描述 | 效率 | 示例 |
|------|------|------|------|
| Coalesced sequential | 连续 thread 访问连续地址 | 100% | `a[tid]` |
| Coalesced with offset | 连续但有偏移 | ~100% | `a[tid + offset]` |
| Strided | 固定步长 | 1/stride | `a[tid * stride]` |
| Random | 随机地址 | ~3-10% | `a[index[tid]]` |
| Broadcast | 所有 thread 同一地址 | 100% (L1 hit) | `a[0]` |

### 11.2 实际 Kernel 分析示例

矩阵转置 `B[j][i] = A[i][j]`：
- 读 A：`A[row][col]` → row-major，coalesced（连续 thread 读连续 col）
- 写 B：`B[col][row]` → column-major，non-coalesced（连续 thread 写不连续地址）
- 解决：先写到 shared memory（coalesced read），再从 shared memory 读出写到 global（coalesced write）

## 12. Parallelism 分析

### 12.1 Latency Hiding

GPU 通过大量并行 warp 隐藏 memory latency：
```
需要的 warp 数 ≈ memory_latency / instruction_throughput
```

A100 示例：
- HBM latency: ~400 cycles
- 每 cycle 可发射 1 条指令
- 需要 ~400 / 1 = 400 条指令来隐藏 → 约 12-13 个 warp（每个 warp 贡献 ~32 条独立指令）

### 12.2 Memory-Level Parallelism

- 每个 SM 可以有多个 outstanding memory request
- 更多活跃 warp → 更多并发 memory request → 更高带宽利用率

## 13. Compute-bound / Memory-bound 判断

### 13.1 Arithmetic Intensity 分析

```
AI = FLOPs / Bytes_accessed

如果 AI > machine_balance_point → compute-bound
如果 AI < machine_balance_point → memory-bound

A100 balance point = 19.5 TFLOPS / 2 TB/s = 9.75 FLOPs/Byte
```

### 13.2 常见算子的 AI

| 算子 | AI (FLOPs/Byte) | Bound |
|------|-----------------|-------|
| Vector Add | 0.17 | Memory |
| Reduction | 0.08 | Memory |
| Softmax | ~0.5 | Memory |
| RMSNorm | ~0.3 | Memory |
| GEMM (M=N=K=4096) | ~341 | Compute |
| GEMM (M=1, N=K=4096) | ~1 | Memory |
| FlashAttention | ~varies | Compute (large seq) |

## 14. Profiling 指标

| 指标 | 工具 | 含义 |
|------|------|------|
| `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum` | Nsight Compute | Global load bytes |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | Nsight Compute | Global load sectors |
| `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct` | Nsight Compute | Coalescing efficiency |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` | Nsight Compute | Shared memory bank conflicts |
| `dram__bytes_read.sum` | Nsight Compute | DRAM read bytes |
| `gpu__time_duration.sum` | Nsight Compute | Kernel duration |

## 15. Benchmark 设计

### 15.1 Coalescing 实验

```python
# 测试不同 stride 对带宽的影响
strides = [1, 2, 4, 8, 16, 32]
for stride in strides:
    time = benchmark_strided_copy(N, stride)
    bandwidth = N * 4 * 2 / time / 1e9  # GB/s (read + write)
    print(f"Stride {stride}: {bandwidth:.1f} GB/s")
```

预期结果：stride=1 接近峰值带宽，stride 增大带宽线性下降。

### 15.2 Bank Conflict 实验

```python
# 测试不同 padding 对 shared memory 性能的影响
paddings = [0, 1, 2, 4]
for pad in paddings:
    time = benchmark_transpose(N, padding=pad)
    print(f"Padding {pad}: {time:.3f} ms")
```

## 16. 常见错误

1. **误以为 L1 cache 能解决所有 coalescing 问题**：L1 只缓存 128B line，stride 访问仍浪费带宽
2. **忘记 `__syncthreads()`**：shared memory 写后读必须同步
3. **Shared memory 大小超限**：A100 最大 164KB/SM（需配置）
4. **Bank conflict 分析忽略 broadcast**：同一地址多 thread 读不算 conflict
5. **混淆 shared memory 和 L1**：它们共享物理空间但逻辑独立
6. **float4 对齐问题**：向量化访问要求地址 16B 对齐

## 17. 实验任务

1. 实现 coalesced vs strided copy kernel，用 Nsight Compute 验证 sector 效率
2. 实现 naive 矩阵转置和 shared memory 优化版本，对比带宽
3. 用 padding 消除 bank conflict，验证性能提升
4. 实现 tiled GEMM，测量不同 tile size 的性能
5. 用 `cudaFuncSetAttribute` 配置 shared memory 大小，观察 occupancy 变化

## 18. 习题 20 道

1. A100 的 HBM 带宽是多少？L2 cache 大小是多少？
2. 什么是 memory coalescing？给出一个 coalesced 和一个 non-coalesced 的访存模式。
3. Warp 内 32 个 thread 访问 stride-4 的 float 数组，需要几个 128B transaction？
4. Shared memory 有多少个 bank？每个 bank 的宽度是多少？
5. 什么情况下多个 thread 访问同一 bank 不算 conflict？
6. 为什么矩阵转置的 naive 实现有 bank conflict？如何用 padding 解决？
7. 计算向量加法的 arithmetic intensity，判断是 compute-bound 还是 memory-bound。
8. A100 上，一个 memory-bound kernel 的理论最大带宽利用率是多少？
9. 解释 L1 cache 和 shared memory 的关系（Ampere 架构）。
10. 什么是 sector？一个 128B cache line 包含几个 sector？
11. 如果 warp 内 thread 0 和 thread 16 访问同一 bank 的不同 word，会发生什么？
12. `__syncthreads()` 的作用是什么？如果省略会怎样？
13. 动态 shared memory 和静态 shared memory 的区别是什么？
14. 如何用 Nsight Compute 测量 global memory coalescing 效率？
15. 什么是 memory-level parallelism？它如何帮助隐藏 latency？
16. Tiled GEMM 中，tile size 增大对 shared memory 使用和 occupancy 的影响是什么？
17. 解释 `float4` 向量化访问的优势和限制。
18. 为什么 reduction kernel 是 memory-bound 的？
19. 如何计算一个 kernel 的有效带宽（effective bandwidth）？
20. Shared memory 的 swizzle 技术是什么？它解决什么问题？

## 19. 标准答案

1. A100 HBM2e 带宽 2039 GB/s（80GB 版本），L2 cache 40MB。
2. Memory coalescing 是将 warp 内多个 thread 的访存合并为少量 transaction。Coalesced: `a[tid]`；Non-coalesced: `a[tid * 32]`。
3. Stride-4 float = stride 16 bytes。32 threads × 16B stride = 512B 范围 → 需要 4 个 128B transaction（但只用了 128B 有效数据）→ 效率 25%。
4. 32 个 bank，每个 bank 宽 4 bytes（32 bits）。
5. 多个 thread 访问同一 bank 的**同一地址**时，硬件执行 broadcast，不算 conflict。
6. Column-wise 读取 `tile[threadIdx.x][threadIdx.y]` 时，相邻 thread 访问 stride-32 的地址，全部落在同一 bank。Padding 为 `tile[32][33]` 后，stride 变为 33，不再对齐到 32 的倍数。
7. AI = 2 FLOPs / (3 × 4 Bytes) = 0.17 FLOPs/Byte << 9.75 → memory-bound。
8. 理论上可达 100%（2 TB/s），实际通常 80-90%（受 TLB、alignment 等影响）。
9. Ampere 架构中 L1 cache 和 shared memory 共享 192KB 物理空间，可配置为 0/64/100/128/164KB shared memory。
10. Sector = 32 bytes。一个 128B cache line = 4 个 sector。
11. 2-way bank conflict，硬件串行化为 2 次访问，延迟翻倍。
12. 确保 block 内所有 thread 到达同一点后再继续。省略可能导致 race condition（读到未写入的数据）。
13. 静态：编译时确定大小 `__shared__ float s[256]`。动态：运行时指定 `extern __shared__ float s[]`，launch 时第三个参数传大小。
14. 查看 metric `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct`，100% 表示完全 coalesced。
15. 多个 warp 同时有 outstanding memory request，使 memory controller 保持忙碌，提高带宽利用率。
16. Tile size 增大 → shared memory 使用增加 → 每 SM 能驻留的 block 减少 → occupancy 可能下降。需要在数据复用和 occupancy 之间平衡。
17. 优势：一次 load 128 bits（4 floats），减少指令数和 transaction 数。限制：要求地址 16B 对齐，数组大小必须是 4 的倍数。
18. Reduction 每个元素只做 1 次加法（1 FLOP），但需要读 4 bytes → AI = 0.25 FLOPs/Byte，远低于 balance point。
19. Effective BW = (bytes_read + bytes_written) / kernel_time。例如 copy 1GB 数据耗时 0.5ms → BW = 2GB / 0.5ms = 4000 GB/s（不可能，说明计时有误）。
20. Swizzle 通过 XOR 操作重新映射 shared memory 地址，使得原本会冲突的访问模式分散到不同 bank。常用于矩阵运算中的 tile 访问。

## 20. 复习卡片 30 张

1. Q: GPU 内存层次从快到慢？ A: Register > Shared Memory ≈ L1 > L2 > HBM
2. Q: A100 HBM 带宽？ A: ~2 TB/s
3. Q: Shared memory 延迟？ A: ~20 cycles
4. Q: Global memory 延迟？ A: ~400 cycles
5. Q: Coalescing 的基本单位？ A: Warp（32 threads）
6. Q: 一个 memory transaction 大小？ A: 32B (sector) 或 128B (cache line)
7. Q: Shared memory bank 数量？ A: 32
8. Q: 每个 bank 宽度？ A: 4 bytes
9. Q: Bank conflict 的定义？ A: 同一 warp 内多个 thread 访问同一 bank 的不同地址
10. Q: Broadcast 条件？ A: 多个 thread 访问同一 bank 的同一地址
11. Q: Padding 解决 bank conflict 的原理？ A: 改变 stride 使地址不再对齐到 32 的倍数
12. Q: A100 L2 cache 大小？ A: 40MB
13. Q: Arithmetic Intensity 定义？ A: FLOPs / Bytes_accessed
14. Q: A100 balance point？ A: ~9.75 FLOPs/Byte
15. Q: 向量加法的 AI？ A: ~0.17 FLOPs/Byte (memory-bound)
16. Q: GEMM 的 AI (大矩阵)？ A: ~O(N) FLOPs/Byte (compute-bound)
17. Q: `__syncthreads()` 作用？ A: Block 内 thread 同步屏障
18. Q: 动态 shared memory 声明方式？ A: `extern __shared__ float s[]`
19. Q: Sector 大小？ A: 32 bytes
20. Q: Stride-2 访问的带宽利用率？ A: 50%
21. Q: 如何测量 coalescing 效率？ A: Nsight Compute 的 sector efficiency metric
22. Q: Shared memory 和 L1 的物理关系？ A: 共享同一物理空间，可配置比例
23. Q: A100 每 SM shared memory 最大？ A: 164KB
24. Q: float4 一次加载多少 bytes？ A: 16 bytes (128 bits)
25. Q: Memory-level parallelism 的含义？ A: 多个并发 memory request 提高带宽利用
26. Q: Tiled GEMM 的数据复用率？ A: 每个元素被复用 TILE_SIZE 次
27. Q: Non-coalesced 访问的最坏情况？ A: 32 个独立 32B transaction
28. Q: Register spill 到哪里？ A: Local memory（实际在 global memory，有 L1/L2 缓存）
29. Q: 如何配置 shared memory 大小？ A: `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, size)`
30. Q: Swizzle 的数学操作？ A: `new_col = col XOR row`（或类似 XOR 变换）
