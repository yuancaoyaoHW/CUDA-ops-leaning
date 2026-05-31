# Parallel Reduction

## 1. 学习目标

- 理解 parallel reduction（并行归约）的算法原理与多种实现方式
- 掌握 tree reduction、warp shuffle reduction、block reduction 的区别
- 理解 reduction 的 memory-bound 特性与带宽利用率优化
- 能够实现高效的 sum、max、min、argmax 等归约操作
- 掌握 multi-block reduction 的两阶段策略

## 2. 前置知识

- CUDA 执行模型（warp、shared memory）
- Memory coalescing
- Warp shuffle 指令（`__shfl_down_sync`）
- Atomic 操作

## 3. 核心术语表

| 术语 | 英文全称 | 定义 |
|------|----------|------|
| Reduction | Reduction | 将 N 个元素通过某种二元操作归约为 1 个结果 |
| Tree Reduction | Tree Reduction | 每步将元素数减半的并行归约方式 |
| Warp Shuffle | Warp Shuffle | warp 内 thread 直接交换 register 值的指令 |
| Block Reduction | Block Reduction | 一个 block 内完成的归约 |
| Grid Reduction | Grid Reduction | 跨 block 的全局归约 |
| Cooperative Groups | Cooperative Groups | CUDA 提供的灵活同步 API |
| Atomic | Atomic Operation | 不可分割的读-改-写操作 |

## 4. 动机

### 4.1 Reduction 在 LLM 中的应用

- Softmax 中的 max 和 sum（每行归约）
- LayerNorm / RMSNorm 中的 mean 和 variance
- Loss 计算中的 sum
- Attention score 的 row-wise max（FlashAttention 中的 online softmax）
- Gradient 的 all-reduce（分布式训练）

### 4.2 为什么需要并行 reduction？

串行 reduction：O(N) 时间
并行 reduction：O(log N) 步，每步 O(N/step) 工作 → 总 O(N) 工作，O(log N) 延迟

关键挑战：
- GPU 上 N 可能很大（百万级）
- 需要跨 warp、跨 block 同步
- Memory-bound：计算量极小（每元素 1 次操作），瓶颈在读取数据

## 5. 数学定义

### 5.1 归约操作

给定数组 `x[0..N-1]` 和二元操作 ⊕（满足结合律）：
```
result = x[0] ⊕ x[1] ⊕ x[2] ⊕ ... ⊕ x[N-1]
```

常见操作：
- Sum: ⊕ = +
- Max: ⊕ = max
- Min: ⊕ = min
- Product: ⊕ = ×
- LogSumExp: log(Σ exp(x_i))

### 5.2 带宽利用率

```
Effective Bandwidth = N × sizeof(element) / kernel_time
Utilization = Effective_Bandwidth / Peak_Bandwidth
```

理想 reduction：读 N 个元素，写 1 个结果
```
Minimum time = N × sizeof(float) / Peak_Bandwidth
// A100: 1M floats → 4MB / 2TB/s = 2μs
```

### 5.3 并行步数

```
Steps = ceil(log2(N))
// N = 1M → 20 steps
```

## 6. 推导逻辑

### 6.1 Naive Tree Reduction（有 bank conflict）

```
Step 1: thread 0 += thread 1, thread 2 += thread 3, ...
Step 2: thread 0 += thread 2, thread 4 += thread 6, ...
Step 3: thread 0 += thread 4, ...
```

问题：stride 从 1 开始递增 → 活跃 thread 不连续 → warp divergence + bank conflict

### 6.2 改进：Reversed Tree（无 divergence）

```
Step 1: stride = blockDim/2
        thread i += thread i+stride  (i < stride)
Step 2: stride /= 2
        thread i += thread i+stride  (i < stride)
...
```

优点：活跃 thread 总是连续的（0 到 stride-1）→ 无 warp divergence

### 6.3 Warp-level Reduction（最快）

```
// 使用 __shfl_down_sync，无需 shared memory
val += __shfl_down_sync(0xffffffff, val, 16);
val += __shfl_down_sync(0xffffffff, val, 8);
val += __shfl_down_sync(0xffffffff, val, 4);
val += __shfl_down_sync(0xffffffff, val, 2);
val += __shfl_down_sync(0xffffffff, val, 1);
// lane 0 持有 warp 内 32 个值的 sum
```

### 6.4 完整 Block Reduction

```
1. 每个 thread 从 global memory 加载多个元素并局部累加（提高带宽利用）
2. Warp-level reduction → 每个 warp 得到一个部分和
3. Warp 0 的 lane 0 收集所有 warp 的部分和（通过 shared memory）
4. Warp 0 内再做一次 warp reduction → block 结果
```

### 6.5 Multi-Block Reduction

方案 A：两次 kernel launch
```
Kernel 1: 每个 block 归约一部分 → 写入中间数组 partial[num_blocks]
Kernel 2: 一个 block 归约 partial[] → 最终结果
```

方案 B：Atomic（简单但可能有竞争）
```
每个 block 归约后 atomicAdd 到全局结果
```

方案 C：Cooperative Groups grid sync（一次 kernel）
```
所有 block 归约 → grid.sync() → block 0 做最终归约
```

## 7. 算子流程

### 7.1 高效 Block Reduction 实现

```cuda
template<int BLOCK_SIZE>
__device__ float block_reduce_sum(float val) {
    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    
    // Block reduction via shared memory
    __shared__ float warp_sums[BLOCK_SIZE / 32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    
    if (lane == 0) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();
    
    // Final reduction in first warp
    if (warp_id == 0) {
        val = (lane < BLOCK_SIZE / 32) ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
    }
    
    return val;  // 只有 thread 0 持有正确结果
}
```

### 7.2 Vector Load + Multi-element Reduction

```cuda
// 每个 thread 处理多个元素，提高带宽利用
__global__ void reduce_sum(const float* input, float* output, int n) {
    float sum = 0.0f;
    
    // Grid-stride loop: 每个 thread 处理多个元素
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // 使用 float4 向量化加载
    const float4* input4 = reinterpret_cast<const float4*>(input);
    int n4 = n / 4;
    
    for (int i = tid; i < n4; i += stride) {
        float4 v = input4[i];
        sum += v.x + v.y + v.z + v.w;
    }
    
    // 处理尾部
    for (int i = n4 * 4 + tid; i < n; i += stride) {
        sum += input[i];
    }
    
    // Block reduction
    sum = block_reduce_sum<256>(sum);
    
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}
```

## 8. PyTorch baseline

```python
import torch

def reduce_sum_pytorch(x: torch.Tensor) -> torch.Tensor:
    return x.sum()

def reduce_max_pytorch(x: torch.Tensor) -> torch.Tensor:
    return x.max()

def row_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    """行归约 - softmax/layernorm 中使用"""
    return x.sum(dim=-1, keepdim=True)

# Benchmark
N = 1 << 20
x = torch.randn(N, device='cuda')

# PyTorch sum 内部调用 CUB 或 custom kernel
result = reduce_sum_pytorch(x)
```

## 9. CUDA 实现思路

### 9.1 完整的高性能 Reduction Kernel

```cuda
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

template<typename T, int BLOCK_SIZE>
__global__ void reduce_kernel(const T* __restrict__ input, 
                              T* __restrict__ output, int n) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    T sum = 0;
    
    // Grid-stride loop with vectorized loads
    int idx = blockIdx.x * BLOCK_SIZE * 4 + threadIdx.x;
    int grid_stride = BLOCK_SIZE * 4 * gridDim.x;
    
    while (idx + 3 * BLOCK_SIZE < n) {
        sum += input[idx];
        sum += input[idx + BLOCK_SIZE];
        sum += input[idx + 2 * BLOCK_SIZE];
        sum += input[idx + 3 * BLOCK_SIZE];
        idx += grid_stride;
    }
    while (idx < n) {
        sum += input[idx];
        idx += BLOCK_SIZE;
    }
    
    // Warp reduction
    for (int offset = warp.size() / 2; offset > 0; offset >>= 1) {
        sum += warp.shfl_down(sum, offset);
    }
    
    // Store warp results
    __shared__ T warp_results[BLOCK_SIZE / 32];
    if (warp.thread_rank() == 0) {
        warp_results[threadIdx.x / 32] = sum;
    }
    block.sync();
    
    // Final reduction
    if (threadIdx.x < BLOCK_SIZE / 32) {
        sum = warp_results[threadIdx.x];
        for (int offset = (BLOCK_SIZE / 32) / 2; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (threadIdx.x == 0) {
            atomicAdd(output, sum);
        }
    }
}
```

### 9.2 Row-wise Reduction（用于 Softmax/LayerNorm）

```cuda
// 每个 block 处理一行
__global__ void row_max(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.x;
    const float* row_data = input + row * cols;
    
    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        max_val = fmaxf(max_val, row_data[i]);
    }
    
    // Block reduction for max
    max_val = block_reduce_max<256>(max_val);
    
    if (threadIdx.x == 0) {
        output[row] = max_val;
    }
}
```

## 10. Triton 实现思路

```python
import triton
import triton.language as tl

@triton.jit
def reduce_sum_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    result = tl.sum(x, axis=0)
    
    if pid == 0:
        # 注意：这只是单 block 的 sum
        # 多 block 需要额外的 reduce 步骤
        tl.store(output_ptr, result)

@triton.jit
def row_reduce_sum_kernel(
    input_ptr, output_ptr,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """每个 program 处理一行"""
    row_idx = tl.program_id(0)
    
    row_start = row_idx * n_cols
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # 累加多个 chunk
    acc = tl.zeros([1], dtype=tl.float32)
    for start in range(0, n_cols, BLOCK_SIZE):
        cols = start + offsets
        mask = cols < n_cols
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=0.0)
        acc += tl.sum(x, axis=0)
    
    tl.store(output_ptr + row_idx, acc)
```

## 11. Memory Access 分析

### 11.1 数据量

- 输入：N × sizeof(float) bytes
- 输出：1 × sizeof(float) bytes（或 num_blocks 个中间结果）
- Shared memory：BLOCK_SIZE / 32 × sizeof(float)

### 11.2 带宽利用率

理想情况：
```
Time = N × 4 bytes / 2 TB/s  (A100)
// N = 1M: Time = 4MB / 2TB/s = 2μs
```

实际影响因素：
- 向量化加载（float4）可提高 ~20% 带宽利用
- Grid-stride loop 减少 block 数量 → 减少 launch overhead
- 多元素累加减少 warp reduction 次数

## 12. Parallelism 分析

- 第一阶段（加载+局部累加）：完全并行，N/4 个 float4 加载
- Warp reduction：5 步，每步 warp 内并行
- Block reduction：log2(warps_per_block) 步
- Grid reduction：atomic 或第二次 kernel

## 13. Compute-bound / Memory-bound 判断

**Reduction 是典型的 memory-bound 操作**：
- 每个元素只做 1 次加法（或 max/min）
- Arithmetic Intensity = 1 FLOP / 4 Bytes = 0.25
- A100 roofline 拐点 ≈ 19.5 TFLOPS / 2 TB/s ≈ 10 FLOPs/Byte
- 0.25 << 10 → 严重 memory-bound

优化方向：最大化带宽利用率，而非增加计算。

## 14. Profiling 指标

| 指标 | 目标值 | 工具 |
|------|--------|------|
| Memory Throughput | > 80% peak | Nsight Compute |
| L2 Hit Rate | 低（streaming access） | Nsight Compute |
| Achieved Occupancy | > 50% | Nsight Compute |
| Warp Execution Efficiency | > 95% | Nsight Compute |
| SM Active Cycles | 高 | Nsight Systems |

## 15. Benchmark 设计

```python
import torch
import triton

sizes = [2**i for i in range(10, 25)]  # 1K to 16M

for N in sizes:
    x = torch.randn(N, device='cuda')
    
    # PyTorch baseline
    torch.cuda.synchronize()
    t_pytorch = benchmark(lambda: x.sum())
    
    # Custom kernel
    t_custom = benchmark(lambda: custom_reduce(x))
    
    # 计算带宽利用率
    bytes_accessed = N * 4
    bw_pytorch = bytes_accessed / t_pytorch / 1e9  # GB/s
    bw_custom = bytes_accessed / t_custom / 1e9
    
    print(f"N={N:>10d} | PyTorch: {bw_pytorch:.0f} GB/s | Custom: {bw_custom:.0f} GB/s")
```

## 16. 常见错误

1. **忘记 `__syncthreads()`**：shared memory 数据不一致
2. **Warp shuffle mask 错误**：使用 `0xffffffff` 而非实际活跃 mask
3. **Bank conflict**：shared memory 写入模式不当
4. **Race condition**：多 block atomic 时未初始化 output
5. **尾部处理遗漏**：N 不是 BLOCK_SIZE 整数倍时丢失数据
6. **Float 精度**：大数组 sum 时精度损失 → 使用 Kahan summation 或 FP64 累加

## 17. 实验任务

1. 实现 naive tree reduction，测量带宽利用率
2. 实现 warp shuffle reduction，对比性能
3. 实现 vectorized load (float4) reduction
4. 实现 row-wise reduction（用于 softmax 前的 max）
5. 对比不同 BLOCK_SIZE (64, 128, 256, 512) 的性能
6. 实现 multi-block reduction（atomic vs two-pass）
7. 用 Nsight Compute 分析带宽利用率

## 18. 习题 20 道

1. Parallel reduction 的时间复杂度是多少？工作复杂度呢？
2. 为什么 naive tree reduction 有 warp divergence？如何修复？
3. `__shfl_down_sync` 的第一个参数 mask 的作用是什么？
4. 一个 BLOCK_SIZE=256 的 block 做 reduction 需要几步 warp shuffle + 几步 shared memory？
5. 为什么 grid-stride loop 比 "一个 thread 一个元素" 更高效？
6. Float32 对 1M 个元素求和的精度误差大约是多少？如何改善？
7. Reduction 的 arithmetic intensity 是多少？它是 compute-bound 还是 memory-bound？
8. 使用 float4 向量化加载能提升多少带宽利用率？为什么？
9. `atomicAdd` 在 reduction 中的优缺点是什么？
10. Cooperative Groups 的 `grid.sync()` 有什么限制？
11. 为什么 reduction kernel 通常不需要高 occupancy？
12. Row-wise reduction 中，如果 cols=128K，一个 block 如何处理？
13. 两阶段 reduction 的中间数组大小如何选择？
14. Warp shuffle reduction 为什么不需要 shared memory？
15. 如何实现 parallel argmax（返回最大值的索引）？
16. CUB 的 `DeviceReduce::Sum` 内部使用什么策略？
17. 在 FlashAttention 中，online softmax 的 reduction 是如何增量进行的？
18. 多 GPU 的 all-reduce 和单 GPU 的 reduction 有什么本质区别？
19. 如何用 Triton 实现跨 block 的 reduction？
20. Reduction 中 shared memory padding 什么时候需要？

## 19. 标准答案

1. 时间 O(log N)，工作 O(N)。N 个元素需要 N-1 次操作（工作量不变），但并行后延迟降为 log N 步。

2. Naive tree reduction 中，step i 只有 thread id 为 2^i 的倍数的 thread 活跃。同一 warp 内活跃/不活跃 thread 交替 → divergence。修复：改为 stride 从 blockDim/2 递减，活跃 thread 始终连续。

3. Mask 指定哪些 lane 参与 shuffle。如果 warp 中部分 thread 已退出（如边界条件），需要正确设置 mask 避免未定义行为。通常用 `0xffffffff` 表示所有 32 个 lane 参与。

4. Warp shuffle: 5 步（32→16→8→4→2→1）。Shared memory: 需要收集 256/32=8 个 warp 的结果，再做一次 warp shuffle（3 步）。总计 5+3=8 步。

5. Grid-stride loop 让少量 block 处理大量数据：(a) 减少 block launch 开销；(b) 每个 thread 累加多个元素后再做 reduction，减少 reduction 步数；(c) 更好的 L2 cache 利用。

6. Float32 有 ~7 位有效数字。1M 个随机数求和，相对误差约 O(√N × ε) ≈ √(10^6) × 10^-7 ≈ 10^-4。改善方法：Kahan summation、分块求和、使用 FP64 累加器。

7. AI = 1 FLOP / 4 Bytes = 0.25 FLOPs/Byte。远低于 roofline 拐点（~10），严重 memory-bound。

8. Float4 一次加载 16 bytes，减少 load 指令数量，更好利用 128-byte cache line。通常提升 10-20% 带宽利用率。

9. 优点：简单，一次 kernel 完成。缺点：高竞争时串行化严重；FP32 atomicAdd 在 Pascal+ 硬件支持，但 FP16 需要 CAS loop。

10. 限制：需要 `cudaLaunchCooperativeKernel`；grid size 不能超过 SM 数量 × 每 SM 最大 block 数；所有 block 必须同时驻留。

11. Reduction 是 memory-bound，瓶颈在带宽而非计算。只要有足够 warp 隐藏 memory latency（~20-30% occupancy），更高 occupancy 不会提升性能。

12. 使用循环：每个 thread 用 grid-stride 方式遍历 128K 个元素，局部累加后再做 block reduction。

13. 中间数组大小 = grid size（block 数量）。通常选择 grid size 使得每个 block 处理足够多元素（如 1024-4096 个），平衡 launch overhead 和并行度。

14. Warp shuffle 通过硬件直接在 register 间传递数据，延迟 ~1 cycle，不经过任何 memory 层次。

15. 使用 struct {float val; int idx}，reduction 时比较 val，保留较大者的 idx。Warp shuffle 时同时传递 val 和 idx。

16. CUB 使用 grid-stride + vectorized load + warp reduction + block reduction + device-wide atomic/two-pass。自动选择最优配置。

17. FlashAttention 维护 running max 和 running sum，每处理一个 block 更新：new_max = max(old_max, block_max)，rescale old_sum，加上 new block 的 exp sum。

18. 单 GPU reduction 最终只有一个结果。All-reduce 要求所有 GPU 都得到相同的全局结果 → 需要通信（ring、tree、butterfly 等拓扑）。

19. Triton 单个 program 内用 `tl.sum`。跨 program 需要：(a) 每个 program 写 partial result 到 global memory；(b) 第二个 kernel 归约 partial results。

20. 当 warp 内 thread 以 stride 访问 shared memory 且 stride 是 32 的倍数时需要 padding。Reduction 中如果 warp_results 数组较小（<32），通常不需要。

## 20. 复习卡片 30 张

1. Q: Reduction 的 arithmetic intensity? A: ~0.25 FLOPs/Byte, memory-bound
2. Q: Warp shuffle reduction 需要几步? A: 5 步 (offset: 16,8,4,2,1)
3. Q: `__shfl_down_sync(mask, val, offset)` 做什么? A: 将 lane+offset 的 val 传给当前 lane
4. Q: Block reduction 的 shared memory 用量? A: (BLOCK_SIZE/32) × sizeof(element)
5. Q: 为什么用 grid-stride loop? A: 减少 block 数，每 thread 多元素累加，提高效率
6. Q: Float4 加载的对齐要求? A: 地址必须 16-byte 对齐
7. Q: Two-pass reduction 的中间数组大小? A: = grid_size (block 数量)
8. Q: atomicAdd 的精度问题? A: 非确定性求和顺序导致浮点结果不可复现
9. Q: Cooperative kernel 的 grid size 限制? A: 所有 block 必须能同时驻留在 GPU 上
10. Q: CUB DeviceReduce 的优势? A: 自动调优，处理任意大小，高带宽利用
11. Q: Row-wise reduction 的 block 分配? A: 通常一个 block 处理一行（或几行）
12. Q: Online reduction 是什么? A: 流式处理数据，不需要存储全部元素（如 online softmax）
13. Q: Reduction 需要高 occupancy 吗? A: 不需要，memory-bound，20-30% 通常足够
14. Q: Warp divergence 在 reduction 中如何避免? A: 活跃 thread 保持连续（stride 递减方式）
15. Q: __syncthreads() 在 reduction 中何时需要? A: shared memory 写后读之前
16. Q: Kahan summation 的额外开销? A: 每次加法多 3 次 FP 操作，但 reduction 是 memory-bound 所以几乎免费
17. Q: Argmax reduction 如何实现? A: 同时传递 (value, index) pair，比较 value 保留 winner
18. Q: 为什么 reduction output 要预先清零? A: atomicAdd 是累加，不清零会得到错误结果
19. Q: Triton 的 tl.sum 对应什么? A: 单 program 内的 reduction，编译为 warp shuffle
20. Q: All-reduce vs reduce 的区别? A: All-reduce 所有节点得到结果，reduce 只有 root 得到
21. Q: Segmented reduction 是什么? A: 对数组的多个 segment 分别做 reduction
22. Q: Reduction 中 NaN 如何处理? A: 需要特殊处理，fmaxf(NaN, x) 行为依赖实现
23. Q: Block size 对 reduction 性能的影响? A: 256 通常最优，太小浪费 launch，太大 occupancy 受限
24. Q: Vectorized load 为什么能提升性能? A: 减少 load 指令数，更好利用 memory controller
25. Q: Persistent kernel reduction 是什么? A: 少量 block 常驻，循环处理所有数据
26. Q: Reduction 的 bank conflict 场景? A: 写 shared memory 时 stride 为 32 的倍数
27. Q: FP16 reduction 的精度策略? A: 用 FP32 累加器，最后转回 FP16
28. Q: Multi-GPU reduction 的通信模式? A: Ring all-reduce, tree all-reduce, butterfly
29. Q: Reduction 在 backward pass 中的角色? A: Gradient aggregation (sum of gradients)
30. Q: 如何验证 reduction 正确性? A: 与 CPU double precision 结果对比，允许 ULP 误差
