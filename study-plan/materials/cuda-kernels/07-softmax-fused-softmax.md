# Softmax 与 Fused Softmax

## 1. 学习目标

- 理解 softmax 的数学定义与数值稳定性问题
- 掌握 online softmax 算法（单 pass 计算 max + sum）
- 理解 fused softmax 的动机与实现策略
- 能够实现 row-wise softmax kernel（适用于 attention score）
- 掌握 softmax 的 memory-bound 特性与优化方向

## 2. 前置知识

- Parallel reduction
- Warp shuffle
- Shared memory 使用
- 数值稳定性（浮点溢出）

## 3. 核心术语表

| 术语 | 英文全称 | 定义 |
|------|----------|------|
| Softmax | Softmax | 将向量映射为概率分布的函数 |
| Safe Softmax | Numerically Stable Softmax | 减去 max 后再计算的稳定版本 |
| Online Softmax | Online Softmax | 单 pass 同时计算 max 和 sum 的算法 |
| Fused Softmax | Fused Softmax | 将 softmax 的多个步骤融合为一个 kernel |
| Row-wise | Row-wise | 沿行方向操作（每行独立计算 softmax） |
| Flash Softmax | Flash Softmax | FlashAttention 中使用的分块 online softmax |

## 4. 动机

### 4.1 Softmax 在 LLM 中的位置

```
Attention Score = Q × K^T / √d_k     → shape [B, H, S, S]
Attention Weight = softmax(Score, dim=-1)  → 每行独立 softmax
Output = Attention Weight × V
```

对于 seq_len=4096, heads=32：
- Score 矩阵大小：32 × 4096 × 4096 × 2 bytes = 1GB (FP16)
- 如果分步计算（读 score → 写 max → 读 score → 写 exp → 读 exp → 写 sum → 读 exp → 写 result）：4 次读写 = 8GB 数据移动
- Fused：1 次读 + 1 次写 = 2GB 数据移动 → 4x 带宽节省

### 4.2 数值稳定性问题

```python
# Naive softmax - 会溢出！
x = [1000, 1001, 1002]
exp(1000) = inf  # FP16 max ≈ 65504, FP32 max ≈ 3.4e38

# Safe softmax
m = max(x) = 1002
exp(1000 - 1002) = exp(-2) = 0.135  # 安全
```

### 4.3 三步 vs 单步

**三步（naive）**：
1. Pass 1: 找 max → 读 N 个元素
2. Pass 2: 计算 exp(x - max) 并求 sum → 读 N 个元素
3. Pass 3: 除以 sum → 读 N 个元素

**单步（online）**：
1. 一次遍历同时维护 running max 和 running sum → 读 N 个元素
2. 最终归一化 → 读 N 个元素（或 fuse 到下一步）

## 5. 数学定义

### 5.1 标准 Softmax

```
softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
```

### 5.2 Safe Softmax

```
m = max(x)
softmax(x_i) = exp(x_i - m) / Σ_j exp(x_j - m)
```

### 5.3 Online Softmax（Milakov & Gimelshein, 2018）

维护两个 running 变量：
```
初始化: m = -inf, d = 0

对每个新元素 x_i:
    m_new = max(m, x_i)
    d_new = d × exp(m - m_new) + exp(x_i - m_new)
    m = m_new
    d = d_new

最终: softmax(x_i) = exp(x_i - m) / d
```

**证明正确性**：
```
d 始终等于 Σ_{j≤i} exp(x_j - m_current)

当处理完所有元素后：
d = Σ_j exp(x_j - m_global) = 正确的分母
```

### 5.4 Log-Softmax

```
log_softmax(x_i) = x_i - m - log(Σ_j exp(x_j - m))
```

更稳定，常用于 loss 计算。

## 6. 推导逻辑

### 6.1 为什么 Online Softmax 正确？

设处理到第 k 个元素时：
```
m_k = max(x_0, ..., x_k)
d_k = Σ_{j=0}^{k} exp(x_j - m_k)
```

当处理第 k+1 个元素：
```
m_{k+1} = max(m_k, x_{k+1})

d_{k+1} = Σ_{j=0}^{k+1} exp(x_j - m_{k+1})
         = Σ_{j=0}^{k} exp(x_j - m_{k+1}) + exp(x_{k+1} - m_{k+1})
         = Σ_{j=0}^{k} exp(x_j - m_k) × exp(m_k - m_{k+1}) + exp(x_{k+1} - m_{k+1})
         = d_k × exp(m_k - m_{k+1}) + exp(x_{k+1} - m_{k+1})
```

### 6.2 并行 Online Softmax（用于 warp/block reduction）

两个部分结果 (m1, d1) 和 (m2, d2) 可以合并：
```
m = max(m1, m2)
d = d1 × exp(m1 - m) + d2 × exp(m2 - m)
```

这使得 online softmax 可以并行化！

### 6.3 Fused Softmax 的 Kernel 设计

**Case 1: 行长度 ≤ warp size (32)**
- 一个 warp 处理一行
- 使用 warp shuffle 做 reduction
- 无需 shared memory

**Case 2: 行长度 ≤ block size (≤1024)**
- 一个 block 处理一行
- 使用 shared memory + warp shuffle

**Case 3: 行长度 > block size**
- 一个 block 处理一行，grid-stride loop
- 需要两次 pass（或 online softmax + 一次 pass 写结果）

## 7. 算子流程

### 7.1 Warp-level Softmax（行长度 ≤ 32）

```cuda
__device__ float warp_softmax(float val, int lane_id, int row_width) {
    // Step 1: Find max
    float max_val = val;
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
    max_val = __shfl_sync(0xffffffff, max_val, 0);  // broadcast
    
    // Step 2: Compute exp(x - max)
    float exp_val = (lane_id < row_width) ? expf(val - max_val) : 0.0f;
    
    // Step 3: Sum
    float sum = exp_val;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    sum = __shfl_sync(0xffffffff, sum, 0);  // broadcast
    
    // Step 4: Normalize
    return exp_val / sum;
}
```

### 7.2 Block-level Fused Softmax

```cuda
template<int BLOCK_SIZE, int ROW_SIZE>
__global__ void fused_softmax_kernel(const float* input, float* output, 
                                      int num_rows, int row_size) {
    __shared__ float smem[BLOCK_SIZE];
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    if (row >= num_rows) return;
    
    const float* row_input = input + row * row_size;
    float* row_output = output + row * row_size;
    
    // Step 1: Load and find max (grid-stride within block)
    float local_max = -INFINITY;
    for (int i = tid; i < row_size; i += BLOCK_SIZE) {
        local_max = fmaxf(local_max, row_input[i]);
    }
    
    // Block reduce max
    local_max = block_reduce_max<BLOCK_SIZE>(local_max, smem);
    __shared__ float row_max;
    if (tid == 0) row_max = local_max;
    __syncthreads();
    
    // Step 2: Compute exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < row_size; i += BLOCK_SIZE) {
        local_sum += expf(row_input[i] - row_max);
    }
    
    // Block reduce sum
    local_sum = block_reduce_sum<BLOCK_SIZE>(local_sum, smem);
    __shared__ float row_sum;
    if (tid == 0) row_sum = local_sum;
    __syncthreads();
    
    // Step 3: Normalize and write
    for (int i = tid; i < row_size; i += BLOCK_SIZE) {
        row_output[i] = expf(row_input[i] - row_max) / row_sum;
    }
}
```

## 8. PyTorch baseline

```python
import torch
import torch.nn.functional as F

def softmax_pytorch(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.softmax(x, dim=dim)

def safe_softmax_manual(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """手动实现 safe softmax"""
    m = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - m)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

# Benchmark: attention score softmax
B, H, S = 4, 32, 4096
scores = torch.randn(B, H, S, S, device='cuda', dtype=torch.float16)

# PyTorch 内部使用 fused kernel
weights = F.softmax(scores, dim=-1)
```

## 9. CUDA 实现思路

### 9.1 Online Softmax Kernel（单 pass 找 max+sum）

```cuda
struct SoftmaxState {
    float max_val;
    float sum;
};

__device__ SoftmaxState combine_states(SoftmaxState a, SoftmaxState b) {
    float new_max = fmaxf(a.max_val, b.max_val);
    float new_sum = a.sum * expf(a.max_val - new_max) 
                  + b.sum * expf(b.max_val - new_max);
    return {new_max, new_sum};
}

__device__ SoftmaxState warp_reduce_softmax_state(SoftmaxState state) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        SoftmaxState other;
        other.max_val = __shfl_down_sync(0xffffffff, state.max_val, offset);
        other.sum = __shfl_down_sync(0xffffffff, state.sum, offset);
        state = combine_states(state, other);
    }
    return state;
}
```

### 9.2 完整 Online Softmax（两 pass：online reduce + normalize）

```cuda
template<int BLOCK_SIZE>
__global__ void online_softmax(const float* input, float* output, 
                                int num_rows, int row_size) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    const float* x = input + row * row_size;
    float* y = output + row * row_size;
    
    // Pass 1: Online reduction (single pass for max + sum)
    SoftmaxState local_state = {-INFINITY, 0.0f};
    for (int i = tid; i < row_size; i += BLOCK_SIZE) {
        float val = x[i];
        SoftmaxState new_elem = {val, 1.0f};
        local_state = combine_states(local_state, new_elem);
    }
    
    // Block reduction of states
    // ... (warp reduce + shared memory)
    __shared__ float final_max, final_sum;
    // (reduction code here)
    __syncthreads();
    
    // Pass 2: Normalize
    for (int i = tid; i < row_size; i += BLOCK_SIZE) {
        y[i] = expf(x[i] - final_max) / final_sum;
    }
}
```

## 10. Triton 实现思路

```python
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    n_cols,
    input_row_stride, output_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    # 一个 program 处理一行
    row_idx = tl.program_id(0)
    
    row_start = row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load row
    row = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=-float('inf'))
    
    # Safe softmax
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator
    
    # Store
    out_start = row_idx * output_row_stride
    tl.store(output_ptr + out_start + col_offsets, result, mask=mask)

# 对于行长度 > BLOCK_SIZE 的情况，需要多次 load + online softmax
@triton.jit
def softmax_kernel_large(
    input_ptr, output_ptr,
    n_cols,
    input_row_stride, output_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * input_row_stride
    
    # Pass 1: find max (online)
    m = tl.full([], -float('inf'), dtype=tl.float32)
    for start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        block = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=-float('inf'))
        m = tl.maximum(m, tl.max(block, axis=0))
    
    # Pass 2: compute sum of exp
    d = tl.zeros([], dtype=tl.float32)
    for start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        block = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=-float('inf'))
        d += tl.sum(tl.exp(block - m), axis=0)
    
    # Pass 3: normalize
    for start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        block = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=-float('inf'))
        result = tl.exp(block - m) / d
        tl.store(output_ptr + row_start + col_offsets, result, mask=mask)
```

## 11. Memory Access 分析

### 11.1 数据移动量

| 实现 | Global Memory 读 | Global Memory 写 | 总数据移动 |
|------|-----------------|-----------------|-----------|
| 3-pass naive | 3N | N (中间) + N (结果) | 5N |
| 2-pass (fused exp+sum) | 2N | N | 3N |
| Online (2-pass) | 2N | N | 3N |
| Fully fused (行 ≤ BLOCK) | N | N | 2N |

### 11.2 Bandwidth 利用率

```
// 对于 row_size = 4096, FP32, A100 (2TB/s)
// 2-pass fused: 读 2×4096×4 + 写 4096×4 = 48KB per row
// 如果 num_rows = 32×4096 = 131072 rows
// Total data = 131072 × 48KB = 6GB
// Minimum time = 6GB / 2TB/s = 3ms
```

## 12. Parallelism 分析

- 行间并行：每行独立，可分配到不同 block
- 行内并行：reduction 需要同步，一个 block 处理一行
- 对于短行（≤32）：一个 warp 处理一行，多行共享一个 block

## 13. Compute-bound / Memory-bound 判断

```
Arithmetic Intensity = (3N ops: max, exp, div) / (2N × 4 bytes) ≈ 0.375 FLOPs/Byte
A100 roofline 拐点 ≈ 10 FLOPs/Byte (FP32)

0.375 << 10 → 严重 memory-bound
```

优化方向：减少 memory access 次数（fusion），而非增加计算效率。

## 14. Profiling 指标

| 指标 | 期望值 | 异常信号 |
|------|--------|---------|
| Memory Throughput | > 80% peak | 低 → coalescing 问题 |
| Compute Utilization | < 20% | 正常（memory-bound） |
| L2 Hit Rate | 取决于行长度 | 高 → 数据在 L2 中被复用 |
| Warp Stall (Memory) | 主要 stall 原因 | 正常 |
| Achieved Occupancy | > 50% | 低 → register/smem 限制 |

## 15. Benchmark 设计

```python
import torch
import triton

configs = [
    # (batch, heads, seq_len) → row_size = seq_len
    (1, 32, 128),
    (1, 32, 512),
    (1, 32, 2048),
    (1, 32, 4096),
    (4, 32, 4096),
    (8, 32, 4096),
]

for B, H, S in configs:
    x = torch.randn(B * H, S, S, device='cuda', dtype=torch.float16)
    
    # Benchmark PyTorch
    torch.cuda.synchronize()
    # ... timing code
    
    # Benchmark custom kernel
    # ... timing code
    
    # Report: time, bandwidth utilization, speedup vs PyTorch
```

## 16. 常见错误

1. **忘记减 max**：FP16 下 exp(x) 对 x > 11 就溢出
2. **Reduction 同步错误**：`__syncthreads()` 放错位置
3. **边界处理**：行长度不是 BLOCK_SIZE 的倍数时 mask 错误
4. **精度问题**：FP16 累加 sum 时精度不够 → 用 FP32 累加
5. **Online softmax 合并错误**：exp(m_old - m_new) 计算顺序错
6. **Warp shuffle mask 错误**：应使用 `0xffffffff` 或正确的活跃 mask
7. **Shared memory 大小不够**：动态分配时忘记传 size
8. **行长度假设错误**：假设行长度 ≤ BLOCK_SIZE 但实际更大

## 17. 实验任务

1. 实现 naive 3-pass softmax kernel，验证正确性
2. 实现 2-pass fused softmax（fuse exp+sum），对比带宽利用率
3. 实现 online softmax（单 pass max+sum），验证数值一致性
4. 实现 warp-level softmax（行长度 ≤ 32），用于小 attention
5. 用 Triton 实现 softmax，对比 CUDA 版本性能
6. 对比不同行长度（128, 512, 2048, 8192）下的性能
7. 测量 FP16 vs FP32 的精度差异（与 PyTorch FP64 对比）
8. Profile kernel，确认 memory-bound 特性

## 18. 习题 20 道

1. 写出 softmax 的数学定义。为什么需要减去 max？
2. Online softmax 的状态更新公式是什么？证明其正确性。
3. 两个 online softmax 部分结果 (m1, d1) 和 (m2, d2) 如何合并？
4. Softmax 是 compute-bound 还是 memory-bound？为什么？
5. 对于 row_size=4096 的 FP16 softmax，理论最小执行时间是多少？（假设 A100）
6. Fused softmax 相比 3-pass 版本节省了多少 memory traffic？
7. 为什么 FlashAttention 需要 online softmax 而不是标准 softmax？
8. 实现 warp-level max reduction（使用 `__shfl_down_sync`）。
9. 如果行长度为 8192 但 block size 为 256，如何设计 kernel？
10. Log-softmax 相比 softmax 有什么数值优势？
11. Softmax 的梯度公式是什么？如何 fuse backward？
12. 在 Triton 中，`tl.max` 和 `tl.sum` 的 reduction 是如何并行化的？
13. 为什么 FP16 softmax 需要用 FP32 做中间累加？
14. 如果 softmax 的输入全是相同值，输出是什么？梯度是什么？
15. 设计一个 benchmark 来验证 softmax kernel 的带宽利用率。
16. Online softmax 中 `exp(m_old - m_new)` 可能下溢吗？如何处理？
17. 对比 PyTorch `F.softmax` 和自定义 Triton kernel 的性能。
18. 如何将 softmax 和前面的 matmul（QK^T）fuse 到一起？
19. Causal mask 如何影响 softmax 的实现？
20. 多头 attention 中，softmax 的并行度如何设计？

## 19. 标准答案

1. softmax(x_i) = exp(x_i) / Σ exp(x_j)。减 max 防止 exp 溢出，不改变结果因为 exp(x-m)/Σexp(x_j-m) = exp(x)/exp(m) / (Σexp(x_j)/exp(m))。

2. m_new = max(m, x_i); d_new = d × exp(m - m_new) + exp(x_i - m_new)。正确性：d 始终等于 Σ_{j≤i} exp(x_j - m_current)。

3. m = max(m1, m2); d = d1 × exp(m1 - m) + d2 × exp(m2 - m)。

4. Memory-bound。AI = ~3 FLOPs / 8 Bytes ≈ 0.375，远低于 roofline 拐点。

5. 读 4096×2B + 写 4096×2B = 16KB。A100 2TB/s → 16KB/2TB/s = 8ns（单行）。实际受 kernel launch 和 occupancy 限制。

6. 3-pass: 5N bytes; Fused 2-pass: 3N bytes → 节省 40%。

7. FlashAttention 分块计算 attention，每块只看到部分 score，需要 online 更新全局 max 和 sum。

8. `float m = val; for(int o=16;o>0;o>>=1) m = fmaxf(m, __shfl_down_sync(0xffffffff, m, o));`

9. Grid-stride loop：每个 thread 处理多个元素，block 内做 reduction。需要 2 pass（或 online + 1 pass normalize）。

10. log_softmax = x - m - log(sum_exp)，避免了除法和 exp 后再取 log 的精度损失。

(后续答案略，完整版见实验手册)

## 20. 复习卡片 30 张

1. Q: Softmax 公式？ A: softmax(x_i) = exp(x_i - max) / Σ exp(x_j - max)
2. Q: 为什么减 max？ A: 防止 exp 溢出，FP16 max=65504，exp(11.1)≈65504
3. Q: Online softmax 状态？ A: (m, d)，m=running max, d=running sum of exp
4. Q: Online softmax 更新？ A: m'=max(m,x); d'=d×exp(m-m')+exp(x-m')
5. Q: 两个状态合并？ A: m=max(m1,m2); d=d1×exp(m1-m)+d2×exp(m2-m)
6. Q: Softmax bound 类型？ A: Memory-bound，AI≈0.375
7. Q: Fused 节省多少？ A: 从 5N 降到 2-3N bytes
8. Q: Warp softmax 适用？ A: 行长度 ≤ 32
9. Q: Block softmax 适用？ A: 行长度 ≤ block_size（或 grid-stride）
10. Q: FlashAttention 为何用 online？ A: 分块计算，无法一次看到全部 score
11. Q: FP16 softmax 精度？ A: 中间用 FP32 累加，最终转回 FP16
12. Q: Log-softmax 优势？ A: 避免 exp→div→log 的精度损失
13. Q: Softmax 梯度？ A: dy_i = y_i × (dout_i - Σ y_j × dout_j)
14. Q: Causal mask 处理？ A: mask 位置设为 -inf，softmax 后自动为 0
15. Q: 行间并行度？ A: B×H×S 行，每行独立
16. Q: Nsight 看什么？ A: Memory Throughput, L2 hit rate
17. Q: exp 下溢处理？ A: exp(large_negative) = 0，不影响正确性
18. Q: Triton softmax 优势？ A: 自动处理 mask、向量化、block 大小选择
19. Q: 大行 softmax 策略？ A: 2-pass（online reduce + normalize）或 3-pass
20. Q: Softmax temperature？ A: softmax(x/T)，T→0 变 argmax，T→∞ 变 uniform
21. Q: Attention score 形状？ A: [B, H, S_q, S_kv]
22. Q: Softmax 输出和？ A: 每行和为 1（概率分布）
23. Q: 全相同输入？ A: 输出 = 1/N（均匀分布）
24. Q: Softmax vs sigmoid？ A: softmax 多类互斥，sigmoid 多标签独立
25. Q: Warp shuffle 优势？ A: 无需 shared memory，延迟更低
26. Q: __syncthreads 位置？ A: reduction 每步之间，或 smem 读写之间
27. Q: 向量化加载？ A: float4 一次读 16B，提高带宽利用
28. Q: Softmax 数值范围？ A: 输出 ∈ (0, 1)，和为 1
29. Q: Online vs 3-pass 精度？ A: 数学等价，精度相同
30. Q: Kernel fusion 收益来源？ A: 减少 global memory 读写次数
