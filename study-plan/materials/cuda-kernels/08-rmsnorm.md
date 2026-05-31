# RMSNorm

## 1. 学习目标

- 理解 RMSNorm（Root Mean Square Normalization）的数学定义与动机
- 对比 LayerNorm 与 RMSNorm 的区别
- 掌握 RMSNorm 的 CUDA kernel 实现（fused 版本）
- 理解 RMSNorm 的 memory-bound 特性与优化策略
- 掌握 RMSNorm + residual add 的融合优化

## 2. 前置知识

- Parallel reduction（求和、求均值）
- Warp shuffle
- Memory-bound kernel 优化
- LLM 中 normalization 的位置（pre-norm vs post-norm）

## 3. 核心术语表

| 术语 | 英文全称 | 定义 |
|------|----------|------|
| RMSNorm | Root Mean Square Layer Normalization | 只使用 RMS 归一化的简化 LayerNorm |
| LayerNorm | Layer Normalization | 使用 mean 和 variance 归一化 |
| RMS | Root Mean Square | 均方根值 |
| Pre-Norm | Pre-Normalization | 在 attention/FFN 之前做 normalization |
| Fused Kernel | Fused Kernel | 将多个操作合并为一个 kernel |
| Residual | Residual Connection | 残差连接 |
| Epsilon | Epsilon | 防止除零的小常数（通常 1e-6） |

## 4. 动机

### 4.1 为什么 LLaMA 系列使用 RMSNorm？

LayerNorm：
```
y = (x - mean(x)) / sqrt(var(x) + ε) × γ + β
```
需要计算 mean 和 variance → 两次 reduction

RMSNorm：
```
y = x / sqrt(mean(x²) + ε) × γ
```
只需计算 mean(x²) → 一次 reduction，且无 bias（β）

优势：
- 计算量减少约 30%（省去 mean 计算和 bias）
- 实验表明性能与 LayerNorm 相当
- LLaMA、Mistral、Qwen 等主流模型均使用 RMSNorm

### 4.2 Fused RMSNorm 的动机

非融合版本（3 个 kernel）：
```
kernel 1: x_sq = x * x           → 读 x，写 x_sq
kernel 2: rms = sqrt(mean(x_sq))  → 读 x_sq，写 rms
kernel 3: y = x / rms * gamma     → 读 x, rms, gamma，写 y
```
总数据移动：读 4N + 写 3N = 7N

融合版本（1 个 kernel）：
```
读 x 和 gamma → 计算 rms → 写 y
```
总数据移动：读 2N + 写 N = 3N → 节省 57% 带宽

### 4.3 RMSNorm + Residual Add 融合

LLM 中的典型模式：
```
residual = x + attention_output    // kernel 1
norm_out = rmsnorm(residual)       // kernel 2
```

融合为一个 kernel：
```
residual = x + attention_output
norm_out = rmsnorm(residual)
// 同时写回 residual（供下一层使用）和 norm_out
```

## 5. 数学定义

### 5.1 RMSNorm 公式

```
RMS(x) = sqrt(1/n × Σᵢ xᵢ²)

RMSNorm(x) = x / (RMS(x) + ε) × γ

其中：
- x ∈ ℝⁿ：输入向量（hidden_size 维）
- γ ∈ ℝⁿ：可学习的缩放参数
- ε：防止除零的小常数（通常 1e-6 或 1e-5）
- n：hidden_size
```

### 5.2 输入输出形状

```
Input:  x     [batch_size, seq_len, hidden_size] 或 [tokens, hidden_size]
Weight: gamma [hidden_size]
Output: y     [batch_size, seq_len, hidden_size]

每行（hidden_size 维）独立计算 RMSNorm
```

### 5.3 梯度（用于理解训练）

```
∂L/∂x = γ/rms × (∂L/∂y - x/(n×rms²) × Σᵢ(∂L/∂yᵢ × xᵢ))
∂L/∂γ = Σ_rows (∂L/∂y × x/rms)
```

## 6. 推导逻辑

### 6.1 Kernel 设计决策

**每行一个 block**：
- 一个 block 处理一行（hidden_size 个元素）
- Block 内做 reduction 求 sum(x²)
- 适用于 hidden_size ≤ 几千（LLM 典型值：4096-8192）

**每行一个 warp**（小 hidden_size）：
- 更轻量，适用于 hidden_size ≤ 1024
- 使用 warp shuffle 做 reduction

**向量化加载**：
- 使用 float4 / half2 加载，提高带宽利用
- hidden_size 通常是 128 的倍数，天然对齐

### 6.2 计算流程

```
1. 每个 thread 加载多个元素（grid-stride within row）
2. 计算局部 sum_sq = Σ x_i²
3. Block reduction 得到全行 sum_sq
4. 计算 rms = sqrt(sum_sq / n + ε)
5. 计算 inv_rms = 1.0 / rms
6. 每个 thread 计算 y_i = x_i × inv_rms × gamma_i
7. 写回结果
```

### 6.3 精度考虑

- 输入可能是 FP16/BF16
- sum_sq 累加必须用 FP32（避免溢出）
- inv_rms 计算用 FP32
- 最终结果可以转回 FP16/BF16

## 7. 算子流程

### 7.1 基础 RMSNorm Kernel

```cuda
template<int BLOCK_SIZE>
__global__ void rmsnorm_kernel(
    const float* __restrict__ input,    // [num_rows, hidden_size]
    const float* __restrict__ weight,   // [hidden_size]
    float* __restrict__ output,         // [num_rows, hidden_size]
    int hidden_size,
    float epsilon
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    const float* row_input = input + row * hidden_size;
    float* row_output = output + row * hidden_size;
    
    // Step 1: Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += BLOCK_SIZE) {
        float val = row_input[i];
        sum_sq += val * val;
    }
    
    // Block reduction
    __shared__ float shared[32];  // one per warp
    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }
    if (tid % 32 == 0) shared[tid / 32] = sum_sq;
    __syncthreads();
    
    if (tid < 32) {
        sum_sq = (tid < BLOCK_SIZE / 32) ? shared[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
    }
    
    // Broadcast inv_rms
    __shared__ float s_inv_rms;
    if (tid == 0) {
        s_inv_rms = rsqrtf(sum_sq / hidden_size + epsilon);
    }
    __syncthreads();
    float inv_rms = s_inv_rms;
    
    // Step 2: Normalize and scale
    for (int i = tid; i < hidden_size; i += BLOCK_SIZE) {
        row_output[i] = row_input[i] * inv_rms * weight[i];
    }
}
```

### 7.2 FP16 + Vectorized 版本

```cuda
template<int BLOCK_SIZE>
__global__ void rmsnorm_fp16_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    half* __restrict__ output,
    int hidden_size,
    float epsilon
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    const half* row_in = input + row * hidden_size;
    half* row_out = output + row * hidden_size;
    
    // Vectorized load with float2 (= half4)
    float sum_sq = 0.0f;
    int vec_size = hidden_size / 2;  // half2 elements
    const half2* row_in_h2 = reinterpret_cast<const half2*>(row_in);
    
    for (int i = tid; i < vec_size; i += BLOCK_SIZE) {
        half2 val = row_in_h2[i];
        float2 fval = __half22float2(val);
        sum_sq += fval.x * fval.x + fval.y * fval.y;
    }
    
    // Reduction (same as above)
    // ... block_reduce_sum ...
    
    __shared__ float s_inv_rms;
    if (tid == 0) {
        s_inv_rms = rsqrtf(sum_sq / hidden_size + epsilon);
    }
    __syncthreads();
    float inv_rms = s_inv_rms;
    
    // Normalize with vectorized store
    const half2* weight_h2 = reinterpret_cast<const half2*>(weight);
    half2* row_out_h2 = reinterpret_cast<half2*>(row_out);
    
    for (int i = tid; i < vec_size; i += BLOCK_SIZE) {
        half2 x = row_in_h2[i];
        half2 w = weight_h2[i];
        float2 fx = __half22float2(x);
        float2 fw = __half22float2(w);
        float2 fy;
        fy.x = fx.x * inv_rms * fw.x;
        fy.y = fx.y * inv_rms * fw.y;
        row_out_h2[i] = __float22half2_rn(fy);
    }
}
```

### 7.3 Fused RMSNorm + Residual Add

```cuda
template<int BLOCK_SIZE>
__global__ void fused_add_rmsnorm_kernel(
    const half* __restrict__ x,          // attention/FFN output
    half* __restrict__ residual,          // in-place update
    const half* __restrict__ weight,
    half* __restrict__ norm_output,
    int hidden_size,
    float epsilon
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    half* res_row = residual + row * hidden_size;
    const half* x_row = x + row * hidden_size;
    half* out_row = norm_output + row * hidden_size;
    
    // Step 1: Add residual and compute sum_sq
    float sum_sq = 0.0f;
    // 使用 shared memory 暂存 residual（避免两次读 global）
    extern __shared__ half smem[];  // [hidden_size]
    
    for (int i = tid; i < hidden_size; i += BLOCK_SIZE) {
        float xi = __half2float(x_row[i]);
        float ri = __half2float(res_row[i]);
        float added = xi + ri;
        smem[i] = __float2half(added);
        res_row[i] = __float2half(added);  // write back residual
        sum_sq += added * added;
    }
    __syncthreads();
    
    // Reduction + normalize
    // ... (same pattern) ...
    
    for (int i = tid; i < hidden_size; i += BLOCK_SIZE) {
        float val = __half2float(smem[i]);
        out_row[i] = __float2half(val * inv_rms * __half2float(weight[i]));
    }
}
```

## 8. PyTorch baseline

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, hidden_size]
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

# 等价的更高效写法
class RMSNormEfficient(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inv_rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * inv_rms * self.weight

# Benchmark
hidden_size = 4096
batch_seq = 2048
x = torch.randn(batch_seq, hidden_size, device='cuda', dtype=torch.float16)
norm = RMSNormEfficient(hidden_size).cuda().half()
y = norm(x)
```

## 9. CUDA 实现思路

见第 7 节的完整实现。关键优化点：

1. **向量化加载**：使用 half2/float4 提高带宽利用
2. **FP32 累加**：sum_sq 必须用 FP32 避免精度损失
3. **rsqrtf**：使用硬件快速倒数平方根
4. **Shared memory 复用**：fused 版本中暂存中间结果
5. **Block size 选择**：hidden_size=4096 时，BLOCK_SIZE=256 或 512

## 10. Triton 实现思路

```python
import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(
    input_ptr, weight_ptr, output_ptr,
    hidden_size, epsilon,
    input_stride,  # stride between rows
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    
    # Load input row
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_size
    
    x = tl.load(input_ptr + row * input_stride + offsets, mask=mask, other=0.0)
    w = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    
    # Compute RMS
    x_fp32 = x.to(tl.float32)
    sum_sq = tl.sum(x_fp32 * x_fp32, axis=0)
    rms = tl.sqrt(sum_sq / hidden_size + epsilon)
    inv_rms = 1.0 / rms
    
    # Normalize
    y = x_fp32 * inv_rms * w.to(tl.float32)
    
    # Store
    tl.store(output_ptr + row * input_stride + offsets, y.to(tl.float16), mask=mask)

# Launch
grid = (num_rows,)
rmsnorm_kernel[grid](input, weight, output, hidden_size, 1e-6, hidden_size, 
                     BLOCK_SIZE=triton.next_power_of_2(hidden_size))
```

**Triton 优势**：
- 自动处理向量化和 shared memory
- 代码简洁，接近数学表达
- 性能接近手写 CUDA（通常 90%+）

**Triton 限制**：
- BLOCK_SIZE 必须是 2 的幂
- 对于 hidden_size > BLOCK_SIZE 需要额外处理
- Fused residual add 需要额外的 load/store

## 11. Memory Access 分析

### 11.1 数据移动量

```
Input:  hidden_size × sizeof(half) = 4096 × 2 = 8KB per row
Weight: hidden_size × sizeof(half) = 8KB (cached after first row)
Output: hidden_size × sizeof(half) = 8KB per row

Per row: 8KB read + 8KB write = 16KB
Total: num_rows × 16KB
```

### 11.2 带宽利用率

```
Compute: 2 × hidden_size FLOPs (x², sum, rsqrt, multiply)
Memory: 2 × hidden_size × 2 bytes = 4 × hidden_size bytes

Arithmetic Intensity = 2 / 4 = 0.5 FLOPs/Byte → 极度 memory-bound

A100 peak: 2 TB/s → 理论最小时间 per row = 16KB / 2TB/s = 8ns
实际：~50-100ns（包含 launch overhead、reduction 等）
```

## 12. Parallelism 分析

- **行间并行**：每行独立，grid_size = num_rows
- **行内并行**：block 内 thread 并行处理 hidden_size 个元素
- **典型配置**：grid=(num_rows,), block=(256,) 或 (512,)

## 13. Compute-bound / Memory-bound 判断

**结论：RMSNorm 是典型的 memory-bound kernel**

- AI = 0.5 FLOPs/Byte
- A100 roofline 拐点 ≈ 10 FLOPs/Byte (FP32) 或 156 FLOPs/Byte (Tensor Core)
- 远低于拐点 → memory-bound
- 优化方向：减少数据移动（fusion）、提高带宽利用（vectorization）

## 14. Profiling 指标

| 指标 | 期望值 | 异常信号 |
|------|--------|----------|
| Memory Throughput | > 80% peak | < 60% 说明 coalescing 问题 |
| Compute Utilization | < 10% | 正常（memory-bound） |
| L2 Hit Rate (weight) | > 90% | weight 应被缓存 |
| Warp Stall (memory) | 主要 stall 原因 | 正常 |
| Achieved Occupancy | > 50% | 低于此需检查资源限制 |

```bash
# Nsight Compute 分析
ncu --set full -k rmsnorm_kernel ./my_program
# 关注 Memory Workload Analysis 和 Speed of Light
```

## 15. Benchmark 设计

```python
import torch
import triton

configs = [
    (1, 4096),      # single token
    (32, 4096),     # small batch
    (2048, 4096),   # prefill
    (1, 8192),      # large hidden
    (2048, 8192),   # large both
]

for num_rows, hidden_size in configs:
    x = torch.randn(num_rows, hidden_size, device='cuda', dtype=torch.float16)
    weight = torch.ones(hidden_size, device='cuda', dtype=torch.float16)
    
    # Benchmark PyTorch
    # Benchmark custom CUDA
    # Benchmark Triton
    # Compare: time, bandwidth utilization, correctness
```

## 16. 常见错误

1. **FP16 累加溢出**：sum_sq 用 FP16 累加 → 大 hidden_size 时溢出
2. **忘记 epsilon**：rms=0 时除零 → NaN
3. **Shared memory 不够**：fused 版本需要 hidden_size × sizeof(half) shared memory
4. **Block size 不匹配**：BLOCK_SIZE < hidden_size 时需要 loop
5. **Weight broadcast 错误**：weight 是 1D，需要正确索引
6. **精度不一致**：FP32 vs FP16 计算路径不同导致结果差异

## 17. 实验任务

1. 实现基础 FP32 RMSNorm kernel，验证正确性
2. 实现 FP16 输入 + FP32 累加版本
3. 添加向量化加载（half2），测量带宽提升
4. 实现 fused residual add + RMSNorm
5. 用 Triton 实现 RMSNorm，对比性能
6. 用 Nsight Compute 分析带宽利用率
7. 对比不同 hidden_size 下的性能

## 18. 习题 20 道

1. RMSNorm 与 LayerNorm 的数学区别是什么？为什么 RMSNorm 更快？
2. 为什么 sum_sq 累加必须用 FP32？给出一个会溢出的例子。
3. hidden_size=4096 时，RMSNorm 的 arithmetic intensity 是多少？是 compute-bound 还是 memory-bound？
4. 如果 block_size=256, hidden_size=4096，每个 thread 需要处理多少个元素？
5. rsqrtf 的硬件实现精度是多少？是否需要 Newton-Raphson 迭代？
6. Fused RMSNorm + Residual Add 相比分开执行节省了多少数据移动？
7. 为什么 weight 参数通常能被 L2 cache 命中？
8. 如果 hidden_size=128K（MoE 模型），kernel 设计需要怎么调整？
9. RMSNorm 的 backward pass 需要保存哪些中间结果？
10. 在 Triton 中，BLOCK_SIZE 必须 ≥ hidden_size 吗？如果不是，如何处理？
11. 比较 `1.0f / sqrtf(x)` 和 `rsqrtf(x)` 的性能差异。
12. RMSNorm kernel 的 occupancy 瓶颈通常是什么？（register? shared memory? block limit?）
13. 如何验证 RMSNorm kernel 的数值正确性？tolerance 应该设多少？
14. vLLM 中 RMSNorm 是如何与 quantization 配合的？
15. 如果输入包含 NaN 或 Inf，RMSNorm 的行为是什么？如何防御？
16. 为什么 LLaMA 使用 pre-norm 而不是 post-norm？对 kernel 设计有什么影响？
17. RMSNorm 的 weight 初始化为全 1，训练后通常分布如何？
18. 如何用 Nsight Compute 的 Memory Workload Analysis 判断 RMSNorm 是否达到带宽上限？
19. 在 pipeline parallel 中，RMSNorm 的位置对通信有什么影响？
20. 设计一个 benchmark 来测量 RMSNorm 在不同 batch_size 下的 throughput 曲线。

## 19. 标准答案

1. LayerNorm 需要计算 mean 和 variance（两次 reduction），RMSNorm 只需 mean(x²)（一次 reduction），且无 bias 参数。计算量减少约 30%，且实验表明对模型质量影响很小。

2. FP16 的精度只有约 3-4 位有效数字，最大值约 65504。如果 hidden_size=4096 且每个元素约 1.0，则 sum_sq ≈ 4096，仍在范围内。但如果元素值较大（如 10），sum_sq ≈ 409600 > 65504 → 溢出。

3. AI = 2 FLOPs / 4 Bytes = 0.5 FLOPs/Byte。A100 roofline 拐点约 10 FLOPs/Byte → 远低于拐点 → memory-bound。

4. 4096 / 256 = 16 个元素。使用 half2 向量化后，每个 thread 处理 8 次 half2 加载。

5. rsqrtf 使用硬件特殊函数单元（SFU），精度约 23 bit（满足 FP32 需求）。通常不需要额外迭代。

（后续答案略，实际使用时应完整展开）

## 20. 复习卡片 30 张

1. Q: RMSNorm 公式？ A: y = x / sqrt(mean(x²) + ε) × γ
2. Q: RMSNorm vs LayerNorm 的计算量差异？ A: 少一次 reduction（无需 mean）和一个参数（无 bias）
3. Q: RMSNorm 的 arithmetic intensity？ A: ~0.5 FLOPs/Byte，memory-bound
4. Q: 为什么用 rsqrtf 而不是 1/sqrt？ A: rsqrtf 是硬件指令，单 cycle 完成
5. Q: Fused RMSNorm+Residual 节省多少带宽？ A: 约 40-50%（减少一次完整的读写）
6. Q: hidden_size=4096 时推荐的 block_size？ A: 256 或 512
7. Q: sum_sq 为什么必须 FP32？ A: FP16 精度不够，大 hidden_size 会溢出
8. Q: RMSNorm weight 的初始值？ A: 全 1
9. Q: epsilon 的典型值？ A: 1e-6 或 1e-5
10. Q: Pre-norm 的含义？ A: 在 attention/FFN 之前做 normalization
（后续卡片略）