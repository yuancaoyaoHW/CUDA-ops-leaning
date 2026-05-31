# RoPE（Rotary Position Embedding）

## 1. 学习目标

- 理解 RoPE 的数学原理与旋转矩阵推导
- 掌握 RoPE 的高效 CUDA 实现（避免显式构造旋转矩阵）
- 理解 RoPE 的相对位置编码特性
- 能够实现支持动态序列长度的 RoPE kernel
- 掌握 RoPE 在长上下文扩展中的变体（NTK-aware、YaRN）

## 2. 前置知识

- 复数运算基础
- 三角函数（sin、cos）
- Attention 机制中 Q、K 的作用
- 位置编码的动机

## 3. 核心术语表

| 术语 | 英文全称 | 定义 |
|------|----------|------|
| RoPE | Rotary Position Embedding | 通过旋转实现的位置编码 |
| Position Embedding | Position Embedding | 为 token 注入位置信息的方法 |
| Absolute PE | Absolute Position Embedding | 直接加到 embedding 上的位置编码 |
| Relative PE | Relative Position Embedding | 编码 token 间相对距离的位置编码 |
| Rotation Matrix | Rotation Matrix | 2D 旋转矩阵 |
| Frequency | Frequency | RoPE 中每个维度对的旋转频率 |
| Base | Base | RoPE 频率计算的基数（通常 10000） |
| NTK-aware | NTK-aware Scaling | 通过修改 base 扩展上下文长度 |
| YaRN | Yet another RoPE extensioN | 结合 NTK 和注意力缩放的扩展方法 |

## 4. 动机

### 4.1 为什么需要 RoPE？

传统位置编码的问题：
- Absolute PE（如 sinusoidal）：加到 embedding 上，attention score 中位置信息不够显式
- Learned PE：无法外推到训练时未见过的长度
- Relative PE（如 ALiBi）：需要修改 attention 计算

RoPE 的优势：
- 通过旋转 Q 和 K，使 attention score 自然包含相对位置信息
- 不增加额外参数
- 支持长度外推（配合 scaling）
- 计算高效（逐元素操作）

### 4.2 RoPE 在 LLM 中的使用

几乎所有主流开源 LLM 都使用 RoPE：
- LLaMA / LLaMA 2 / LLaMA 3
- Mistral / Mixtral
- Qwen / Qwen2
- DeepSeek
- Gemma

## 5. 数学定义

### 5.1 核心思想

对于位置 m 的 query 向量 q 和位置 n 的 key 向量 k：
```
RoPE(q, m) · RoPE(k, n) = f(q, k, m-n)
```
即旋转后的内积只依赖于相对位置 m-n。

### 5.2 2D 旋转

将 d 维向量的每两个相邻维度视为一个 2D 平面，应用旋转：
```
[q_{2i}  ]     [cos(mθ_i)  -sin(mθ_i)] [q_{2i}  ]
[q_{2i+1}]  =  [sin(mθ_i)   cos(mθ_i)] [q_{2i+1}]
```

其中频率：
```
θ_i = base^(-2i/d) = 1 / base^(2i/d)

base = 10000（默认）
i = 0, 1, ..., d/2 - 1
d = head_dim（通常 128）
```

### 5.3 等价的复数形式

将 (q_{2i}, q_{2i+1}) 视为复数 q_{2i} + j·q_{2i+1}：
```
RoPE(q, m)_i = q_i × e^(j·m·θ_i)
             = q_i × (cos(mθ_i) + j·sin(mθ_i))
```

### 5.4 高效计算形式（避免矩阵乘法）

```
RoPE(q, m)_{2i}   = q_{2i} × cos(mθ_i) - q_{2i+1} × sin(mθ_i)
RoPE(q, m)_{2i+1} = q_{2i} × sin(mθ_i) + q_{2i+1} × cos(mθ_i)
```

只需要逐元素乘法和加法！

### 5.5 验证相对位置特性

```
<RoPE(q,m), RoPE(k,n)> = Σ_i Re(q_i* × k_i × e^(j(m-n)θ_i))
```
只依赖 m-n ✓

## 6. 推导逻辑

### 6.1 频率设计

```
θ_i = 10000^(-2i/d)

i=0:  θ_0 = 1.0          → 高频（相邻 token 旋转角度大）
i=63: θ_63 = 10000^(-1) = 0.0001  → 低频（远距离 token 才有明显旋转）
```

这种设计使得：
- 低维度对：编码短距离关系
- 高维度对：编码长距离关系
- 类似 sinusoidal PE 的多尺度特性

### 6.2 长上下文扩展

**NTK-aware Scaling**：
```
base_new = base × α^(d/(d-2))
// α = target_length / train_length
```
效果：拉伸低频部分，保持高频部分

**YaRN**：
```
// 分段处理不同频率
低频维度：不缩放
高频维度：线性插值
中间维度：NTK 插值
+ attention 缩放因子
```

### 6.3 Kernel 设计

RoPE 的计算特点：
- 每个 (position, head, dim_pair) 独立 → 完美并行
- 计算量小（2 次乘法 + 1 次加法 per element）
- Memory-bound：主要开销在读写 Q/K
- 可以与 QKV projection 融合

## 7. 算子流程

### 7.1 预计算 cos/sin 表

```cuda
// 预计算 [max_seq_len, head_dim/2] 的 cos/sin 表
__global__ void precompute_freqs(
    float* cos_table,  // [max_seq_len, head_dim/2]
    float* sin_table,  // [max_seq_len, head_dim/2]
    int max_seq_len,
    int head_dim,
    float base
) {
    int pos = blockIdx.x;
    int i = threadIdx.x;  // dim pair index
    
    if (pos < max_seq_len && i < head_dim / 2) {
        float theta = powf(base, -2.0f * i / head_dim);
        float angle = pos * theta;
        cos_table[pos * (head_dim/2) + i] = cosf(angle);
        sin_table[pos * (head_dim/2) + i] = sinf(angle);
    }
}
```

### 7.2 Apply RoPE Kernel

```cuda
__global__ void apply_rope_kernel(
    float* __restrict__ q,        // [batch, seq_len, num_heads, head_dim]
    float* __restrict__ k,        // [batch, seq_len, num_kv_heads, head_dim]
    const float* __restrict__ cos_table,  // [max_seq_len, head_dim/2]
    const float* __restrict__ sin_table,  // [max_seq_len, head_dim/2]
    const int* __restrict__ positions,    // [batch, seq_len] or NULL
    int batch_size,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim
) {
    // 每个 thread 处理一个 dim pair
    int batch = blockIdx.z;
    int seq_pos = blockIdx.y;
    int head = blockIdx.x;
    int pair_idx = threadIdx.x;  // 0 to head_dim/2 - 1
    
    if (pair_idx >= head_dim / 2) return;
    
    // 获取位置
    int pos = (positions != NULL) ? positions[batch * seq_len + seq_pos] : seq_pos;
    
    // 获取 cos/sin
    float cos_val = cos_table[pos * (head_dim/2) + pair_idx];
    float sin_val = sin_table[pos * (head_dim/2) + pair_idx];
    
    // Apply to Q
    if (head < num_heads) {
        int base_idx = ((batch * seq_len + seq_pos) * num_heads + head) * head_dim;
        float q0 = q[base_idx + 2 * pair_idx];
        float q1 = q[base_idx + 2 * pair_idx + 1];
        q[base_idx + 2 * pair_idx]     = q0 * cos_val - q1 * sin_val;
        q[base_idx + 2 * pair_idx + 1] = q0 * sin_val + q1 * cos_val;
    }
    
    // Apply to K (only for kv_heads)
    if (head < num_kv_heads) {
        int base_idx = ((batch * seq_len + seq_pos) * num_kv_heads + head) * head_dim;
        float k0 = k[base_idx + 2 * pair_idx];
        float k1 = k[base_idx + 2 * pair_idx + 1];
        k[base_idx + 2 * pair_idx]     = k0 * cos_val - k1 * sin_val;
        k[base_idx + 2 * pair_idx + 1] = k0 * sin_val + k1 * cos_val;
    }
}
```

## 8. PyTorch baseline

```python
import torch

def precompute_freqs_cis(dim: int, max_seq_len: int, base: float = 10000.0):
    """预计算 RoPE 的 cos/sin 值"""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)  # [max_seq_len, dim/2]
    cos = freqs.cos()  # [max_seq_len, dim/2]
    sin = freqs.sin()  # [max_seq_len, dim/2]
    return cos, sin

def apply_rope(q: torch.Tensor, k: torch.Tensor, 
               cos: torch.Tensor, sin: torch.Tensor,
               positions: torch.Tensor = None):
    """
    q: [batch, seq_len, num_heads, head_dim]
    k: [batch, seq_len, num_kv_heads, head_dim]
    cos, sin: [max_seq_len, head_dim/2]
    """
    if positions is None:
        seq_len = q.shape[1]
        cos = cos[:seq_len]  # [seq_len, head_dim/2]
        sin = sin[:seq_len]
    else:
        cos = cos[positions]  # [batch, seq_len, head_dim/2]
        sin = sin[positions]
    
    # Reshape for broadcasting
    cos = cos.unsqueeze(-2)  # [..., 1, head_dim/2]
    sin = sin.unsqueeze(-2)
    
    # Split into pairs
    q_even = q[..., 0::2]  # [..., head_dim/2]
    q_odd = q[..., 1::2]
    k_even = k[..., 0::2]
    k_odd = k[..., 1::2]
    
    # Apply rotation
    q_rot_even = q_even * cos - q_odd * sin
    q_rot_odd = q_even * sin + q_odd * cos
    k_rot_even = k_even * cos - k_odd * sin
    k_rot_odd = k_even * sin + k_odd * cos
    
    # Interleave back
    q_rot = torch.stack([q_rot_even, q_rot_odd], dim=-1).flatten(-2)
    k_rot = torch.stack([k_rot_even, k_rot_odd], dim=-1).flatten(-2)
    
    return q_rot, k_rot

# Usage
head_dim = 128
max_seq_len = 8192
cos, sin = precompute_freqs_cis(head_dim, max_seq_len)
cos, sin = cos.cuda(), sin.cuda()

B, S, H, D = 4, 2048, 32, 128
q = torch.randn(B, S, H, D, device='cuda')
k = torch.randn(B, S, 8, D, device='cuda')  # GQA: 8 kv heads
q_rot, k_rot = apply_rope(q, k, cos, sin)
```

## 9. CUDA 实现思路

### 9.1 Fused RoPE（与 QKV split 融合）

```cuda
// 在 QKV projection 输出后直接 apply RoPE
// 避免额外的 kernel launch 和 memory read/write
__global__ void fused_qkv_rope_kernel(
    const half* __restrict__ qkv,     // [B, S, (H_q + 2*H_kv) * D]
    half* __restrict__ q_out,          // [B, S, H_q, D]
    half* __restrict__ k_out,          // [B, S, H_kv, D]
    half* __restrict__ v_out,          // [B, S, H_kv, D]
    const float* __restrict__ cos_sin, // [max_seq, D] interleaved cos,sin
    int B, int S, int H_q, int H_kv, int D
) {
    // ... split QKV and apply RoPE to Q,K in one pass
}
```

### 9.2 Half2 向量化 RoPE

```cuda
__global__ void apply_rope_half2(
    half2* __restrict__ q,  // treat as half2 for vectorized access
    const half2* __restrict__ cos_sin,  // interleaved [cos, sin] pairs
    int total_pairs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pairs) return;
    
    half2 q_pair = q[idx];
    half2 cs = cos_sin[idx];  // .x = cos, .y = sin
    
    float q0 = __half2float(q_pair.x);
    float q1 = __half2float(q_pair.y);
    float c = __half2float(cs.x);
    float s = __half2float(cs.y);
    
    half2 result;
    result.x = __float2half(q0 * c - q1 * s);
    result.y = __float2half(q0 * s + q1 * c);
    q[idx] = result;
}
```

## 10. Triton 实现思路

```python
@triton.jit
def rope_kernel(
    Q, COS, SIN, OUT,
    stride_qb, stride_qs, stride_qh, stride_qd,
    seq_len, num_heads, head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_seq = tl.program_id(1)
    pid_head = tl.program_id(2)
    
    # Load cos/sin for this position
    half_dim = head_dim // 2
    dim_offsets = tl.arange(0, BLOCK_SIZE)
    mask = dim_offsets < half_dim
    
    cos_vals = tl.load(COS + pid_seq * half_dim + dim_offsets, mask=mask)
    sin_vals = tl.load(SIN + pid_seq * half_dim + dim_offsets, mask=mask)
    
    # Load q pairs
    base = pid_batch * stride_qb + pid_seq * stride_qs + pid_head * stride_qh
    q_even = tl.load(Q + base + 2 * dim_offsets, mask=mask)
    q_odd = tl.load(Q + base + 2 * dim_offsets + 1, mask=mask)
    
    # Rotate
    out_even = q_even * cos_vals - q_odd * sin_vals
    out_odd = q_even * sin_vals + q_odd * cos_vals
    
    # Store
    tl.store(OUT + base + 2 * dim_offsets, out_even, mask=mask)
    tl.store(OUT + base + 2 * dim_offsets + 1, out_odd, mask=mask)
```

## 11. Memory Access 分析

### 11.1 数据量

```
Q: B × S × H × D × sizeof(dtype)
K: B × S × H_kv × D × sizeof(dtype)
cos/sin: S × D/2 × sizeof(float) (预计算，可复用)

典型值 (B=4, S=2048, H=32, H_kv=8, D=128, FP16):
Q: 4 × 2048 × 32 × 128 × 2 = 64 MB
K: 4 × 2048 × 8 × 128 × 2 = 16 MB
cos/sin: 2048 × 64 × 4 = 0.5 MB (negligible)
```

### 11.2 访存模式

- Q/K 的读写是连续的（coalesced）
- cos/sin 表被所有 head 共享 → L2 cache 命中率高
- 完全 memory-bound：每个元素只做 2 次乘法 + 1 次加法

### 11.3 优化方向

1. 与 QKV projection 融合（省一次读写）
2. 向量化加载（half2/float4）
3. 预计算 cos/sin 避免重复 sinf/cosf 调用

## 12. Parallelism 分析

- 完美并行：每个 (batch, seq, head, dim_pair) 独立
- Grid: (num_heads, seq_len, batch_size)
- Block: (head_dim / 2) threads
- 无 reduction，无同步，无 shared memory 需求

## 13. Compute-bound / Memory-bound 判断

```
FLOPs per element pair: 4 (2 mul + 1 add for each of 2 outputs... actually 6: 4 mul + 2 add)
Bytes per element pair: 2 × 2 (read) + 2 × 2 (write) + 2 × 4 (cos/sin) = 16 bytes (FP16 Q/K)

AI = 6 / 16 = 0.375 FLOPs/Byte

A100 roofline 拐点: 19.5 TFLOPS / 2 TB/s ≈ 10 FLOPs/Byte
0.375 << 10 → 严重 memory-bound
```

## 14. Profiling 指标

| 指标 | 期望值 | 异常信号 |
|------|--------|----------|
| Memory Throughput | >80% peak | 低 → coalescing 问题 |
| Compute Utilization | <10% | 正常（memory-bound） |
| L2 Hit Rate (cos/sin) | >90% | 低 → 表太大或访问模式差 |
| Achieved Occupancy | >50% | 低 → block 配置问题 |

## 15. Benchmark 设计

```python
import torch
import time

configs = [
    # (batch, seq_len, num_heads, head_dim)
    (1, 1, 32, 128),       # decode single token
    (32, 1, 32, 128),      # decode batch=32
    (1, 4096, 32, 128),    # prefill long
    (4, 2048, 32, 128),    # prefill batch
    (1, 131072, 32, 128),  # very long context
]

for B, S, H, D in configs:
    q = torch.randn(B, S, H, D, device='cuda', dtype=torch.float16)
    cos = torch.randn(S, D//2, device='cuda')
    sin = torch.randn(S, D//2, device='cuda')
    
    # Warmup
    for _ in range(10):
        apply_rope_inplace(q, cos, sin)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(100):
        apply_rope_inplace(q, cos, sin)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / 100
    
    bytes_moved = B * S * H * D * 2 * 2  # read + write, FP16
    bw = bytes_moved / elapsed / 1e9
    print(f"B={B}, S={S}: {elapsed*1e6:.1f} μs, BW={bw:.0f} GB/s")
```

## 16. 常见错误

| 错误 | 现象 | 原因 | 修复 |
|------|------|------|------|
| 维度配对错误 | 输出数值错误 | 将 (0,1), (2,3) 配对而非 (0,64), (1,65) | 确认模型使用的 interleave 方式 |
| 频率计算错误 | 长序列性能下降 | base 或指数计算有误 | 对照 HuggingFace 实现 |
| 位置偏移 | KV cache 场景结果错误 | decode 时位置应为当前 step 而非 0 | 传入正确的 position_ids |
| 精度丢失 | 与 FP32 baseline 差异大 | sin/cos 用 FP16 计算 | cos/sin 表用 FP32 |
| GQA 处理错误 | K 的 RoPE 应用到错误 head | num_kv_heads ≠ num_heads | 分别处理 Q 和 K |

## 17. 实验任务

1. 实现 PyTorch 版 RoPE 并验证 `<q_rot, k_rot>` 只依赖相对位置
2. 实现 CUDA RoPE kernel，对比 PyTorch 版本的正确性和速度
3. 实现 Triton RoPE kernel
4. 测试不同 head_dim (64, 128, 256) 下的带宽利用率
5. 实现 NTK-aware scaling，验证长序列外推效果
6. 将 RoPE 与 QKV projection 融合，测量 kernel launch 节省

## 18. 习题 20 道

1. RoPE 的数学本质是什么？为什么旋转能编码相对位置？
2. 写出 head_dim=4 时，位置 m 的完整旋转矩阵。
3. 为什么 RoPE 的频率设计为 θ_i = base^(-2i/d)？如果所有频率相同会怎样？
4. RoPE 与 sinusoidal position embedding 的联系和区别？
5. 在 GQA 中，RoPE 应该应用到哪些张量？为什么 V 不需要 RoPE？
6. 如果 head_dim=128，base=10000，计算 θ_0 和 θ_63 的值。
7. 为什么 RoPE kernel 是 memory-bound？计算其 arithmetic intensity。
8. 如何将 RoPE 与 QKV projection 融合？画出数据流。
9. NTK-aware scaling 的原理是什么？为什么能扩展上下文长度？
10. 在 KV cache 场景中，decode 阶段的 RoPE position 应该是什么？
11. 比较 RoPE 的两种实现方式：interleaved (0,1,2,3→0,2,1,3) vs sequential (0,1,2,3→0,1,2,3)。
12. 如何验证 RoPE 实现的正确性？设计测试用例。
13. RoPE 的 cos/sin 表应该用什么精度存储？为什么？
14. 如果要支持 position_ids 不连续（如 prefix caching），kernel 需要怎么修改？
15. 计算 B=4, S=4096, H=32, D=128 时 RoPE 的理论最小执行时间（A100）。
16. YaRN 相比 NTK-aware 的改进是什么？
17. 为什么 RoPE 不需要 shared memory？
18. 如何用 Nsight Compute 验证 RoPE kernel 的带宽利用率？
19. RoPE 在 Triton 中实现时，BLOCK_SIZE 应该设为多少？为什么？
20. 如果模型使用 head_dim=256（如某些 MoE 模型），RoPE 的实现需要注意什么？

## 19. 标准答案

1. RoPE 将向量的每两个维度视为复平面上的点，通过旋转角度 mθ 编码位置 m。两个旋转后向量的内积 = 原始内积 × 旋转差角的函数，因此只依赖相对位置 m-n。

2. head_dim=4, 位置 m:
```
[cos(mθ₀) -sin(mθ₀)    0         0    ] [q₀]
[sin(mθ₀)  cos(mθ₀)    0         0    ] [q₁]
[   0         0      cos(mθ₁) -sin(mθ₁)] [q₂]
[   0         0      sin(mθ₁)  cos(mθ₁)] [q₃]
```

3. 指数递减的频率使不同维度对编码不同尺度的位置关系。低维度（高频）区分相邻 token，高维度（低频）区分远距离 token。如果所有频率相同，所有维度对提供相同信息，表达能力大幅下降。

4. 联系：都使用 sin/cos 函数，都有多尺度特性。区别：sinusoidal PE 是加到 embedding 上的绝对编码；RoPE 是乘到 Q/K 上的旋转操作，天然编码相对位置。

5. RoPE 应用到 Q 和 K。V 不需要因为 attention 的输出是 softmax(QK^T)V，位置信息已经通过 QK^T 的 score 体现，V 只是被加权求和。

(后续答案略，完整版见实际教学材料)

## 20. 复习卡片 30 张

1. Q: RoPE 的全称？ A: Rotary Position Embedding
2. Q: RoPE 的频率公式？ A: θ_i = base^(-2i/d), base=10000
3. Q: RoPE 对一个 dim pair 的操作？ A: [x₀,x₁] → [x₀cos-x₁sin, x₀sin+x₁cos]
4. Q: RoPE 为什么能编码相对位置？ A: 旋转后内积只依赖角度差 (m-n)θ
5. Q: RoPE 是 compute-bound 还是 memory-bound？ A: Memory-bound (AI≈0.375)
6. Q: RoPE 需要 shared memory 吗？ A: 不需要，每个元素独立计算
7. Q: V 需要 apply RoPE 吗？ A: 不需要
8. Q: NTK-aware scaling 修改什么？ A: 修改 base 值来扩展上下文
9. Q: cos/sin 表应该用什么精度？ A: FP32（避免精度丢失）
10. Q: decode 时 RoPE 的 position 是什么？ A: 当前 token 的绝对位置（step number）
(后续卡片略)