# Attention 机制

## 1. 学习目标

- 理解 Multi-Head Attention（MHA）的数学定义与计算流程
- 掌握 Grouped-Query Attention（GQA）和 Multi-Query Attention（MQA）的区别
- 理解 attention 的计算复杂度与内存瓶颈
- 能够分析 prefill 和 decode 阶段 attention 的不同特性
- 掌握 attention kernel 的基本 CUDA 实现思路

## 2. 前置知识

- 矩阵乘法（GEMM）
- Softmax
- RoPE
- Memory-bound vs compute-bound 判断

## 3. 核心术语表

| 术语 | 英文全称 | 定义 |
|------|----------|------|
| MHA | Multi-Head Attention | 多头注意力，每个头有独立的 Q/K/V |
| GQA | Grouped-Query Attention | 分组查询注意力，多个 Q 头共享一组 K/V |
| MQA | Multi-Query Attention | 多查询注意力，所有 Q 头共享一组 K/V |
| SDPA | Scaled Dot-Product Attention | 缩放点积注意力 |
| Causal Mask | Causal Mask | 因果掩码，防止看到未来 token |
| KV Cache | Key-Value Cache | 缓存历史 K/V 避免重复计算 |
| Prefill | Prefill Phase | 处理完整 prompt 的阶段 |
| Decode | Decode Phase | 逐 token 生成的阶段 |
| Head Dim | Head Dimension | 每个注意力头的维度（通常 128） |

## 4. 动机

### 4.1 Attention 的计算量

对于 seq_len=S, num_heads=H, head_dim=D, hidden_size=E=H×D：

**QKV Projection**：
```
Q = X × W_Q: [B,S,E] × [E,E] → 2BSE² FLOPs
K = X × W_K: [B,S,E] × [E,E_kv] → 2BSE×E_kv FLOPs  
V = X × W_V: [B,S,E] × [E,E_kv] → 2BSE×E_kv FLOPs
```

**Attention Score**：
```
Score = Q × K^T: [B,H,S,D] × [B,H,D,S] → 2BHS²D FLOPs
```

**Attention Output**：
```
Out = Softmax(Score) × V: [B,H,S,S] × [B,H,S,D] → 2BHS²D FLOPs
```

**Output Projection**：
```
O = Out × W_O: [B,S,E] × [E,E] → 2BSE² FLOPs
```

### 4.2 内存瓶颈

Attention score 矩阵大小：
```
[B, H, S, S] × sizeof(float16) = B × H × S² × 2 bytes

B=1, H=32, S=4096: 32 × 4096² × 2 = 1GB
B=1, H=32, S=128K: 32 × 128K² × 2 = 1TB → 不可能存储！
```

这就是 FlashAttention 的动机：不显式存储 S×S 矩阵。

### 4.3 GQA 的动机

MHA: num_kv_heads = num_heads → KV cache 大
MQA: num_kv_heads = 1 → KV cache 最小，但质量下降
GQA: num_kv_heads = num_heads / group_size → 平衡

LLaMA 2 70B: 64 Q heads, 8 KV heads (GQA-8)
LLaMA 3: 32 Q heads, 8 KV heads (GQA-4)

KV cache 节省：GQA-8 比 MHA 节省 8x KV cache 内存

## 5. 数学定义

### 5.1 Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V

其中：
Q ∈ ℝ^{S_q × d_k}  (query)
K ∈ ℝ^{S_kv × d_k} (key)
V ∈ ℝ^{S_kv × d_v} (value)
d_k = head_dim
```

### 5.2 Causal Attention

```
Score[i][j] = Q[i] · K[j] / √d_k    if j ≤ i
Score[i][j] = -∞                       if j > i (masked)
```

### 5.3 Multi-Head Attention

```
MultiHead(X) = Concat(head_1, ..., head_H) × W_O

head_i = Attention(X × W_Q^i, X × W_K^i, X × W_V^i)
```

### 5.4 GQA

```
// num_heads = 32, num_kv_heads = 8, group_size = 4
// Q heads 0-3 共享 KV head 0
// Q heads 4-7 共享 KV head 1
// ...

kv_head_idx = q_head_idx // group_size
```

## 6. 推导逻辑

### 6.1 Prefill vs Decode 的区别

**Prefill**（处理 prompt）：
```
Q: [B, S, H, D]    S = prompt_length (可能很长)
K: [B, S, H_kv, D]
V: [B, S, H_kv, D]
Score: [B, H, S, S]  → compute-bound (大矩阵乘法)
```

**Decode**（生成一个 token）：
```
Q: [B, 1, H, D]     只有 1 个新 token 的 query
K: [B, S_kv, H_kv, D]  所有历史 key（从 KV cache 读取）
V: [B, S_kv, H_kv, D]
Score: [B, H, 1, S_kv]  → memory-bound (读取整个 KV cache)
```

### 6.2 Decode Attention 的 Memory-Bound 分析

```
数据读取：
- Q: B × H × D × 2 bytes (很小)
- K cache: B × S_kv × H_kv × D × 2 bytes (大)
- V cache: B × S_kv × H_kv × D × 2 bytes (大)

计算量：
- Q×K^T: B × H × S_kv × D × 2 FLOPs
- Score×V: B × H × S_kv × D × 2 FLOPs

Arithmetic Intensity:
AI = 4BH×S_kv×D / (2B×S_kv×H_kv×D×2 × 2)
   ≈ H / (2×H_kv) = group_size / 2

GQA-4: AI ≈ 2 → 严重 memory-bound
```

### 6.3 优化方向

| 阶段 | 瓶颈 | 优化方向 |
|------|------|---------|
| Prefill | Compute | FlashAttention（减少 HBM 访问） |
| Decode | Memory | GQA（减少 KV cache）、量化 KV cache、PagedAttention |

## 7. 算子流程

### 7.1 标准 Attention（非 Flash）

```cuda
// Step 1: Score = Q × K^T / sqrt(d_k)
// [B,H,S_q,D] × [B,H,D,S_kv] → [B,H,S_q,S_kv]
cublasSgemmStridedBatched(...);

// Step 2: Apply causal mask
apply_causal_mask<<<...>>>(score, S_q, S_kv);

// Step 3: Softmax along last dim
softmax<<<...>>>(score, S_kv);

// Step 4: Output = Score × V
// [B,H,S_q,S_kv] × [B,H,S_kv,D] → [B,H,S_q,D]
cublasSgemmStridedBatched(...);
```

### 7.2 Decode Attention Kernel（单 query）

```cuda
__global__ void decode_attention_kernel(
    const half* __restrict__ q,       // [num_heads, head_dim]
    const half* __restrict__ k_cache, // [seq_len, num_kv_heads, head_dim]
    const half* __restrict__ v_cache, // [seq_len, num_kv_heads, head_dim]
    half* __restrict__ output,        // [num_heads, head_dim]
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float scale
) {
    // 每个 block 处理一个 head
    int head_idx = blockIdx.x;
    int kv_head_idx = head_idx / (num_heads / num_kv_heads);
    int tid = threadIdx.x;
    
    // Load q to shared memory
    extern __shared__ float smem[];
    float* s_q = smem;  // [head_dim]
    
    for (int i = tid; i < head_dim; i += blockDim.x) {
        s_q[i] = __half2float(q[head_idx * head_dim + i]);
    }
    __syncthreads();
    
    // Online softmax over seq_len
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float acc[HEAD_DIM_PER_THREAD] = {0};  // partial output accumulator
    
    // Each thread handles a chunk of seq positions
    for (int pos = tid; pos < seq_len; pos += blockDim.x) {
        // Compute dot(q, k[pos])
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            float k_val = __half2float(
                k_cache[(pos * num_kv_heads + kv_head_idx) * head_dim + d]);
            score += s_q[d] * k_val;
        }
        score *= scale;
        
        // Online softmax update
        float new_max = fmaxf(max_score, score);
        float exp_diff = expf(max_score - new_max);
        float exp_score = expf(score - new_max);
        
        // Rescale previous accumulator
        for (int d = 0; d < HEAD_DIM_PER_THREAD; d++) {
            acc[d] *= exp_diff;
        }
        sum_exp = sum_exp * exp_diff + exp_score;
        max_score = new_max;
        
        // Accumulate v[pos] * exp_score
        for (int d = 0; d < HEAD_DIM_PER_THREAD; d++) {
            float v_val = __half2float(
                v_cache[(pos * num_kv_heads + kv_head_idx) * head_dim + d]);
            acc[d] += v_val * exp_score;
        }
    }
    
    // Cross-thread reduction (online softmax merge)
    // ... (complex, involves merging (max, sum, acc) across threads)
    
    // Final: output = acc / sum_exp
    // ...
}
```

## 8. PyTorch baseline

```python
import torch
import torch.nn.functional as F

def attention_pytorch(q, k, v, causal=True):
    """
    q: [B, H, S_q, D]
    k: [B, H_kv, S_kv, D]
    v: [B, H_kv, S_kv, D]
    """
    # GQA: expand kv heads
    if k.shape[1] != q.shape[1]:
        group_size = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(group_size, dim=1)
        v = v.repeat_interleave(group_size, dim=1)
    
    # PyTorch 2.0+ SDPA (自动选择 FlashAttention/Memory-Efficient/Math)
    output = F.scaled_dot_product_attention(
        q, k, v, 
        is_causal=causal,
        scale=1.0 / (q.shape[-1] ** 0.5)
    )
    return output

# Benchmark
B, H, S, D = 1, 32, 4096, 128
H_kv = 8  # GQA
q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
k = torch.randn(B, H_kv, S, D, device='cuda', dtype=torch.float16)
v = torch.randn(B, H_kv, S, D, device='cuda', dtype=torch.float16)

out = attention_pytorch(q, k, v)
```

## 9. CUDA 实现思路

### 9.1 Naive Attention（教学用）

```cuda
// 分步实现，不适合生产
// Step 1: S = Q @ K^T * scale
// Step 2: S = causal_mask(S)
// Step 3: P = softmax(S, dim=-1)
// Step 4: O = P @ V
```

### 9.2 Fused Decode Attention（生产级）

vLLM 的 decode attention kernel 设计：
- 每个 thread block 处理一个 head 的一个 query
- 使用 online softmax 避免两次 pass
- 向量化加载 KV cache
- 支持 PagedAttention（非连续 KV 块）

## 10. Triton 实现思路

```python
@triton.jit
def attention_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    seq_len_q, seq_len_kv, head_dim,
    scale,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # 每个 program 处理一个 (batch, head, query_block)
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_q = tl.program_id(2)
    
    # Load Q block
    q_offs = pid_q * BLOCK_S + tl.arange(0, BLOCK_S)
    d_offs = tl.arange(0, BLOCK_D)
    
    q = tl.load(Q + pid_b * stride_qb + pid_h * stride_qh + 
                q_offs[:, None] * stride_qs + d_offs[None, :] * stride_qd,
                mask=(q_offs[:, None] < seq_len_q) & (d_offs[None, :] < head_dim))
    
    # Online softmax loop over K/V blocks
    m = tl.full([BLOCK_S], float('-inf'), dtype=tl.float32)
    l = tl.zeros([BLOCK_S], dtype=tl.float32)
    acc = tl.zeros([BLOCK_S, BLOCK_D], dtype=tl.float32)
    
    for start_kv in range(0, seq_len_kv, BLOCK_S):
        kv_offs = start_kv + tl.arange(0, BLOCK_S)
        
        # Load K block
        k = tl.load(K + pid_b * stride_kb + pid_h * stride_kh +
                    kv_offs[:, None] * stride_ks + d_offs[None, :] * stride_kd,
                    mask=(kv_offs[:, None] < seq_len_kv))
        
        # Score = Q @ K^T * scale
        s = tl.dot(q, tl.trans(k)) * scale  # [BLOCK_S, BLOCK_S]
        
        # Causal mask
        causal_mask = q_offs[:, None] >= kv_offs[None, :]
        s = tl.where(causal_mask, s, float('-inf'))
        
        # Online softmax update
        m_new = tl.maximum(m, tl.max(s, axis=1))
        alpha = tl.exp(m - m_new)
        p = tl.exp(s - m_new[:, None])
        l = l * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        
        # Load V and accumulate
        v = tl.load(V + pid_b * stride_vb + pid_h * stride_vh +
                    kv_offs[:, None] * stride_vs + d_offs[None, :] * stride_vd,
                    mask=(kv_offs[:, None] < seq_len_kv))
        acc += tl.dot(p.to(v.dtype), v)
        m = m_new
    
    # Final normalization
    acc = acc / l[:, None]
    
    # Store
    tl.store(Out + pid_b * stride_ob + pid_h * stride_oh +
             q_offs[:, None] * stride_os + d_offs[None, :] * stride_od,
             acc.to(Out.dtype.element_ty),
             mask=(q_offs[:, None] < seq_len_q))
```

## 11. Memory Access 分析

### Prefill Attention
```
Q: [B,H,S,D] → 读一次
K: [B,H,S,D] → 被 S 个 query 复用 → 读 S/BLOCK 次（tiled）
V: [B,H,S,D] → 同 K
Score: [B,H,S,S] → FlashAttention 不写到 HBM

FlashAttention HBM 访问: O(S²D / SRAM_size)
Naive HBM 访问: O(S²D + S²)
```

### Decode Attention
```
Q: [B,H,1,D] → 很小，常驻 register/shared
K cache: [B,H_kv,S_kv,D] → 必须全部读取
V cache: [B,H_kv,S_kv,D] → 必须全部读取

总读取: 2 × B × H_kv × S_kv × D × 2 bytes
B=1, H_kv=8, S_kv=4096, D=128: 2×8×4096×128×2 = 16MB
```

## 12. Compute-bound / Memory-bound 判断

| 场景 | AI (FLOPs/Byte) | 判断 |
|------|-----------------|------|
| Prefill S=4096 | ~S×D/(4D) = S/4 = 1024 | Compute-bound |
| Decode S_kv=4096, GQA-4 | ~4/2 = 2 | Memory-bound |
| Decode S_kv=4096, MHA | ~1 | 严重 Memory-bound |

## 13. Profiling 指标

- `sm__throughput.avg.pct_of_peak_sustained_elapsed`：SM 利用率
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`：HBM 带宽利用率
- `l2_throughput`：L2 cache 吞吐
- Tensor Core utilization（prefill 应该高）
- Warp stall reasons（decode 应该是 memory）

## 14. Benchmark 设计

```python
import torch
from torch.utils.benchmark import Timer

configs = [
    # (B, H, H_kv, S_q, S_kv, D, label)
    (1, 32, 8, 4096, 4096, 128, "prefill-4k"),
    (1, 32, 8, 1, 4096, 128, "decode-4k"),
    (1, 32, 8, 1, 32768, 128, "decode-32k"),
    (32, 32, 8, 1, 4096, 128, "decode-4k-batch32"),
]

for B, H, H_kv, S_q, S_kv, D, label in configs:
    q = torch.randn(B, H, S_q, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H_kv, S_kv, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H_kv, S_kv, D, device='cuda', dtype=torch.float16)
    
    t = Timer(
        stmt="F.scaled_dot_product_attention(q, k.expand(-1,H,-1,-1), v.expand(-1,H,-1,-1), is_causal=(S_q>1))",
        globals={"F": torch.nn.functional, "q": q, "k": k, "v": v, "H": H, "S_q": S_q}
    )
    print(f"{label}: {t.blocked_autorange().median * 1000:.3f} ms")
```

## 15. 常见错误

1. **忘记 scale**：不除以 √d_k → attention 值过大 → softmax 饱和
2. **GQA expand 错误**：K/V 的 head 维度 repeat 方式错误
3. **Causal mask 方向错误**：mask 了不该 mask 的位置
4. **KV cache 索引错误**：decode 时 position 计算错误
5. **精度问题**：FP16 attention score 溢出（需要 FP32 累加）

## 16-20. 实验任务、习题、答案、复习卡片

### 实验任务

1. 实现 naive attention（分步 GEMM + softmax）并与 `F.scaled_dot_product_attention` 对比
2. 测量 prefill vs decode 的 latency 随 seq_len 的变化
3. 对比 MHA vs GQA-4 vs GQA-8 的 decode latency 和 KV cache 大小
4. 使用 Nsight Compute 分析 decode attention 的瓶颈

### 习题（选 5 道）

1. 为什么 attention score 要除以 √d_k？如果不除会怎样？
2. GQA-8 相比 MHA 节省多少 KV cache 内存？decode latency 预期降低多少？
3. Decode attention 的 arithmetic intensity 是多少？在 A100 上理论最大吞吐是多少？
4. 为什么 FlashAttention 不适用于 decode phase？decode 应该用什么优化？
5. 如果 seq_len=128K，MHA 的 attention score 矩阵需要多少内存？为什么必须用 FlashAttention？
