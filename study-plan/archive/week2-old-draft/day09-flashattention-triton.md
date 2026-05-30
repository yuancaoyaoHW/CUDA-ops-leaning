# Day 9：FlashAttention Triton 实现

## 学习目标
- 用 Triton 实现 FlashAttention forward
- 通过正确性测试
- Benchmark 对比 PyTorch SDPA
- 理解 vLLM scheduler 概览

---

## 上午（3h）- 实现

### Triton FlashAttention Forward

```python
import torch
import triton
import triton.language as tl
import math

@triton.jit
def flash_attention_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qm, stride_qd,
    stride_km, stride_kd,
    stride_vm, stride_vd,
    stride_om, stride_od,
    N,  # sequence length
    D: tl.constexpr,  # head dimension
    BLOCK_M: tl.constexpr,  # Q block size
    BLOCK_N: tl.constexpr,  # K/V block size
    scale: tl.constexpr,
):
    # 每个 program 处理 Q 的一个 block
    pid_m = tl.program_id(0)

    # Q block 的行范围
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)

    # 加载 Q block [BLOCK_M, D]
    q_ptrs = Q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N, other=0.0)

    # 初始化 output accumulator 和 softmax stats
    o = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], value=-float('inf'), dtype=tl.float32)  # running max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # running sum

    # 遍历 K/V 的所有 blocks
    for start_n in range(0, N, BLOCK_N):
        cur_offs_n = start_n + offs_n

        # 加载 K block [BLOCK_N, D]
        k_ptrs = K_ptr + cur_offs_n[:, None] * stride_km + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=cur_offs_n[:, None] < N, other=0.0)

        # 加载 V block [BLOCK_N, D]
        v_ptrs = V_ptr + cur_offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=cur_offs_n[:, None] < N, other=0.0)

        # 计算 S = Q @ K^T * scale  → [BLOCK_M, BLOCK_N]
        s = tl.dot(q, tl.trans(k)) * scale

        # Causal mask (可选)
        # causal_mask = offs_m[:, None] >= cur_offs_n[None, :]
        # s = tl.where(causal_mask, s, -float('inf'))

        # Online softmax update
        m_ij = tl.max(s, axis=1)  # [BLOCK_M] 当前块的 row max
        m_new = tl.maximum(m_i, m_ij)  # [BLOCK_M] 新的 running max

        # Rescale 因子
        alpha = tl.exp(m_i - m_new)  # 旧的 rescale
        beta = tl.exp(m_ij - m_new)  # 新块的 scale

        # 更新 l (running sum)
        p = tl.exp(s - m_new[:, None])  # [BLOCK_M, BLOCK_N] 当前块的 exp
        l_new = l_i * alpha + tl.sum(p, axis=1)  # [BLOCK_M]

        # 更新 O
        # O_new = O_old * (l_old * alpha / l_new) + p @ V / l_new
        o = o * (l_i[:, None] * alpha[:, None] / l_new[:, None])
        o += tl.dot(p.to(tl.float16), v).to(tl.float32) / l_new[:, None]

        # 更新 stats
        m_i = m_new
        l_i = l_new

    # 写回 O
    o_ptrs = O_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, o.to(tl.float16), mask=offs_m[:, None] < N)


def flash_attention_triton(Q, K, V, causal=False):
    N, D = Q.shape
    O = torch.empty_like(Q)
    scale = 1.0 / math.sqrt(D)

    BLOCK_M = 64
    BLOCK_N = 64
    grid = (triton.cdiv(N, BLOCK_M),)

    flash_attention_fwd_kernel[grid](
        Q, K, V, O,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        V.stride(0), V.stride(1),
        O.stride(0), O.stride(1),
        N, D, BLOCK_M, BLOCK_N, scale,
    )
    return O
```

### 正确性测试

```python
def test_flash_attention():
    torch.manual_seed(42)
    for N in [128, 256, 512, 1024]:
        for D in [64, 128]:
            Q = torch.randn(N, D, device='cuda', dtype=torch.float16)
            K = torch.randn(N, D, device='cuda', dtype=torch.float16)
            V = torch.randn(N, D, device='cuda', dtype=torch.float16)

            # Reference: standard attention
            scale = 1.0 / math.sqrt(D)
            S = (Q.float() @ K.float().T) * scale
            P = torch.softmax(S, dim=-1)
            ref = (P @ V.float()).half()

            # Our implementation
            out = flash_attention_triton(Q, K, V)

            # 允许一定误差（fp16 累积误差）
            max_diff = (out - ref).abs().max().item()
            assert max_diff < 0.05, f"N={N}, D={D}: max_diff={max_diff}"
            print(f"N={N}, D={D}: PASS (max_diff={max_diff:.4f})")
```

---

## 下午（2h）- Benchmark

### 性能对比

```python
import triton

for N in [512, 1024, 2048, 4096]:
    D = 128
    Q = torch.randn(N, D, device='cuda', dtype=torch.float16)
    K = torch.randn(N, D, device='cuda', dtype=torch.float16)
    V = torch.randn(N, D, device='cuda', dtype=torch.float16)

    # PyTorch SDPA (uses flash attention internally)
    Q_b = Q.unsqueeze(0).unsqueeze(0)  # [1, 1, N, D]
    K_b = K.unsqueeze(0).unsqueeze(0)
    V_b = V.unsqueeze(0).unsqueeze(0)

    t_sdpa = triton.testing.do_bench(
        lambda: torch.nn.functional.scaled_dot_product_attention(Q_b, K_b, V_b)
    )
    t_ours = triton.testing.do_bench(lambda: flash_attention_triton(Q, K, V))
    t_naive = triton.testing.do_bench(
        lambda: torch.softmax(Q.float() @ K.float().T / math.sqrt(D), -1) @ V.float()
    )

    flops = 2 * N * N * D * 2  # QK^T + PV
    print(f"N={N}: naive={t_naive:.2f}ms, ours={t_ours:.2f}ms, "
          f"SDPA={t_sdpa:.2f}ms, speedup_vs_naive={t_naive/t_ours:.1f}x")
```

### 为什么我们的实现比 flash-attn 库慢

```
1. 没有 double buffering: 加载 K/V 和计算没有 overlap
2. 没有 warp specialization: 所有 warp 做相同的事
3. 没有 vectorized load: 没有用 float4/int4 加载
4. 没有 causal mask 优化: 可以跳过全 mask 的块
5. 单 head: 没有利用 multi-head 的并行性
6. 没有 backward: 只实现了 forward

但这不重要！面试目标是能讲清楚算法，不是写出最快的实现。
```

---

## 晚上（1.5h）- vLLM Scheduler 概览

### 架构总览

```
vLLM 的核心组件:

LLMEngine
  ├── Tokenizer: 文本 → token ids
  ├── Scheduler: 决定哪些 request 进入当前 batch
  │   ├── waiting queue: 等待 prefill 的 request
  │   ├── running queue: 正在 decode 的 request
  │   └── swapped queue: KV cache 被换出到 CPU 的 request
  ├── BlockManager: 管理 KV cache 的物理块分配
  │   ├── GPU block allocator
  │   └── CPU block allocator (for swap)
  └── Worker: 执行实际的模型 forward
      ├── ModelRunner: 准备输入、调用模型
      └── CacheEngine: 管理 KV cache tensor
```

### Scheduler 决策流程（每个 step）

```
1. 尝试 swap in (swapped → running):
   - 检查是否有足够的 GPU blocks
   - 如果有，把 CPU 上的 KV cache 拷回 GPU

2. 尝试 prefill (waiting → running):
   - 从 waiting 队列取 request
   - 检查是否有足够的 GPU blocks 存储新 request 的 KV cache
   - 如果有，分配 blocks，加入 running

3. 如果 GPU blocks 不够:
   - Preempt: 从 running 中选择 request 驱逐
   - 策略 1 (swap): 把 KV cache 拷贝到 CPU → swapped 队列
   - 策略 2 (recompute): 丢弃 KV cache → waiting 队列（下次重新 prefill）
   - 选择哪个 request 驱逐: FCFS (先来的优先保留) 或 priority-based

4. 构建当前 step 的 batch:
   - 所有 running 中的 request 组成一个 batch
   - Prefill request: 处理所有 input tokens
   - Decode request: 只处理 1 个新 token
```

### 关键设计决策

```
Q: 为什么 prefill 和 decode 放在同一个 batch？
A: Continuous batching 的核心。如果分开，decode 的 GPU 利用率很低。
   混合 batch 让 prefill 的 compute 和 decode 的 memory access 可以 overlap。
   但长 prefill 会阻塞 decode → chunked prefill 解决这个问题。

Q: swap vs recompute 怎么选？
A: swap 适合 KV cache 大的情况（避免重新计算）
   recompute 适合 prompt 短的情况（重新 prefill 很快）
   vLLM 默认用 swap
```

---

## 日检（20 分钟）

1. **闭卷手写**（10min）：写出 FlashAttention 的核心循环伪代码（遍历 K/V blocks，online softmax update O）
2. **口述**（5min）：你的 Triton 实现比 flash-attn 库慢，原因是什么？（至少说 3 个）
3. **口述**（5min）：vLLM scheduler 的三个队列是什么？preempt 有哪两种策略？

---

## 参考资料

- Triton tutorial: Flash Attention
- vLLM source: vllm/core/scheduler.py
- Aleksa Gordić, "Inside vLLM: Anatomy of a High-Throughput LLM Inference System"
