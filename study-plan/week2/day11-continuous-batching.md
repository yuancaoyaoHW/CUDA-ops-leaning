# Day 11：Continuous Batching + Chunked Prefill + Decode Attention

## 学习目标
- 理解 continuous batching vs static batching 的本质区别
- 理解 chunked prefill 解决的问题
- 理解 decode attention 为什么是 memory-bound
- 理解 Context Parallelism

---

## 上午（3h）- Continuous Batching

### Static Batching 的问题

```
Static Batching (传统方式):
  1. 收集 B 个 request 组成一个 batch
  2. 所有 request 一起 prefill
  3. 所有 request 一起 decode，直到最长的那个完成
  4. 返回所有结果，开始下一个 batch

问题：
  Request 1: 生成 10 tokens → 第 10 步就完成了
  Request 2: 生成 500 tokens → 第 500 步才完成
  → Request 1 完成后，它的 GPU 资源空闲了 490 步！
  → GPU 利用率 = 有效计算 / 总计算 ≈ 30-50%

时间线:
  |---Prefill---|---Decode (all B requests)---...---Decode---|---Return---|
                 ↑ Request 1 done here, but GPU still busy
```

### Continuous Batching (Orca 论文)

```
核心 idea: Iteration-level scheduling
  每个 decode step 后检查:
  - 哪些 request 完成了？→ 移出 batch
  - 有新 request 等待吗？→ 加入 batch

时间线:
  Step 1: [R1, R2, R3, R4] decode
  Step 2: [R1, R2, R3, R4] decode  ← R1 完成，R5 加入
  Step 3: [R5, R2, R3, R4] decode  ← R5 做 prefill，其他继续 decode
  Step 4: [R5, R2, R3, R4] decode  ← R3 完成，R6 加入
  ...

好处:
  - GPU 始终满载（有 request 完成就立即替换）
  - 利用率从 30-50% 提升到 80-90%
  - 吞吐量提升 2-3x

实现关键:
  - Prefill 和 decode 可以在同一个 batch 中混合
  - 需要 padding 或 ragged tensor 处理不同长度
  - vLLM 的做法：prefill 和 decode 分开执行（不混合）
```

### vLLM 的 Batching 策略

```
vLLM 的选择：prefill 和 decode 分开

每个 step:
  如果有 waiting request 且有足够 blocks:
    → 做 prefill（一个或多个 request 的 prefill）
  否则:
    → 做 decode（所有 running request 一起 decode）

为什么分开？
  - Prefill: compute-bound (大矩阵乘)，用 FlashAttention
  - Decode: memory-bound (GEMV)，用 PagedAttention kernel
  - 两者的 kernel 不同，混合执行效率低
  - 分开可以各自用最优的 kernel
```

### Chunked Prefill

```
问题：长 prompt 的 prefill 阻塞 decode

例：prompt = 8192 tokens
  Prefill 时间 ≈ 200ms（8192 tokens 的 forward）
  这 200ms 内所有 decode request 都在等待
  → decode 的 TTFT (Time To First Token) 飙升

解决：把长 prefill 切成小块

Chunked Prefill:
  chunk_size = 512 tokens
  8192 tokens → 16 个 chunk
  每个 step:
    - 做 1 个 prefill chunk (512 tokens)
    - 做所有 running request 的 decode
  → prefill 不再阻塞 decode
  → TTFT P99 大幅降低

代价:
  - Prefill 总时间变长（每个 chunk 有 kernel launch overhead）
  - 实现更复杂（需要管理 partial prefill 状态）
  - 但对 SLA 敏感的场景非常值得
```

---

## 下午（2h）- Decode Attention 的特殊性

### Prefill vs Decode 的本质区别

```
Prefill Attention:
  Q: [seq_len, num_heads, head_dim]  → seq_len 个 query
  K: [seq_len, num_heads, head_dim]  → seq_len 个 key
  计算: Q @ K^T → [seq_len, seq_len] 矩阵乘
  → compute-bound（大矩阵乘，AI 很高）
  → 用 FlashAttention 优化

Decode Attention:
  Q: [1, num_heads, head_dim]  → 只有 1 个新 query！
  K: [seq_len, num_heads, head_dim]  → 整个 KV cache
  计算: Q @ K^T → [1, seq_len] → 本质是 GEMV
  → memory-bound（需要读取整个 KV cache，但只算 1 个 dot product）

Roofline 分析:
  Decode attention for one head:
    读: K cache = seq_len * head_dim * 2 bytes (fp16)
        V cache = seq_len * head_dim * 2 bytes
        Q = head_dim * 2 bytes
    计算: seq_len * head_dim * 2 FLOPs (Q@K^T) + seq_len * head_dim * 2 (P@V)
    AI = 4 * seq_len * head_dim / (4 * seq_len * head_dim + head_dim * 2)
       ≈ 1 FLOP/byte (当 seq_len >> 1)
    → 严重 memory-bound!
```

### Batching 为什么对 Decode 重要

```
单个 request decode:
  读 KV cache: seq_len * head_dim * 2 * 2 bytes
  计算: seq_len * head_dim * 4 FLOPs
  AI ≈ 1 → memory-bound，GPU 计算单元大量空闲

Batch of B requests decode:
  读 KV cache: B * seq_len * head_dim * 2 * 2 bytes（每个 request 的 KV 不同）
  计算: B * seq_len * head_dim * 4 FLOPs
  AI 仍然 ≈ 1（每个 request 的 KV 独立，无法复用）

  但！模型权重可以复用：
  MLP/Attention 的 weight matrix 被 B 个 request 共享
  Weight 读一次，用 B 次
  → MLP 部分的 AI = B * 2NK / (2NK + 2BN) ≈ B（当 K >> B）
  → Batch 越大，MLP 越接近 compute-bound

结论：
  - Attention 部分：batching 不改变 AI（每个 request 的 KV 独立）
  - MLP 部分：batching 线性提高 AI
  - 总体：batching 提高 GPU 利用率，但 attention 仍是瓶颈
```

### FlashDecoding

```
问题：decode attention 的并行度太低
  标准实现：每个 head 一个 thread block
  如果 batch=1, num_heads=32 → 只有 32 个 blocks → GPU 利用率低

FlashDecoding 解决方案：
  把 KV cache 的 seq_len 维度也并行化
  每个 head 用多个 thread blocks，每个 block 处理一段 KV

  Block 0: 处理 KV[0:1024]，得到 partial_O_0, partial_m_0, partial_l_0
  Block 1: 处理 KV[1024:2048]，得到 partial_O_1, partial_m_1, partial_l_1
  ...
  最终: reduce 所有 partial 结果（用 online softmax rescaling）

  并行度: num_heads * ceil(seq_len / chunk_size)
  → 即使 batch=1，也能充分利用 GPU

代价：
  - 需要额外的 reduce kernel
  - 多了一次 global memory 写+读（partial results）
  - 但对长序列（seq_len > 4096）效果显著
```

---

## 晚上（1.5h）- Context Parallelism

### 动机

```
长序列训练（8K-128K tokens）的问题：
  Activation 内存 = batch * seq_len * hidden * num_layers * sizeof
  seq_len = 128K, hidden = 4096, 32 layers, fp16:
    = 1 * 128K * 4096 * 32 * 2 = 32 GB（单层 activation 就 1GB）
  → 单卡放不下

解决：把 sequence 切分到多个 GPU
```

### Ring Attention

```
N 个 GPU，每个 GPU 持有 seq_len/N 的 Q 和对应的 KV

问题：attention 需要每个 Q 和所有 K/V 交互
解决：K/V 在 GPU 间 ring 传递

Step 1: GPU_i 用自己的 Q 和自己的 KV 计算 partial attention
Step 2: GPU_i 把 KV 发给 GPU_(i+1)，同时从 GPU_(i-1) 接收 KV
Step 3: GPU_i 用自己的 Q 和新收到的 KV 计算 partial attention
...
N-1 步后：每个 GPU 的 Q 都和所有 KV 交互过了

通信量：
  每步每个 GPU 发送: seq_len/N * head_dim * 2 * sizeof (K 和 V)
  总步数: N-1
  总通信: (N-1) * seq_len/N * head_dim * 2 * sizeof * num_heads * num_layers

关键：通信和计算可以 overlap（发送当前 KV 的同时计算上一个 KV 的 attention）
```

### CP vs TP 的选择

```
TP (Tensor Parallelism):
  切分 hidden dimension
  通信: AllReduce，通信量 = batch * seq * hidden * sizeof
  适合: hidden 大的模型

CP (Context Parallelism):
  切分 sequence dimension
  通信: Ring（P2P），通信量 = seq/N * hidden * sizeof * (N-1)
  适合: sequence 长的场景

组合使用:
  TP 放 node 内（需要高带宽）
  CP 可以跨 node（P2P 通信量相对小）
  例: 8 GPU/node, TP=8 (node 内), CP=4 (跨 4 nodes)
```

---

## 日检（20 分钟）

1. **口述**（5min）：Continuous batching vs static batching 的区别？对吞吐的影响？
2. **口述**（5min）：Decode attention 为什么是 memory-bound？AI 大约是多少？
3. **口述**（5min）：Chunked prefill 解决什么问题？代价是什么？
4. **口述**（5min）：FlashDecoding 的核心 idea？为什么对长序列有效？

---

## 参考资料

- Orca: A Distributed Serving System for Transformer-Based Generative Models (Yu et al., 2022)
- vLLM source code: `vllm/core/scheduler.py`
- Flash-Decoding for long-context LLM inference (Dao et al., blog post)
- Ring Attention with Blockwise Transformers for Near-Infinite Context (Liu et al., 2023)
