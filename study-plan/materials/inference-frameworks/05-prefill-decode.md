# Prefill 与 Decode 阶段分析

## 1. 学习目标

- 理解 LLM 推理的两阶段模型：prefill（预填充）和 decode（解码）
- 掌握两阶段的计算特性差异：compute-bound vs memory-bound
- 理解 TTFT 和 TPOT 的定义与优化方向
- 能够分析不同 batch size 和 sequence length 下的性能瓶颈
- 掌握 prefill-decode 分离（disaggregation）的动机与实现

## 2. 系统动机

### 2.1 两阶段的本质区别

**Prefill（预填充）**：
- 输入：完整 prompt（S 个 token）
- 计算：所有 token 的 QKV projection + attention + FFN
- 特点：大矩阵乘法，**compute-bound**
- 输出：第一个生成 token + 完整 KV cache

**Decode（解码）**：
- 输入：上一步生成的 1 个 token
- 计算：1 个 token 的 QKV + 与所有历史 KV 做 attention
- 特点：GEMV（矩阵向量乘），**memory-bound**
- 输出：下一个 token

### 2.2 性能指标

| 指标 | 英文 | 定义 | 影响因素 |
|------|------|------|---------|
| TTFT | Time To First Token | 从请求到第一个 token 的时间 | Prefill 速度 |
| TPOT | Time Per Output Token | 每个输出 token 的生成时间 | Decode 速度 |
| ITL | Inter-Token Latency | token 间延迟（≈TPOT） | Decode + scheduling |
| Throughput | Throughput | 每秒生成的 token 数 | Batch size × 1/TPOT |
| Latency | End-to-End Latency | 完整请求的总时间 | TTFT + output_len × TPOT |

### 2.3 计算量对比

```
Prefill (seq_len=S, hidden=E, layers=L):
  Per layer: 
    QKV proj: 3 × 2SE² FLOPs (或 GQA: (2+2×ratio)SE²)
    Attention: 2S²×head_dim×num_heads FLOPs
    FFN: 2 × 2SE×intermediate FLOPs
  Total: O(L × (SE² + S²D))

Decode (1 token, KV cache len=S):
  Per layer:
    QKV proj: 3 × 2E² FLOPs (S=1)
    Attention: 2S×D×H FLOPs (读取整个 KV cache)
    FFN: 2 × 2E×intermediate FLOPs
  Total: O(L × (E² + SD))
```

### 2.4 Arithmetic Intensity 对比

```
Prefill GEMM (S=2048, E=4096):
  AI = 2×S×E×E / ((S×E + E×E + S×E)×2) ≈ S/3 ≈ 683 → compute-bound

Decode GEMV (S=1, E=4096):
  AI = 2×1×E×E / ((1×E + E×E + 1×E)×2) ≈ 1 → memory-bound

Decode Attention (KV_len=2048, D=128):
  AI = 2×1×2048×128 / (2×2048×128×2) ≈ 1 → memory-bound
```

## 3. 核心术语表

| 术语 | 英文全称 | 定义 |
|------|----------|------|
| Prefill | Prefill Phase | 处理完整 prompt，生成 KV cache 的阶段 |
| Decode | Decode Phase | 逐 token 自回归生成的阶段 |
| TTFT | Time To First Token | 首 token 延迟 |
| TPOT | Time Per Output Token | 每 token 生成时间 |
| ITL | Inter-Token Latency | token 间延迟 |
| GEMM | General Matrix Multiply | 矩阵乘矩阵（prefill） |
| GEMV | General Matrix-Vector Multiply | 矩阵乘向量（decode） |
| Batching | Batching | 将多个请求合并处理 |
| PD Disaggregation | Prefill-Decode Disaggregation | 将两阶段分离到不同硬件 |

## 4. 执行流程

```
Request arrives
    │
    ▼
┌─────────────────────────────────────────┐
│ PREFILL PHASE                           │
│                                         │
│ Input: [tok_1, tok_2, ..., tok_S]       │
│                                         │
│ For each layer:                         │
│   QKV = X @ W_qkv    (GEMM: S×E × E×3E)│
│   Score = Q @ K^T     (GEMM: S×D × D×S) │
│   Attn = softmax(Score) @ V             │
│   Out = Attn @ W_o    (GEMM: S×E × E×E) │
│   FFN = SwiGLU(X @ W_up, X @ W_gate)   │
│   X = FFN @ W_down                      │
│                                         │
│ Output: first_token, KV_cache           │
│ Time: TTFT                              │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ DECODE PHASE (repeated N times)         │
│                                         │
│ Input: [last_token] (1 token)           │
│                                         │
│ For each layer:                         │
│   qkv = x @ W_qkv    (GEMV: 1×E × E×3E)│
│   Append k,v to KV_cache                │
│   score = q @ K_cache^T (dot: 1×D × D×S)│
│   attn = softmax(score) @ V_cache       │
│   out = attn @ W_o    (GEMV: 1×E × E×E) │
│   ffn = SwiGLU(...)                     │
│   x = ffn @ W_down                      │
│                                         │
│ Output: next_token                      │
│ Time per step: TPOT                     │
└─────────────────────────────────────────┘
    │
    ▼ (repeat until EOS or max_len)
```

## 5. 参数解释

| 参数 | 影响 Prefill | 影响 Decode | 调优方向 |
|------|-------------|-------------|---------|
| batch_size | 提高 GPU 利用率 | 提高 throughput，增加 TPOT | 根据 SLA 平衡 |
| max_seq_len | 增加 TTFT | 增加 KV cache 读取量 | 按需设置 |
| tensor_parallel | 减少 TTFT | 减少 TPOT（通信开销） | 大模型必需 |
| quantization | 减少计算量 | 减少 KV cache 读取 | decode 收益更大 |
| chunked_prefill | 控制 prefill 对 decode 的干扰 | 减少 decode 被阻塞时间 | 混合场景必需 |

## 6. 调优目标

### 6.1 Prefill 优化（降低 TTFT）

- 使用 FlashAttention（减少 HBM IO）
- Tensor Parallel（分摊计算）
- 量化（减少计算量）
- Chunked prefill（避免长 prefill 阻塞 decode）

### 6.2 Decode 优化（降低 TPOT）

- 增大 batch size（提高 GPU 利用率）
- GQA/MQA（减少 KV cache 读取）
- KV cache 量化（INT8/FP8）
- CUDA Graph（减少 launch overhead）
- Speculative decoding（一次验证多个 token）

### 6.3 Throughput 优化

```
Throughput = batch_size / TPOT

提高 throughput:
1. 增大 batch_size（直到显存或 latency 限制）
2. 减少 TPOT（优化 decode）
3. Continuous batching（动态调整 batch）
```

## 7. 适用场景

| 场景 | 优先优化 | 原因 |
|------|---------|------|
| 实时对话 | TTFT + TPOT | 用户感知延迟 |
| 批量翻译 | Throughput | 总处理时间 |
| 代码补全 | TTFT | 用户等待首个建议 |
| 长文档摘要 | TTFT (prefill) | 长 prompt 的 prefill 时间 |
| Streaming | TPOT | 流式输出的流畅度 |

## 8-20. 关键内容

### Benchmark 设计

```python
import time
import torch

def measure_prefill_decode(model, tokenizer, prompt, max_new_tokens=100):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    
    # Measure TTFT (prefill)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(input_ids, max_new_tokens=1)
    torch.cuda.synchronize()
    ttft = time.perf_counter() - t0
    
    # Measure TPOT (decode)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(input_ids, max_new_tokens=max_new_tokens)
    torch.cuda.synchronize()
    total_time = time.perf_counter() - t0
    tpot = (total_time - ttft) / (max_new_tokens - 1)
    
    return {"ttft_ms": ttft * 1000, "tpot_ms": tpot * 1000}
```

### 关键习题

1. 为什么 prefill 是 compute-bound 而 decode 是 memory-bound？
2. 计算 LLaMA-3 8B 在 A100 上的理论最小 TPOT（batch=1）。
3. 增大 batch size 对 TTFT 和 TPOT 分别有什么影响？
4. 什么是 PD Disaggregation？它解决了什么问题？
5. Chunked prefill 如何平衡 TTFT 和 decode 延迟？

### 调优 checklist

- [ ] 测量 baseline TTFT 和 TPOT
- [ ] 确认 prefill 使用 FlashAttention
- [ ] 确认 decode 使用 CUDA Graph
- [ ] 设置合理的 batch size 上限
- [ ] 启用 continuous batching
- [ ] 配置 chunked prefill（如果混合 prefill/decode）
- [ ] 监控 GPU utilization（prefill 应 > 80%，decode 看 memory BW）
- [ ] 验证 TTFT 和 TPOT 满足 SLA
- [ ] 测试不同 input/output length 组合的性能
- [ ] 考虑 PD disaggregation（如果 prefill 和 decode 需求差异大）
