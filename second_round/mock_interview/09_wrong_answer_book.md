# 错题本（Wrong Answer Book）

## 使用说明
- 每个错误按类别分类，包含：错误示例、为什么错、正确答案、记忆要点
- 重点标注候选人当前能力下最容易犯的错误
- 用于面试前快速复习，避免踩坑

---

## Category 1: concept_gap（概念理解不到位）

### CG-01：混淆 Prefill 和 Decode 的计算特性

**错误示例**：
"Prefill 和 decode 都是 memory-bound 的，因为都需要读取模型权重。"

**为什么错**：
- Prefill 处理整个 input sequence，是一个大的 GEMM 操作，arithmetic intensity 高 → **compute-bound**
- Decode 每次只生成一个 token，batch dimension 小，arithmetic intensity 低 → **memory-bound**
- 这是 prefill-decode disaggregation 的理论基础

**正确答案**：
"Prefill 是 compute-bound：输入 N 个 token 做一次大 GEMM（矩阵乘法），计算量 O(N × d²)，arithmetic intensity 高。Decode 是 memory-bound：每步只生成 1 个 token，需要读取全部模型权重但只做少量计算，arithmetic intensity ≈ 1（每读一个 weight 只做一次乘加）。这就是为什么 decode 的 latency 主要取决于 memory bandwidth 而非 compute。"

**记忆要点**：
- Prefill = 大 batch GEMM = compute-bound
- Decode = batch=1 GEMV = memory-bound
- A100: 312 TFLOPS compute, 2TB/s bandwidth → arithmetic intensity 分界点 ≈ 156 ops/byte

---

### CG-02：认为 Occupancy 越高性能越好

**错误示例**：
"我的 kernel occupancy 只有 50%，需要优化到 100% 才能达到最佳性能。"

**为什么错**：
- Occupancy 高有利于隐藏 memory latency（更多 warp 可以切换）
- 但降低 occupancy 可以让每个 thread 使用更多 register/shared memory
- 对于 compute-bound kernel，更多 register → 减少 spilling → 性能更好
- 实际最优 occupancy 通常在 50-75% 之间

**正确答案**：
"Occupancy 和性能不是线性关系。对于 memory-bound kernel，高 occupancy 有利于隐藏 latency。但对于 compute-bound kernel（如 GEMM），降低 occupancy 换取更多 register 和 shared memory 可能更快。需要用 profiler 实际测量，而不是盲目追求 100% occupancy。"

**记忆要点**：
- Memory-bound → 高 occupancy 好（隐藏 latency）
- Compute-bound → 适中 occupancy 好（更多 register）
- 用 NCU 的 occupancy analysis 确认是否是瓶颈

---

### CG-03：混淆 PagedAttention 的 page 和 OS 的 page

**错误示例**：
"PagedAttention 就是用操作系统的虚拟内存来管理 KV cache，让 OS 自动做 page fault 处理。"

**为什么错**：
- PagedAttention 是在 GPU memory 中的用户态内存管理，不涉及 OS page fault
- 它借鉴了 OS 虚拟内存的思想（page table, 逻辑→物理映射），但完全在应用层实现
- GPU memory 没有 OS 级别的虚拟内存管理（CUDA unified memory 除外，但那不是 PagedAttention 用的）

**正确答案**：
"PagedAttention 借鉴了 OS 虚拟内存的设计思想，但完全在用户态实现。它将 KV cache 分为固定大小的 block（如 16 tokens），用 block table 维护逻辑 block → 物理 block 的映射。好处是：1）消除内存碎片（不需要连续分配）；2）支持动态增长（按需分配新 block）；3）支持 copy-on-write（beam search 共享 prefix）。这一切都在 vLLM 的 block manager 中实现，不依赖 OS。"

**记忆要点**：
- PagedAttention = 用户态内存管理，不是 OS 虚拟内存
- Block size 通常 16 tokens
- Block table 类似 OS page table，但在 GPU memory 中

---

### CG-04：认为 FlashAttention 减少了计算量

**错误示例**：
"FlashAttention 比标准 attention 快是因为它减少了计算量，用了更高效的算法。"

**为什么错**：
- FlashAttention 的 FLOPs 和标准 attention 完全相同（甚至略多，因为 recomputation）
- 它快是因为减少了 HBM 访问量（IO complexity 从 O(N²) 降到 O(N²d/M)）
- 核心思想是 IO-aware：通过 tiling 让计算在 SRAM 中完成，避免写回 HBM

**正确答案**：
"FlashAttention 不减少计算量，FLOPs 相同甚至略多（backward 需要 recomputation）。它快是因为是 IO-aware 的：标准 attention 需要将 N×N 的中间矩阵写回 HBM 再读回来，FlashAttention 通过 tiling + online softmax 让所有中间结果留在 SRAM 中。IO 复杂度从 O(N²) 降到 O(N²d/M)，其中 M 是 SRAM 大小。对于 N=4096, d=128, M=100KB，这是约 5x 的 IO 减少。"

**记忆要点**：
- FlashAttention: 相同 FLOPs，更少 IO
- 核心：tiling + online softmax + recomputation
- 快的原因：避免 N×N 中间矩阵写回 HBM

---

### CG-05：混淆 Tensor Parallelism 的通信模式

**错误示例**：
"Tensor Parallelism 每层需要两次 AllReduce，一次在 attention 后，一次在 MLP 后。"

**为什么错**：
- Megatron-style TP 中，MLP 层用 column-then-row 切分，只需要一次 AllReduce（在 row parallel 之后）
- Attention 层类似：QKV 按 head 切分，output projection 按行切分，一次 AllReduce
- 所以每个 transformer layer 需要 2 次 AllReduce（attention 后 1 次 + MLP 后 1 次）
- 不是"每层两次"的说法错，而是要准确说明在哪里做

**正确答案**：
"Megatron-style TP 中，每个 transformer layer 有 2 次 AllReduce：
1. Attention 的 output projection 是 row parallel → 之后做 AllReduce
2. MLP 的第二个 linear 是 row parallel → 之后做 AllReduce
通信量：每次 AllReduce 的数据量 = 2 × (N-1)/N × hidden_size × seq_len × batch_size × dtype_size
对于 70B 模型（hidden=8192），TP=4，seq_len=1，batch=1：每次约 32KB，延迟主要是 latency 而非 bandwidth。"

**记忆要点**：
- 每个 transformer layer = 2 次 AllReduce
- Column parallel 不需要通信，Row parallel 后需要 AllReduce
- 通信量与 hidden_size 成正比

---

### CG-06：不理解 Continuous Batching 的本质

**错误示例**：
"Continuous batching 就是动态调整 batch size，请求多的时候 batch 大，请求少的时候 batch 小。"

**为什么错**：
- 这描述的是 dynamic batching，不是 continuous batching
- Continuous batching 的核心是 **iteration-level scheduling**：每个 decode iteration 都可以加入新请求或移除已完成请求
- Static batching 要等整个 batch 都完成才能开始下一个 batch

**正确答案**：
"Continuous batching（iteration-level scheduling）的核心是：不等所有请求完成就可以处理新请求。每个 decode step 结束后，已完成的请求立即释放资源，新请求立即加入。这样 GPU 始终在处理最大可能的 batch，不会因为一个 long request 阻塞整个 batch。相比 static batching，throughput 可以提升 2-10x，因为消除了'短请求等长请求'的浪费。"

**记忆要点**：
- Static batching: batch 级调度，等最慢的请求
- Continuous batching: iteration 级调度，随时进出
- 关键实现：需要 PagedAttention 支持动态内存管理

---

## Category 2: weak_phrasing（表达不够有力）

### WP-01：描述项目成果时缺乏量化

**错误示例**：
"我实现了推测解码，性能有明显提升。"

**为什么弱**：没有具体数字，面试官无法判断"明显"是 10% 还是 100%。

**正确表达**：
"我在 Ascend 910B NPU 上实现了 EAGLE-3 推测解码，num_spec_tokens=3 时输出吞吐从 9.22 tok/s 提升至 14.30 tok/s（+55%），TPOT 从 108ms 降至 66ms（-39%）。Token-1 接受率 70%，mean acceptance length 1.63。"

**记忆要点**：永远给具体数字 + 测试条件 + 对比 baseline

---

### WP-02：回答"不知道"时太生硬

**错误示例**：
"这个我不知道。"（然后沉默）

**为什么弱**：面试官无法判断你是完全不了解还是只是没深入研究。

**正确表达**：
"这个我没有实战经验。但基于我对 [相关概念] 的理解，我推测 [给出合理推断]。如果要深入，我会 [说明学习路径]。"

例如被问 CUDA kernel 优化时：
"我没有写过 CUDA kernel，但基于我对 GPU memory hierarchy 的理解，这个 kernel 的瓶颈可能在 shared memory bank conflict。如果让我优化，我会先用 NCU 确认瓶颈，然后尝试 padding 或 swizzle 来消除 conflict。"

**记忆要点**：不知道 → 说出你知道的相关知识 → 给出合理推断 → 说明学习路径

---

### WP-03：描述技术方案时没有 tradeoff

**错误示例**：
"我选择了 HNSW 作为向量索引，因为它 recall 高。"

**为什么弱**：没有讨论为什么不选其他方案，没有 tradeoff 分析。

**正确表达**：
"我选择 HNSW 而非 IVF-PQ，原因是：
- HNSW recall@10 > 95%，IVF-PQ 约 85-90%（我的场景对 recall 要求高）
- HNSW 内存占用大（每个向量额外 ~500 bytes 存图结构），但我的数据量（数万 chunk）内存不是瓶颈
- Trade-off：如果数据量增长到百万级，需要切换到 IVF-PQ + reranking 来控制内存
- 查询延迟：HNSW < 5ms，满足我的 latency 要求"

**记忆要点**：每个技术选择都要说：选了什么 + 为什么不选其他 + 什么条件下会改变选择

---

### WP-04：系统设计时只说"用 XXX"

**错误示例**：
"监控用 Prometheus + Grafana，日志用 ELK，告警用 PagerDuty。"

**为什么弱**：只列了工具名，没有说监控什么指标、告警阈值是什么、怎么用这些数据做决策。

**正确表达**：
"监控分三层：
1. 业务指标：TTFT P99 < 300ms（告警阈值 250ms 预警，300ms critical），TPOT P99 < 50ms，throughput tokens/s
2. 系统指标：GPU utilization > 80%（低于 60% 说明 batch 不够），KV cache utilization（> 90% 预警）
3. 硬件指标：GPU 温度 < 83°C，ECC error count，NVLink error rate

告警策略：预警 → Slack 通知；Critical → PagerDuty 叫人；连续 3 次 critical → 自动 failover"

**记忆要点**：工具只是手段，重要的是：监控什么 + 阈值多少 + 超了怎么办

---

## Category 3: missing_metric（缺少具体数据）

### MM-01：KV Cache 内存计算

**错误示例**：
"70B 模型的 KV cache 很大，需要很多内存。"

**正确计算**：
```
Llama-70B (GQA, 8 KV heads, head_dim=128, 80 layers, FP16):
- Per token per layer: 2 (K+V) × 8 (heads) × 128 (dim) × 2 (bytes) = 4KB
- Per token all layers: 4KB × 80 = 320KB
- 1024 tokens: 320MB
- Batch=32, seq_len=2048: 32 × 2048 × 320KB = 20GB
```

**记忆要点**：Llama-70B KV cache ≈ 320KB/token（GQA 8 heads）

---

### MM-02：GPU Bandwidth 与 Decode Latency

**错误示例**：
"Decode 很快，每个 token 只需要几毫秒。"

**正确计算**：
```
A100 HBM bandwidth: 2TB/s
70B FP16 model weights: 140GB
Minimum decode latency (batch=1): 140GB / 2TB/s = 70ms per token
实际（考虑 overhead）: ~80-100ms per token

W8A8 quantized (70GB): 70GB / 2TB/s = 35ms per token
W4A16 quantized (35GB): 35GB / 2TB/s = 17.5ms per token
```

**记忆要点**：
- Decode latency ≈ model_size / bandwidth（memory-bound）
- A100: 2TB/s, H100: 3.35TB/s
- 70B FP16 on A100: ~70ms/token minimum

---

### MM-03：Speculative Decoding Speedup 公式

**错误示例**：
"Acceptance rate 70% 意味着 speedup 是 1.7x。"

**正确计算**：
```
设 draft length = K, acceptance rate = α (per token)
Expected accepted tokens = Σ(i=1 to K) α^i ≈ α/(1-α) (geometric series, when K large)
α=0.7, K=3: expected accepted = 0.7 + 0.7² + 0.7³ = 0.7 + 0.49 + 0.343 = 1.533

Speedup = (1 + expected_accepted) / (1 + K × draft_cost/verify_cost)
如果 draft_cost ≈ 0.1 × verify_cost:
Speedup = (1 + 1.533) / (1 + 3 × 0.1) = 2.533 / 1.3 = 1.95x

实际我的结果: 14.30/9.22 = 1.55x (低于理论值，因为 NPU overhead)
```

**记忆要点**：
- Speedup ≠ 1 + acceptance_rate
- 需要考虑 draft model 的 overhead
- 实际 speedup 通常是理论值的 70-80%

---

## Category 4: missing_tradeoff（缺少权衡分析）

### MT-01：Quantization 的 tradeoff

**错误示例**：
"量化可以减少内存和加速推理，应该总是使用。"

**正确 tradeoff**：
```
| 方案 | Memory 节省 | Speedup | 精度损失 | 适用场景 |
|------|------------|---------|---------|---------|
| FP16 | baseline | baseline | 0 | 精度敏感任务 |
| W8A8 | 2x | 1.5-2x | < 0.5% | 通用推理 |
| W4A16 | 4x | 1.5-2x (decode) | 1-3% | decode-heavy, memory-limited |
| W4A4 | 4x | 2-3x | 3-5% | 极致成本优化 |
```

不应该用量化的场景：
- 数学推理任务（对精度敏感）
- 小模型（7B 量化后精度损失大于大模型）
- Prefill-heavy 场景（W4A16 对 prefill 没有加速，因为 prefill 是 compute-bound）

**记忆要点**：量化对 decode（memory-bound）收益大，对 prefill（compute-bound）收益小

---

### MT-02：Prefix Caching 的 tradeoff

**错误示例**：
"Prefix caching 总是好的，可以减少重复计算。"

**正确 tradeoff**：
- 好处：相同 system prompt 的请求共享 KV cache，减少 prefill 计算
- 代价：
  1. 额外内存：cached prefix 占用 GPU memory，减少可用于新请求的 KV cache
  2. 管理开销：hash 计算、cache lookup、eviction 策略
  3. 碎片化：cached blocks 不能被其他请求使用
- 适用条件：大量请求共享相同 prefix（如 system prompt）
- 不适用：每个请求的 prefix 都不同（cache hit rate 低）

**记忆要点**：Prefix caching 收益 = hit_rate × prefix_length × prefill_cost - cache_memory_cost

---

## Category 5: missing_incident_handling（缺少故障处理）

### IH-01：系统设计中忽略 failure mode

**错误示例**：
"系统架构是 Load Balancer → Worker Pool → GPU。请求进来分发到 worker 处理。"

**缺少的 failure handling**：
- Worker 挂了怎么办？→ Health check + 自动重启 + 请求重试
- GPU OOM 怎么办？→ Memory watermark + admission control + graceful rejection
- 网络分区怎么办？→ Timeout + circuit breaker + fallback
- 模型加载失败怎么办？→ Retry + rollback to previous version
- 请求超时怎么办？→ Per-request timeout + 资源释放 + 用户通知

**记忆要点**：每个组件都要问"如果它挂了会怎样？怎么检测？怎么恢复？"

---

### IH-02：RAG 系统的 failure mode

**错误示例**：
"RAG 系统就是检索 + 生成，很简单。"

**需要考虑的 failure mode**：
1. Embedding 服务不可用 → 降级到 BM25 检索
2. 向量数据库超时 → 返回缓存结果或 fallback 到 keyword search
3. LLM 生成超时 → 返回检索到的原文片段
4. 检索结果为空 → 明确告知用户"未找到相关信息"
5. LLM 幻觉 → 添加 faithfulness check，不确定时标注"可能不准确"

**记忆要点**：RAG 的每个环节都可能失败，需要 graceful degradation 而非直接报错

---

## Category 6: missing_benchmark_evidence（缺少 benchmark 支撑）

### BE-01：声称"性能好"但没有对比

**错误示例**：
"我的 RAG 系统准确率 90%，性能很好。"

**需要的 benchmark context**：
- 90% 是什么指标？Faithfulness? Answer Relevancy? 综合？
- 测试集多大？怎么构造的？
- Baseline 是什么？（没有 RAG 直接问 LLM 是多少？简单 RAG 是多少？）
- 和业界对比如何？（RAGAS leaderboard 上的 SOTA 是多少？）
- 不同难度的 query 分别是多少？

**正确表达**：
"RAGAS 评测结果：Faithfulness 92%, Answer Relevancy 88%, 综合 90%。测试集 XX 条，覆盖简单/中等/困难三个难度。对比 baseline（无 RAG 直接问 LLM）：准确率从 45% 提升到 90%。主要失败 case 集中在需要跨文档推理的复杂问题。"

**记忆要点**：任何性能数字都需要：指标定义 + 测试条件 + baseline 对比 + 失败分析

---

### BE-02：Throughput 数字缺乏上下文

**错误示例**：
"吞吐提升了 55%。"

**需要的上下文**：
- 55% 是在什么条件下？（模型大小、batch size、hardware、input/output length）
- Baseline 是什么？（原始 vLLM without speculative decoding）
- 这个数字在不同条件下会怎么变？（batch 增大时 speedup 会降低）
- 和其他方案对比如何？（Medusa 在同条件下是多少？）

**正确表达**：
"在 Ascend 910B + Qwen2.5-7B + batch=1 + MT-Bench 80 轮的条件下，EAGLE-3 (spec=3) 相比无推测解码的 baseline，输出吞吐从 9.22 tok/s 提升至 14.30 tok/s（+55%）。注意：这是 batch=1 的结果，大 batch 下 speedup 会降低（预计 batch=8 时约 20-30%）。"

**记忆要点**：性能数字 = 具体值 + 测试条件 + 适用范围 + 局限性
