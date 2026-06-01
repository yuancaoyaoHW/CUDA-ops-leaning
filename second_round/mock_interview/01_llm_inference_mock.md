# 模拟面试 1：45 分钟 LLM Inference 系统设计

## 面试官 Profile

- **公司**：某 AI Inference 创业公司（对标 Anyscale / Together AI）
- **级别**：Staff Engineer，前 Google Brain Serving Infra
- **风格**：数字驱动，追求量化分析，对模糊回答零容忍。会打断候选人要求给具体数字。
- **偏好**：喜欢候选人先给 high-level 再 drill down，讨厌上来就说细节不给全局。

---

## Opening Question

> "我们需要设计一个 LLM serving system，模型是 Llama-70B FP16，目标 QPS 500，TTFT P99 < 300ms，TPOT P99 < 40ms，平均 input 1024 tokens，平均 output 256 tokens。你有 32 张 A100-80GB。请从架构开始，给我一个完整的设计。"

---

## Candidate Expected Answer（基于候选人真实经验）

"首先做容量规划：
- 70B FP16 模型权重 = 140GB，需要至少 TP=2（2×80GB），留给 KV cache 约 20GB/卡
- KV cache per token per layer：2 × num_heads × head_dim × 2 bytes = 2 × 8 × 128 × 2 = 4KB（GQA 8 KV heads）
- 80 layers → 每个 token 的 KV cache = 80 × 4KB = 320KB
- 20GB 可存约 65000 tokens 的 KV cache，即约 50 个 1280-token 请求
- TP=2 的 instance 需要 2 卡，32 卡可部署 16 个 instance
- 单 instance decode throughput 约 30-50 tokens/s per request，batch size 32 时约 1000 tokens/s
- 500 QPS × 256 output tokens = 128000 tokens/s 需要约 128 个 instance... 这远超 16 个

需要重新考虑：用 INT8 量化将模型压缩到 70GB，TP=1 即可放下，32 个 instance。或者用 FP8/W4A16 进一步压缩。

架构上：
1. Load Balancer → Request Router → Worker Pool
2. 每个 Worker 运行 vLLM engine，continuous batching
3. Prefill 和 Decode 可以考虑分离部署（disaggregated serving）
4. 用 PagedAttention 管理 KV cache，最大化内存利用率
5. Prefix caching 对重复 system prompt 的场景有收益

我在 vLLM-Ascend 上实现过 EAGLE-3 speculative decoding，throughput 提升 55%，这在 decode-heavy 场景下可以显著降低 TPOT。"

---

## Weak Answer（常见错误）

"用 vLLM 部署就行了，开 continuous batching，模型用 tensor parallelism 分到多张卡上。如果 QPS 不够就加机器。监控 TTFT 和 TPOT，超了就扩容。"

**为什么弱**：
- 没有任何数字计算
- 没有容量规划
- 没有讨论 memory 约束
- 没有讨论 scheduling 策略
- "加机器"不是设计

---

## Strong Answer（工程深度 + 数据 + tradeoff）

"让我分层来设计：

**1. 容量规划（Back-of-envelope）**

Llama-70B FP16：
- 模型权重：70B × 2 bytes = 140GB → TP=2 最小（每卡 70GB 权重）
- 但 80GB 卡只剩 10GB 给 KV cache，太少
- 方案 A：TP=4，每卡 35GB 权重，45GB 给 KV cache → 8 个 instance
- 方案 B：W8A8 量化，70GB 权重，TP=1，每卡 10GB 给 KV cache → 32 个 instance 但 KV cache 太小
- 方案 C：W8A8 + TP=2，每卡 35GB 权重，45GB KV cache → 16 个 instance ✓

KV cache 容量（方案 C，GQA 8 heads，head_dim 128，80 layers）：
- Per token per layer：2 × 8 × 128 × 2 bytes = 4KB
- Per token all layers：4KB × 80 = 320KB
- 45GB / 320KB ≈ 147,000 tokens per GPU
- 每个请求平均 1024+256 = 1280 tokens
- 每个 instance 可同时 serve ≈ 115 个请求

Throughput 估算：
- Decode 阶段 memory-bound：A100 HBM bandwidth 2TB/s
- 每个 decode step 需要读取全部权重：70GB（W8A8）
- 理论最小 decode latency per step：70GB / 2TB/s = 35ms（batch=1）
- Batch=32 时 amortize：35ms per step，32 tokens output → ~1000 tokens/s per instance
- 16 个 instance → 16,000 tokens/s
- 需要 128,000 tokens/s → 差 8x

**解决方案**：
1. Speculative decoding：我的 EAGLE-3 实现可以 1.55x throughput → 24,800 tokens/s
2. 更激进的量化：W4A16 → 模型 35GB，TP=1，32 instances → 32,000 tokens/s
3. Prefill-decode disaggregation：prefill 用 TP=4 高算力配置，decode 用 TP=1 + W4A16 高并发配置
4. 最终方案：8 卡做 prefill（2 个 TP=4 instance），24 卡做 decode（24 个 W4A16 instance）

**2. 架构设计**

```
Client → API Gateway (rate limiting, auth)
       → Request Router (prefix matching, load-aware)
       → Prefill Workers (2 × TP=4, high compute)
       → KV Cache Transfer (RDMA/NVLink)
       → Decode Workers (24 × TP=1 W4A16, high throughput)
       → Response Streamer (SSE/WebSocket)
```

**3. Scheduling 策略**
- Continuous batching with iteration-level scheduling
- Chunked prefill：chunk size 512 tokens，避免 long prefill 阻塞 decode
- Priority queue：VIP 请求优先 prefill
- Preemption：当 KV cache 不足时，swap 低优先级请求到 CPU memory

**4. TTFT 保证**
- Prefill 1024 tokens on TP=4 A100：约 50-80ms compute
- Queue wait target < 100ms → admission control
- Prefix caching：对 system prompt 复用 KV cache，减少 prefill 量
- 如果 queue depth > threshold，reject 新请求返回 429

**5. TPOT 保证**
- Decode step latency = model_size / bandwidth / batch_size + overhead
- W4A16 35GB / 2TB/s = 17.5ms per step（batch=1）
- Batch=32：仍然约 17.5ms（memory-bound，batch 不增加 latency 直到 compute-bound）
- EAGLE-3 speculative decoding：平均 accept 1.6 tokens/step → effective TPOT ≈ 11ms ✓

**6. 容错**
- Health check per worker，3 次连续失败下线
- 请求级 retry：prefill 失败重试到其他 worker
- KV cache 不做持久化（重新 prefill 比恢复快）
- Graceful degradation：过载时降低 max_tokens，拒绝新请求

**7. 监控**
- SLI：TTFT P50/P99, TPOT P50/P99, throughput (tokens/s), queue depth
- GPU metrics：utilization, memory usage, temperature
- 告警：TTFT P99 > 250ms（预警），> 300ms（critical）
- Dashboard：per-instance throughput, batch size distribution, KV cache utilization"

---

## Follow-up Chain（5 层递进追问）

### Follow-up 1：Prefill-Decode Disaggregation 深入
> "你提到 prefill-decode 分离。KV cache 从 prefill worker 传到 decode worker 的 overhead 是多少？值得吗？"

**期望回答**：
- 1024 tokens × 320KB/token = 320MB KV cache 需要传输
- NVLink 900GB/s：0.35ms；PCIe Gen4 64GB/s：5ms；RDMA 200Gbps：12.8ms
- 如果跨机（RDMA）：12.8ms overhead vs prefill 50-80ms → 约 15-25% overhead
- 值得的条件：prefill 和 decode 的 compute profile 差异大（prefill compute-bound, decode memory-bound），分离后各自可以用最优配置
- 不值得的条件：请求量小、input 短、或者网络带宽不足

### Follow-up 2：Speculative Decoding 在高并发下的表现
> "你说 EAGLE-3 提升 55% throughput。但那是 batch=1 的数据。batch=32 时 speculative decoding 还有收益吗？"

**期望回答**：
- Batch=1 时 decode 严重 memory-bound，speculative decoding 收益最大
- Batch 增大后，decode 逐渐接近 compute-bound，speculative decoding 的额外 compute 开始成为瓶颈
- 经验数据：batch=1 speedup 1.5-2x，batch=8 speedup 1.2-1.4x，batch=32 speedup 可能 < 1.1x
- 我的 55% 数据是在 batch=1 的 MT-Bench 上测的，大 batch 场景需要重新评估
- Trade-off：speculative decoding 适合 latency-sensitive 低并发场景，不适合 throughput-maximizing 高并发场景

### Follow-up 3：Quantization 精度影响
> "W4A16 量化后模型质量下降多少？你怎么评估？如果客户说'回答质量变差了'你怎么处理？"

**期望回答**：
- W4A16（如 AWQ/GPTQ）在 70B 模型上 perplexity 增加约 0.1-0.3，大部分 benchmark 下降 < 1%
- 评估方法：offline eval（MMLU, HumanEval, MT-Bench）+ online A/B test
- 如果客户投诉：先确认是否是量化导致（对比 FP16 输出），如果是则提供 FP16 tier（更贵）
- 分层服务：Free tier 用 W4A16，Pro tier 用 FP8，Enterprise 用 FP16
- **候选人诚实标注**：我没有量化部署的实战经验，以上是基于论文和社区讨论的理解

### Follow-up 4：Burst Traffic 处理
> "黑五当天流量突然 10x。你的 32 张卡不够了。怎么办？"

**期望回答**：
- 短期（秒级）：Admission control，queue + reject，保护已有请求的 SLA
- 中期（分钟级）：Auto-scaling，预热新 instance（模型加载约 30-60s）
- 长期（小时级）：Spot instance pool，预留 buffer capacity
- Graceful degradation 策略：
  1. 降低 max_output_tokens（256→128）
  2. 关闭 speculative decoding（释放 draft model memory）
  3. 增大 batch size（牺牲 latency 换 throughput）
  4. 启用更激进的量化（W4A16 → W3A16）
  5. 最后手段：reject 低优先级请求

### Follow-up 5：成本优化
> "你的方案每月 GPU 成本约 $150K（32×A100 on-demand）。CEO 说太贵了，要砍一半。你怎么做？"

**期望回答**：
- 方案 1：Spot instance（节省 60-70%，但需要处理 preemption）
- 方案 2：更激进量化 W4A16 → 用更少的卡（16 卡可能够）
- 方案 3：混合部署——peak 时用 on-demand，off-peak 用 spot
- 方案 4：模型蒸馏——用 70B 蒸馏出 13B，大部分请求用小模型，复杂请求路由到大模型
- 方案 5：Prefix caching 减少重复 prefill 的 compute
- 量化分析：如果 70% 请求可以用 13B 处理，成本降低约 5x for those requests
- Trade-off：每个方案都有质量/延迟/可靠性的代价，需要和 PM 对齐 SLA

---

## Pressure Follow-up（故意挑战候选人）

> "你说你在 vLLM-Ascend 上做了 EAGLE-3，throughput 提升 55%。但你的测试是在 NPU 上，不是 GPU。NPU 和 GPU 的 memory bandwidth、compute throughput 完全不同。你怎么确定你的经验能迁移到 GPU serving？"

**期望应对**：
- 承认硬件差异：NPU（Ascend 910B）的 HBM bandwidth 约 1.5TB/s vs A100 2TB/s，compute 特性也不同
- 但核心算法逻辑是通用的：rejection sampling、KV cache 管理、draft model 设计
- 迁移需要的工作：重新 benchmark、调整 draft length、可能需要不同的 batch size 策略
- 我的价值在于理解 speculative decoding 的系统级设计，而不是特定硬件的调优
- 诚实承认：我没有 CUDA GPU 上的 profiling 经验，这是我需要补强的

---

## Debugging Scenario

> "线上告警：TPOT P99 从 35ms 突然飙升到 120ms，持续了 10 分钟后自动恢复。没有代码变更，没有流量变化。你怎么排查？"

**排查思路**：

1. **第一步（30s）**：看 GPU metrics dashboard
   - GPU utilization 是否正常？如果下降 → 可能是 GPU throttling（温度/功耗）
   - Memory usage 是否突增？如果是 → 可能是 KV cache 碎片导致频繁 eviction

2. **第二步（2min）**：看 batch size 和 queue depth
   - Batch size 是否突然增大？→ 可能是某些请求的 output 特别长，占用了 KV cache
   - Queue depth 是否增加？→ 可能是某个 worker 变慢导致请求堆积

3. **第三步（5min）**：看 per-request metrics
   - 是所有请求都慢还是部分请求？
   - 如果部分请求：检查 input length 分布是否有异常（某个用户发了超长 input）
   - 如果所有请求：检查 GPU 硬件状态（ECC error, NVLink error, PCIe bandwidth）

4. **可能的 root cause**：
   - GPU thermal throttling（温度过高降频）→ 10 分钟后散热恢复
   - 某个用户的超长请求占满了 KV cache，触发大量 preemption/swap
   - NVLink 间歇性错误导致 TP AllReduce 变慢
   - Host CPU GC pause 导致 scheduling 延迟

5. **预防措施**：
   - 设置 max_input_length 和 max_output_length 限制
   - GPU 温度监控 + 预警
   - Per-request timeout + 自动 kill
   - NVLink/PCIe error counter 监控

---

## System Design Extension（扩展到更大规模）

> "现在需求变了：QPS 从 500 涨到 50000，模型从 1 个变成 10 个（7B 到 405B），需要支持多租户。重新设计。"

**扩展设计要点**：

1. **Multi-model serving**：
   - Model registry + dynamic loading
   - 小模型（7B/13B）可以多个共享一张卡（GPU sharing / MPS）
   - 大模型（70B/405B）需要 TP=4/8
   - 按需加载：cold model 在首次请求时加载（cold start 30-60s），hot model 常驻

2. **Multi-tenancy**：
   - Per-tenant rate limiting + quota
   - Priority-based scheduling（付费用户优先）
   - Resource isolation：不同 tenant 的请求不共享 KV cache
   - Noisy neighbor 防护：单 tenant 不能占满所有 GPU

3. **Cluster-level scheduling**：
   - 中心化 scheduler（类似 Kubernetes + custom GPU scheduler）
   - Bin-packing：将小模型 pack 到同一组 GPU
   - Affinity：同一模型的请求路由到已加载该模型的 worker
   - Auto-scaling per model：根据 QPS 动态调整 replica 数

4. **50000 QPS 的挑战**：
   - 需要约 500-1000 张 A100（取决于模型 mix）
   - 网络成为瓶颈：需要 spine-leaf 网络架构
   - 单点故障影响大：需要多 AZ 部署
   - 成本优化更重要：spot instance + reserved instance 混合

---

## Hire/No-Hire Evaluation

| 维度 | 权重 | 评分标准 |
|------|------|---------|
| 容量规划 | 20% | 能否做 back-of-envelope 计算，数字是否合理 |
| 架构完整性 | 20% | 是否覆盖了 routing, scheduling, caching, fault tolerance |
| 技术深度 | 25% | 对 PagedAttention, continuous batching, speculative decoding 的理解深度 |
| Tradeoff 分析 | 20% | 能否讨论不同方案的 pros/cons，给出选择依据 |
| Production 思维 | 15% | 是否考虑了 monitoring, alerting, degradation, cost |

---

## Scorecard

| 维度 | 候选人预期得分 | 说明 |
|------|--------------|------|
| 容量规划 | 5/10 | 能做基本计算但不够精确，可能遗漏 KV cache 计算 |
| 架构完整性 | 6/10 | 基本架构正确，但 scheduling 和 fault tolerance 细节不足 |
| 技术深度 | 7/10 | 对 speculative decoding 和 PagedAttention 有实际经验，但 quantization 和 disaggregation 偏理论 |
| Tradeoff 分析 | 5/10 | 能给出方向但缺乏量化的 tradeoff 分析 |
| Production 思维 | 3/10 | 缺乏 production 运维经验，monitoring 和 incident response 偏弱 |
| **总分** | **5.2/10** | **Lean No Hire — 有潜力但 production 经验不足** |

### 决策依据
- **Hire 信号**：有 vLLM 源码级经验，理解 speculative decoding 原理和实现，有开源协作能力
- **No Hire 信号**：缺乏 GPU profiling 经验，无法做精确的 capacity planning，production debugging 能力弱
- **建议**：如果是 junior/mid-level 岗位可以 Hire（有成长潜力），Staff level 则 No Hire
