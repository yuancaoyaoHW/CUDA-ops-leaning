# 模拟面试 7：45 分钟 Production Debugging 面

## 面试官 Profile

- **公司**：某 AI SaaS 公司（对标 OpenAI / Anthropic / Cohere）
- **级别**：On-call Lead / Senior SRE，负责 LLM serving 的 production reliability
- **风格**：场景驱动，给出真实的 production incident 让候选人现场排查。关注思维过程而非最终答案。
- **偏好**：看重系统性排查方法、对 failure mode 的直觉、以及"第一反应"是否正确。

---

## Opening Question

> "我会给你 3 个真实的 production incident 场景。对于每个场景，请描述你的排查步骤、可能的 root cause、以及修复方案。我关注的是你的思维过程，不是最终答案。准备好了吗？"

---

## Scenario 1：LLM Serving Latency Spike

> **告警内容**：`[CRITICAL] LLM serving TPOT P99 从 45ms 飙升到 350ms，持续 15 分钟。影响 30% 的请求。`
> 
> **已知信息**：
> - 没有代码部署
> - 总 QPS 没有明显变化
> - GPU utilization 从 75% 上升到 95%
> - 只有部分 instance 受影响（3/8 个 instance）

### Candidate Expected Answer

**排查步骤**：

"我的排查思路是从外到内、从宏观到微观：

**Step 1（30s）：确认影响范围**
- 只有 3/8 instance 受影响 → 不是全局问题，是特定 instance 的问题
- 这 3 个 instance 有什么共同点？同一个 node？同一个 GPU 型号？同一个 rack？

**Step 2（2min）：检查受影响 instance 的 metrics**
- GPU memory usage：是否接近上限？→ 可能 KV cache 满了导致频繁 eviction
- Batch size：是否异常增大？→ 可能有 long-running request 占用资源
- Request queue depth：是否堆积？

**Step 3（5min）：检查 request 特征**
- 受影响 instance 上的请求有什么特殊？
- 检查 input length 分布：是否有异常长的 input？
- 检查 output length 分布：是否有请求生成了异常长的 output？

**Step 4（5min）：深入分析**
- 如果是 long request 导致：检查是否有用户发送了超长 input（如 paste 了整本书）
- 如果是 memory 问题：检查 KV cache utilization，是否有 memory leak
- 如果是 GPU 问题：检查 GPU clock frequency（是否 thermal throttle）

**可能的 Root Cause（按概率排序）**：

1. **最可能（60%）**：某些用户发送了异常长的请求（input 10K+ tokens），这些请求的 prefill 占用了大量 GPU compute，导致同 batch 的其他请求被阻塞。3 个 instance 恰好被 load balancer 分配到了这些长请求。

2. **次可能（25%）**：3 个 instance 所在的 GPU 出现 thermal throttling。GPU utilization 95% 但实际 clock 降低了，导致每个 step 变慢。

3. **较少见（15%）**：KV cache memory leak——某些请求完成后 KV cache 没有正确释放，导致可用 memory 减少，新请求被 swap 到 CPU memory。

**修复方案**：

- 短期（立即）：
  - 将受影响 instance 的流量切走（从 load balancer 摘除）
  - 重启受影响 instance（清除可能的 memory leak）
  
- 中期（本周）：
  - 添加 max_input_length 限制（如 8192 tokens）
  - 添加 per-request timeout（如 60s）
  - 实现 chunked prefill 避免 long prefill 阻塞

- 长期（本月）：
  - 实现 request-level isolation（long request 不影响 short request）
  - 添加 GPU temperature monitoring + 自动降频告警
  - 实现 KV cache memory 的 leak detection"

---

## Scenario 2：RAG 系统 Retrieval Quality 下降

> **告警内容**：`[WARNING] RAG retrieval recall@10 从 85% 降到 62%，过去 24 小时持续下降。`
> 
> **已知信息**：
> - 没有代码变更
> - 没有 embedding 模型更新
> - 文档数量从 50 万增长到 55 万（过去一周新增 5 万）
> - 向量数据库 CPU usage 从 40% 上升到 80%

### Candidate Expected Answer

**排查步骤**：

"Retrieval quality 下降 + 文档增长 + CPU 升高，这几个信号组合起来指向索引问题。

**Step 1（2min）：确认 recall 下降的模式**
- 是所有 query 都下降还是特定类型？
- 新增的 5 万文档是什么类型？和原有文档有什么不同？
- recall 是突然下降还是渐进下降？

**Step 2（5min）：检查向量索引状态**
- 索引类型是什么？（HNSW / IVF / Flat）
- 如果是 HNSW：
  - 新增文档是否正确插入了图结构？
  - ef_search 参数是否足够？（文档增多后可能需要增大）
  - 图的连通性是否退化？（大量插入后 HNSW 质量会下降）
- 如果是 IVF：
  - 聚类中心是否需要重新训练？（数据分布变化后旧聚类不准）
  - nprobe 参数是否需要调大？

**Step 3（5min）：检查新增文档**
- 新文档的 embedding 分布是否和原有文档一致？
- 是否有大量重复或低质量文档？（噪声增加）
- 新文档的 chunk 策略是否正确？（可能有格式解析错误）

**可能的 Root Cause（按概率排序）**：

1. **最可能（50%）**：HNSW 索引在大量增量插入后质量退化。HNSW 的增量插入不如全量构建的图质量好，5 万新文档的插入导致图结构变差，recall 下降。CPU 升高是因为搜索时需要遍历更多节点。

2. **次可能（30%）**：新增文档的内容分布与原有文档差异大（如新增了大量英文文档而原来是中文），embedding 空间中形成了新的 cluster，但 HNSW 的连接没有很好地覆盖这些新 cluster。

3. **较少见（20%）**：向量数据库的内存不足，部分索引被 swap 到磁盘，导致搜索变慢且不完整（timeout 导致返回不完整结果）。

**修复方案**：

- 短期（立即）：
  - 增大 ef_search / nprobe 参数（牺牲 latency 换 recall）
  - 如果 CPU 过高导致 timeout：增加副本分担负载

- 中期（本周）：
  - 全量重建索引（offline rebuild + atomic swap）
  - 评估新文档质量，过滤低质量 chunk

- 长期（本月）：
  - 建立索引质量监控：定期用 ground truth 测试 recall
  - 设置自动 rebuild 策略：每增加 10% 文档触发 rebuild
  - 考虑 hybrid search（BM25 + vector）提高鲁棒性"

---

## Scenario 3：GPU OOM in Production

> **告警内容**：`[CRITICAL] GPU OOM on instance-05. Process killed. 3 requests lost.`
> 
> **已知信息**：
> - 该 instance 运行 Llama-70B，TP=4
> - Memory usage 在 OOM 前 30 分钟内从 72GB 缓慢增长到 80GB
> - 正常运行时 memory 稳定在 70-72GB
> - 该 instance 已运行 7 天未重启
> - 其他 instance 没有 OOM

### Candidate Expected Answer

**排查步骤**：

"缓慢增长 + 单 instance + 长时间运行 → 高度怀疑 memory leak。

**Step 1（1min）：确认 OOM 的 memory 组成**
- 模型权重：固定 ~140GB / 4 GPUs = 35GB per GPU
- KV cache：动态，取决于 active requests
- Activation memory：临时，每个 forward pass 后释放
- 其他：CUDA context, NCCL buffers, framework overhead

正常 70GB = 35GB 权重 + 30GB KV cache + 5GB overhead
OOM 时 80GB = 35GB 权重 + 40GB KV cache + 5GB overhead
→ KV cache 多了 10GB，为什么？

**Step 2（5min）：检查 KV cache 管理**
- 是否有请求完成后 KV cache 没有释放？
- 检查 block manager 的 free block 数量趋势
- 是否有 prefix caching 导致 cache 不断积累？

**Step 3（5min）：检查请求模式**
- 过去 7 天是否有特殊的请求模式？
- 是否有大量使用相同 prefix 的请求（触发 prefix caching 积累）？
- 是否有请求被 stuck（永远不完成，KV cache 不释放）？

**可能的 Root Cause（按概率排序）**：

1. **最可能（40%）**：Prefix caching 的 eviction 策略有 bug。当 cache 满时应该 evict 最久未使用的 prefix，但某些 prefix 的引用计数没有正确减少，导致永远不会被 evict。

2. **次可能（30%）**：某些请求因为异常（如 client disconnect）没有正确完成，KV cache 没有释放。7 天内积累了足够多的 leaked blocks。

3. **较少见（20%）**：CUDA memory fragmentation。频繁的 allocate/free 导致虽然总 free memory 够，但没有连续的大块可用，最终 allocation 失败报 OOM。

4. **边缘情况（10%）**：Framework 的 memory pool 有 bug，某些 tensor 的引用被意外持有。

**修复方案**：

- 短期（立即）：
  - 重启 instance-05（清除 leaked memory）
  - 设置定期重启策略（每 3 天重启一次，作为 workaround）

- 中期（本周）：
  - 添加 memory usage 监控 + 告警（> 75GB 预警，> 78GB critical）
  - 添加 KV cache block 使用量的 metric（free blocks, used blocks, cached blocks）
  - 实现 memory leak detection：如果 free blocks 持续下降超过阈值，自动触发 GC

- 长期（本月）：
  - 排查 prefix caching 的引用计数逻辑
  - 添加 request timeout + 强制清理（超过 5 分钟的请求强制终止并释放资源）
  - 实现 memory 使用的 watermark 机制：高水位时停止接收新请求，等待 memory 释放"

---

## Pressure Follow-up（故意挑战候选人）

> "你描述的排查步骤很有条理，但这些都是理论。你实际处理过 production incident 吗？凌晨 3 点被叫醒，系统挂了，你的真实反应是什么？"

**期望应对**：
- 诚实承认：我没有 production on-call 经验，没有被凌晨叫醒过
- 但展示相关经验：
  1. 我在 vLLM-Ascend 开发中遇到过类似的 debugging 场景——推理结果不正确、性能不达预期
  2. 我的 RAG 系统有过 Research 链路状态流转 bug 的排查经验
  3. 我的排查方法论来自：开源社区的 issue 分析 + 系统性学习
- 展示正确的 mindset：
  - 第一反应：确认影响范围（多少用户受影响？有没有 workaround？）
  - 第二反应：快速止血（能不能重启？能不能切流量？）
  - 第三反应：根因分析（不急于修复，先理解问题）
- 承认需要成长：production debugging 是需要积累的能力，我需要在实际 on-call 中锻炼

---

## Debugging Scenario（额外场景）

> "你的 LLM serving 系统突然开始返回乱码。不是所有请求，大约 5% 的请求输出是乱码（random tokens）。其他 95% 正常。怎么排查？"

**排查思路**：

1. **特征分析**：
   - 5% 的请求 → 不是模型本身的问题（否则应该全部异常）
   - 乱码 = random tokens → 可能是 memory corruption 或 sampling bug

2. **检查 pattern**：
   - 乱码请求是否集中在某个 instance？→ 硬件问题
   - 乱码请求是否有共同特征（长 input？特定 token？）→ 触发条件
   - 乱码是从第一个 token 开始还是中间某个位置开始？→ 定位问题阶段

3. **可能的 Root Cause**：
   - **GPU memory corruption**：ECC uncorrectable error 导致权重或 KV cache 被破坏
     - 检查：`nvidia-smi -q -d ECC`
     - 如果有 uncorrectable error → 立即下线该 GPU
   - **KV cache 错误**：PagedAttention 的 block table 映射错误，读到了其他请求的 KV cache
     - 检查：对比正常和异常请求的 block table
   - **Quantization 精度问题**：某些 weight 的量化误差在特定 input 下被放大
     - 检查：用 FP16 重新推理相同 input，看是否正常
   - **Race condition**：多线程/多 stream 的 memory 访问冲突
     - 检查：单线程模式下是否能复现

4. **修复**：
   - 短期：识别并下线有问题的 GPU/instance
   - 中期：添加 output validation（检测乱码并重试）
   - 长期：添加 ECC error 监控 + 自动下线

---

## System Design Extension

> "设计一个 LLM serving 系统的 observability stack。需要覆盖：metrics, logging, tracing, alerting。"

**设计要点**：

**Metrics（Prometheus + Grafana）**：
```
# Request-level
llm_request_ttft_seconds{model, instance, priority}
llm_request_tpot_seconds{model, instance}
llm_request_total_tokens{model, direction=input|output}
llm_request_queue_time_seconds{model}

# System-level
gpu_utilization_percent{instance, gpu_id}
gpu_memory_used_bytes{instance, gpu_id}
gpu_temperature_celsius{instance, gpu_id}
kv_cache_utilization_percent{instance}
batch_size_current{instance}

# Business-level
llm_request_total{model, status=success|error|timeout}
llm_cost_per_request_dollars{model, tenant}
```

**Logging（结构化日志）**：
```json
{
  "timestamp": "2024-01-01T00:00:00Z",
  "request_id": "req-123",
  "model": "llama-70b",
  "input_tokens": 1024,
  "output_tokens": 256,
  "ttft_ms": 150,
  "tpot_ms": 35,
  "total_latency_ms": 9100,
  "instance": "instance-03",
  "gpu_ids": [0, 1, 2, 3],
  "kv_cache_blocks_used": 128,
  "status": "success"
}
```

**Tracing（OpenTelemetry）**：
```
Span: api_gateway (5ms)
  └── Span: model_router (1ms)
      └── Span: queue_wait (50ms)
          └── Span: prefill (80ms)
              └── Span: decode (8900ms)
                  ├── Span: kv_cache_alloc (0.5ms)
                  └── Span: attention_compute (8899ms)
```

**Alerting Rules**：
```yaml
- alert: HighTTFT
  expr: histogram_quantile(0.99, llm_request_ttft_seconds) > 0.5
  for: 2m
  severity: warning

- alert: CriticalTTFT
  expr: histogram_quantile(0.99, llm_request_ttft_seconds) > 1.0
  for: 1m
  severity: critical

- alert: GPUMemoryLeak
  expr: increase(gpu_memory_used_bytes[1h]) > 2e9  # 2GB/hour increase
  severity: warning

- alert: KVCacheExhaustion
  expr: kv_cache_utilization_percent > 95
  for: 5m
  severity: critical
```

---

## Hire/No-Hire Evaluation

| 维度 | 权重 | 评分标准 |
|------|------|---------|
| 排查方法论 | 25% | 是否有系统性的排查步骤 |
| Root Cause 直觉 | 25% | 能否快速缩小范围到最可能的原因 |
| 修复方案 | 20% | 短期止血 + 长期根治的思维 |
| Production 经验 | 20% | 是否有实际 incident handling 经验 |
| 沟通能力 | 10% | 能否清晰描述排查过程 |

---

## Scorecard

| 维度 | 候选人预期得分 | 说明 |
|------|--------------|------|
| 排查方法论 | 5/10 | 有基本的系统性思维，但缺乏实战磨练 |
| Root Cause 直觉 | 4/10 | 能列出可能原因但缺乏"第一直觉" |
| 修复方案 | 5/10 | 能给出合理方案但缺乏 production 约束考虑 |
| Production 经验 | 2/10 | 无 on-call 经验，无 incident handling 经验 |
| 沟通能力 | 6/10 | 表达清晰，能结构化描述 |
| **总分** | **4.2/10** | **No Hire for SRE/On-call role, Lean Hire for dev role with on-call rotation** |

### 决策依据
- **No Hire 原因**：Production debugging 岗位需要实战经验，候选人只有理论
- **候选人亮点**：排查思路有条理，能从 metrics 推断 root cause，有学习意愿
- **建议**：适合有 senior mentor 带的团队，在 on-call rotation 中积累经验
- **成长路径**：需要 3-6 个月的 production 环境浸泡才能独立处理 incident
