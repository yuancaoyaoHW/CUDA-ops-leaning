# 分布式推理系统设计
## 学习目标
1. 能解释 Prefill-Decode 分离的动机、架构和 KV transfer 成本。
2. 能说明 Mooncake 的 KV-cache-centric 思路，以及它和传统 GPU-centric 调度的差异。
3. 能设计 KV routing，在 locality 和 load balance 之间做可量化权衡。
4. 能覆盖 checkpoint、request migration、redundancy、多模型 registry/routing/canary。
5. 能处理 long context 的 ring attention、sequence parallel、context caching。
6. 能从 spot、mixed precision、batching efficiency 角度做成本优化。
## 前置知识
- LLM inference 有 prefill 和 decode 两种阶段。Prefill 处理整段输入，算力密集；decode 每次追加一个 token，带宽和调度敏感。
- 分布式推理不只是模型分片，还包括请求路由、KV cache 放置、跨节点通信、故障迁移和多模型控制面。
- RDMA/InfiniBand 适合大块低延迟传输，NVLink/NVSwitch 适合同节点多 GPU 通信。
- 长上下文会让 attention、KV cache 和网络传输都变成一等瓶颈。
## 核心内容
### 1. PD 分离动机
Prefill 和 decode 的资源特征不同：
| 维度 | Prefill | Decode |
|---|---|---|
| 输入形态 | prompt tokens 批量处理 | 每序列每步 1 token |
| 主要瓶颈 | GEMM/attention compute | HBM bandwidth、KV lookup、调度 |
| SLA | TTFT | TPOT |
| 批处理偏好 | 大 tokens batch | 大 active sequence batch |
| 扩缩容指标 | waiting prefill tokens | active sequences、output tokens/s |
混部问题：
- 长 prompt prefill 会阻塞 decode step，导致 TPOT 抖动。
- Prefill 需要大 chunk 提高吞吐，decode 需要稳定小步推进。
- 两阶段资源比例随流量变化。RAG 高峰、长上下文高峰、短问答高峰的最优比例不同。
PD 分离目标：让 prefill pool 和 decode pool 独立扩缩容、独立调度、独立优化硬件。
### 2. PD 分离架构
```text
Client
  -> Gateway
  -> Request Router
  -> Prefill Scheduler
     -> Prefill Workers
        - compute optimized
        - chunked prefill
        - produce KV blocks
  -> KV Transfer Plane
     - RDMA, NVLink, PCIe, shared memory, object/KV store fallback
  -> Decode Scheduler
     -> Decode Workers
        - bandwidth optimized
        - continuous batching
        - PagedAttention block manager
  -> Streaming Response
Control Plane:
  model registry
  worker registry
  KV metadata service
  placement optimizer
  autoscaler
  metrics and tracing
```
请求生命周期：
1. Router 根据 model、prompt length、priority、adapter、region 选择 prefill pool。
2. Prefill worker 计算 prompt KV，并把每层 KV block 写入目标 decode worker 或中间 KV store。
3. KV metadata service 记录 `request_id -> decode_worker -> block_table`。
4. Decode scheduler 将请求加入 active set，开始逐 token 生成。
5. 请求结束、取消或超时后释放 KV blocks，并更新 cache/refcount。
### 3. KV transfer 成本
KV transfer 大小：
```text
kv_transfer_bytes =
  prompt_tokens * 2 * layers * kv_heads * head_dim * bytes
```
以 70B GQA BF16 估算：
```text
kv_per_token ~= 320 KiB
prompt 2048 tokens ~= 640 MiB
prompt 8192 tokens ~= 2.5 GiB
```
网络传输时间：
```text
transfer_time = kv_bytes / effective_bandwidth + setup_latency
```
示例：
| 链路 | 有效带宽估算 | 640 MiB KV | 2.5 GiB KV |
|---|---:|---:|---:|
| 100 Gbps RDMA | ~12 GB/s | ~53 ms | ~213 ms |
| 200 Gbps RDMA | ~24 GB/s | ~27 ms | ~107 ms |
| NVLink local | 数百 GB/s | 数 ms | 十几 ms |
优化：
- layer-by-layer streaming: 第 i 层 KV 生成后立即传输，隐藏部分延迟。
- FP8 KV transfer: 带宽减半，但要验证质量。
- decode placement before prefill: 先选定 decode worker，让 prefill 直接写目标。
- avoid transfer on prefix hit: 如果 decode worker 已有 prefix KV，就路由到本地。
### 4. Mooncake: KV-cache-centric 架构
Mooncake 的核心思想可以概括为：把 KV cache 当作调度中心，而不是只把 GPU 当作计算槽位。
传统 GPU-centric：
```text
find available GPU -> run prefill/decode -> KV follows request
```
KV-cache-centric：
```text
find or place KV near future decode -> schedule compute around KV locality
```
关键设计点：
- KV metadata 是一等对象：位置、大小、引用计数、生命周期、冷热度。
- 调度考虑 KV reuse：相同 system prompt、RAG template、多轮对话前缀优先路由到已有 KV 的 worker。
- Prefill 和 decode 解耦：prefill 生产 KV，decode 消费 KV，KV transfer plane 负责移动。
- 全局优化目标不是单 worker 利用率，而是 `TTFT + TPOT + transfer_cost + eviction_cost`。
一个简化打分函数：
```text
score(worker, request) =
  w_load * normalized_queue_delay(worker)
+ w_transfer * estimated_kv_transfer_ms(worker, request)
+ w_cache * cache_miss_penalty(worker, request)
+ w_sla * sla_risk(request, worker)
+ w_frag * kv_fragmentation_penalty(worker)
```
选择 score 最低的 worker，而不是简单 least-connections。
### 5. KV Routing: locality vs load balance
KV locality 策略：
- 优点：减少 KV transfer，提升 prefix cache 命中，降低 TTFT。
- 缺点：热门前缀会造成热点 worker，TPOT 变差。
Load balance 策略：
- 优点：平摊 active sequences 和 GPU 负载。
- 缺点：可能频繁跨节点搬 KV，TTFT 和网络成本上升。
混合策略：
```text
candidate_workers = workers_with_prefix_cache(request.prefix_hash)
if candidate_workers not empty:
  choose by score(locality, queue, kv_free, tpot_risk)
else:
  choose by score(queue, kv_free, network_distance)
if best_local.queue_wait - best_global.queue_wait > transfer_savings:
  route_global()
else:
  route_local()
```
要监控的指标：
```text
kv_local_hit_rate
kv_remote_transfer_bytes
kv_remote_transfer_latency_p95
prefix_cache_hit_rate
worker_decode_queue_wait_p95
worker_active_sequences
hot_prefix_topk
eviction_rate_by_reason
```
经验：当 prompt 很长或 prefix 可复用时，locality 权重高；当 prompt 短且 decode 长时，load balance 更重要。
### 6. Checkpoint、request migration、redundancy
请求状态包括：
```text
request_state =
  prompt token ids
  generated token ids
  sampling rng state
  KV block table
  adapter/model version
  stream offset
  client delivery ack
```
迁移方式：
| 方式 | 代价 | 适用 |
|---|---|---|
| recompute prefill | 重新算 prompt + generated prefix | 短上下文、故障恢复 |
| KV copy migration | 复制 KV blocks 到目标 worker | 中长上下文、计划 drain |
| periodic checkpoint | 定期保存 KV metadata 和采样状态 | 高价值长会话 |
| redundant decode | 两个 worker 冗余生成 | 极高 SLA，成本翻倍 |
Drain 伪代码：
```text
mark_worker_draining(worker)
stop_admitting_new_requests(worker)
for req in active_requests(worker):
  if req.remaining_tokens_small:
    let_finish(req)
  elif can_copy_kv(req):
    migrate_kv_and_resume(req)
  else:
    return_retryable_partial(req)
```
注意：decode 的完全透明迁移很难，因为流式输出、采样随机性和 KV cache 必须一致。生产系统要明确语义：partial response、retry token、幂等 request id、是否允许重算。
### 7. 多模型部署: registry、routing、canary
Model registry 需要记录：
```text
model_id
version
weight_uri_or_local_cache_key
tokenizer_version
quantization
max_context
parallel_config
adapter_compatibility
health_state
traffic_policy
```
Routing 输入：
- model_id/version 或 alias，例如 `chat-prod -> llama-70b-v42`。
- tenant、quota、priority、region、data residency。
- prompt length、expected output length、adapter/LoRA。
- canary 百分比、denylist、fallback policy。
Canary 流程：
1. 新版本只加载到少量 replica。
2. shadow traffic 或 1% live traffic。
3. 比较 TTFT、TPOT、error rate、output length、user feedback、quality eval。
4. 逐步扩大到 5%、25%、50%、100%。
5. 任一硬指标退化超过阈值自动回滚 alias。
多模型资源策略：
- 小模型合并到同一 worker 池，提高利用率。
- 大模型独占 replica，避免权重切换造成冷启动。
- 热模型保留 warm pool，冷模型允许较长 TTFT 或排队。
- Adapter 多租户要隔离 cache key，避免串租户。
### 8. Long context: ring attention、sequence parallel、context caching
长上下文挑战：
```text
attention_compute_prefill ~= O(seq_len^2)
kv_capacity ~= O(seq_len)
kv_transfer ~= O(seq_len)
```
Ring attention：
- 将长序列分片到多个 GPU。
- 每个 GPU 只持有部分 K/V，并通过 ring 传递 block 计算 attention。
- 优点：支持超长上下文，单 GPU 内存压力下降。
- 缺点：通信步数增加，延迟和实现复杂度上升。
Sequence parallel：
- 按 sequence 维度切分 activation 或 attention 输入。
- 常与 TP 结合，用于降低单卡 activation 和 attention workspace。
- 对 decode 帮助有限，因为 decode 每步 query 很短，但对 prefill 长上下文有价值。
Context caching：
- 对稳定长前缀做 KV cache，例如 system prompt、文档集合、代码仓库上下文。
- 使用 prefix tree 或 radix tree 管理共享前缀。
- 热 context 可 pin 在 HBM，温 context 放 CPU/NVMe，冷 context 失效。
长上下文调度建议：
```text
if prompt_tokens > long_context_threshold:
  route_to_long_context_pool()
  use_chunked_prefill()
  reserve_decode_budget()
  lower_priority_if_batch_job()
  enforce_max_new_tokens()
```
### 9. 成本优化: spot、mixed precision、batching efficiency
Spot/preemptible：
- 适合 prefill pool、batch/offline、可重试任务。
- 不适合无 checkpoint 的高优先级 decode。
- 需要 termination notice 处理：drain、KV migration、停止接新请求。
Mixed precision：
| 对象 | 可选精度 | 风险 |
|---|---|---|
| weights | BF16、FP8、INT8、INT4 | 质量、kernel 支持、校准成本 |
| KV cache | BF16、FP8、INT8 | 长上下文质量、dequant overhead |
| activation | BF16、FP8 | 数值稳定、算子覆盖 |
Batching efficiency：
```text
batching_efficiency =
  useful_tokens_processed / max_possible_tokens_under_sla
```
提升方式：
- 按 prompt length bucket，减少 padding 和极端长 prompt 干扰。
- decode 使用 continuous batching，保持 active sequences。
- prefill 使用 max batched tokens，而不是固定 request batch size。
- 限制 best_of、beam、max_new_tokens，防止单请求放大资源。
- 针对租户计费 input tokens、output tokens、reserved KV time。
成本指标：
```text
cost_per_1k_output_tokens
gpu_seconds_per_request
kv_gb_seconds_per_request
tokens_per_gpu_second
effective_batch_size_p50/p95
```
### 10. 分布式排障指标
TTFT 高：
- prefill queue wait 高：prefill pool 不够或长 prompt 混入。
- KV transfer latency 高：网络拥塞、路由远、KV 太大。
- prefix cache hit 低：模板变化、hash key 错误、版本切换。
TPOT 高：
- decode active sequences 过高。
- HBM bandwidth 接近上限。
- KV fragmentation 或 eviction 上升。
- NCCL/RDMA latency 抖动。
局部热点：
- hot prefix 或 hot tenant 集中到少数 worker。
- consistent hash 缺少虚拟节点或负载反馈。
- canary 版本 replica 太少。
## 完整的问答/题目
### 题目
设计一个分布式 LLM 推理系统，要求支持 prefill/decode 分离、多模型部署、长上下文、KV cache 复用和故障迁移。请说明架构、KV transfer、KV routing、Mooncake 类 KV-cache-centric 调度、多模型 canary、long context 优化和成本控制。
### 参考回答结构
1. 先说明 prefill compute-bound、decode memory-bound，混部会让 TTFT/TPOT 相互干扰。
2. 设计 Router、Prefill Pool、KV Transfer Plane、Decode Pool、KV Metadata Service、Model Registry。
3. 用 KV 公式估算 transfer bytes，并比较 RDMA/NVLink 的延迟。
4. 说明 Mooncake 的核心是围绕 KV cache 放置和复用做调度。
5. KV routing 用 score 函数同时考虑 locality、queue、KV free blocks、transfer cost。
6. 故障恢复区分重算、KV copy migration、checkpoint、redundant decode。
7. 多模型用 registry 管理 version、tokenizer、quant、parallel config，并用 alias/canary 控制流量。
8. 长上下文用 ring attention、sequence parallel、context caching、独立队列。
9. 成本优化用 spot、mixed precision、batching efficiency 和 token/KV 计费指标。
### 可写出的伪代码
```text
def route_request(req):
    candidates = registry.ready_workers(req.model, req.version)
    scored = []
    for w in candidates:
        score = (
            queue_cost(w)
          + transfer_cost(req.kv_meta, w)
          + cache_miss_cost(req.prefix_hash, w)
          + kv_pressure_cost(w)
          + sla_risk(req, w)
        )
        scored.append((score, w))
    return min(scored).worker
```
### 可写出的容量估算
```text
kv_bytes = prompt_tokens * 2 * layers * kv_heads * head_dim * bytes
transfer_ms = kv_bytes / effective_bandwidth * 1000
decode_slots = qps * avg_output_tokens * tpot
```
## 追问方向与深入点
- 为什么 PD 分离后 TTFT 可能反而变差？因为多了 KV transfer 和跨池排队，需要 locality 和 streaming transfer。
- 如何避免热门 prefix 造成热点？使用负载反馈、replication、cache admission、虚拟节点和上限。
- KV metadata service 挂了怎么办？worker 本地保留 block table，控制面降级只影响新请求路由。
- 长上下文请求是否应该和普通请求混跑？通常不应该，至少要单独队列和 token budget。
- 什么时候 recompute 比 KV migration 更划算？短 prompt、低优先级、网络拥塞或目标 worker 很近但 prefill 很快时。
- Canary 为什么要比较 tokenizer version？tokenizer 改动会改变 token ids，影响 cache、质量和输出长度。
- Spot 节点能不能跑 decode？可以跑低优先级 decode，但高 SLA decode 需要 checkpoint 或快速迁移语义。
- KV cache 量化如何验收？看长上下文 retrieval、困惑度、业务 eval、线上 A/B 和异常 token rate。
## 评分标准
| 维度 | 满分 | 要点 |
|---|---:|---|
| PD 分离 | 15 | 动机、架构、独立扩缩容、TTFT/TPOT 影响 |
| KV transfer | 15 | 公式、带宽估算、streaming、压缩 |
| Mooncake/KV-centric | 15 | KV metadata、调度目标、cache reuse |
| KV routing | 15 | locality/load balance tradeoff 和 score 函数 |
| 容错迁移 | 10 | checkpoint、migration、redundancy、drain 语义 |
| 多模型 | 10 | registry、routing、canary、alias、adapter 隔离 |
| 长上下文 | 10 | ring attention、sequence parallel、context caching |
| 成本优化 | 10 | spot、mixed precision、batching efficiency、成本指标 |
优秀回答应把分布式问题和 LLM 特有的 KV cache 生命周期联系起来，而不是只讲普通微服务负载均衡。
## 复习卡片 15 张
1. Q: PD 分离的核心动机是什么？ A: Prefill 和 decode 资源特征不同，分离后可独立调度和扩缩容。
2. Q: Prefill 主要优化什么 SLA？ A: TTFT。
3. Q: Decode 主要优化什么 SLA？ A: TPOT。
4. Q: KV transfer bytes 公式是什么？ A: `tokens * 2 * layers * kv_heads * head_dim * bytes`。
5. Q: Mooncake 类架构的关键词是什么？ A: KV-cache-centric scheduling。
6. Q: KV locality 的好处是什么？ A: 减少 transfer，提高 prefix/cache 命中，降低 TTFT。
7. Q: KV locality 的风险是什么？ A: 热点 worker，decode queue 上升，TPOT 变差。
8. Q: Load balance 的风险是什么？ A: KV 远程传输增加，TTFT 和网络成本上升。
9. Q: 请求迁移至少要保存哪些状态？ A: token ids、generated tokens、sampling state、KV block table、model version、stream offset。
10. Q: Decode crash 为什么难恢复？ A: 本地 KV 和采样状态丢失时无法直接继续。
11. Q: Model registry 至少记录什么？ A: model/version、tokenizer、quant、max context、parallel config、traffic policy。
12. Q: Canary 回滚依据有哪些？ A: TTFT、TPOT、error rate、质量 eval、输出长度、用户反馈。
13. Q: Ring attention 解决什么？ A: 长上下文 attention 和 KV/activation 单卡压力。
14. Q: Spot 更适合哪个池？ A: 可重试的 prefill 或 batch/offline，谨慎用于高 SLA decode。
15. Q: 成本指标除 GPU utilization 外还看什么？ A: cost per 1k tokens、tokens/GPU-second、KV GB-seconds、batching efficiency。
