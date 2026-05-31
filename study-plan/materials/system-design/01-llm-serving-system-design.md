# 1000 QPS LLM Serving 系统设计
## 学习目标
1. 能把 1000 QPS 从业务目标拆成 TTFT、TPOT、并发、token 吞吐和 GPU 数量。
2. 能用公式估算 model weights、KV cache、activation、通信 buffer 和 runtime overhead。
3. 能解释 A100、H100、H200 在 LLM serving 中的差异，以及 NVLink 对 TP/PP 的影响。
4. 能设计 TP/PP/EP、continuous batching、chunked prefill、priority queue、PagedAttention、prefix caching。
5. 能设计 autoscaling、冷启动、容错、Prometheus metrics、alerting 和 SLA dashboard。
## 前置知识
- Transformer 推理分为 prefill 和 decode。Prefill 一次处理输入序列，主要受算力和 attention 复杂度影响；decode 每步生成 1 个 token，主要受权重读取和 KV cache 访问影响。
- TTFT 指从请求进入系统到第一个 token 返回的时间；TPOT 指生成阶段每个输出 token 的平均或分位耗时。
- TP 是 tensor parallelism，PP 是 pipeline parallelism，EP 是 expert parallelism，DP 是 data parallelism。
- GQA/MQA 会减少 KV head 数，显著降低 KV cache。
- GPU serving 的瓶颈通常不是单个请求 latency，而是 batch 调度、KV cache 容量、HBM 带宽和队列等待。
## 核心内容
### 1. 需求分析
目标场景：为一个聊天类 LLM 服务设计 1000 QPS 峰值容量，模型以 70B dense 或 MoE 等价容量为主。
建议先把 SLA 写成可计算约束：
| 指标 | 目标 | 设计含义 |
|---|---:|---|
| 峰值 QPS | 1000 req/s | 入口、调度、KV 容量、decode slots 都要按峰值或可降级峰值设计 |
| TTFT P95/P99 | < 500 ms | prefill 排队 + prefill compute + KV 分配 + 首 token decode |
| TPOT P95/P99 | < 50 ms/token | decode batch 的每步延迟不能长期超过 50 ms |
| 输入长度 | avg 512, P95 2048, max 8192 | 影响 TTFT、KV cache 和 prefix cache 命中 |
| 输出长度 | avg 256, P95 1024, max 2048 | 影响 decode 并发驻留时间 |
| 模型规模 | 70B BF16 或 INT4/FP8 量化 | 决定权重显存、TP 数和吞吐 |
关键估算：
```text
avg_decode_time = avg_output_tokens * TPOT
                = 256 * 50 ms
                = 12.8 s
decode_concurrency ~= QPS * avg_decode_time
                   = 1000 * 12.8
                   = 12800 active sequences
decode_token_throughput = QPS * avg_output_tokens
                        = 256000 output tokens/s
```
注意：1000 QPS 不等于同时跑 1000 条序列。decode 是长驻留工作负载，真正要容量化的是 active sequence 数、tokens/s 和 KV cache。
### 2. 显存规划
总显存公式：
```text
VRAM_total =
  model_weights
+ kv_cache
+ activation_peak
+ scheduler_blocks
+ communication_buffers
+ framework_overhead
+ fragmentation_margin
```
模型权重：
```text
model_weights = parameter_count * bytes_per_param
70B BF16 ~= 70e9 * 2 bytes = 140 GB
70B FP8  ~= 70e9 * 1 byte  = 70 GB
70B INT4 ~= 70e9 * 0.5 byte + scales/zeros ~= 38-45 GB
```
KV cache 单 token 公式：
```text
kv_bytes_per_token =
  2 * num_layers * num_kv_heads * head_dim * bytes_per_element
```
以 LLaMA-70B 类配置估算：80 层，GQA 的 KV heads=8，head_dim=128，BF16。
```text
kv_bytes_per_token = 2 * 80 * 8 * 128 * 2
                   = 327680 bytes
                   ~= 320 KiB/token
kv_per_request_avg = (avg_input_tokens + avg_generated_tokens) * 320 KiB
                   = (512 + 256) * 320 KiB
                   ~= 240 MiB
kv_for_12800_active ~= 12800 * 240 MiB
                    ~= 3.0 TiB
```
这说明 1000 QPS 不能靠少量 GPU 硬扛，需要分片、队列控制、prefix caching、KV 量化和 admission control。
Activation 估算：
```text
activation_peak ~= micro_batch * seq_len * hidden_size * bytes * layer_factor
```
在 inference 中 activation 通常可以逐层释放，峰值小于训练，但 prefill 的长 prompt、大 batch 和 attention workspace 仍会造成尖峰。实际容量规划中可预留 5%-15% HBM 给 activation/workspace，另留 5%-10% 给 runtime 和碎片。
### 3. 硬件选型
| GPU | HBM | HBM 带宽 | NVLink | 适合点 |
|---|---:|---:|---:|---|
| A100 80GB | 80 GB | ~2.0 TB/s | ~600 GB/s | 成本较低，适合学习、INT4、较小 batch |
| H100 80GB | 80 GB | ~3.35 TB/s | ~900 GB/s | decode 带宽更强，FP8 支持更好 |
| H200 141GB | 141 GB | ~4.8 TB/s | ~900 GB/s | KV cache 容量更大，长上下文更友好 |
选型判断：
- Decode 常见瓶颈是 HBM bandwidth，因为每步要读取权重和 KV cache。H100/H200 通常比 A100 更适合高 QPS decode。
- 70B BF16 单卡放不下，至少需要 TP=2，生产上常用 TP=4 或 TP=8 来留出 KV 空间。
- NVLink 对 TP 关键。TP 每层需要 all-reduce 或 reduce-scatter/all-gather，跨 PCIe 会明显拉高 TPOT。
- H200 的价值主要是容量和带宽，能减少 KV eviction、CPU swap 和跨节点迁移。
### 4. 并行配置: TP/PP/EP
Dense 70B 推荐起点：
```text
单节点 8xH100:
  TP=8, PP=1, DP=N
  优点: 通信在 NVLink 内，TTFT/TPOT 稳定
  缺点: 每 replica 占 8 卡，弹性粒度粗
双节点 16xH100:
  TP=8, PP=2, DP=N
  优点: 权重和 activation 压力更小
  缺点: pipeline bubble、跨节点通信、故障域更大
4xH200:
  TP=4, PP=1, DP=N
  优点: 单 replica 卡数少，KV 空间大
  缺点: TP=4 下单卡权重更大，对量化和 KV 管理要求高
```
MoE 模型增加 EP：
```text
EP = experts sharded across GPUs
TP = each expert internal tensor split
DP = replicas for traffic
```
EP tradeoff：
- 优点：每 token 只激活 top-k experts，FLOPs 低于 dense。
- 缺点：expert routing 会引入 all-to-all，负载不均会让热点 expert 拖慢 batch。
- 设计要求：监控 per-expert token count、drop rate、all-to-all latency。
### 5. 调度: continuous batching、chunked prefill、priority queue
Continuous batching 将 decode 过程拆成 token step，每个 step 从等待队列补入新请求，而不是等一个 batch 全部结束。
核心循环伪代码：
```text
while server_running:
  finished = collect_finished_sequences()
  release_kv_blocks(finished)
  new_prefill = admit_prefill_requests(
    queue,
    kv_free_blocks,
    ttft_budget,
    priority_policy
  )
  decode_batch = active_sequences.select(
    max_tokens_per_step,
    fairness_window,
    priority_policy
  )
  run_chunked_prefill(new_prefill, chunk_size)
  run_one_decode_step(decode_batch)
  stream_tokens_to_clients()
```
Chunked prefill 把长 prompt 拆成多个 token chunk，避免一个 32K prompt 长时间占住 GPU。
```text
TTFT = queue_wait + sum(prefill_chunks_before_first_decode) + first_decode
```
Tradeoff：
- chunk 太大：prefill 吞吐好，但 decode 被阻塞，TPOT 抖动。
- chunk 太小：调度更公平，但 kernel launch、attention overhead 和 cache locality 变差。
- 经验做法：按 `max_num_batched_tokens` 限制一个 step 的 prefill tokens，并给 decode 保留固定 token budget。
Priority queue：
- interactive chat: 高优先级，限制 max output，保证 TTFT。
- batch/offline: 低优先级，可被抢占或延后。
- long context: 单独队列或 admission penalty，避免挤占普通请求。
- retry/migration: 带 request id，避免重复计费和重复输出。
### 6. KV cache: PagedAttention、prefix caching、KV 量化
PagedAttention 将 KV cache 切成固定大小 block，逻辑序列到物理 block 用 block table 映射。
```text
logical token positions -> block ids -> physical GPU pages
```
收益：
- 减少连续大块显存分配失败。
- 支持请求结束后回收部分 block。
- 支持 beam search、prefix sharing 和 copy-on-write。
Prefix caching：
```text
prefix_key = hash(model_id, tokenizer_version, sampling_invariant_prefix)
cache_hit if prefix_key exists and kv_blocks alive
```
适用：系统提示词、RAG 模板、few-shot 示例、多轮对话公共前缀。
风险：
- sampling 参数通常不影响 prefill KV，但 tokenizer、模型版本、LoRA adapter、prompt bytes 会影响。
- 缓存命中提升 TTFT，但会增加 KV 常驻压力，需要 LRU/LFU + TTL。
KV cache 量化：
```text
BF16 KV -> FP8 KV: capacity x2, bandwidth x2 theoretical
BF16 KV -> INT8 KV: capacity x2, more dequant overhead
```
Tradeoff：长上下文更受益；短上下文可能被 dequant overhead 抵消。必须用 perplexity、exact-match、long-context retrieval 和线上 A/B 验证质量。
### 7. Autoscaling 与冷启动
扩容触发信号不要只看 GPU utilization。LLM serving 更应该看：
```text
queue_wait_p95
ttft_p95 / ttft_sla
tpot_p95 / tpot_sla
kv_cache_used_blocks / total_blocks
decode_active_sequences
prefill_waiting_tokens
```
Replica 级容量估算：
```text
required_replicas =
  ceil(target_output_tokens_per_sec / measured_tokens_per_sec_per_replica)
```
冷启动路径：
1. 拉起容器和 runtime。
2. 加载 tokenizer、config、engine。
3. 权重从本地 NVMe 或对象存储加载到 CPU。
4. 权重搬到 GPU 并建立 CUDA graph / kernel autotune。
5. warmup 常见 batch 和 seq_len。
6. health ready 后再接流量。
优化：
- 镜像内置 runtime，不内置大模型权重时使用节点本地缓存。
- 每个 AZ 保留 warm pool。
- 扩容先 shadow warmup，再加入路由。
- scale down 前 drain active sequences，超时后迁移或返回可重试错误。
### 8. 容错设计
故障类型：
- 单请求失败：OOM、timeout、client disconnect、bad prompt。
- 单 worker 失败：CUDA error、进程崩溃、NCCL hang。
- 单节点失败：GPU reset、网络故障、电源故障。
- 控制面失败：registry、scheduler、metrics 后端不可用。
处理策略：
| 故障 | 策略 | 注意 |
|---|---|---|
| prefill worker crash | 请求可重试到其他 worker | 确保 request id 幂等 |
| decode worker crash | 已生成 token 可返回 partial 或重试 | 不保存 KV 时无法无缝恢复 |
| KV block OOM | admission control 或降级 max tokens | 不要等 CUDA OOM 才失败 |
| NCCL timeout | kill replica 并重建 | 进程内恢复通常不可靠 |
| metrics backend down | 本地 ring buffer + 降级告警 | 服务不应依赖 metrics 可用性 |
Admission control：
```text
if kv_required_blocks > kv_free_blocks * safety_ratio:
  reject_or_queue(429, retry_after)
if estimated_ttft > sla_budget:
  route_to_less_loaded_pool_or_degrade()
```
### 9. Prometheus metrics、alerting、SLA dashboard
核心 metrics：
```text
llm_requests_total{model,route,status}
llm_request_duration_seconds_bucket{model}
llm_ttft_seconds_bucket{model,priority}
llm_tpot_seconds_bucket{model,priority}
llm_tokens_total{model,type=input|output}
llm_queue_wait_seconds_bucket{queue}
llm_prefill_tokens_waiting{model}
llm_decode_active_sequences{model}
llm_kv_blocks_used{model,gpu}
llm_kv_blocks_total{model,gpu}
llm_prefix_cache_hit_total{model}
llm_prefix_cache_lookup_total{model}
llm_gpu_hbm_used_bytes{gpu}
llm_gpu_sm_utilization_ratio{gpu}
llm_gpu_hbm_bandwidth_ratio{gpu}
llm_worker_restarts_total{reason}
```
告警示例：
- TTFT P99 > 500 ms 持续 5 分钟：page。
- TPOT P99 > 50 ms 持续 5 分钟且 queue_wait 不高：检查 decode 带宽、batch size、NCCL。
- KV used blocks > 90% 持续 3 分钟：扩容或启用更激进 eviction。
- prefix cache hit rate 突降：检查 prompt 模板、tokenizer 版本、cache key。
- worker restart rate 上升：检查 OOM、CUDA error、驱动、节点健康。
SLA dashboard 布局：
1. 顶部：QPS、error rate、TTFT P50/P95/P99、TPOT P50/P95/P99。
2. 容量：active sequences、input/output tokens/s、queue depth、waiting tokens。
3. GPU：SM util、HBM bandwidth、HBM used、KV blocks used、NCCL latency。
4. 调度：batch size、batched tokens、chunked prefill tokens、priority queue wait。
5. 质量与缓存：prefix hit rate、KV quant mode、eviction rate。
### 10. 完整文字架构图
```text
Client
  -> API Gateway
     - auth, quota, rate limit, request id
  -> Global Load Balancer
     - region/AZ selection, failover
  -> Model Router
     - model_id, version, adapter, priority, context length
  -> Admission Controller
     - estimate tokens, KV blocks, SLA feasibility
  -> Scheduler
     - continuous batching
     - chunked prefill
     - priority queue
     - prefix cache lookup
  -> Inference Replica Pool
     -> Prefill/Decode Workers
        - TP/PP/EP group
        - CUDA graphs
        - PagedAttention block manager
        - KV cache allocator
        - tokenizer sidecar or service
  -> Streaming Response
     - SSE/gRPC stream
     - partial output, cancellation, timeout
Control Plane:
  Model Registry -> rollout/canary/version pinning
  Autoscaler -> warm pool/scale up/drain
  Metrics -> Prometheus -> Alertmanager -> SLA dashboard
  Logs/Traces -> request timeline, kernel stage, queue events
  Failure Manager -> health check, replica quarantine, retry policy
```
## 完整的问答/题目
### 题目
设计一个服务，支持 70B 级聊天模型，峰值 1000 QPS，TTFT P99 < 500 ms，TPOT P99 < 50 ms。输入平均 512 tokens，输出平均 256 tokens，最长上下文 8192 tokens。要求说明容量估算、显存规划、硬件和并行策略、调度、KV cache、扩缩容、容错和监控。
### 参考回答结构
1. 先澄清：模型 dense/MoE、量化策略、输入输出分布、SLA 分位、是否多租户、是否允许降级。
2. 用 `QPS * output_tokens * TPOT` 得到 decode 并发约 12800。
3. 用 KV 公式估算 70B GQA BF16 约 320 KiB/token，平均请求约 240 MiB KV，整体需要 TiB 级 KV 容量。
4. 选择 H100/H200，70B BF16 用 TP=8 单节点作为 baseline，H200 可降低 KV 压力。
5. 调度采用 continuous batching + chunked prefill + priority queue。
6. KV 采用 PagedAttention、prefix caching、必要时 FP8 KV。
7. Autoscaling 以 TTFT/TPOT、queue wait、KV 使用率和 tokens/s 为核心，不只看 GPU util。
8. 容错区分 prefill 可重试和 decode 难恢复，使用 drain、request id、admission control。
9. 监控覆盖 SLA、队列、GPU、KV、缓存、错误和重启。
### 面试中可写出的关键公式
```text
active_sequences = qps * avg_output_tokens * tpot
output_tokens_per_sec = qps * avg_output_tokens
kv_bytes_per_token = 2 * layers * kv_heads * head_dim * bytes
model_weights = params * bytes_per_param
replicas = ceil(required_tokens_per_sec / measured_tokens_per_sec_per_replica)
```
## 追问方向与深入点
- 如果 TTFT 超标但 TPOT 正常，优先查 queue_wait、prefill batch tokens、chunk size、prefix cache hit rate。
- 如果 TPOT 超标但 TTFT 正常，优先查 decode batch size、HBM bandwidth、KV fragmentation、NCCL latency。
- 如果 GPU util 低但排队高，可能是内存带宽、KV blocks、调度锁、tokenizer 或网络成为瓶颈。
- 如果长 prompt 用户拖垮普通用户，使用独立队列、chunked prefill、max prompt policy 和 priority scheduling。
- 如果 prefix cache 命中率低，检查系统提示词是否稳定、cache key 是否包含错误字段、tokenizer 是否变更。
- 如果线上随机 OOM，检查 max_tokens 估算、best_of/beam、KV block 碎片、并发取消是否及时释放。
- 如果跨节点 TP 慢，减少跨节点 TP，改 PP 或提高单节点 replica 数。
- 如果 H200 成本高，比较 H100 + KV 量化 + 更严格 admission 是否满足 SLA。
## 评分标准
| 维度 | 满分 | 要点 |
|---|---:|---|
| 需求拆解 | 15 | 能把 QPS 转成 active sequences、tokens/s、TTFT/TPOT 预算 |
| 显存估算 | 20 | 写出权重、KV、activation 公式，并能代入 70B 数字 |
| 硬件与并行 | 15 | 能解释 A100/H100/H200、NVLink、TP/PP/EP tradeoff |
| 调度设计 | 15 | continuous batching、chunked prefill、priority queue 讲清楚 |
| KV 优化 | 15 | PagedAttention、prefix caching、KV 量化、eviction 风险 |
| 可靠性与扩缩容 | 10 | admission、drain、冷启动、重试、故障域 |
| 监控排障 | 10 | metrics、alert、dashboard 与故障定位关联 |
优秀回答应能量化，不只背组件名；应主动说明 tradeoff 和降级策略。
## 复习卡片 15 张
1. Q: 1000 QPS、平均输出 256、TPOT 50 ms 时 active sequences 约多少？ A: `1000 * 256 * 0.05 = 12800`。
2. Q: KV cache 单 token 公式是什么？ A: `2 * layers * kv_heads * head_dim * bytes`。
3. Q: 70B BF16 权重大约多少？ A: `70B * 2 bytes = 140 GB`。
4. Q: 为什么 decode 常常 memory-bandwidth bound？ A: 每步只生成 1 token，矩阵维度小但要反复读取权重和 KV。
5. Q: H200 相比 H100 对 serving 的主要优势是什么？ A: 更大 HBM 和更高 HBM 带宽，利于 KV 容量和 decode。
6. Q: NVLink 为什么影响 TP？ A: TP 每层有集合通信，低带宽高延迟会放大 TPOT。
7. Q: continuous batching 解决什么问题？ A: 避免静态 batch 等最慢请求，decode step 间动态补入新请求。
8. Q: chunked prefill 的目的是什么？ A: 防止长 prompt 长时间阻塞 decode，改善 TTFT/TPOT 公平性。
9. Q: PagedAttention 的核心抽象是什么？ A: 用 block table 把逻辑 token 映射到物理 KV blocks。
10. Q: prefix cache key 至少包含什么？ A: model/version、tokenizer、adapter、prompt prefix bytes 或 token ids。
11. Q: KV FP8 的主要 tradeoff 是什么？ A: 容量和带宽收益 vs 质量损失与 dequant overhead。
12. Q: Autoscaling 为什么不能只看 GPU util？ A: LLM 瓶颈可能在 KV 容量、队列、HBM 带宽、tokenizer 或调度。
13. Q: Decode worker crash 后为什么难无缝恢复？ A: KV cache 和采样状态在 worker 本地，未 checkpoint 时需重算或返回 partial。
14. Q: TTFT 高、TPOT 正常通常查什么？ A: queue wait、prefill tokens、chunk size、prefix cache、admission。
15. Q: TPOT 高、TTFT 正常通常查什么？ A: decode batch、HBM bandwidth、KV blocks、NCCL、热点 priority 队列。
