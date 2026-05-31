# 小红书 JD 定制深度问答

## 学习目标

1. 能解释 Mooncake/RBG 这类 KV cache-centric serving 架构的核心动机和 tradeoff。
2. 能说明 Dynamic PD disaggregation 如何根据负载动态调整 prefill/decode 资源。
3. 能设计 KV cache prefix caching, compression, offloading 的工程方案和指标。
4. 能把 speculative decoding 讲清楚: draft, verify, acceptance, rollback, serving 集成。
5. 能针对业务流量选择推理框架, 调整调度, batch, cache, 并设计 benchmark。
6. 能用 TTFT, TPOT, throughput, p99, cache hit rate, GPU utilization 分析 serving 问题。

## 前置知识

- LLM serving 两阶段: prefill compute-bound, decode memory-bound。
- KV cache: 每层每 head 保存历史 token 的 K/V, 显存随 batch 和 context 线性增长。
- Continuous batching, paged KV cache, prefix cache, tensor parallel, pipeline parallel。
- PD disaggregation: prefill 和 decode 使用不同资源池, 通过 KV transfer 衔接。
- Speculative decoding: 小模型或轻量 draft 先生成候选 token, 大模型并行验证。
- Benchmark 基础: 流量分布, prompt/output length, warmup, p50/p95/p99。

## 核心内容

### 1. Mooncake/RBG 架构理解

- Mooncake 的核心是把 KV cache 从单机 GPU 内部状态提升为全局可调度资源。
- Prefill 和 decode 分离后, 系统可以分别优化 compute utilization 和 memory bandwidth utilization。
- RBG 可以理解为 request-based grouping, 通过请求特征和 cache locality 分组, 提升 prefix/KV 复用。

### 2. Dynamic PD Disaggregation

- 静态 PD 分离的问题是资源比例固定, 遇到长 prompt 或长生成流量变化会产生一侧空闲一侧排队。
- Dynamic PD 根据队列, SLA, GPU 利用率, KV 传输压力动态改变节点角色或请求路由。
- 关键难点是切换成本, cache 迁移, drain 策略和避免频繁震荡。

### 3. KV Cache 优化

- Prefix caching 省 prefill 计算, 但占用显存/内存, 需要 eviction 和一致性。
- Compression 降低 KV footprint, 但引入精度损失或额外解压成本。
- Offloading 把 KV 放到 CPU/SSD/远端内存, 提升容量, 但可能拉高 TPOT 和 p99。

### 4. Speculative Decoding

- 性能收益来自用一次 target model forward 并行验证多个 draft token。
- 工程重点是 draft 模型选择, token 对齐, acceptance sampling, KV cache 回滚, batch 调度。
- 接受率低时 speculative decoding 可能变慢。

### 5. 框架选型和 Benchmark

- 框架选型要围绕业务: 单机吞吐, 长上下文, 多租户, LoRA, prefix cache, PD 分离, 运维复杂度。
- benchmark 不能只跑固定 batch, 要模拟真实 arrival rate 和长度分布。
- 分析时要区分 queue time, prefill time, decode time, transfer time 和 sampling time。

## 完整的问答/题目

### Q1: 请解释 Mooncake 架构为什么叫 KV cache-centric, 它解决了传统 serving 的什么问题?

**考察点:** 前沿 serving 架构, KV cache 资源化, PD 分离。

**参考答案:**

传统 serving 常把 KV cache 视为某个 GPU worker 的内部状态。请求在哪个 worker 上 prefill, 后续 decode 通常也留在该 worker, 否则 KV cache 迁移成本高。这会造成两个问题: 一是 prefill 和 decode 的资源需求不同, 同一批 GPU 很难同时高效服务; 二是 prefix/KV 复用被限制在单节点或局部范围。

Mooncake 的思路是把 KV cache 作为系统的一等资源管理。prefill pool 负责计算 prompt 的 KV, decode pool 负责基于 KV 生成 token, 中间通过高速网络或分布式 KV store 传递 KV。全局调度器维护 KV 的位置, 引用计数, 热度, 可复用前缀和节点负载, 请求路由不只看 GPU 空闲, 还看 KV locality。

优势是 prefill/decode 可以独立扩缩容, 长 prompt 不一定阻塞 decode, 多请求共享 prefix 时可以跨节点复用 KV。代价是系统复杂度增加, KV transfer 可能影响 TTFT, 调度器需要处理一致性, 容错和 cache eviction。

**追问方向:**

- KV transfer 延迟如何进入 TTFT?
- 如果全局调度器故障, 如何保证服务可用?
- KV cache locality 和 load balance 冲突时怎么决策?

### Q2: RBG 在推理服务中可以如何理解和落地?

**考察点:** 请求分组, cache locality, 多租户路由。

**参考答案:**

RBG 可以理解为 request-based grouping: 根据请求属性把相似请求分到同一组, 再把组路由到更合适的 worker 或 decode pool。分组依据可以包括 system prompt hash, user/app id, 模板 id, 模型版本, LoRA adapter, prompt 长度, SLA 等。

它的目标不是简单聚类, 而是让 cache locality 和调度效率更好。例如小红书内容理解场景中, 大量请求可能共享固定审核规则, 固定 system prompt 或固定多模态模板。把这些请求聚到同一组, 可以提升 prefix cache 命中率, 减少重复 prefill。

落地时要有动态退化机制。如果某组请求太热, 不能为了 cache hit 把它们全部压到一个 worker, 否则排队会抵消收益。调度评分可以是:

```text
score = cache_hit_benefit - queue_penalty - transfer_cost - SLA_risk
```

最终选择 score 最高的节点或资源池。

**追问方向:**

- 分组粒度太细或太粗分别有什么问题?
- 多租户场景下 prefix cache 是否可以跨租户共享?
- 如何度量 RBG 带来的收益?

### Q3: Dynamic PD disaggregation 为什么需要动态? 如何设计控制策略?

**考察点:** 弹性调度, prefill/decode 资源比例, 稳定性。

**参考答案:**

prefill 和 decode 的瓶颈不同。prefill 处理 prompt 矩阵计算, 更偏 compute-bound; decode 每轮生成少量 token, 不断读取 KV, 更偏 memory-bound。静态 PD 把 GPU 固定分成 prefill pool 和 decode pool, 当流量从短问答变成长文档总结时, prefill 可能排队, decode 空闲; 当生成很长时, decode 可能成为瓶颈。

Dynamic PD 的目标是根据实时负载调整资源。策略可以有三层:

第一层是请求路由动态化。在不改变节点角色的情况下, 调整 prefill 请求和 decode 请求进入不同 pool 的比例。

第二层是节点角色切换。某些 GPU worker 同时具备 prefill/decode 能力, 当 prefill queue pressure 连续高于阈值, decode pressure 低于阈值, 选择低负载 decode 节点 drain 后切到 prefill。

第三层是预测式调度。根据时间段, 活动入口, 历史 prompt/output length 分布提前调整比例。

为了避免震荡, 需要 hysteresis, cooldown, 最小角色保持时间和容量下限。例如 prefill pool 不能低于总 GPU 的 30%, decode pool 不能低于 40%, 切换后至少保持 5 分钟。

**追问方向:**

- 节点切换时已有 KV cache 怎么处理?
- 如何判断是增加 prefill 还是降低 max batched tokens?
- Dynamic PD 和 autoscaling 如何配合?

### Q4: Prefix caching 如何设计? 需要解决哪些一致性问题?

**考察点:** KV cache key, longest prefix match, eviction, model/version 隔离。

**参考答案:**

Prefix caching 的目标是复用相同前缀的 KV cache, 省掉重复 prefill。实现通常以 token block 为单位。缓存 key 不能只用文本, 应包含 model id, model version, tokenizer version, adapter id, dtype, block tokens hash, 以及必要的 system setting。否则模型升级或 tokenizer 变化后可能复用错误 KV。

查找时做 longest prefix match。可以用 hash map 按 block hash 连续匹配, 也可以用 radix tree 存 token 前缀。命中后, 请求的 block table 指向已缓存的 physical block, ref_count 增加。后续生成新 token 时如果写入共享 block, 需要 copy-on-write。

eviction 要综合 LRU, LFU, block size, 重算成本和租户配额。长 prefix 占用大, 但命中后收益也大。可以用 estimated_saved_prefill_time / memory_cost 做价值评分。

一致性问题包括模型版本失效, adapter 隔离, 多租户数据隔离, hash 冲突校验, ref_count 正确性。跨节点 prefix cache 还要考虑远端拉取是否比本地重算更划算。

**追问方向:**

- Radix tree 相比 block hash map 的优劣?
- prefix cache 命中率高但 TTFT 不降的原因?
- 如何防止跨租户信息泄露?

### Q5: KV cache compression 可以怎么做? 如何评估是否值得?

**考察点:** 量化, 稀疏, sliding window, 精度/延迟 tradeoff。

**参考答案:**

KV cache compression 有几类。第一类是低比特量化, 例如把 FP16 KV 存成 INT8/FP8, 每个 block 或每个 head 保存 scale。decode attention 时读取压缩 KV 后反量化参与计算。优点是容量和带宽下降, 缺点是反量化开销和精度风险。

第二类是结构性裁剪或稀疏, 例如 sliding window 只保留最近窗口, sink token 保留开头少量重要 token, 或按注意力重要性淘汰部分 KV。这类对模型质量影响更大, 需要任务级评估。

第三类是分层压缩, 热 KV 保持 FP16 在 HBM, 冷 KV 量化后放 CPU 或远端内存。decode 需要时再取回。

评估要同时看显存节省, decode bandwidth, TPOT, p99, perplexity 或业务质量指标。不能只看 memory reduction。如果压缩让每步 decode 多出昂贵反量化, 小 batch 下可能不划算。

**追问方向:**

- per-tensor, per-channel, per-block scale 怎么选?
- KV 量化对长上下文质量影响如何测试?
- compression 和 offloading 能否组合?

### Q6: KV cache offloading 的工程设计是什么?

**考察点:** HBM/CPU/SSD 分层, prefetch, p99 风险。

**参考答案:**

offloading 是把暂时不活跃或低优先级请求的 KV cache 从 GPU HBM 移到 CPU DRAM, 甚至 SSD 或远端内存。目的是扩大可服务上下文和并发数。基本流程是: 选择 victim sequence -> 将其 KV block 异步拷出 -> 更新 block location metadata -> 释放 GPU block -> 恢复时 prefetch 或同步拷回。

关键是隐藏传输延迟。对于被抢占的请求, 可以在重新进入 decode 前提前 prefetch。对于长上下文, 可以按 block 粒度分批拉取, 但 attention 需要访问历史 KV, 因此 decode 通常希望当前 sequence 的历史 KV 都在可访问层级。

风险是 p99。PCIe 传输和 CPU 内存带宽会让单请求抖动明显。offload 策略要有上限, 并把高优先级, 即将完成, SLA 严格的请求排除在 victim 之外。

**追问方向:**

- offloading 和 recompute 哪个更适合长 prompt?
- 如何用异步 copy 和 stream 降低阻塞?
- 什么指标说明 offloading 过度了?

### Q7: Speculative decoding 的工程流程是什么?

**考察点:** draft/target 协作, acceptance, KV 管理。

**参考答案:**

Speculative decoding 用一个 draft 模型先生成 k 个候选 token, 再用 target 模型一次 forward 验证这些 token。target 模型输出每个位置的分布, 按 speculative sampling 规则接受前缀 token, 遇到第一个拒绝的位置后用 target 分布重新采样, 丢弃后面的 draft token。

工程流程是:

1. 为每个请求运行 draft, 生成候选 token 和 draft logits。
2. 将原上下文加候选 token 送入 target 做并行验证。
3. 按接受率规则决定 accepted length。
4. 更新请求 token 序列和 target KV cache。
5. 对未接受 token 做 rollback, draft KV 也要同步到新状态。

难点包括 tokenizer 必须一致, draft 和 target 的采样参数要对齐, batch 内每个请求 accepted length 不同, KV cache 要正确截断或追加。接受率越高, target 每次 forward 产出的有效 token 越多, TPOT 越低; 接受率低时, draft 开销可能超过收益。

**追问方向:**

- draft 模型如何选择?
- accepted length 不同如何做 continuous batching?
- greedy decoding 和 sampling 下验证规则有什么差异?

### Q8: Speculative decoding 为什么有时会变慢? 如何诊断?

**考察点:** 接受率, draft 成本, batch 调度, workload 适配。

**参考答案:**

变慢通常有几类原因。第一是接受率低。draft 和 target 分布差异大, 候选 token 很快被拒绝, target 每轮只接受 0 到 1 个 token, 但额外付出了 draft 计算。

第二是 draft 成本过高。如果 draft 模型仍然很大, 或者 draft 运行无法和 target 并行, 总 GPU 时间增加。

第三是 batching 变差。speculative decoding 让每个请求每轮消耗的 token 数不一致, 调度和 KV 更新更复杂, 可能降低 continuous batching 效率。

第四是 memory/KV 管理开销。候选 token 写入后又 rollback, 或 target verify 需要更大临时 buffer, 都会影响 p99。

诊断指标包括 acceptance rate, average accepted tokens per target forward, draft latency, target verify latency, rollback 次数, batch size, GPU utilization, end-to-end TPOT。只有当 accepted tokens per target forward 明显大于 1, 且 draft 开销足够低, 才有稳定收益。

**追问方向:**

- k 取太大或太小分别有什么问题?
- 可以用 n-gram draft 或 prompt lookup 替代小模型 draft 吗?
- 如何对不同请求动态启用 speculative decoding?

### Q9: 小红书业务下如何选择推理框架?

**考察点:** 框架选型, 业务约束, 运营成本。

**参考答案:**

我会先定义业务画像。小红书可能同时有内容理解, 搜索问答, 评论生成, 审核辅助, 多轮对话等场景。要看模型大小, QPS, prompt/output length 分布, 是否多租户, 是否需要 LoRA, 是否长上下文, 是否严格 p99, 是否要多模态。

如果重点是通用 LLM 高吞吐和成熟生态, vLLM 是强候选, 它有 continuous batching, PagedAttention, prefix caching, 多种量化和较成熟的 OpenAI-compatible server。如果重点是 NVIDIA GPU 上极致性能和企业部署, TensorRT-LLM 可考虑, 但构建和调参复杂。如果要深度自研 PD 分离, 全局 KV cache, 或业务路由, 可以在 vLLM/SGLang 等框架基础上扩展调度层。

选型不是只看单机 tokens/s。要评估功能覆盖, kernel 性能, scheduler 可定制性, observability, failure recovery, 模型支持速度, 团队维护成本。最终要用真实流量 benchmark, 而不是只用固定 prompt 的离线压测。

**追问方向:**

- vLLM 和 TensorRT-LLM 的主要 tradeoff?
- 什么时候需要自研 serving layer?
- 多 LoRA 场景如何影响框架选择?

### Q10: 推理服务调优时, 你会优先调哪些参数?

**考察点:** serving 参数, latency/throughput tradeoff, cache。

**参考答案:**

优先调度相关参数包括 max_num_batched_tokens, max_num_seqs, chunked prefill, scheduling policy。max_num_batched_tokens 太小会吞吐低, 太大可能 TTFT 和 p99 上升。chunked prefill 能避免超长 prompt 阻塞 decode, 但可能增加 prefill 完成轮数。

内存相关参数包括 GPU memory utilization, KV block size, swap/offload 上限, prefix cache 开关和容量。KV block size 太大浪费显存, 太小增加 block table 和 kernel 间接访问开销。

模型执行相关包括 dtype, quantization, tensor parallel size, attention backend, CUDA graph, speculative decoding。调优顺序应基于 profiling, 不要一次改很多参数。

验证时要同时看 TTFT, TPOT, throughput, p95/p99, error rate, OOM, cache hit rate 和 GPU 利用率。业务上还要检查输出质量, 因为量化和 KV compression 会影响结果。

**追问方向:**

- max_num_batched_tokens 增大为什么可能提高吞吐但恶化 TTFT?
- block size 如何影响碎片和 attention 访存?
- CUDA graph 对动态 shape 有什么限制?

### Q11: Serving benchmark 应该如何设计才可信?

**考察点:** benchmark 方法论, 负载建模, 指标分析。

**参考答案:**

可信 benchmark 必须模拟真实请求, 而不是固定 batch 的离线吞吐。首先采样真实或近似的 prompt length/output length 分布, 包括短问答, 长文档, 多轮对话。其次用 arrival rate 模拟在线流量, 例如 Poisson 或 trace replay, 让系统出现排队和 batching 行为。

指标要拆分: TTFT, TPOT, end-to-end latency, throughput tokens/s, requests/s, p50/p95/p99, GPU utilization, HBM usage, KV block usage, prefix cache hit rate, swap/offload 次数, error/OOM。还要区分 warmup 和 steady state。

实验设计要固定模型, tokenizer, dtype, GPU, driver, framework commit, sampling 参数。每组实验跑足够长时间, 丢弃 warmup, 给出置信区间或多次重复结果。对比优化时只改变一个变量, 否则无法归因。

**追问方向:**

- 为什么固定 batch benchmark 不能代表 online serving?
- 如何构造长尾请求压测 p99?
- prefix cache benchmark 如何避免过于理想化?

### Q12: 如果线上 TTFT 突然升高, 你怎么排查?

**考察点:** 故障诊断, 队列拆解, PD/cache/transfer。

**参考答案:**

先确认 TTFT 升高来自 queue time 还是 prefill execution time。看入口 QPS, waiting queue, scheduler delay, prefill batch size, GPU utilization。如果 queue time 上升且 GPU 满, 可能容量不足或长 prompt 流量增加; 如果 GPU 不满, 可能调度, 限流或上游阻塞。

再看 prompt length 分布和 prefix cache hit rate。如果突然出现大量长 prompt 或 cache miss, prefill 时间会增加。对于 PD 分离架构, 还要看 KV transfer latency, prefill pool/decode pool 队列是否失衡, 是否有节点角色不足。

然后看系统事件: 新模型版本, tokenizer 变化, cache 失效, 节点故障, NCCL/network 抖动, offload/swap 增加。排查要用分段指标: arrival -> scheduler queue -> prefill start -> prefill end -> KV ready -> first decode -> first token emitted。

修复可以是临时限流长 prompt, 提高 prefill 资源, 开启/调大 chunked prefill, 回滚导致 cache miss 的版本, 或调整 PD 比例。最终用 p50/p99 TTFT 和错误率验证。

**追问方向:**

- cache hit rate 下降为什么会影响 TTFT?
- PD 架构下 KV transfer 慢如何定位是网络还是调度?
- 如何快速止血而不牺牲所有用户体验?

## 追问方向与深入点

- Mooncake 把 KV cache 资源化后, 调度状态如何持久化和恢复?
- RBG 的分组特征是否会泄露用户或租户信息?
- Dynamic PD 中角色切换的最小单位是 GPU, worker, 还是请求?
- Prefix cache 命中后, 为什么仍可能受 decode 队列影响?
- KV compression 是否应该按 layer/head/token 使用不同策略?
- Offloading 的 victim selection 如何兼顾公平性和 SLA?
- Speculative decoding 如何和 prefix caching, paged KV cache 一起工作?
- benchmark 中如何区分模型质量下降和 serving 速度提升?

## 评分标准

### 优秀

- 能把 Mooncake/RBG, Dynamic PD, KV cache 优化和 benchmark 串成完整 serving 设计。
- 能主动讨论 cache locality, transfer cost, queue penalty, p99 风险和多租户隔离。
- 能对 speculative decoding 给出工程链路, 不是只说小模型加速大模型。
- 能设计真实 online benchmark, 并用分段指标定位问题。

### 合格

- 能解释 PD 分离, prefix caching, speculative decoding 的基本原理。
- 能说出常见框架选型依据和主要 serving 指标。
- 对动态调度和工程故障有概念, 但细节不够完整。

### 风险

- 只背 Mooncake 名词, 说不出 KV cache 如何被调度和迁移。
- 把 speculative decoding 说成无条件提速, 不提接受率和 draft 成本。
- benchmark 只看 tokens/s, 不看 TTFT/TPOT/p99 和真实流量。
- 忽视多租户隔离, 模型版本一致性和 cache 失效。

## 复习卡片 15 张

1. **Mooncake 的核心是什么?** KV cache-centric, 把 KV cache 当作全局可调度资源。
2. **PD 分离解决什么?** prefill compute-bound 和 decode memory-bound 的资源错配。
3. **Dynamic PD 为什么需要 hysteresis?** 避免资源角色频繁切换造成震荡。
4. **RBG 的目标是什么?** 按请求特征分组, 平衡 cache locality 和 load balance。
5. **Prefix cache key 应包含什么?** model/version, tokenizer, adapter, dtype, token block hash, 配置。
6. **Prefix cache 最大风险是什么?** 错误复用导致输出污染或跨租户泄露。
7. **KV compression 的收益是什么?** 降低显存占用和带宽, 但可能影响质量和延迟。
8. **Offloading 的主要代价是什么?** 传输延迟导致 TPOT 和 p99 抖动。
9. **Speculative decoding 的收益来源是什么?** target 一次 forward 验证多个 draft token。
10. **Speculative decoding 何时变慢?** 接受率低, draft 成本高, batching/KV rollback 开销大。
11. **框架选型不能只看什么?** 不能只看单机 tokens/s, 还要看功能, 可运维性和真实 workload。
12. **max_num_batched_tokens 的 tradeoff?** 大可提高吞吐, 但可能增加 TTFT/p99。
13. **可信 benchmark 需要什么输入?** 真实或近似的 prompt/output length 分布和 arrival process。
14. **TTFT 排查先拆什么?** queue time, prefill time, KV transfer time, first decode time。
15. **Serving 优化如何验证?** 对比 baseline, 固定变量, 看 TTFT/TPOT/throughput/p99/质量/稳定性。

## 参考链接索引

本文出现的项目、论文、技术报告和博客链接集中维护在 [09-reference-links.md](./09-reference-links.md)。
