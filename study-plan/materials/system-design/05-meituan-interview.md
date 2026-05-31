# 美团 JD 定制深度问答

## 学习目标

1. 能从 serving 系统角度分析 LongCat/long context 的 TTFT, TPOT, KV cache 和调度瓶颈。
2. 能解释 MoE 推理中的 expert parallel, load balancing, all-to-all 和 grouped GEMM。
3. 能判断哪些 op 适合 kernel fusion, 说明 Triton/CUDA/CUTLASS 实现路径和收益边界。
4. 能讲清 N-gram speculative decoding 的命中, 验证, 回退和线上开关策略。
5. 能设计训练-推理一体化链路, 覆盖 checkpoint, eval, serving, feedback 和 rollback。

## 前置知识

- LLM inference: prefill, decode, KV cache, batching, sampling。
- 分布式并行: TP, PP, DP, EP, all-reduce, all-to-all。
- GPU 性能模型: HBM bandwidth, Tensor Core, occupancy, register pressure。
- Kernel 基础: tiling, masking, reduction, vectorized load/store。
- Speculative decoding: draft, target verify, acceptance rate。


## 核心内容

| 主题 | 核心瓶颈 | 关键指标 | 回答抓手 |
|------|----------|----------|----------|
| Long context | prefill O(N^2), decode 读 KV, HBM 容量 | TTFT, TPOT, cache hit, eviction | chunked prefill, prefix cache, PD 分离 |
| MoE | expert imbalance, token dispatch, all-to-all | load variance, all-to-all time, drop/reroute | EP, capacity, hot expert replica |
| Fusion | 中间 tensor HBM 往返, launch overhead | HBM bytes, occupancy, spill | elementwise/reduction/epilogue |
| N-gram speculation | draft 命中率和 target verify 成本 | hit rate, accept length, speedup | 局部 n-gram index, adaptive disable |
| Train-infer | 版本一致性和上线风险 | eval pass, rollback time, online badcase | artifact contract, canary, feedback loop |

回答时先定位瓶颈, 再给工程方案, 最后给指标和失败场景。面试官通常不是只听名词, 而是看你能否把方案落到 latency, throughput, memory, communication 和 correctness。


## 完整的问答/题目
### Q1: LongCat 类长上下文模型上线 serving, 你会怎么优化?
**考察点:** long context serving 架构, prefill/decode 拆解, KV cache 和调度。
**参考答案:**

长上下文的主要矛盾是 prefill 计算重, decode 读 KV 重。长 prompt 会拉高 TTFT, decode 阶段又因为历史 KV 很大导致 TPOT 上升和 batch capacity 下降。

可分五层优化:

1. 请求分桶: 按 prompt length 分 short/medium/long/ultra-long, 长请求独立队列, 避免阻塞短请求。
2. Chunked prefill: 将长 prompt 切成 2K/4K token chunk, chunk 间允许插入 decode batch, 降低 head-of-line blocking。
3. Prefix cache: 对系统 prompt, 工具说明, 商家模板做 block-level cache。key 需要包含 model_id, tokenizer_version, rope_config, token hash。
4. KV 分层: 热 KV 在 HBM, 温 KV 在 CPU pinned memory, 冷 KV 远端或重算。能接受质量损失时可做旧 KV FP8/INT8。
5. PD disaggregation: prefill worker 追求大 GEMM 吞吐, decode worker 追求高并发读 KV。prefill 后通过 NVLink/IB/RDMA 转移 KV。

KV 估算公式:

```text
KV bytes = batch * seq_len * layers * kv_heads * head_dim * 2 * bytes_per_elem
```

例如 32 layers, kv_heads=8, head_dim=128, FP16 时单 token KV 约 128 KB。128K context 单请求 KV 约 16 GB, 所以容量规划不能只看权重显存。
**追问方向:**

- chunk size 如何选? 过小调度开销大, 过大短请求仍被阻塞。
- remote KV hit 和 local recompute 如何取舍?
- RoPE scaling 改动后旧 KV cache 能不能复用?
- FlashAttention 能降低 IO, 为什么仍解决不了 KV 容量?

### Q2: Long context 下如何设计 KV cache 管理?
**考察点:** PagedAttention, fragmentation, prefix sharing, eviction。
**参考答案:**

我会用 paged KV cache。KV 被切成固定 block, 例如 16/32 tokens, 请求维护逻辑 block 到物理 block 的映射。这样能减少连续显存分配需求, 支持非连续增长, 也方便 prefix cache 共享。

核心数据结构:

- block table: request_id -> logical_block_id -> physical_block_id。
- free list: 管理可复用物理 block。
- ref count: prefix block 被多请求共享时防止误释放。
- cache metadata: model_id, tokenizer_version, rope_config, token_hash。

eviction 策略要考虑业务价值而不是只看 LRU:

- 正在 decode 的请求不可随意驱逐。
- 长时间未命中的 prefix block 可以淘汰。
- 大 block 和低命中率 block 优先淘汰。
- 超长低优先级请求可 preempt, 但要评估 CPU swap 的 TPOT 影响。

监控指标包括 HBM usage, free block count, allocation failure, fragmentation, prefix hit rate, eviction rate, swap latency。
**追问方向:**

- KV block size 太大和太小分别有什么问题?
- Prefix cache 如何做最长前缀匹配?
- 多租户场景如何避免一个租户占满 cache?

### Q3: MoE 推理和 Dense 推理最大的系统差异是什么?
**考察点:** MoE routing, token dispatch, expert compute, 通信。
**参考答案:**

Dense FFN 是所有 token 走同一组 FFN 权重。MoE 则由 router 为每个 token 选择 top-k experts, 只激活少数 experts。它降低了每 token FLOPs, 但引入 routing, token dispatch, all-to-all 和 expert load imbalance。

MoE layer 流程:

```text
hidden -> router logits -> top-k expert ids
-> token 按 expert 分桶
-> all-to-all 到 expert 所在 GPU
-> local grouped GEMM
-> all-to-all 返回
-> 按 router weight combine
```

三个关键难点:

- 负载不均: 热 expert 成为 straggler, p99 latency 上升。
- 通信放大: top-2 会复制 token hidden state, all-to-all bytes 增加。
- 小矩阵低效: 每个 expert 分到的 token 数不稳定, GEMM shape 碎片化。

因此 MoE serving 不是只看 FLOPs, 更要看 routing 分布和网络拓扑。
**追问方向:**

- top-1 和 top-2 routing 的质量/延迟 tradeoff?
- 为什么参数更多但推理 FLOPs 不一定更多?
- 如何在线检测 hot expert?

### Q4: Expert parallel 如何设计? 如何做 load balancing?
**考察点:** EP/TP 组合, expert placement, capacity factor, all-to-all。
**参考答案:**

Expert parallel 是把 experts 分布到不同 GPU。Attention 和 dense projection 常用 TP, MoE FFN 用 EP, 大模型再叠加 PP/DP。

设计步骤:

1. 设并行组: 例如 16 卡中 TP=4, EP=4。TP 尽量在 NVLink 域内, EP 也避免跨慢网络。
2. 放置 expert: 默认均匀放置, 高频 expert 可复制多份, 拓扑上把同一 EP group 放近。
3. dispatch: 统计 send counts, all-to-all hidden states, 本地按 expert 分段 grouped GEMM, 再 all-to-all 回原 token 顺序。
4. overlap: 分 bucket 发送, 边接收边计算, intra-node 和 inter-node 分层通信。

训练阶段 load balancing 常用辅助损失:

```text
balance_loss = num_experts * sum_i(f_i * p_i)
```

`f_i` 是实际 token fraction, `p_i` 是平均 router probability。capacity:

```text
capacity = ceil(tokens * top_k / num_experts * capacity_factor)
```

训练中超过 capacity 的 token 可 drop/reroute。推理中通常不希望 drop, 更常用 hot expert replica, routing-aware batching, 请求限流和动态调度。
**追问方向:**

- hot expert 是复制好还是迁移好?
- all-to-all 如何做 topology-aware?
- capacity factor 越大是否一定越好?

### Q5: Kernel fusion 的判断原则是什么?
**考察点:** fusion 收益边界, memory-bound vs compute-bound, register pressure。
**参考答案:**

Fusion 的本质收益是减少中间结果写回 HBM 和 kernel launch。适合 fuse 的算子通常是 producer-consumer 紧邻, 中间 tensor 单消费者, 以 elementwise/reduction 为主, shape 简单。

适合 fuse:

- Elementwise chain: bias add + activation + dropout, residual add + scale + cast。
- Reduction + elementwise: LayerNorm/RMSNorm, softmax, router top-k 后的 mask/scale。
- GEMM epilogue: matmul + bias + GELU/SwiGLU, matmul + scale + quantize。
- Attention 内部: QK + scale + mask + softmax + PV, 典型是 FlashAttention。

不适合 fuse:

- 两个大 GEMM 强行合并, 可能破坏 tiling, 收益小。
- 中间结果有多个消费者, fuse 会导致重复计算。
- 需要跨 block 全局同步的链路。
- fuse 后 register pressure 过大, occupancy 降低甚至 spill。

判断流程: 先 profiler 看 kernel launch count, HBM bytes, achieved bandwidth, occupancy, local memory spill。memory-bound 小算子优先 fuse, compute-bound GEMM 优先做 tile/epilogue。
**追问方向:**

- LayerNorm 为什么适合 fuse?
- register spill 如何发现?
- fused kernel 数值误差如何验证?

### Q6: 设计 fused RMSNorm + residual + quantization kernel, 怎么实现?
**考察点:** Triton/CUDA 实现细节, masking, quantization。
**参考答案:**

目标:

```text
v = x + residual
y = v / sqrt(mean(v^2) + eps) * weight
q = clamp(round(y / scale), -128, 127)
```

Triton 可以一个 program 处理一行 hidden。读取 x, residual, weight 到 FP32, 做 block 内 reduction 得到 rms, 再 normalize 和量化。hidden 非 2 的幂时用 mask, block 取 next_power_of_2(hidden)。

注意点:

- hidden 太大时单 block reduction 不够, 要 split reduction 或两阶段 kernel。
- per-token scale 适合动态激活量化, per-channel scale 更适合权重量化。
- 如果后续还需要 FP16 y, 需要额外写出, 可能抵消部分收益。
- INT8 量化必须 clamp, 并验证 saturation rate。
- aligned 和 non-aligned hidden size 都要测。

验证方式: PyTorch reference 对比误差, profiler 对比 HBM bytes 和 latency, 检查 occupancy 和 local memory。
**追问方向:**

- RMSNorm 和 LayerNorm reduction 差异是什么?
- per-token scale 如何存储?
- 为什么 fuse 后可能变慢?

### Q7: N-gram speculative decoding 是什么?
**考察点:** n-gram draft, target verify, correctness, 适用场景。
**参考答案:**

N-gram speculation 不加载 draft model, 而是在上下文中查找重复 n-gram, 把历史相同 n-gram 后面的 token 当作 draft。它适合代码补全, JSON/SQL, 表格, 客服模板等重复结构强的场景。

流程:

1. 对当前请求维护 n-gram index。
2. 用最近 n 个 token 查找历史匹配位置。
3. 命中后取后续 k 个 token 作为 draft。
4. target model 一次 forward 验证 k 个 token。
5. 接受最长连续正确前缀, 第一个失败处回退到 target 分布。

和小模型 speculation 对比:

| 项 | N-gram | Draft model |
|----|--------|-------------|
| 草稿来源 | 历史 token | 小模型预测 |
| 额外显存 | 很低 | 需要 draft model |
| 场景 | 重复文本强 | 更通用 |
| 风险 | 命中率不稳定 | draft-target 不一致 |

正确性来自 target verification。greedy 时可比较 argmax。采样场景需要严格 rejection sampling, 不能简单比较 draft token。
**追问方向:**

- n 和 draft length k 如何自适应?
- 全局 n-gram cache 有什么隐私风险?
- 低接受率时如何自动关闭?

### Q8: N-gram speculation 如何集成到线上 serving?
**考察点:** scheduler, batch verify, fallback, metrics。
**参考答案:**

我会把它做成 decode scheduler 的 optional fast path。

组件:

- NGramIndex: 请求级局部索引, 可选租户级共享索引。
- DraftGenerator: suffix lookup 后产生 draft span。
- Verifier: target model 批量验证 draft tokens。
- Scheduler: speculative 和普通 decode 混合调度。
- Metrics: hit rate, accept rate, accepted tokens/verify, fallback rate, speedup。

Batch 内 draft length 不同, 可以 padding 或按 draft length 分桶。一个请求如果连续多次 miss 或接受率低于阈值, 就关闭该请求的 speculation, 避免拖慢整体吞吐。

收益粗估:

```text
speedup ~= avg_accepted_tokens / verify_cost_multiplier
```

例如 verify 成本是普通 decode 的 1.3 倍, 平均接受 2.6 token, step 级加速约 2x。但端到端还要扣除 index lookup, scheduler, padding 和 KV update 开销。
**追问方向:**

- speculative 和 non-speculative 请求如何共 batch?
- 接受多个 token 后 KV cache 如何更新?
- streaming 输出如何处理回退?

### Q9: 训练-推理一体化应该包含哪些链路?
**考察点:** LLMOps, artifact contract, eval, serving release。
**参考答案:**

训练-推理一体化不是把训练和推理放进同一进程, 而是把 checkpoint 到线上服务的链路标准化, 自动化, 可回滚。

关键链路:

1. Checkpoint 标准化: sharded checkpoint 转 serving artifact, 记录 tensor parallel shard, tokenizer, rope_config, chat template, generation_config。
2. 自动评测: perplexity/domain eval/safety/tool-call 格式, 以及 TTFT/TPOT/throughput/memory peak。
3. 发布策略: canary, shadow traffic, A/B, 快速 rollback。
4. 版本隔离: model version, tokenizer version, KV cache version 绑定, 防止跨版本 cache 复用。
5. 反馈回流: badcase, latency trace, 用户反馈脱敏后进入 eval set 或训练数据。

训练侧 kernel 和推理侧 kernel 可以共享思想, 但目标不同。训练重吞吐和 backward, 推理重 latency, KV cache 和 batch scheduler。同一个 fused kernel 未必两边都最优。
**追问方向:**

- 新 checkpoint 上线前必须检查哪些元数据?
- Quantization 应该在链路哪一步做?
- 线上 badcase 如何避免污染训练集?

### Q10: 线上模型质量退化, 怀疑和训练 loss spike 有关, 怎么定位?
**考察点:** 训练日志, 转换链路, 推理配置, rollback。
**参考答案:**

先 rollback 或切 canary, 保住线上 SLA。定位分四层:

1. 模型本身: 查看 loss spike 是否发生在数据切换, LR peak, resume, optimizer state 异常附近。对 spike 前后 checkpoint 跑同一 eval set。
2. 转换链路: 检查 shard 完整性, TP 切分维度, tokenizer/chat template/RoPE config, quantization calibration。
3. 推理配置: 固定 seed, greedy decoding, 关闭 speculation/prefix cache/quantization 做 A/B。
4. Serving 系统: 检查跨版本 KV cache, position ids, attention mask, long context 溢出。

关键是拿同一 prompt 做 logits diff。如果 PyTorch reference 和 serving logits 从某层开始分叉, 可以继续二分定位是权重转换, kernel, mask 还是 position。
**追问方向:**

- tokenizer 不一致为什么可能不报错但质量很差?
- logits diff 应该看哪些位置?
- long context position id 错误有什么症状?


## 追问方向与深入点

1. Long context: chunked prefill fairness, remote KV transfer ROI, 128K context 下 sliding window 的质量风险。
2. MoE: EP/TP group 拓扑, all-to-all overlap, hot expert replica, grouped GEMM padding 浪费。
3. Fusion: memory-bound 判断, register spill, epilogue fusion, CI 数值回归阈值。
4. Speculation: greedy vs sampling correctness, low acceptance fallback, KV update 和 streaming 语义。
5. Train-infer: artifact 版本契约, online eval, badcase 脱敏, canary 和 rollback。


## 评分标准

| 等级 | 表现 |
|------|------|
| A | 能给架构, 公式, 调度策略, 通信细节, 指标和失败场景。 |
| B | 主要方向正确, 但容量估算, all-to-all 或验证方案不够具体。 |
| C | 只会堆术语, 不能说明为什么有效或如何验证。 |
| D | 混淆 prefill/decode, Dense/MoE, speculation/beam search 等基本概念。 |

加分点: 能主动量化 KV cache, all-to-all bytes, HBM traffic, acceptance rate, p95/p99 latency。扣分点: 只说 "加缓存", "做并行", "写 fused kernel", 但没有数据结构和边界条件。


## 复习卡片 15 张

1. Long context 两个核心成本: prefill O(N^2) 计算, decode O(N) 读 KV。
2. Chunked prefill: 把长 prompt 分块, 降低短请求排队和 TTFT。
3. KV cache 公式: `batch * seq_len * layers * kv_heads * head_dim * 2 * bytes`。
4. GQA 省 KV: 多个 Q heads 共享更少 KV heads。
5. PagedAttention: 固定 block 管理 KV, 减少碎片, 支持 prefix 共享。
6. Prefix cache key: model_id, tokenizer_version, rope_config, token hash。
7. MoE 额外开销: router, dispatch, all-to-all, grouped GEMM, combine。
8. Expert parallel: experts 分布在多 GPU, token 路由到 expert 所在卡。
9. Load balance loss: 用 `f_i * p_i` 约束 router 使用更多 experts。
10. Capacity factor: 控制每个 expert 最大 token 容量和 padding/drop tradeoff。
11. Fusion 收益: 减少 HBM 中间读写和 kernel launch。
12. 适合 fuse: elementwise chain, reduction+elementwise, GEMM epilogue, attention 内部。
13. Fusion 风险: register pressure, occupancy 降低, spill, shape 分支复杂。
14. N-gram speculation: 用历史重复 n-gram 生成 draft, target model 验证。
15. 训练-推理一体化: checkpoint, eval, serving artifact, canary, feedback, rollback。
