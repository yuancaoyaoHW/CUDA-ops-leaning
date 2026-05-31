# Company Interview Cards: Tencent / Xiaohongshu / Meituan

Reader: candidate preparing for LLM inference framework, kernel, and serving-system interviews.
Post-read action: rehearse concise answers, then use the final checklist for a 15-minute last pass.

## 腾讯

### 1. 如果让你手写 FP16 GEMM kernel，你会如何优化到接近 cuBLAS？
- One-sentence answer: 先建立 correctness baseline，再按 coalesced load、shared-memory tiling、register tiling、vectorized load、pipeline/Tensor Core 的顺序优化，并用 Nsight 验证瓶颈变化。
- Expanded explanation: naive GEMM 重复从 HBM 读取 A/B，算术强度低；shared memory tile 让 A/B 在 block 内复用，register tile 让每个线程维护多个 accumulator，cp.async/double buffering 隐藏 global memory 延迟，MMA/Tensor Core 提升 FP16/BF16 吞吐。
- Formula/diagram hook: `C[BM,BN] += A[BM,BK] x B[BK,BN]`; roofline 看 `FLOPs / bytes` 是否越过机器平衡点。
- Follow-up risk: tile 太大会增加 register/shared memory 占用，降低 occupancy；非整除 M/N/K 必须有 mask 和 non-aligned 测试。

### 2. Triton softmax kernel 的 mask 和数值稳定性怎么做？
- One-sentence answer: 对越界位置用 `-inf` mask，按行减去最大值后再 exp/sum，最后只把合法位置写回。
- Expanded explanation: softmax 需要处理 block 大于真实列数、causal mask 或 padding mask；`x - max(x)` 避免 exp 溢出，mask 的位置不能参与 max 和 sum，否则会污染归一化。
- Formula/diagram hook: `softmax(x_i)=exp(x_i-m)/sum_j exp(x_j-m), m=max_j x_j`。
- Follow-up risk: 如果 mask 在 max 前处理不对，可能产生 NaN；如果 block size 过大，register pressure 和 SRAM 使用会上升。

### 3. FlashAttention 为什么比标准 attention 快？
- One-sentence answer: 它不 materialize 完整 `QK^T` 和 softmax 矩阵，而是分块在片上做 online softmax，主要减少 HBM IO。
- Expanded explanation: 标准 attention 会写出并读回 `S = QK^T` 和概率矩阵；FlashAttention 扫描 K/V blocks，维护每行最大值和归一化分母，把中间分数留在 SRAM/shared memory，FLOPs 类似但 HBM 往返少。
- Formula/diagram hook: `m_new=max(m_old,rowmax(S_j))`; `l_new=e^(m_old-m_new)l_old + rowsum(e^(S_j-m_new))`。
- Follow-up risk: causal mask、GQA/MQA、backward recompute 和 decode attention 的访问模式都可能改变 kernel 组织。

### 4. cuBLAS、CUTLASS、Triton 分别适合什么？
- One-sentence answer: cuBLAS 适合标准 GEMM baseline，CUTLASS 适合可控的高性能 C++/CUDA GEMM 定制，Triton 适合快速迭代自定义融合算子。
- Expanded explanation: cuBLAS 成熟但可定制性低；CUTLASS 提供 threadblock/warp/instruction tile、iterator 和 epilogue 抽象；Triton 开发效率高，适合 softmax、norm、简单 matmul 和 fused kernels。
- Formula/diagram hook: CUTLASS mental model: `threadblock tile -> warp tile -> MMA instruction tile -> epilogue`。
- Follow-up risk: Triton 不一定自动达到手写 CUDA/CUTLASS 上限；CUTLASS 调参复杂，alignment、layout、stage 数和 epilogue 都会影响结果。

### 5. CUTLASS epilogue 可以做哪些 LLM fusion？
- One-sentence answer: Epilogue 可以在 accumulator 写回前融合 bias、activation、scale、residual、dequant 或 quantize，减少中间 HBM 读写。
- Expanded explanation: GEMM mainloop 通常累加到 FP32 accumulator，epilogue 负责输出类型转换和后处理；LLM 中常见 `linear+bias+activation`、INT8/FP8 scale、residual add 等。
- Formula/diagram hook: `D = activation(alpha * Acc + beta * C + bias)`。
- Follow-up risk: 复杂 epilogue 会增加寄存器和访存，可能拖慢已经 compute-bound 的 GEMM。

### 6. Nsight Compute 里如何判断 kernel 是算力瓶颈还是访存瓶颈？
- One-sentence answer: 先看 roofline 和 achieved occupancy，再结合 SM throughput、DRAM throughput、Tensor Core utilization、eligible warps、bank conflict 等指标定位。
- Expanded explanation: 如果 DRAM 带宽接近峰值但 SM 利用不足，多半是 memory-bound；如果 Tensor Core/SM 接近饱和而带宽不高，多半是 compute-bound；如果两者都低，要查 occupancy、依赖 stall、访存不合并或 launch/batch 太小。
- Formula/diagram hook: `arithmetic_intensity = FLOPs / bytes_moved`；roofline 上限是 `min(peak_FLOPs, AI * peak_bandwidth)`。
- Follow-up risk: 单个指标容易误判，必须固定 workload、shape、dtype、GPU 和 baseline 后再比较。

### 7. vLLM Scheduler 和 BlockSpaceManager 分别解决什么？
- One-sentence answer: Scheduler 决定每步哪些请求执行，BlockSpaceManager 管理逻辑 KV block 到物理 KV block 的分配、释放、swap 和复用。
- Expanded explanation: Scheduler 在 waiting/running/swapped 队列和 token/KV budget 下做 continuous batching；BlockSpaceManager 像虚拟内存页表一样维护 block table，支持 PagedAttention、copy-on-write 和碎片控制。
- Formula/diagram hook: `request -> scheduler -> block table -> attention backend -> sampler`。
- Follow-up risk: 过度 preemption 或 swap 会造成 TPOT/P99 抖动；block size 太大浪费显存，太小增加间接寻址开销。

### 8. TensorRT-LLM 与 vLLM 的部署 tradeoff 是什么？
- One-sentence answer: vLLM 更灵活适合快速迭代，TensorRT-LLM 更适合稳定模型在 NVIDIA GPU 上追求编译优化和极限性能。
- Expanded explanation: TensorRT-LLM 通过 engine build、plugins、FMHA、paged KV、quantized kernels 和 inflight batching 优化；代价是构建时间、shape/version/GPU 绑定和调试不透明。具体默认行为 needs source verification。
- Formula/diagram hook: `HF weights -> convert -> trtllm-build -> serialized engine -> runtime/Triton server`。
- Follow-up risk: engine 的 max batch、max input、max sequence 和量化配置不匹配真实流量时，性能或可用性会退化。

### 9. INT4 weight-only quantization 的 pack/dequant 怎么讲？
- One-sentence answer: 权重按 group 存成 4-bit packed values 和 scale/zero，GEMM 时按 tile 读出并 dequant 到计算 dtype 参与累加。
- Expanded explanation: weight-only INT4 主要降低权重 HBM 带宽和显存占用，activation 通常仍是 FP16/BF16；group size 决定 scale 粒度，pack layout 要配合 vectorized load 和 Tensor Core/自定义 kernel。
- Formula/diagram hook: `w_fp ~= scale_g * (w_int4 - zero_g)`。
- Follow-up risk: group 太大质量差，group 太小 metadata 多；dequant 如果不能融合到 matmul，会抵消收益。

### 10. 分布式推理中 TP/PP/EP 怎么取舍？
- One-sentence answer: TP 降低单卡权重压力但引入每层通信，PP 降低单阶段显存但有 pipeline bubble，EP 适合 MoE 但引入 all-to-all 和负载均衡问题。
- Expanded explanation: Dense 70B 常用单节点 TP=4/8 让通信走 NVLink；跨节点 PP 要考虑 bubble 和故障域；MoE 的 EP 需要监控 per-expert token count、drop rate 和 all-to-all latency。
- Formula/diagram hook: `TP: split hidden/GEMM`; `PP: split layers`; `EP: split experts`。
- Follow-up risk: 跨 PCIe 或跨节点 TP 会显著影响 TPOT；MoE 热点 expert 会拖慢整个 batch。

## 小红书

### 1. SGLang 的 RadixAttention 解决什么问题？
- One-sentence answer: RadixAttention 用 radix tree 管理共享前缀 KV，让新请求复用最长匹配前缀，只计算未命中的 suffix。
- Expanded explanation: 多轮对话、few-shot、固定 system prompt 和模板化请求会产生稳定前缀；把前缀 KV 作为可复用 cache 能降低 TTFT 和 prefill 计算。
- Formula/diagram hook: `root -> system prompt -> examples -> user suffix`; lookup 是 longest prefix match。
- Follow-up risk: tokenizer、chat template、model version 或 adapter 变化都会破坏或污染 cache key。

### 2. RadixAttention 和 vLLM PagedAttention 的差异？
- One-sentence answer: RadixAttention 强调 prefix KV 复用，PagedAttention 强调 KV 物理内存分页和碎片管理。
- Expanded explanation: SGLang 的 radix tree 让相同 token 前缀共享 KV；vLLM 的 block table 让不同长度序列不需要连续 KV 内存，并支持 copy-on-write、swap 和动态 batch。
- Formula/diagram hook: `Radix: token prefix tree`; `Paged: logical block id -> physical block id`。
- Follow-up risk: 二者不是互斥能力，面试中要避免把 prefix caching 和 paged allocation 混为一谈。

### 3. Mooncake 为什么叫 KV-cache-centric？
- One-sentence answer: 它把 KV cache 从单 worker 内部状态提升为可调度资源，让请求路由围绕 KV locality、复用和传输成本做决策。
- Expanded explanation: 传统 GPU-centric 调度先找空闲 GPU，KV 跟着请求走；KV-cache-centric 调度先看可复用 KV 在哪里、迁移成本多大、decode 队列是否可承受，再决定 prefill/decode 放置。具体 Mooncake 实现细节 needs source verification。
- Formula/diagram hook: `score = queue_delay + transfer_cost + cache_miss_penalty + SLA_risk`。
- Follow-up risk: 全局 KV metadata 的一致性、故障恢复、eviction 和跨节点传输都会成为系统复杂度来源。

### 4. External KV cache 的收益和代价是什么？
- One-sentence answer: external KV 扩大了跨节点复用和 PD 分离空间，但会引入网络传输、metadata 正确性和 P99 抖动。
- Expanded explanation: Prefill worker 可以生产 KV，decode worker 消费 KV；若 KV 在远端，需要 RDMA/NVLink/PCIe 或 KV store 传输。长 prompt 的 KV 可能非常大，传输可能吃掉 PD 分离收益。
- Formula/diagram hook: `kv_transfer_bytes = prompt_tokens * 2 * layers * kv_heads * head_dim * bytes`。
- Follow-up risk: 如果 transfer time 进入 TTFT，长上下文会显著抬高首 token 延迟。

### 5. RBG 可以如何落地？
- One-sentence answer: RBG 可理解为按请求特征分组，再把相似请求路由到 prefix/KV locality 更好的 worker 或 pool。
- Expanded explanation: 分组依据可包括 system prompt hash、业务模板、模型版本、adapter、prompt length、SLA 和租户；目标是提升 prefix cache 命中并减少重复 prefill。RBG 的公开细节 needs source verification。
- Formula/diagram hook: `score = cache_hit_benefit - queue_penalty - transfer_cost - SLA_risk`。
- Follow-up risk: 过度追求 locality 会造成热门前缀热点，队列等待反而抵消 cache 收益。

### 6. KV Router 如何在 locality 和 load balance 之间权衡？
- One-sentence answer: 先找有 prefix/KV 命中的候选 worker，再用队列、KV 空间、传输成本和 SLA 风险打分选择。
- Expanded explanation: 长 prompt 或高命中前缀时 locality 权重大；短 prompt、长 decode 或某 worker 过热时 load balance 权重大。路由需要实时观测 cache hit、transfer latency、active sequences 和 queue wait。
- Formula/diagram hook: `if local_queue_wait - global_queue_wait > transfer_savings: route_global()`。
- Follow-up risk: 打分函数没有 hysteresis 会在热点和空闲节点之间来回震荡。

### 7. PD/EPD 分离的动机是什么？
- One-sentence answer: Prefill 偏 compute-bound、decode 偏 memory/scheduling-bound，分离后可以独立扩缩容和优化两类资源。
- Expanded explanation: 长 prompt prefill 会阻塞 decode step，导致 TPOT 抖动；PD 分离让 prefill pool 做大 token batch，decode pool 做 continuous batching。EPD 可理解为进一步把扩展阶段或外部 KV/执行资源纳入分离调度，具体术语细节 needs source verification。
- Formula/diagram hook: `TTFT = queue + prefill + KV transfer + first decode`; `TPOT = decode_step_latency`。
- Follow-up risk: KV transfer、角色切换和调度元数据可能让系统收益小于复杂度。

### 8. Dynamic PD 怎么设计控制策略？
- One-sentence answer: 根据 prefill waiting tokens、decode active sequences、TTFT/TPOT、GPU 利用率和 KV transfer 压力动态调整路由或节点角色。
- Expanded explanation: 轻量做法是动态路由；重做法是 worker drain 后从 decode 切到 prefill 或反向切换。需要 cooldown、hysteresis、最小角色保持时间和 pool 容量下限。
- Formula/diagram hook: `pressure_prefill = waiting_tokens / prefill_capacity`; `pressure_decode = active_sequences / decode_capacity`。
- Follow-up risk: 角色频繁切换会迁移 KV、扰动 cache，并导致 SLA 抖动。

### 9. Prefix cache 的 key 和 eviction 怎么设计？
- One-sentence answer: key 必须包含模型、tokenizer、adapter、dtype、版本和 token block hash，eviction 要按命中价值、内存成本和租户配额综合决策。
- Expanded explanation: 查找用 longest prefix match 或 block hash 连续匹配；命中后增加 refcount，写共享尾 block 时 copy-on-write。eviction 可用 LRU/LFU 加重算收益估计。
- Formula/diagram hook: `value = estimated_saved_prefill_time / memory_cost`。
- Follow-up risk: 跨租户共享、hash 冲突、版本漂移和 refcount 错误都可能造成数据泄露或错误输出。

### 10. 滚动升级或请求迁移时 KV cache 怎么处理？
- One-sentence answer: 对即将完成的请求 drain，长会话可 KV copy migration，短上下文可 recompute prefill，语义上要明确 partial/retry/idempotency。
- Expanded explanation: 请求状态包括 prompt tokens、generated tokens、sampling RNG、block table、model/adaptor version、stream offset 和 client ack；迁移要保证目标 worker 的模型版本和 cache key 兼容。
- Formula/diagram hook: `mark draining -> stop admit -> finish small -> copy KV or recompute -> resume/retry`。
- Follow-up risk: 流式输出和采样随机性让完全透明迁移很难，错误处理会导致重复 token 或输出不一致。

## 美团

### 1. LongCat/长上下文推理的主要瓶颈是什么？
- One-sentence answer: 长上下文同时放大 prefill attention 计算、KV cache 容量和 KV transfer 成本，必须结合 chunked prefill、context caching、KV 压缩和 admission control。
- Expanded explanation: 本地材料只把 LongCat 作为美团长上下文方向提示，具体实现 needs source verification；可回答通用长上下文设计：prefill 近似 `O(seq_len^2)`，KV 容量和传输是 `O(seq_len)`，decode 每步还要读历史 KV。
- Formula/diagram hook: `attention_prefill ~= O(L^2)`; `kv_capacity ~= O(L)`; `kv_transfer ~= O(L)`。
- Follow-up risk: 长上下文请求如果和短交互请求混部，容易拖高 TTFT/TPOT P99。

### 2. MoE routing 在推理里怎么做？
- One-sentence answer: Router 为每个 token 选择 top-k experts，系统按 expert 分发 token 并在 expert 计算后聚合结果。
- Expanded explanation: MoE 降低激活 FLOPs，但引入 expert parallel all-to-all、token reorder、load balance 和热点 expert 问题；推理要关注 per-expert token count、capacity、drop/overflow 和通信延迟。
- Formula/diagram hook: `y = sum_{e in TopK(router(x))} gate_e * Expert_e(x)`。
- Follow-up risk: top-k 分布不均会让最慢 expert 决定 batch latency。

### 3. TopK router fusion 为什么有价值？
- One-sentence answer: 将 router logits 的 softmax、top-k、scaling 和 token dispatch 前处理融合，可减少 HBM 往返和 kernel launch。
- Expanded explanation: 本地材料把 TopK router fusion 列为美团重点，具体公司实现 needs source verification；通用思路是 router 输出后立即完成概率归一化、选 expert、计算 gate weight 和必要 metadata，避免多个小 kernel 串联。
- Formula/diagram hook: `experts, gates = topk(softmax(router_logits), k)`。
- Follow-up risk: top-k 是不规则选择，fusion 后可能增加寄存器、分支和临时 buffer 管理复杂度。

### 4. N-gram cache / speculative decoding 如何加速？
- One-sentence answer: 用 prompt lookup 或 n-gram draft 先提出候选 token，再让 target model 一次验证多个 token，从而摊薄 target forward 次数。
- Expanded explanation: 当候选 token 接受率高时，每次 target forward 产生多个有效 token，TPOT 降低；n-gram 适合重复文本、模板化输出或上下文中可直接复制的片段。美团具体 N-gram 方案 needs source verification。
- Formula/diagram hook: `speedup ~= accepted_tokens_per_target_forward / (1 + draft_overhead)`。
- Follow-up risk: 接受率低、draft 成本高或 rollback/KV 管理复杂时可能变慢。

### 5. PDL scheduling 和普通 continuous batching 有什么不同？
- One-sentence answer: PDL scheduling 显式感知 prefill、decode 和 length 分布，在调度时控制长 prompt/长输出对短请求的干扰。
- Expanded explanation: 普通 continuous batching 每步动态补入请求；PDL 会把 prompt length、expected output length、TTFT/TPOT budget 纳入 admission 和 batch 选择，避免长请求长期占用 token budget。本地对 PDL 的公司细节 needs source verification。
- Formula/diagram hook: `priority = SLA_urgency - length_penalty + cache_benefit`。
- Follow-up risk: length 预测不准会导致错误优先级，影响公平性和 P99。

### 6. AllReduce + Residual Add + RMSNorm 为什么能融合？
- One-sentence answer: TP 场景下 AllReduce 得到完整 hidden 后常立刻做 residual add 和 RMSNorm，融合可减少一次或多次 HBM 读写与 launch。
- Expanded explanation: 分片 GEMM 后需要跨 GPU 聚合 hidden；如果聚合输出先写回 HBM，再单独 add/norm，会产生额外访存。融合思路是在通信完成或通信 epilogue 附近直接做 add、sum of squares、rsqrt 和 scale。具体美团实现 needs source verification。
- Formula/diagram hook: `rmsnorm(x+r)=((x+r)/sqrt(mean((x+r)^2)+eps))*w`。
- Follow-up risk: 融合 communication 和 reduction 会受 NCCL/custom all-reduce 接口限制，且 hidden 维 reduction 需要处理跨线程/跨块同步。

### 7. Softmax + TopK + Scaling fusion 有哪些边界？
- One-sentence answer: 它适合 router 这类小向量归一化和选择，但要小心数值稳定、top-k tie、稀疏输出和寄存器压力。
- Expanded explanation: router logits 先减 max 做稳定 softmax，再 top-k，最后输出 expert id 和 gate scale；融合后少写概率矩阵，但 top-k 的比较网络和动态索引会让 kernel 更复杂。
- Formula/diagram hook: `p_i = exp(l_i-m)/sum_j exp(l_j-m); select top-k(p)`。
- Follow-up risk: 如果 expert 数很大或 top-k 逻辑复杂，专用 kernel 可能被寄存器和局部内存拖慢。

### 8. MoE Expert Parallelism 的 all-to-all 怎么优化？
- One-sentence answer: 通过 token bucketing、容量控制、负载均衡 loss/策略、通信计算重叠和局部 expert 放置降低 all-to-all 开销。
- Expanded explanation: 推理中每步要把 token 发到对应 expert 所在 GPU，再收回结果；可以按 expert 聚合 token，减少小包；热 expert 可复制或调整路由；通信可与 local expert compute overlap。
- Formula/diagram hook: `route tokens -> all-to-all -> expert GEMM -> all-to-all/gather -> combine`。
- Follow-up risk: batch 太小通信效率差，batch 太大又拉高 latency；热点 expert 会造成尾延迟。

### 9. 长上下文 KV cache 如何减压？
- One-sentence answer: 用 GQA/MQA、KV 量化、sliding/window attention、context caching、offload 和长上下文专用队列一起控制容量和延迟。
- Expanded explanation: KV 单 token 大小由层数、KV heads、head dim 和 dtype 决定；GQA/MQA 减少 KV heads，KV FP8/INT8 减少容量和带宽，offload 扩容但影响 P99，context caching 对稳定长前缀收益大。
- Formula/diagram hook: `kv_bytes_per_token = 2 * layers * kv_heads * head_dim * bytes`。
- Follow-up risk: KV 压缩和窗口裁剪可能影响长程检索质量，必须用任务级 eval 或 A/B 验证。

### 10. 量化部署怎么从方案走到上线？
- One-sentence answer: 先选权重量化/KV 量化/activation 精度，再做校准、离线质量评估、性能 benchmark、灰度和回滚。
- Expanded explanation: INT4 weight-only 省权重带宽，FP8/INT8 KV 省上下文容量，activation/FP8 需要更严格 kernel 和数值支持；上线不能只看 tokens/s，还要看输出质量、TTFT/TPOT、OOM、错误率和长上下文表现。
- Formula/diagram hook: `memory_saved -> latency_delta -> quality_delta -> rollout decision`。
- Follow-up risk: 量化 scale 粒度、校准集不匹配和 kernel fallback 都可能让线上收益低于离线预期。

## 15-Minute Review Checklist

1. 2 min: 复述 prefill/decode 差异：prefill 影响 TTFT、decode 影响 TPOT，分别偏 compute 和 bandwidth/scheduling。
2. 2 min: 背 KV 公式：`2 * layers * kv_heads * head_dim * bytes * tokens`，并能估算 transfer cost。
3. 2 min: 腾讯重点：GEMM 优化顺序、FlashAttention online softmax、Nsight roofline、CUTLASS epilogue、vLLM/TensorRT-LLM tradeoff。
4. 2 min: 小红书重点：RadixAttention、Mooncake external KV、RBG/KV Router、PD/EPD、prefix cache key、rolling migration。
5. 2 min: 美团重点：LongCat/long context、MoE routing、TopK fusion、N-gram speculative decoding、PDL、AllReduce + RMSNorm fusion、quant deploy。
6. 2 min: 每题都用四段式：one-sentence answer -> expanded explanation -> formula/diagram hook -> follow-up risk。
7. 2 min: 主动说验证：用 TTFT/TPOT/P99、GPU util、HBM bandwidth、KV usage、prefix hit rate、acceptance rate 和 quality eval 闭环。
8. 1 min: 对公司或版本相关细节加一句 `needs source verification`，避免把本地材料提示说成已验证事实。
