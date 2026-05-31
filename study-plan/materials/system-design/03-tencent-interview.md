# 腾讯 JD 定制深度问答

## 学习目标

1. 能把 CUDA kernel 优化讲成完整闭环: profiling -> bottleneck -> optimization -> verification。
2. 能从手写 GEMM 推导 shared memory tiling, register tiling, vectorized load, Tensor Core 的收益和约束。
3. 能解释 FlashAttention 为什么减少 HBM IO, online softmax 如何保证数值正确。
4. 能说明 fused kernel 的适用边界, 以及 register pressure, occupancy, memory traffic 的 tradeoff。
5. 能用 CUTLASS 的 tile 抽象, epilogue 和 threadblock/warp/instruction 分层描述自定义 kernel。
6. 能从源码结构层面解释 vLLM 的 Scheduler, BlockSpaceManager, Worker, ModelRunner 和 PagedAttention。

## 前置知识

- CUDA execution model: grid, block, thread, warp, SM, occupancy。
- Memory hierarchy: register, shared memory, L1/L2, HBM, coalescing, bank conflict。
- GEMM 基础: C = A x B, M/N/K 分块, arithmetic intensity, roofline。
- Attention 基础: QK^T, softmax, V, causal mask, MHA/MQA/GQA。
- LLM serving 基础: prefill/decode, KV cache, continuous batching, tensor parallel。
- vLLM 基础结构: request, sequence group, block table, worker 执行模型。

## 核心内容

### 1. CUDA Kernel 优化

- 手写 GEMM 的优化路线不是只背 shared memory, 而是从数据复用和访存带宽推导。
- 优化顺序通常是: 正确性 -> coalesced global load/store -> shared memory tile -> register tile -> vectorized load -> double buffering -> Tensor Core。
- 面试中要能说清楚每一步改变了什么瓶颈, 如何用 Nsight Compute 指标验证。

### 2. FlashAttention 与 Fused Kernel

- FlashAttention 的关键是 IO-aware tiling 和 online softmax, 不是近似算法。
- Fused kernel 的目标是减少中间结果 HBM 往返和 launch overhead, 但 fusion 会增加寄存器压力和实现复杂度。
- 对腾讯这类岗位, 重点是能把原理落到 kernel 组织, mask, backward, decode 变体。

### 3. CUTLASS 自定义 Kernel

- CUTLASS 把 GEMM 拆成 threadblock tile, warp tile, instruction tile。
- Epilogue 负责把 accumulator 写回前做 bias, activation, scaling, residual 等后处理。
- 自定义 kernel 时要说清楚 iterator, layout, alignment, pipeline stage 和 epilogue visitor 的作用。

### 4. vLLM 源码与 PagedAttention

- Scheduler 负责选择哪些 sequence group 进入本轮执行, 并处理 prefill/decode 混排。
- BlockSpaceManager 负责逻辑 block 到物理 KV block 的分配, 释放, 复用和换出。
- Worker/ModelRunner 负责把 scheduler 输出转成模型前向所需 metadata, 调用 attention backend 和采样。
- PagedAttention 用 block table 解耦请求逻辑位置和 KV cache 物理位置, 支持 copy-on-write 和 swap。

### 5. 性能优化闭环

- profiling: 先定义 workload, 输入尺寸, batch, dtype, GPU, baseline。
- bottleneck: 用 roofline 判断 compute-bound 还是 memory-bound, 再看具体指标。
- optimization: 每次只改一个主要因素, 保留可回滚 baseline。
- verification: 数值正确性, latency/throughput, p50/p99, 显存占用, 稳定性都要验证。

## 完整的问答/题目

### Q1: 如果让你手写一个 FP16 GEMM kernel, 你会如何逐步优化到接近 cuBLAS?

**考察点:** CUDA 编程基本功, GEMM 数据复用, Tensor Core, 性能分析。

**参考答案:**

我会先写 naive GEMM 做 correctness baseline: 每个线程计算一个 C[m,n], 循环 K, 直接从 global memory 读取 A[m,k] 和 B[k,n]。这个版本的问题是同一行 A 和同一列 B 被大量重复读取, arithmetic intensity 很低, 通常受 HBM 带宽限制。

第一步做 shared memory tiling。一个 thread block 负责 C 的 BM x BN tile, K 维按 BK 分块。线程协作把 A 的 BM x BK 和 B 的 BK x BN 加载到 shared memory, 再在 shared memory 内循环累加。这样 A tile 被 BN 方向复用, B tile 被 BM 方向复用。

第二步做 register tiling。每个线程不只算一个元素, 而是算 TM x TN 个元素, accumulator 放在寄存器里。这样从 shared memory 读入的一组 A/B 值可以服务多个 accumulator, 降低 shared memory 压力, 提升计算密度。

第三步做访存优化。global load 要 coalesced, 尽量使用 half2, float4, int4 等向量化 load/store。shared memory layout 要避免 bank conflict, 常见做法是 padding 或 swizzle。K tile 的加载可以用 double buffering 或 cp.async, 在计算当前 tile 时预取下一 tile。

第四步使用 Tensor Core。FP16/BF16 输入通常用 MMA 指令, accumulator 用 FP32。kernel 组织会变成 threadblock tile -> warp tile -> mma instruction tile。要保证输入矩阵 layout, alignment, K 维倍数和 shared memory 排布满足 Tensor Core 高效访问。

最后用 profiling 验证。看 achieved occupancy, sm throughput, dram throughput, tensor core utilization, shared memory bank conflict, eligible warps per cycle。如果 tensor core 利用率低, 查 tile size 和 pipeline; 如果 dram 带宽高但 FLOPS 低, 查数据复用; 如果 occupancy 很低, 查 register 和 shared memory 使用。

**追问方向:**

- BM, BN, BK 怎么选? 受 shared memory, register, occupancy, warp 数量共同约束。
- 为什么 register tiling 可能降低 occupancy? accumulator 太多会增加寄存器使用。
- cp.async 和普通 global load 的区别是什么? 如何用多 stage pipeline 隐藏延迟。
- Tensor Core 对 alignment 和 layout 有什么要求?

### Q2: 手写 GEMM 中如何处理非整除尺寸和边界 mask?

**考察点:** 工程正确性, block mask, aligned/non-aligned 测试意识。

**参考答案:**

真实输入的 M/N/K 不一定是 tile size 的整数倍。加载 A/B tile 时要检查 row < M, col < N, k < K, 越界位置填 0。写回 C 时也要检查 row < M 且 col < N。这个 mask 必须同时覆盖 global load 和 store, 否则会读越界或写坏结果。

性能上, 边界 block 通常占比小, 可以在主 kernel 内用 predicate 处理。对于极端小矩阵或大量 ragged shape, 可以单独走 specialized kernel。测试上必须覆盖 aligned shape, 例如 4096x4096x4096, 也要覆盖 non-aligned shape, 例如 4097x4093x4111, 并和 torch.matmul 或 cuBLAS 对比误差。

**追问方向:**

- mask 分支会不会影响 warp divergence? 如何减少影响?
- 为什么越界 load 通常填 0 而不是跳过整个循环?
- FP16 GEMM 的误差阈值如何设定?

### Q3: FlashAttention 为什么能更快? 它的 online softmax 怎么保证正确?

**考察点:** Attention IO 复杂度, 数值稳定 softmax, kernel tiling。

**参考答案:**

标准 attention 会显式生成 S = QK^T, 大小是 seq_len x seq_len。长序列下 S 很大, 需要写入 HBM, softmax 再读回, 然后和 V 相乘。FlashAttention 不 materialize 完整 S, 而是把 Q/K/V 分块放进 SRAM/shared memory, 对每个 Q block 依次扫描 KV block, 边算 score 边更新 softmax 统计量和输出。

online softmax 维护每行的 m 和 l。m 是目前见过的最大 score, l 是以 m 为基准的 exp sum。处理新 block 时得到局部 score S_j, 计算 m_new = max(m_old, rowmax(S_j))。旧分母要乘 exp(m_old - m_new), 新分子用 exp(S_j - m_new)。输出 O 也要同样 rescale:

```text
l_new = exp(m_old - m_new) * l_old + rowsum(exp(S_j - m_new))
O_new = exp(m_old - m_new) * O_old + exp(S_j - m_new) @ V_j
```

最后 O = O_new / l_new。这个过程和全量 softmax 在数学上等价, 只是把归一化分块增量完成。

FlashAttention 更快的核心不是少算 QK^T 的 FLOPs, 而是减少 HBM IO。它把中间 score 和概率矩阵留在片上, 只读 Q/K/V, 写 O 和少量 logsumexp。对长序列尤其有效。

**追问方向:**

- causal mask 在 block 级和元素级分别怎么处理?
- FlashAttention backward 为什么可以 recompute score?
- FlashAttention-2 主要改进在哪里?
- GQA/MQA 下 K/V head 共享对 kernel 有什么影响?

### Q4: Fused kernel 什么时候值得做? 举一个 LLM 中的例子。

**考察点:** kernel fusion 判断, memory traffic, register pressure。

**参考答案:**

值得 fusion 的典型场景是多个轻量 elementwise 或 reduction kernel 串联, 中间结果只被下一步使用。如果不 fusion, 每一步都要读写 HBM, 还要付 kernel launch overhead。LLM 中常见例子包括 bias + GELU, residual add + RMSNorm, softmax 的 max/sub/exp/sum/div, GEMM epilogue 中的 bias/activation。

例如 RMSNorm + residual。输入 hidden 和 residual 先相加, 然后对 hidden dimension 做 sum(x^2), 计算 inv_rms, 最后乘 weight。融合后每个 row 一个 block 或多个 block, 中间的 x 可以尽量留在寄存器或 shared memory, 避免 add 结果先写 HBM 再读回来。

但 fusion 不是越多越好。如果融合后每个线程需要保存太多中间变量, register pressure 增大, occupancy 下降, 可能更慢。如果两个 kernel 都是大 GEMM 且 compute-bound, 强行融合通常收益有限。判断方法是 profiling: 如果瓶颈是 HBM traffic 或 launch overhead, fusion 更可能有效; 如果 tensor core 已经满, 优先调整 GEMM 本身。

**追问方向:**

- 怎么量化 fusion 是否减少了 memory traffic?
- fusion 后 occupancy 降低怎么办?
- 为什么 GEMM + epilogue 是常见 fusion 形态?

### Q5: CUTLASS 的 tile 抽象如何帮助你写自定义 GEMM?

**考察点:** CUTLASS 分层模型, tile shape, pipeline。

**参考答案:**

CUTLASS 把 GEMM 映射成多层 tile。threadblock tile 定义一个 CTA 计算的 C 子矩阵, 例如 128x128x32。warp tile 把 threadblock tile 分给多个 warp, 例如每个 warp 负责 64x64。instruction tile 对应 mma 指令, 例如 m16n8k16。

这种抽象让开发者用 shape 参数表达数据复用和并行层级, 而不用从零写所有加载, 排布和 MMA 细节。CUTLASS iterator 负责从 global memory 按 layout 和 alignment 加载 A/B tile, shared memory layout 负责减少 bank conflict, mainloop 负责多 stage pipeline, epilogue 负责从 accumulator 写回。

调参时我会关注 ThreadblockShape, WarpShape, InstructionShape, stages, alignment, layout, math instruction。大矩阵追求 Tensor Core 利用率和 pipeline 饱和; 小矩阵或 skinny GEMM 要避免过大的 tile 造成浪费。

**追问方向:**

- ThreadblockShape 和 WarpShape 不匹配会有什么问题?
- stages 增加一定更好吗?
- CUTLASS iterator 解决了哪些手写 kernel 的复杂度?

### Q6: CUTLASS epilogue 能做什么? 自定义 epilogue 有哪些坑?

**考察点:** epilogue fusion, accumulator 类型, 数值和访存。

**参考答案:**

Epilogue 是 GEMM mainloop 完成后, accumulator 写回 C 之前的阶段。它可以做 alpha * accumulator + beta * C, bias add, activation, clamp, scale, residual add, quantize 等。LLM 中常见的是 linear 后接 bias/GELU, 或者 int8/fp8 GEMM 后做 dequantize 和 scale。

自定义 epilogue 的关键是理解 accumulator 通常是 FP32, 输出可能是 FP16/BF16/FP8。要处理好类型转换, rounding, saturation 和误差。还要注意 epilogue 的访存模式: 如果需要读取 bias 或 residual, 要保证 coalesced, 并处理 broadcasting。复杂 epilogue 会增加寄存器和指令数量, 可能拖慢原本高效的 MMA mainloop。

工程上我会先实现 reference epilogue 对比 PyTorch, 再在 CUTLASS profiler 或自建 benchmark 中比较只做 GEMM 和 GEMM+epilogue 的端到端收益。

**追问方向:**

- Bias 是 per-column 时如何映射到 C 的 tile?
- int8/fp8 epilogue 中 scale 放在哪里更合理?
- epilogue fusion 如何影响数值回归测试?

### Q7: vLLM Scheduler 在 serving 中解决什么问题?

**考察点:** continuous batching, prefill/decode 调度, SLA tradeoff。

**参考答案:**

vLLM Scheduler 的核心任务是每一轮决定哪些 sequence group 能进入模型执行。它要在 token budget, KV block budget, running/waiting/swapped 队列之间做选择, 让 GPU 尽量满, 同时避免单个长 prompt 或长 decode 请求拖垮延迟。

prefill 阶段一次处理 prompt 的多个 token, compute-bound 更明显; decode 阶段每个请求通常只生成一个 token, memory-bound 更明显。Scheduler 要支持 prefill 和 decode 混排, 也要支持 chunked prefill, 把超长 prompt 切成多轮, 避免 TTFT 抖动。

它还会和 BlockSpaceManager 协作。只有当某个请求能分配到足够 KV cache block 时, 才能被调度。若显存不足, 可能触发抢占, recompute 或 swap。调度结果会被转成 model input metadata, 交给 Worker/ModelRunner 执行。

**追问方向:**

- continuous batching 为什么比静态 batching 更适合 LLM serving?
- chunked prefill 改善了什么, 又牺牲了什么?
- 抢占策略选择 recompute 还是 swap 的依据是什么?

### Q8: BlockSpaceManager 和 PagedAttention 的 block table 是怎么配合的?

**考察点:** KV cache 内存管理, logical/physical block 映射。

**参考答案:**

PagedAttention 把每个 sequence 的 KV cache 切成固定大小 block。sequence 逻辑上第 i 个 block 不要求在物理显存中连续, 而是通过 block table 映射到物理 block id。attention kernel 根据 block table 找到每个 token 的 K/V 地址。

BlockSpaceManager 负责物理 block 的生命周期。新请求进入时分配 block; decode 增加 token 时如果当前 block 满了就追加 block; 请求结束时释放 block; prefix cache 或 beam search 共享 block 时维护引用计数。

这种设计类似 OS page table。好处是减少连续大块显存分配需求, 支持不同长度请求的动态增长, 也支持把部分 block swap 到 CPU。代价是 attention kernel 多了一层间接寻址, block size 选择会影响碎片和访存效率。

**追问方向:**

- block size 选大或选小分别有什么问题?
- block table 在 attention kernel 中如何影响 memory coalescing?
- 为什么 PagedAttention 能缓解 KV cache 碎片?

### Q9: PagedAttention 中 copy-on-write 怎么工作?

**考察点:** prefix sharing, beam search, 引用计数, 正确性。

**参考答案:**

copy-on-write 用于多个 sequence 共享同一段 KV cache 的情况, 例如 prompt prefix caching 或 beam search。共享 block 的 ref_count 大于 1 时, 如果某个 sequence 要向这个 block 写入新 token, 不能直接原地写, 否则会污染其它 sequence。系统会先分配一个新物理 block, 把原 block 内容复制过去, 更新当前 sequence 的 block table, 然后在新 block 上写入。

如果追加 token 时落在一个全新的 block, 可以直接分配新 block, 不需要 copy。只有写入共享且未满的最后一个 block 时才触发 COW。实现重点是 ref_count 更新必须和 block table 更新一致, 否则会出现悬挂引用或错误复用。

**追问方向:**

- COW 会带来什么额外开销? 如何减少?
- prefix cache 的 block 什么时候可以被 eviction?
- beam search 中不同 beam 何时共享, 何时分叉?

### Q10: vLLM 的 swap 机制解决什么问题? 它有什么风险?

**考察点:** 显存压力处理, latency tradeoff, 调度策略。

**参考答案:**

swap 用于 GPU KV cache block 不足时, 把部分 sequence 的 KV block 换到 CPU 内存, 给当前更高优先级的请求腾出 GPU block。之后该 sequence 恢复执行时, 再把 block swap in 回 GPU。

它解决的是 serving 中的瞬时显存压力和长尾请求问题。相比直接拒绝请求或等待, swap 可以提高系统吞吐和接纳能力。但风险也明显: PCIe/NVLink 传输会增加延迟, swap 过多会导致抖动, 如果调度策略不当可能反复 swap in/out。

工程上要限制 swap 队列大小, 记录每次 swap 的 block 数和耗时, 对接近完成的请求或高优先级请求减少 swap。对于 decode 阶段, swap in 的延迟会直接影响 TPOT 和 p99。

**追问方向:**

- swap 和 recompute 如何取舍?
- 哪些请求适合被 preempt?
- 如何监控 swap 导致的性能退化?

### Q11: Worker 和 ModelRunner 在 vLLM 执行链路中分别做什么?

**考察点:** 源码结构, runtime metadata, distributed worker。

**参考答案:**

Scheduler 做的是控制面决策, Worker/ModelRunner 做的是执行面。Worker 通常对应一个 GPU 进程或一个 tensor parallel rank, 负责接收 scheduler 产出的本轮执行计划, 管理 device 上的 cache, 通信和模型执行。

ModelRunner 把 sequence metadata 转成模型 forward 需要的张量: input_ids, positions, slot_mapping, block_tables, attention metadata, sampling metadata。然后调用模型 forward, attention backend 会根据 block table 读取 KV cache 并写入新 KV。最后 logits 进入 sampler, 生成下一个 token。

在多 GPU 下, Worker 还要处理 tensor parallel 通信, NCCL group, 权重加载和 device 初始化。面试时可以把链路讲成: request -> scheduler -> scheduled metadata -> worker execute_model -> model runner prepare input -> model forward -> sampler -> scheduler 状态更新。

**追问方向:**

- slot_mapping 的作用是什么?
- prefill 和 decode 的 attention metadata 有什么差异?
- tensor parallel worker 之间需要同步哪些结果?

### Q12: 给你一个线上 LLM serving 性能问题, 你如何从 profiling 到验证完成优化?

**考察点:** 性能优化经验, 指标体系, 工程闭环。

**参考答案:**

先定义问题和 workload。比如用户反馈 p99 TTFT 高, 要确认模型, GPU, batch, prompt length 分布, output length 分布, QPS, prefix cache 命中率, 是否混合 prefill/decode。不要一上来改 kernel。

第二步采集指标。系统层看 TTFT, TPOT, throughput, queue time, GPU utilization, HBM usage, KV block usage, swap 次数。kernel 层用 Nsight Systems 看 timeline 和 launch gap, 用 Nsight Compute 看热点 kernel 的 dram throughput, sm throughput, tensor core utilization, occupancy。

第三步定位瓶颈。如果 queue time 高且 GPU 不满, 可能是 scheduler 或 batching 问题。如果 decode kernel dram throughput 高, 可能是 KV cache 访存瓶颈。如果 prefill GEMM tensor core utilization 低, 查 shape, dtype, kernel selection。如果 p99 高且 swap 多, 查 block budget 和 eviction。

第四步提出单点优化并验证。例如开启 chunked prefill 降低长 prompt 阻塞; 调整 max_num_batched_tokens 提高吞吐; 优化 fused RMSNorm 降低 HBM traffic; 调整 block size 降低碎片。每次改动都和 baseline 对比, 验证数值正确性, p50/p99, throughput, 显存, 稳定运行时间。

**追问方向:**

- Nsight Systems 和 Nsight Compute 分别适合看什么?
- 如何避免 benchmark 只优化平均值而牺牲 p99?
- 怎么设计 A/B 实验证明优化有效?

## 追问方向与深入点

- 如果用户 prompt 长度分布变化, Scheduler 参数怎么调?
- 如果 attention kernel 是 memory-bound, 你会先改 block table 访问还是改 batch 策略?
- 如果 fused kernel 数值误差变大, 如何定位是 reduction 顺序还是 dtype cast?
- 如果 CUTLASS kernel 比 cuBLAS 慢 40%, 你会检查哪些 shape 和 profiler 指标?
- 如果 prefix cache 命中率高但延迟没有下降, 可能是什么原因?
- 如果 swap 次数很少但 p99 很高, 是否还应该盯 KV cache?
- 如果 Tensor Core 利用率很高但端到端吞吐低, 说明什么?
- 如何证明一次 kernel 优化没有破坏 aligned 和 non-aligned shape?

## 评分标准

### 优秀

- 能把 kernel 优化和系统调度联起来, 不只背单点术语。
- 能讲清楚 GEMM/FlashAttention/PagedAttention 的数据流和内存流。
- 能主动提 profiling 指标, baseline, correctness check 和 p99 验证。
- 能描述 vLLM 关键组件职责及它们之间的调用关系。

### 合格

- 能解释 shared memory tiling, online softmax, block table 的基本原理。
- 能说出 Scheduler, BlockSpaceManager, Worker, ModelRunner 的大致作用。
- 能给出常见优化手段, 但对验证指标和 tradeoff 说明不够完整。

### 风险

- 把 FlashAttention 说成近似 attention 或只说减少计算量。
- 只说 fusion 一定更快, 不提 register pressure 和 occupancy。
- 不了解 PagedAttention 的 logical block 到 physical block 映射。
- 性能优化没有 baseline, 没有 profiler, 没有正确性验证。

## 复习卡片 15 张

1. **GEMM naive 瓶颈是什么?** Global memory 重复读取导致 arithmetic intensity 低, 通常带宽受限。
2. **shared memory tiling 的收益是什么?** 把 A/B tile 放到片上复用, 减少 HBM 访问。
3. **register tiling 的风险是什么?** accumulator 增多会提高寄存器使用, 可能降低 occupancy。
4. **Tensor Core kernel 的关键约束是什么?** dtype, layout, alignment, tile shape, K 维粒度。
5. **FlashAttention 快在哪里?** 不 materialize NxN score/probability, 减少 HBM IO。
6. **online softmax 维护什么?** 每行最大值 m, 分母 l, 以及需要 rescale 的输出 O。
7. **fused kernel 主要减少什么?** 中间结果 HBM 读写和 kernel launch overhead。
8. **fusion 的主要副作用是什么?** register pressure, occupancy 下降, 代码复杂度和数值验证成本。
9. **CUTLASS 三层 tile 是什么?** threadblock tile, warp tile, instruction tile。
10. **CUTLASS epilogue 做什么?** accumulator 写回前做 bias, activation, scale, residual, quantization 等。
11. **vLLM Scheduler 决定什么?** 每轮哪些 sequence group 执行, 消耗多少 token/block budget。
12. **BlockSpaceManager 管什么?** KV physical block 的分配, 释放, 引用计数, swap。
13. **PagedAttention block table 解决什么?** 逻辑 KV 位置到非连续物理 block 的映射。
14. **copy-on-write 何时触发?** 共享 block ref_count > 1 且某 sequence 要写入该 block。
15. **性能优化闭环是什么?** 明确 workload, profiling, 定位瓶颈, 单点优化, 正确性和性能验证。
