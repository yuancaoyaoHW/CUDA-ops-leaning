# 技术面试题答案与深度解析

对应 `01_technical_questions.md` 中的 55 道题，提供标准答案、追问方向和加分要点。

---

## 一、CUDA 与 GPU 架构

### A1 [L2] CUDA 执行模型

**标准答案**：
- Grid → Block → Thread 是逻辑层次；硬件上 Block 被调度到 SM，Thread 以 32 个为一组形成 Warp 执行
- 一个 SM 上最多同时调度的 warp 数取决于架构（A100: 64 warps/SM, H100: 64 warps/SM）
- Occupancy 受限于三个因素：每个 block 使用的 shared memory、每个 thread 使用的 register 数量、每个 SM 的最大 block 数

**追问方向**：
1. Occupancy 100% 一定性能最好吗？（不一定，有时降低 occupancy 换取更多 register/shared memory 反而更快）
2. 如何用 CUDA Occupancy Calculator 分析？
3. Warp scheduler 如何隐藏延迟？

**加分要点**：能结合具体 kernel 分析 occupancy 瓶颈，知道 latency hiding 的原理。

---

### A2 [L2] Warp Divergence

**标准答案**：
- 同一 warp 内的 32 个 thread 必须执行相同指令；遇到分支时，不同路径串行执行，未执行路径的 thread 被 mask
- 示例：`if (threadIdx.x % 2 == 0) { ... } else { ... }` 导致 50% 的 thread 在每个分支被 mask
- 优化：将分支条件对齐到 warp 边界，如 `if (threadIdx.x / 32 < N) { ... }`
- Volta+ 的 Independent Thread Scheduling：每个 thread 有独立 PC 和 call stack，divergent thread 可以被独立调度，但仍有性能损失

**追问方向**：
1. 如何用 NCU 检测 divergence？（看 branch efficiency 指标）
2. Volta 之前和之后 divergence 的行为差异？
3. 循环中的 divergence 如何处理？

**加分要点**：知道 Volta 架构变化的具体影响，能用 profiler 定位。

---

### A3 [L2] GPU 内存层次

**标准答案**：
| 层次 | 容量 | 延迟 | 带宽 |
|------|------|------|------|
| Register | ~256KB/SM | 0 cycle | - |
| Shared Memory | 48-164KB/SM | ~20 cycles | ~19 TB/s |
| L1 Cache | 与 shared memory 共享 | ~30 cycles | ~19 TB/s |
| L2 Cache | 40-50MB (A100) | ~200 cycles | ~5 TB/s |
| Global Memory (HBM) | 40-80GB | ~400 cycles | 1.5-3.35 TB/s |

- Ampere 架构：L1 和 Shared Memory 共享同一物理 SRAM，可配置比例（如 A100 最大 164KB shared memory）
- Hopper 架构：引入 TMA（Tensor Memory Accelerator）加速数据搬运

**追问方向**：
1. 如何判断 kernel 是 compute-bound 还是 memory-bound？（Roofline model）
2. L2 cache 的 residency control 是什么？
3. HBM 带宽如何计算理论峰值？

**加分要点**：能画 Roofline model，知道如何计算 arithmetic intensity。

---

### A4 [L3] Bank Conflict

**标准答案**：
- Shared Memory 被分为 32 个 bank，每个 bank 宽 4 bytes，连续 4 bytes 映射到连续 bank
- 当同一 warp 中多个 thread 访问同一 bank 的不同地址时，产生 bank conflict，访问被串行化
- 2D 数组示例：`shared[threadIdx.x][threadIdx.y]` 如果列数是 32 的倍数，同一列的访问会冲突
- Padding 解决：声明为 `shared[N][M+1]`，错开 bank 映射
- NCU 检测：查看 `l1tex__data_bank_conflicts_pipe_lsu_mem_shared` 指标

**追问方向**：
1. broadcast 机制是什么？（多个 thread 读同一地址不算 conflict）
2. 64-bit 和 128-bit 访问的 bank conflict 规则？
3. 除了 padding 还有什么方法？（swizzle pattern）

**加分要点**：知道 swizzle 技术，能在实际 kernel 中应用。

---

### A5 [L3] GEMM 优化

**标准答案**：
1. **Naive**：每个 thread 计算 C 的一个元素，global memory 访问量 O(N³)
2. **Tiling**：将 C 分块，每个 block 负责一个 tile，用 shared memory 缓存 A/B 的子矩阵
3. **Register blocking**：每个 thread 计算多个输出元素（如 8×8），减少 shared memory 访问
4. **Double buffering**：用两块 shared memory 交替，overlap 数据加载和计算
5. **Vectorized load**：用 float4/int4 一次加载 128 bits，提高带宽利用率
6. **Warp-level MMA**：使用 Tensor Core（wmma/mma PTX 指令）

理论性能上限 = min(计算峰值, 带宽 × arithmetic_intensity)

**追问方向**：
1. CUTLASS 的 epilogue fusion 是什么？
2. Split-K 策略适用什么场景？
3. Tensor Core 的数据布局要求？

**加分要点**：了解 CUTLASS 架构，能解释 warp tile 和 thread tile 的关系。

---

### A6 [L2] Coalesced Memory Access

**标准答案**：
- Coalesced access：同一 warp 的 32 个 thread 访问连续的 128 bytes 内存区域，可合并为一次事务
- AoS vs SoA：AoS 中相邻 thread 访问的数据不连续（stride = struct size），SoA 中连续
- 矩阵转置：naive 实现中写入是 non-coalesced（stride = N），优化方法是用 shared memory 做中转——coalesced 读入 shared memory，再 coalesced 写出

**追问方向**：
1. 128B vs 32B sector 的区别？
2. 如何用 NCU 检测 non-coalesced access？
3. 对于无法避免的 non-coalesced 访问，有什么缓解方法？

**加分要点**：知道 sector 粒度的访问模式分析。

---

### A7 [L2] CUDA Stream 与异步执行

**标准答案**：
- Stream 是 GPU 上的命令队列，同一 stream 内操作按序执行，不同 stream 可并行
- Pipeline 重叠：Stream 0 做 H2D，Stream 1 做 Kernel，Stream 2 做 D2H
- 无法重叠的情况：同一 stream 内的操作；使用 default stream（与所有 stream 同步）；硬件资源冲突（如只有一个 copy engine）

**追问方向**：
1. cudaStreamSynchronize vs cudaDeviceSynchronize 的区别？
2. CUDA Event 如何实现跨 stream 依赖？
3. CUDA Graph 与 Stream 的关系？

**加分要点**：了解 CUDA Graph 的使用场景和性能收益。

---

### A8 [L3] Reduction 优化

**标准答案**：
优化层次：
1. 避免 warp divergence：用 stride 从大到小递减
2. 避免 bank conflict：sequential addressing
3. Warp-level reduction：最后 32 个元素用 `__shfl_down_sync`
4. Grid-level reduction：多 block 结果用 atomic 或两次 kernel launch 汇总
5. Cooperative groups：更灵活的同步原语

**追问方向**：
1. 为什么最后一个 warp 不需要 `__syncthreads()`？
2. `__shfl_down_sync` 的 mask 参数作用？
3. 如何处理非 2 的幂次长度？

**加分要点**：能手写完整的 warp shuffle reduction。

---

### A9 [L3] Nsight Compute 使用

**标准答案**：
- NCU 是 kernel 级 profiler，分析单个 kernel 的性能瓶颈
- 关键指标：SM throughput、memory throughput、occupancy、warp stall reasons
- Speed of Light (SOL)：显示 kernel 达到硬件峰值的百分比
- Memory chart：显示各级内存的 hit rate 和 throughput
- 使用流程：`ncu --set full -o profile ./app`，然后用 GUI 分析

**追问方向**：
1. 如何判断 kernel 是 compute-bound 还是 memory-bound？
2. Warp stall 的常见原因有哪些？
3. NCU 和 Nsight Systems 的区别？

**加分要点**：有实际 profiling 经验，能解读 SOL 图表。

---

### A10 [L2] Atomic 操作

**标准答案**：
- Atomic 操作保证对 global/shared memory 的读-改-写是原子的
- 性能问题：高竞争时串行化严重
- 优化策略：先在 shared memory 做局部 reduction，最后一个 thread 做 atomic；使用 warp-level vote/ballot 减少 atomic 次数
- Ampere+ 支持 `atomicAdd` 对 float 的硬件加速

**追问方向**：
1. `atomicCAS` 如何实现自定义 atomic 操作？
2. Atomic 在 L2 cache 和 global memory 上的性能差异？
3. 如何用 cooperative groups 替代部分 atomic？

**加分要点**：知道 Red (reduction) 指令和 atomic 的关系。

---

### A11-A15 略（参见题库 Q11-Q15，答案模式相同）

---

## 二、LLM 推理系统

### A16 [L2] KV Cache 机制

**标准答案**：
- Transformer 的 autoregressive decoding 中，每个 token 生成需要之前所有 token 的 K/V
- KV Cache 缓存已计算的 K/V，避免重复计算，将 decode 复杂度从 O(n²) 降为 O(n)
- 内存占用：每层 2 × hidden_size × seq_len × batch_size × dtype_size
- 70B 模型、2048 seq_len、fp16：约 40GB KV cache

**追问方向**：
1. KV cache 是推理的主要内存瓶颈，有哪些压缩方法？（GQA, MQA, quantization, eviction）
2. Prefill 和 Decode 阶段 KV cache 的行为差异？
3. 如何计算给定 GPU 内存下能支持的最大 batch size？

**加分要点**：能做具体的内存计算，知道 GQA/MQA 的节省比例。

---

### A17 [L3] PagedAttention

**标准答案**：
- 动机：传统方式为每个请求预分配最大 seq_len 的连续内存，导致严重碎片和浪费（平均利用率仅 20-40%）
- 原理：将 KV cache 分为固定大小的 page（如 16 tokens），用 page table 管理逻辑→物理映射
- 优势：内存利用率接近 100%、支持动态增长、支持 copy-on-write（beam search 共享）
- 开销：page table 查找增加少量延迟，但相比内存节省可忽略

**追问方向**：
1. Page size 如何选择？太大太小的 trade-off？
2. 如何实现 prefix caching？（共享相同 prefix 的 page）
3. PagedAttention 的 kernel 实现与标准 attention 有什么不同？

**加分要点**：读过 vLLM 源码，能描述 block_table 的数据结构。

---

### A18 [L3] 推测解码（Speculative Decoding）

**标准答案**：
- 核心思想：用小的 draft model 快速生成 K 个候选 token，再用 target model 并行验证
- 数学保证：通过 rejection sampling，输出分布与 target model 完全一致
- 验证过程：对每个位置 i，计算 p_target(x_i) / p_draft(x_i)，以 min(1, ratio) 的概率接受
- 收益条件：draft model 足够快且与 target model 分布接近
- EAGLE-3 特点：使用 target model 的 hidden states 作为 draft model 输入，提高 acceptance rate

**追问方向**：
1. 为什么 rejection sampling 保证分布一致？（数学证明）
2. acceptance rate 和 speedup 的关系？（speedup ≈ accepted_length / (1 + draft_cost/verify_cost)）
3. Tree-based speculative decoding 的优势？

**加分要点**：能推导 speedup 公式，知道 EAGLE 系列的演进。

---

### A19 [L2] Continuous Batching

**标准答案**：
- Static batching：等所有请求完成才开始下一批，短请求等长请求
- Continuous batching（iteration-level scheduling）：每个 iteration 可以加入新请求或移除已完成请求
- 优势：GPU 利用率大幅提升，throughput 可提升 2-10x
- 实现要点：需要动态内存管理（PagedAttention）、请求队列、调度策略

**追问方向**：
1. Prefill 和 Decode 混合调度的挑战？（prefill 是 compute-bound，decode 是 memory-bound）
2. Chunked prefill 是什么？解决什么问题？
3. 如何避免 decode 请求被 prefill 饿死？

**加分要点**：了解 vLLM 的 scheduler 实现细节。

---

### A20 [L3] FlashAttention

**标准答案**：
- 动机：标准 attention 需要 O(N²) 的中间矩阵（QK^T），对长序列内存爆炸
- 核心技术：tiling + online softmax + recomputation
- 实现：将 Q/K/V 分块加载到 SRAM，在 SRAM 中完成 attention 计算，避免写回 HBM
- Online softmax：维护 running max 和 running sum，一次遍历完成 softmax
- IO 复杂度：从 O(N²) 降为 O(N²d/M)，其中 M 是 SRAM 大小

**追问方向**：
1. FlashAttention-2 相比 v1 的改进？（更好的 parallelism, 减少 non-matmul FLOPs）
2. FlashAttention-3 的新特性？（Hopper 架构优化, FP8, asynchronous）
3. 为什么 recomputation 在 backward 中是值得的？

**加分要点**：能解释 online softmax 的数学推导。

---

### A21 [L2] 量化技术

**标准答案**：
- INT8 量化：将 FP16 权重/激活映射到 INT8，减少内存和计算
- GPTQ：post-training weight-only 量化，基于 Hessian 信息逐列量化
- AWQ：Activation-aware Weight Quantization，保护重要权重通道
- SmoothQuant：将激活的量化难度转移到权重上（per-channel scaling）
- W4A16：4-bit 权重 + 16-bit 激活，decode 阶段 memory-bound 时收益最大

**追问方向**：
1. 为什么 weight-only 量化在 decode 阶段收益大？（memory-bound，减少内存访问）
2. KV cache 量化的方法和影响？
3. FP8 vs INT8 的 trade-off？

**加分要点**：了解量化对不同模型/任务的精度影响。

---

### A22-A30 答案模式相同（Prefill/Decode 分离、Batching 策略、模型并行、Serving 架构等）

---

## 三、RAG 与检索系统

### A31 [L2] RAG 架构

**标准答案**：
- 核心流程：Query → Retrieval → Context Assembly → Generation
- 检索方式：Dense retrieval（embedding 相似度）、Sparse retrieval（BM25）、Hybrid
- 关键组件：Document parser → Chunker → Embedder → Vector store → Retriever → Reranker → LLM
- 评估维度：Retrieval quality（Recall, MRR）+ Generation quality（Faithfulness, Relevancy）

**追问方向**：
1. 如何处理 query 和 document 的语义鸿沟？
2. Multi-hop RAG 如何实现？
3. RAG vs Fine-tuning 的选择标准？

**加分要点**：有实际系统经验，能讨论工程 trade-off。

---

### A32 [L2] Chunk 策略

**标准答案**：
- 固定大小：简单但可能切断语义
- 语义分割：基于句子/段落边界
- 递归分割：先按大结构（章节），再按小结构（段落、句子）
- Overlap：相邻 chunk 重叠 10-20%，保证边界信息不丢失
- 特殊内容：表格保持完整、代码按函数/类分割、标题作为 metadata

**追问方向**：
1. Chunk size 对检索质量的影响？（太小：缺少上下文；太大：噪声多）
2. 如何评估 chunk 策略的好坏？
3. Late chunking 是什么？

**加分要点**：能结合实际项目讨论 chunk 策略的迭代过程。

---

### A33-A40 答案模式相同（向量索引、Reranking、评测、多模态 RAG 等）

---

## 四、分布式系统与通信

### A41 [L2] Tensor Parallelism

**标准答案**：
- 将模型的每一层按列/行切分到多个 GPU
- MLP 层：第一个线性层按列切分（Column Parallel），第二个按行切分（Row Parallel）
- Attention 层：Q/K/V 投影按 head 切分，output 投影按行切分
- 通信：每层需要一次 AllReduce（forward）+ 一次 AllReduce（backward）
- 通信量：2 × (N-1)/N × hidden_size × batch_size × seq_len × dtype_size

**追问方向**：
1. TP 的 scaling efficiency 受什么限制？（通信带宽）
2. 为什么 TP 通常限制在单机内？（需要高带宽 NVLink）
3. Sequence Parallelism 是什么？与 TP 的关系？

**加分要点**：能计算具体场景下的通信开销。

---

### A42 [L2] Pipeline Parallelism

**标准答案**：
- 将模型按层切分到多个 GPU，每个 GPU 负责若干连续层
- Bubble 问题：naive PP 中 GPU 利用率 = 1/num_stages
- Micro-batching：将 batch 切分为多个 micro-batch，pipeline 执行
- GPipe schedule：所有 micro-batch forward 完再 backward
- 1F1B schedule：交替执行 forward 和 backward，减少内存占用
- Interleaved schedule：每个 stage 负责非连续的层，进一步减少 bubble

**追问方向**：
1. PP 的 bubble ratio 如何计算？
2. PP vs TP 如何选择？（PP 适合跨机，TP 适合机内）
3. 推理场景下 PP 的特殊考虑？

**加分要点**：了解 Megatron-LM 的并行策略组合。

---

### A43 [L3] NCCL AllReduce

**标准答案**：
- Ring AllReduce：N 个 GPU 组成环，分 2(N-1) 步完成，通信量 2(N-1)/N × data_size
- Tree AllReduce：树形结构，延迟 O(log N) 但带宽利用率较低
- NCCL 实际使用：小数据用 Tree（低延迟），大数据用 Ring（高带宽）
- Double binary tree：NCCL 的默认算法，结合两者优势

**追问方向**：
1. AllReduce 和 ReduceScatter + AllGather 的等价关系？
2. 如何 overlap 通信和计算？（gradient bucketing）
3. InfiniBand vs RoCE 对 NCCL 性能的影响？

**加分要点**：了解 NCCL 的拓扑感知和通道选择。

---

### A44-A50 答案模式相同（Expert Parallelism、通信优化、故障恢复等）

---

## 五、系统设计

### A51 [L3] 设计高吞吐 LLM Serving 系统

**标准答案框架**：

1. **容量规划**：
   - 70B 模型 FP16 需要 140GB，至少 2×A100-80G（TP=2）或 4×A100-40G
   - 1000 QPS × 平均 512 output tokens ÷ 单卡吞吐 → 需要 N 组副本

2. **架构设计**：
   - Load Balancer → Router → Worker Pool
   - 每个 Worker 是一组 TP GPU
   - Prefill 和 Decode 分离部署（disaggregated serving）

3. **调度策略**：
   - Continuous batching + priority queue
   - 长请求限流/降级
   - Prefill chunking 避免阻塞 decode

4. **容错**：
   - Health check + 自动重启
   - Request retry with idempotency
   - Graceful degradation：负载过高时拒绝新请求

5. **监控**：
   - TTFT、TPOT、throughput、queue depth
   - GPU utilization、memory usage
   - 告警：P99 延迟 > 阈值、queue depth 持续增长

**追问方向**：
1. 如何做 A/B 测试？
2. 如何处理 burst traffic？
3. 多模型部署如何共享 GPU？

---

### A52-A55 答案模式相同（RAG 系统设计、实时推荐、多模态推理等）

---

## 使用建议

1. **每天练习 3-5 题**，先不看答案自己回答，再对照补充
2. **重点关注追问方向**，面试中追问才是区分度所在
3. **结合自己的项目经验**回答，比纯理论更有说服力
4. **对于 Gap 领域的题目**（CUDA kernel、分布式），需要先学习再练习
5. **模拟面试时计时**，每题控制在 3-5 分钟内回答完
