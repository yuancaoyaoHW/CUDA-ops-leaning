# 技术面试题库

## 一、CUDA 与 GPU 架构（15 题）

### Q1 [L2] CUDA 执行模型
**题目**：解释 CUDA 的 Grid → Block → Thread → Warp 层次结构。一个 SM 上最多能同时调度多少个 warp？Occupancy 受哪些因素限制？

**考察点**：CUDA 执行模型基础、Occupancy 分析能力

---

### Q2 [L2] Warp Divergence
**题目**：什么是 warp divergence？给出一个会导致严重 divergence 的代码示例，并说明如何优化。在 Volta+ 架构上，independent thread scheduling 如何影响 divergence 的行为？

**考察点**：warp 执行机制、分支优化、架构演进理解

---

### Q3 [L2] GPU 内存层次
**题目**：详细描述 GPU 的内存层次（Register → Shared Memory → L1 → L2 → Global Memory），各自的容量、延迟、带宽量级。Shared Memory 和 L1 Cache 在 Ampere 架构上是什么关系？

**考察点**：内存层次理解、架构细节

---

### Q4 [L3] Bank Conflict
**题目**：Shared Memory 的 bank conflict 是如何产生的？32 个 bank 的情况下，给出一个 2D 数组访问导致 bank conflict 的例子，以及用 padding 解决的方法。如何用 NCU 检测 bank conflict？

**考察点**：Shared Memory 细节、性能调优经验

---

### Q5 [L3] GEMM 优化
**题目**：从 naive GEMM 到高性能 GEMM，描述关键优化步骤：tiling、shared memory、register blocking、double buffering、vectorized load。每一步解决什么瓶颈？最终版本的理论性能上限如何计算？

**考察点**：GEMM 优化全链路、性能建模能力

---

### Q6 [L2] Coalesced Memory Access
**题目**：什么是 coalesced memory access？为什么 AoS（Array of Structures）布局通常比 SoA（Structure of Arrays）在 GPU 上性能差？给出一个矩阵转置中 non-coalesced 访问的例子和优化方法。

**考察点**：Global Memory 访问模式优化

---

### Q7 [L2] CUDA Stream 与异步执行
**题目**：CUDA Stream 的作用是什么？如何用多 Stream 实现 H2D、Kernel、D2H 的 pipeline 重叠？什么情况下两个 Stream 中的操作无法重叠？

**考察点**：异步执行模型、pipeline 设计

---

### Q8 [L3] Reduction 优化
**题目**：实现一个高性能的 parallel reduction（求和）。描述从 naive 到最优的优化路径：避免 divergence → sequential addressing → warp shuffle → 多级 reduction。最终版本能达到理论带宽的多少比例？

**考察点**：经典 CUDA 优化模式、warp-level 原语

---

### Q9 [L2] Tensor Core
**题目**：什么是 Tensor Core？它与 CUDA Core 的区别是什么？WMMA API 的基本用法是什么？为什么 Tensor Core 对 GEMM 的加速如此显著？

**考察点**：现代 GPU 架构特性

---

### Q10 [L3] NSight Compute 分析
**题目**：你拿到一个 kernel 的 NCU 报告，发现 Compute Throughput 只有 30%，Memory Throughput 有 85%。这说明什么？你会如何优化？如果反过来（Compute 85%，Memory 30%）呢？

**考察点**：性能分析实战能力、Roofline 思维

---

### Q11 [L2] Triton vs CUDA
**题目**：对比 Triton 和 CUDA 的编程模型差异。Triton 的 block-level programming 相比 CUDA 的 thread-level programming 有什么优势和局限？什么场景下你会选择 Triton，什么场景下必须用 CUDA？

**考察点**：工具选型能力、对两种编程模型的理解

---

### Q12 [L3] FlashAttention IO 分析
**题目**：推导标准 Attention 的 IO 复杂度，然后推导 FlashAttention 的 IO 复杂度。解释 FlashAttention 为什么是 IO-aware 的，以及 tiling size 如何选择。

**考察点**：FlashAttention 核心原理、IO 复杂度分析

---

### Q13 [L2] GPU 架构对比
**题目**：对比 A100、H100、H200 的关键规格（SM 数量、内存带宽、Tensor Core 代次、NVLink 带宽）。这些差异对 LLM 推理性能有什么影响？

**考察点**：硬件知识、性能预估能力

---

### Q14 [L3] Kernel Fusion
**题目**：什么是 kernel fusion？为什么它对 LLM 推理很重要？给出 3 个 LLM 推理中常见的 fusion 模式。Fusion 的收益上限如何估算？什么情况下 fusion 反而会降低性能？

**考察点**：优化策略、系统思维

---

### Q15 [L3] Mixed Precision 与数值稳定性
**题目**：在 FP16/BF16 混合精度训练和推理中，哪些操作需要保持 FP32？为什么 softmax 在 FP16 下容易出现数值问题？BF16 相比 FP16 的优势是什么？

**考察点**：数值精度理解、工程实践

---

## 二、LLM 推理系统（15 题）

### Q16 [L2] KV Cache
**题目**：解释 KV Cache 在自回归生成中的作用。计算一个 70B 模型、seq_len=4096、batch_size=32 时 KV Cache 的显存占用。如何优化 KV Cache 的内存效率？

**考察点**：KV Cache 原理与工程实践

---

### Q17 [L3] PagedAttention
**题目**：详细解释 PagedAttention 的设计。它如何解决 KV Cache 的内存碎片问题？Block table 的映射机制是什么？与操作系统的虚拟内存有什么类比？对 kernel 实现有什么影响？

**考察点**：vLLM 核心创新的深入理解

---

### Q18 [L2] Continuous Batching
**题目**：对比 static batching 和 continuous batching。为什么 continuous batching 能显著提升吞吐？它的实现复杂度在哪里？iteration-level scheduling 的具体含义是什么？

**考察点**：推理调度核心概念

---

### Q19 [L3] Speculative Decoding
**题目**：解释 Speculative Decoding 的原理。为什么它能在不改变输出分布的情况下加速生成？acceptance rate 受什么因素影响？draft model 的选择有什么 trade-off？

**考察点**：推测解码原理（候选人有实际经验）

---

### Q20 [L2] Prefill vs Decode
**题目**：LLM 推理的 prefill 和 decode 阶段有什么本质区别？它们分别是 compute-bound 还是 memory-bound？为什么有些系统选择将 prefill 和 decode 分离部署？

**考察点**：推理阶段特性理解

---

### Q21 [L3] vLLM Scheduler
**题目**：描述 vLLM Scheduler 的核心逻辑。waiting/running/swapped 三个队列如何交互？preemption 策略是什么？如何保证公平性？Chunked Prefill 如何与 Scheduler 配合？

**考察点**：vLLM 源码级理解

---

### Q22 [L2] Prefix Caching
**题目**：什么是 Prefix Caching？它在什么场景下收益最大？vLLM 中的 automatic prefix caching 是如何实现的（hash 机制）？缓存淘汰策略是什么？

**考察点**：推理优化技术

---

### Q23 [L3] Quantization for Inference
**题目**：对比 W4A16、W8A8、W4A4 三种量化方案在推理中的特点。各自适合什么场景？对 kernel 实现有什么要求？量化后的 GEMM 如何利用 Tensor Core？

**考察点**：量化与推理性能的关系

---

### Q24 [L2] TTFT vs TPOT
**题目**：解释 TTFT（Time To First Token）和 TPOT（Time Per Output Token）的含义。它们分别受什么因素影响？在 SLA 设计中如何平衡这两个指标？

**考察点**：推理性能指标理解

---

### Q25 [L3] TensorRT-LLM vs vLLM
**题目**：从架构设计、优化策略、易用性三个维度对比 TensorRT-LLM 和 vLLM。各自的优势场景是什么？如果你要部署一个 70B 模型服务 1000 QPS，你会选择哪个？为什么？

**考察点**：推理框架选型能力

---

### Q26 [L2] GQA/MQA
**题目**：解释 Multi-Head Attention、Multi-Query Attention、Grouped-Query Attention 的区别。GQA 如何在推理效率和模型质量之间取得平衡？对 KV Cache 大小有什么影响？

**考察点**：注意力机制变体理解

---

### Q27 [L3] Decode 调度优化
**题目**：在高并发 decode 场景下，如何优化 GPU 利用率？讨论 batch size 与 latency 的 trade-off。什么是 iteration-level scheduling？如何处理不同请求的 priority？

**考察点**：调度优化深度

---

### Q28 [L2] Model Parallelism for Inference
**题目**：推理场景下的模型并行与训练有什么不同？为什么推理通常只用 TP 而较少用 PP？TP degree 的选择依据是什么？

**考察点**：分布式推理理解

---

### Q29 [L3] KV Cache Compression
**题目**：列举 3 种 KV Cache 压缩技术（量化、eviction、merging）。各自的原理和 trade-off 是什么？在什么场景下值得使用 KV Cache 压缩？

**考察点**：前沿优化技术

---

### Q30 [L3] Serving System Capacity Planning
**题目**：给定：模型 70B FP16，目标 TTFT P99 < 500ms，TPOT P99 < 50ms，QPS = 200，平均输入 500 tokens，平均输出 200 tokens。计算需要多少张 A100-80G，并说明计算过程。

**考察点**：容量规划实战能力

---

## 三、RAG 与检索系统（10 题）

### Q31 [L2] RAG 架构
**题目**：描述一个生产级 RAG 系统的完整架构。从文档摄入到最终回答，经过哪些阶段？每个阶段的关键设计决策是什么？

**考察点**：RAG 全链路理解（候选人有实际经验）

---

### Q32 [L2] Chunking 策略
**题目**：对比不同的文档切分策略：固定长度、按段落、递归切分、语义切分。各自的优缺点？chunk size 和 overlap 如何选择？对检索效果有什么影响？

**考察点**：文档处理工程经验

---

### Q33 [L3] Embedding 模型选择与优化
**题目**：如何选择 embedding 模型？对比 OpenAI Ada、BGE、E5 等模型。如何评估 embedding 质量？在大规模场景下如何优化 embedding 推理延迟？

**考察点**：向量化技术选型

---

### Q34 [L2] 向量检索
**题目**：对比 FAISS 的不同索引类型（Flat、IVF、HNSW、PQ）。各自的时间/空间复杂度和适用场景？如何在召回率和延迟之间取得平衡？

**考察点**：向量检索工程

---

### Q35 [L3] 混合检索
**题目**：什么是混合检索（Hybrid Search）？如何结合稀疏检索（BM25）和稠密检索（向量）？RRF（Reciprocal Rank Fusion）的原理是什么？什么场景下混合检索优于纯向量检索？

**考察点**：检索策略深度

---

### Q36 [L2] RAG 评测
**题目**：如何评测 RAG 系统的质量？解释 RAGAS 框架的核心指标（Faithfulness、Answer Relevancy、Context Precision、Context Recall）。如何构建评测数据集？

**考察点**：RAG 评测经验（候选人使用过 RAGAS）

---

### Q37 [L3] 长文档处理
**题目**：当文档超过 embedding 模型的 max_length 时如何处理？对比 truncation、chunking+pooling、hierarchical embedding 等方案。如何处理跨 chunk 的信息？

**考察点**：长文档 RAG 挑战

---

### Q38 [L2] RAG vs Fine-tuning
**题目**：什么场景下应该用 RAG，什么场景下应该 fine-tune？两者可以结合吗？如何判断 RAG 系统的回答质量瓶颈是检索还是生成？

**考察点**：技术选型判断力

---

### Q39 [L3] RAG 系统优化
**题目**：你的 RAG 系统准确率只有 60%，如何诊断和优化？从检索、重排、prompt、生成四个环节分别讨论可能的问题和解决方案。

**考察点**：RAG 调优实战能力

---

### Q40 [L2] 多模态 RAG
**题目**：如何构建支持图片、表格、PDF 的多模态 RAG 系统？文档解析的挑战是什么？如何处理表格数据的检索？

**考察点**：多模态文档处理

---

## 四、分布式系统与通信（10 题）

### Q41 [L2] NCCL 通信原语
**题目**：解释 AllReduce、AllGather、ReduceScatter、Broadcast 的语义和通信量。在 LLM 推理中，哪些操作需要用到哪些通信原语？

**考察点**：分布式通信基础

---

### Q42 [L3] Ring AllReduce
**题目**：详细描述 Ring AllReduce 的工作原理。N 个节点、数据量 D 的情况下，总通信量是多少？带宽利用率是多少？与 Tree AllReduce 对比有什么优劣？

**考察点**：通信算法理解

---

### Q43 [L2] Tensor Parallelism
**题目**：解释 Megatron-style Tensor Parallelism。对于一个 Linear 层 Y = XW，如何做列切分和行切分？各需要什么通信操作？为什么 MLP 层用 column-then-row 的方式？

**考察点**：TP 实现细节

---

### Q44 [L3] Pipeline Parallelism
**题目**：解释 Pipeline Parallelism 的 1F1B 调度策略。Pipeline bubble 的比例如何计算？micro-batch 数量如何选择？PP 在推理中为什么不如训练中常用？

**考察点**：PP 原理与适用性

---

### Q45 [L2] NVLink vs PCIe vs InfiniBand
**题目**：对比 NVLink、PCIe、InfiniBand 的带宽和延迟。在多 GPU 推理中，通信拓扑如何影响并行策略的选择？

**考察点**：硬件互联知识

---

### Q46 [L3] 通信与计算重叠
**题目**：如何实现通信与计算的重叠（overlap）？在 TP 推理中，哪些计算可以与 AllReduce 重叠？实现 overlap 的技术手段有哪些（CUDA Stream、kernel 拆分）？

**考察点**：分布式优化技术

---

### Q47 [L2] Expert Parallelism (MoE)
**题目**：MoE 模型的推理如何做并行？Expert Parallelism 的通信模式是什么？All-to-All 通信的开销如何估算？

**考察点**：MoE 推理理解

---

### Q48 [L3] 分布式 KV Cache 管理
**题目**：在 TP 推理中，KV Cache 如何分布在多个 GPU 上？如果要做 KV Cache 的跨请求共享（prefix caching），分布式场景下有什么额外挑战？

**考察点**：分布式推理细节

---

### Q49 [L2] Fault Tolerance
**题目**：在多 GPU 推理服务中，如果一张卡故障，系统应该如何处理？讨论 graceful degradation、request rerouting、health check 等机制。

**考察点**：系统可靠性设计

---

### Q50 [L3] Disaggregated Serving
**题目**：解释 prefill-decode disaggregation（如 DistServe）的设计思路。为什么要将 prefill 和 decode 分离到不同的 GPU 上？KV Cache 如何在两者之间传输？这种架构的适用条件是什么？

**考察点**：前沿架构理解

---

## 五、系统设计（5 题）

### Q51 [L3] 设计 LLM Serving 系统
**题目**：设计一个支持 100 QPS、P99 TTFT < 1s、P99 TPOT < 100ms 的 LLM Serving 系统。模型为 70B，需要支持多租户和优先级调度。请从架构、资源规划、调度策略、容错等方面完整设计。

**考察点**：系统设计综合能力

---

### Q52 [L3] 设计 RAG 平台
**题目**：设计一个企业级 RAG 平台，支持：多数据源接入（PDF/网页/数据库）、实时更新、多租户隔离、QPS 1000、P99 < 3s。讨论数据管道、索引策略、缓存设计、评测闭环。

**考察点**：RAG 系统设计（候选人有实际经验）

---

### Q53 [L3] 设计模型推理网关
**题目**：设计一个统一的模型推理网关，支持：多模型路由、A/B 测试、流量控制、自动扩缩容、成本优化。讨论 API 设计、负载均衡、模型版本管理。

**考察点**：平台工程设计能力

---

### Q54 [L3] 设计 GPU 集群调度器
**题目**：设计一个 GPU 集群调度器，支持：多任务混部（训练+推理）、GPU 碎片整理、优先级抢占、资源配额。讨论调度算法、资源抽象、监控告警。

**考察点**：基础设施设计能力

---

### Q55 [L3] 设计实时对话系统
**题目**：设计一个支持流式输出、多轮对话、上下文管理的实时对话系统。需要支持 10K 并发连接、会话持久化、对话历史压缩。讨论 WebSocket 管理、状态存储、内存优化。

**考察点**：实时系统设计能力
