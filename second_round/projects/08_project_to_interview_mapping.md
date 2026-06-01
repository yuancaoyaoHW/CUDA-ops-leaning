# 项目到面试映射

## 映射原则

1. 每个项目准备 **30s / 2min / 10min** 三个版本的叙事
2. 每个项目准备 **5 个深度追问** 及回答框架
3. 使用 **STAR 框架**：Situation → Task → Action → Result
4. 所有回答必须有**具体数据**支撑

---

## 项目 A: CUDA Ops Learning Lab

### 30 秒版本（Elevator Pitch）

"我从零实现了一个 CUDA kernel library，覆盖 LLM 推理的 8 个核心算子，包括 GEMM、FlashAttention、RMSNorm 等。每个 kernel 都有从 naive 到 optimized 的完整优化路径，配套 Nsight Compute profiling 报告。"

### 2 分钟版本（面试开场）

"**Situation**: LLM 推理的核心性能瓶颈在 GPU kernel 层，我需要深入理解底层优化才能做好推理系统工作。

**Task**: 从零实现 8 个核心算子，覆盖 memory-bound（vector_add, reduction, softmax）和 compute-bound（GEMM, attention）两类，每个都要有 benchmark 数据。

**Action**: 我按照从简单到复杂的路径实现：先通过 vector_add 理解 memory coalescing 和 grid-stride loop，再通过 reduction 掌握 warp shuffle，然后实现 tiled GEMM 理解 shared memory tiling 策略，最后实现简化版 FlashAttention 理解 online softmax 如何将 memory 从 O(N²) 降到 O(N)。每个 kernel 都用 Nsight Compute 分析瓶颈。

**Result**: GEMM 达到 cuBLAS X% 性能，FlashAttention 比 naive 快 X×，所有 memory-bound kernel 达到 80%+ bandwidth utilization。"

### 深度追问准备

| # | 问题 | 回答框架 |
|---|------|---------|
| 1 | "你的 GEMM 达到 cuBLAS 多少？瓶颈在哪？" | 具体百分比 + NCU 数据（compute throughput / memory throughput）+ 下一步优化方向 |
| 2 | "Tile size 怎么选？" | shared memory 容量约束（48KB/SM）+ occupancy tradeoff + 实验数据 |
| 3 | "FlashAttention 为什么快？" | 减少 HBM 访问（从 O(N²) 到 O(N)）+ online softmax 数学推导 + tiling 策略 |
| 4 | "怎么判断 kernel 是 memory-bound 还是 compute-bound？" | arithmetic intensity + roofline model + NCU 指标对比 |
| 5 | "RMSNorm fusion 省了什么？" | 减少一次 global memory round-trip + 具体 bandwidth 节省数据 |

### 适用面试场景

- CUDA/GPU 编程能力考察
- 性能优化方法论考察
- 对 LLM 推理底层的理解

---

## 项目 B: Triton Kernels Lab

### 30 秒版本

"我用 Triton 重新实现了 CUDA Lab 中的 6 个核心 kernel，量化对比了两者的性能差距和开发效率差异。Triton 达到 CUDA 80-95% 性能，但开发时间只需 1/3-1/5。"

### 2 分钟版本

"**Situation**: 越来越多团队用 Triton 替代手写 CUDA，我需要理解两者的 tradeoff。

**Task**: 用 Triton 实现 6 个 kernel（GEMM, softmax, attention 等），与 CUDA 版本做公平对比。

**Action**: 利用 Triton 的 block-level programming model 和 autotune 机制实现各算子，用相同的 benchmark 框架测量性能，同时记录开发时间和代码行数。

**Result**: Triton 在 softmax/layernorm 等 memory-bound kernel 上接近 CUDA（95%+），在 GEMM 上达到 70-90%，开发效率提升 3-5×。结论是：快速原型用 Triton，极致性能用 CUDA。"

### 深度追问准备

| # | 问题 | 回答框架 |
|---|------|---------|
| 1 | "Triton 和 CUDA 怎么选？" | 按场景分：原型/中等复杂度→Triton，极致性能/warp-level→CUDA |
| 2 | "Triton 的局限性？" | 不支持 warp primitives、block-level 抽象限制、编译时间长 |
| 3 | "Autotune 怎么工作？" | 枚举 tile 配置 + 运行时 benchmark + 缓存最优配置 |
| 4 | "为什么 Triton GEMM 比 CUDA 慢？" | 无法做 register-level 优化、double buffering 受限 |
| 5 | "Triton 生成的 PTX 质量如何？" | 用 NCU 分析 Triton 编译后的 kernel，对比手写 CUDA |

---

## 项目 C: LLM Inference Benchmark Lab

### 30 秒版本

"我设计了一个系统化的 LLM 推理 benchmark 框架，覆盖 vLLM/SGLang 在 200+ 配置下的性能表现，通过 roofline 分析和 latency breakdown 定位瓶颈，产出可复现的优化建议。"

### 2 分钟版本

"**Situation**: 在 Atlas 3000 上做推理优化时，我发现缺乏系统化的 benchmark 方法论，很多优化是 trial-and-error。

**Task**: 建立完整的 benchmark 框架，覆盖 batch/seq/concurrency 三个维度的实验矩阵，对比 vLLM 和 SGLang。

**Action**: 设计单变量控制实验：batch sweep 找 throughput 饱和点，seq sweep 找 TTFT 拐点，concurrency sweep 找最大可支持并发。用 Nsight Systems 做 latency breakdown，用 roofline 定位各 kernel 瓶颈。

**Result**: 在 Atlas 3000 上已实现 55% throughput 提升（9.22→14.30 tok/s）。本项目在 NVIDIA GPU 上扩展实验，识别出 batch=X 为饱和点，SGLang 在 prefix-heavy workload 下比 vLLM 快 X%。"

### 深度追问准备

| # | 问题 | 回答框架 |
|---|------|---------|
| 1 | "Batch size 增大为什么 throughput 先升后降？" | GPU memory 限制 KV cache → scheduling overhead → 具体拐点数据 |
| 2 | "TTFT 和 TPOT 分别受什么影响？" | TTFT=prefill(compute-bound, ∝seq_len), TPOT=decode(memory-bound, ∝KV cache size) |
| 3 | "vLLM 和 SGLang 核心区别？" | PagedAttention vs RadixAttention, 适用场景不同 |
| 4 | "怎么做公平对比？" | 相同硬件/模型/workload + 预热 + 统计显著性 + 隔离 |
| 5 | "如果 p99 latency 突然升高怎么排查？" | Nsight timeline → 是否有 preemption → queue depth → memory pressure |

### 已有面试素材（可直接使用）

- "在 Atlas 3000 上通过系统化 profiling 实现 55% throughput 提升"
- "TPOT 从 108.18ms 降到 65.76ms（39%↓）"
- "有 vLLM-Ascend PR #1032 合入经验"

---

## 项目 D: RAG Infrastructure Evaluation Lab

### 30 秒版本

"我构建了一个完整的 RAG 评测体系，使用 RAGAS 框架评测质量（faithfulness 0.87, accuracy 90%），通过消融实验量化了混合检索、重排序、metadata filter 各组件的贡献度。"

### 2 分钟版本

"**Situation**: RAG 系统的质量评测需要系统化方法，不能只看 end-to-end accuracy。

**Task**: 建立覆盖检索质量和生成质量的完整评测体系，量化每个组件的贡献。

**Action**: 使用 RAGAS 框架评测 faithfulness/relevancy/precision/recall 四个维度。设计消融实验：逐一去掉 reranker、BM25、metadata filter，测量 accuracy 变化。同时做延迟 benchmark，确保 p95 < 200ms。

**Result**: 混合检索比纯向量检索 precision 提升 X%，reranker 贡献 answer relevancy +X%，整体 QA accuracy 90%，p95 retrieval latency < 200ms。"

### 深度追问准备

| # | 问题 | 回答框架 |
|---|------|---------|
| 1 | "Chunk size 怎么选？" | 实验数据：512+overlap=50 最优 + precision/recall tradeoff 分析 |
| 2 | "向量检索 miss 了怎么办？" | hybrid retrieval + query expansion + reranking |
| 3 | "怎么保证 faithfulness？" | citation tracking + RAGAS 持续监控 + 阈值告警 |
| 4 | "文档更新怎么处理？" | incremental indexing + versioning + stale detection |
| 5 | "怎么处理多跳推理？" | iterative retrieval + query decomposition + chain-of-thought |

---

## 项目 E: GPU Serving Observability Demo

### 30 秒版本

"我设计了一个 GPU serving 可观测性系统，覆盖 GPU/请求/系统三层指标，实现了基于多指标的 autoscaling 模拟，在突发流量下 2 分钟内完成扩容。"

### 2 分钟版本

"**Situation**: 生产级 LLM serving 需要完整的监控和自动扩缩容能力。

**Task**: 设计三层监控体系 + autoscaling 策略 + 告警规则。

**Action**: 用 Prometheus + Grafana 构建监控栈，设计 GPU util / queue depth / latency 三指标联合判断的 autoscaling 策略，模拟突发流量场景验证响应时间。

**Result**: 突发流量下 < 2min 完成扩容，SLA 违反率 < 5%，告警覆盖 GPU OOM / 高延迟 / 队列堆积等关键场景。"

### 深度追问准备

| # | 问题 | 回答框架 |
|---|------|---------|
| 1 | "Autoscaling 指标怎么选？" | 多指标联合：GPU util + queue depth + latency，避免单指标误判 |
| 2 | "冷启动时间长怎么办？" | 预热 replica pool + 模型预加载 + 分层启动 |
| 3 | "多租户怎么隔离？" | per-tenant queue + priority + resource quota |
| 4 | "GPU OOM 怎么预防？" | memory 预估 + admission control + graceful rejection |
| 5 | "怎么做 zero-downtime 更新？" | rolling update + health check + graceful drain |

---

## 面试场景到项目映射

| 面试场景 | 主打项目 | 辅助项目 | 核心数据点 |
|---------|---------|---------|-----------|
| "讲一个你做过的性能优化" | C (Benchmark) | A (CUDA) | 55% throughput↑, TPOT 39%↓ |
| "讲一个你从零搭建的系统" | D (RAG) | E (Observability) | 90% accuracy, p95<200ms |
| "讲一个你深入底层的经历" | A (CUDA) | B (Triton) | GEMM X% cuBLAS, FlashAttention O(N) |
| "讲一个你做过的 benchmark" | C (Benchmark) | A (CUDA) | 200+ 配置, roofline 分析 |
| "讲一个生产化的经验" | E (Observability) | D (RAG) | <2min 扩容, <5% SLA 违反 |
| "CUDA 编程能力" | A (CUDA) | B (Triton) | 8 个算子, NCU profiling |
| "系统设计" | D (RAG) + E (Obs) | C (Benchmark) | 架构图 + 指标体系 |

---

## 高压追问应对策略

### 当被问到不会的问题

1. **承认边界**: "这个我还没有实际实践过，但基于我的理解..."
2. **关联已有经验**: "虽然我没做过 X，但我做过类似的 Y，原理是..."
3. **展示学习能力**: "我会通过 [具体方法] 来快速补上这个知识"

### 当被质疑数据

1. **说明方法论**: "这个数据是通过 [具体方法] 测量的，统计方法是..."
2. **承认局限**: "这个数据是在 [具体环境] 下测的，不同环境可能有差异"
3. **展示严谨性**: "我做了 N 次重复实验，报告的是 median 值，std 是..."
