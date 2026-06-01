# 项目到简历映射

## 映射原则

1. 每个 bullet 必须有**量化数据**支撑（完成项目后填入实际数字）
2. 使用 **Action + Context + Result** 结构
3. 标注 bullet 的**适用岗位类型**
4. 标注**前置条件**（哪些实验必须完成才能使用该 bullet）

---

## 项目 A: CUDA Ops Learning Lab

| # | Resume Bullet | 适用岗位 | 前置条件 |
|---|--------------|---------|---------|
| A1 | "Built CUDA kernel library for LLM inference operators (GEMM, FlashAttention, RMSNorm, RoPE) achieving X% of cuBLAS/FlashAttention-2 performance on [GPU]" | GPU Kernel Engineer, Inference Optimization | GEMM + FlashAttention 完成 + benchmark 数据 |
| A2 | "Implemented tiled GEMM kernel with shared memory optimization achieving X% of cuBLAS peak performance through systematic tiling and register blocking" | CUDA Engineer, Performance Engineer | GEMM 4 个版本完成 + NCU profiling |
| A3 | "Built simplified FlashAttention kernel reducing memory complexity from O(N²) to O(N) with X× speedup over naive implementation" | Inference Engineer, ML Systems | FlashAttention 完成 + memory 对比数据 |
| A4 | "Profiled and optimized GPU kernels using Nsight Compute, improving memory bandwidth utilization from X% to Y%" | Performance Engineer, GPU Systems | 至少 3 个 kernel 有 NCU 报告 |
| A5 | "Implemented fused RMSNorm kernel reducing global memory accesses by 50% through operator fusion, achieving X% bandwidth utilization" | Kernel Engineer, Inference Optimization | RMSNorm fused 版本完成 |
| A6 | "Implemented numerically-stable online softmax kernel with single-pass algorithm, foundational to FlashAttention" | ML Systems, Research Engineer | Softmax online 版本完成 |

### 岗位匹配

| 岗位类型 | 推荐 Bullets | 组合策略 |
|---------|-------------|---------|
| NVIDIA Kernel Engineer | A1 + A2 + A4 | 强调性能数字和 profiling |
| Inference Optimization (DeepSeek/Moonshot) | A1 + A3 + A5 | 强调 LLM 相关算子 |
| ML Systems (Meta/Google) | A3 + A4 + A6 | 强调算法理解和系统思维 |

---

## 项目 B: Triton Kernels Lab

| # | Resume Bullet | 适用岗位 | 前置条件 |
|---|--------------|---------|---------|
| B1 | "Implemented 6 Triton kernels (GEMM, FlashAttention, LayerNorm, Quantization) achieving 80-95% of hand-written CUDA performance with 3-5× less development time" | ML Compiler, Kernel Engineer | 6 个 kernel 完成 + 对比数据 |
| B2 | "Conducted systematic Triton vs CUDA comparison across 6 operators, quantifying performance-productivity tradeoff for GPU kernel development" | ML Systems, Performance Engineer | 对比报告完成 |

### 岗位匹配

| 岗位类型 | 推荐 Bullets |
|---------|-------------|
| Meta AI Infra | B1（Meta 大量使用 Triton） |
| ML Compiler | B1 + B2 |
| 通用 Inference | B2（作为补充） |

---

## 项目 C: LLM Inference Benchmark Lab

| # | Resume Bullet | 适用岗位 | 前置条件 |
|---|--------------|---------|---------|
| C1 | "Designed comprehensive LLM inference benchmark suite evaluating vLLM/SGLang across 200+ configurations (batch/seq/concurrency), identifying optimal serving parameters" | Inference Engineer, Performance Engineer | 完整实验矩阵完成 |
| C2 | "Achieved X% throughput improvement through systematic Nsight profiling and parameter tuning on [GPU]" | Performance Engineer, SRE | 参数调优实验完成 |
| C3 | "Identified performance bottlenecks through roofline analysis and latency breakdown, providing actionable optimization recommendations for LLM serving" | Systems Engineer, Performance Engineer | Roofline + breakdown 分析完成 |
| C4 | "Benchmarked LLM serving frameworks (vLLM/SGLang) under diverse workload patterns, achieving X tok/s throughput with p99 latency < Y ms" | Inference Platform, MLOps | 框架对比实验完成 |

### 已有可用 Bullet（基于 Ascend 经验）

| # | Resume Bullet | 状态 |
|---|--------------|------|
| C0 | "Achieved 55% throughput improvement on Atlas 3000 (9.22→14.30 tok/s) through systematic profiling and parameter tuning" | ✅ 可直接使用 |

### 岗位匹配

| 岗位类型 | 推荐 Bullets |
|---------|-------------|
| vLLM/SGLang 团队 | C0 + C1 + C4 |
| Performance Engineer | C0 + C2 + C3 |
| Inference Platform | C1 + C4 |

---

## 项目 D: RAG Infrastructure Evaluation Lab

| # | Resume Bullet | 适用岗位 | 前置条件 |
|---|--------------|---------|---------|
| D1 | "Built RAG evaluation framework achieving 90% QA accuracy (RAGAS), with hybrid retrieval improving precision by X% over vector-only search" | RAG Engineer, ML Engineer | RAGAS 评测 + 策略对比完成 |
| D2 | "Designed RAG system architecture supporting 10K+ documents with p95 retrieval latency < 200ms, using hybrid search (vector + BM25 + rerank)" | Backend Engineer, Systems Engineer | 延迟 benchmark 完成 |
| D3 | "Conducted systematic RAG ablation study quantifying contribution of each component (reranker +X%, hybrid +Y%, metadata filter +Z%)" | ML Engineer, Research Engineer | 消融实验完成 |

### 已有可用 Bullet（基于已有 RAG 经验）

| # | Resume Bullet | 状态 |
|---|--------------|------|
| D0 | "Built production RAG infrastructure with RAGAS evaluation achieving 90% QA accuracy, supporting multi-format document parsing" | ✅ 可直接使用 |

### 岗位匹配

| 岗位类型 | 推荐 Bullets |
|---------|-------------|
| RAG/Search Engineer | D0 + D1 + D3 |
| Full-stack ML Engineer | D0 + D2 |
| Applied Scientist | D1 + D3 |

---

## 项目 E: GPU Serving Observability Demo

| # | Resume Bullet | 适用岗位 | 前置条件 |
|---|--------------|---------|---------|
| E1 | "Designed GPU serving observability system with Prometheus/Grafana monitoring covering GPU utilization, request latency (TTFT/TPOT), and autoscaling metrics" | MLOps, SRE, Platform Engineer | Dashboard + 告警规则完成 |
| E2 | "Implemented GPU-aware autoscaling simulation achieving < 2min scale-up response under burst traffic with < 5% SLA violation" | Platform Engineer, SRE | Autoscaling 模拟完成 |

### 岗位匹配

| 岗位类型 | 推荐 Bullets |
|---------|-------------|
| ML Platform (CoreWeave/Baseten) | E1 + E2 |
| SRE / DevOps | E1 |
| 通用 Inference | E1（作为补充） |

---

## 简历组合策略

### 针对不同 JD 的 Bullet 组合

#### GPU Inference Optimization（NVIDIA/DeepSeek/Moonshot）
1. A1 — CUDA kernel library
2. A2 或 A3 — GEMM 或 FlashAttention 具体成果
3. C0 — 已有 throughput 提升经验
4. C1 或 C2 — Benchmark 方法论
5. B1 — Triton 能力（加分）

#### ML Systems Engineer（Meta/Google）
1. A1 — CUDA kernel 能力
2. A3 — FlashAttention 理解
3. C1 — Benchmark 方法论
4. D0 — RAG 系统经验
5. B2 — Triton vs CUDA 分析

#### RAG / Search Infrastructure
1. D0 — RAG 生产经验
2. D1 — RAGAS 评测
3. D2 — 系统设计
4. C4 — LLM serving 理解
5. E1 — 监控能力（加分）

#### ML Platform / MLOps
1. C1 — Benchmark 框架
2. E1 — Observability
3. E2 — Autoscaling
4. D2 — 系统设计
5. A4 — Profiling 能力

---

## 使用注意事项

- ⚠️ 标注"完成后使用"的 bullet 必须有实际数据支撑后才能写入简历
- ✅ 标注"可直接使用"的 bullet 基于已有经验，可立即使用
- 所有 X/Y/Z 占位符需替换为实际 benchmark 数据
- 根据目标 JD 动态选择 5-8 条最相关的 bullets
