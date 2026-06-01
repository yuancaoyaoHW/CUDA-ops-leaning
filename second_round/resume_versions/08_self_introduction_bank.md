# 自我介绍库

> 每个版本中英文各一份，按时长分为 60 秒、3 分钟、5 分钟

---

## 一、60 秒版本（电梯演讲）

### 中文版

面试官您好，我是袁曹尧，浙江大学计算机技术硕士。

我的核心方向是大模型推理系统优化。我在 vLLM-Ascend 社区独立实现了 EAGLE-3 推测解码，PR #1032 已合入主线，在昇腾 NPU 上实现了 55% 的吞吐提升。同时我独立负责过一个 RAG 系统后端，基于 RAGAS 评测准确率达到 90%。

我的目标是继续在推理系统方向深入，目前正在系统补强 CUDA 和 GPU profiling 能力。我对贵团队的推理优化方向很感兴趣，希望能有机会进一步交流。

### English Version

Hi, I'm Yuan Caoyao, a Master's graduate in Computer Science from Zhejiang University.

I focus on LLM inference optimization. I independently implemented EAGLE-3 speculative decoding for vLLM-Ascend — my PR #1032 was merged into the main branch, achieving 55% throughput improvement on Ascend NPU. I also independently built a RAG system backend with 90% accuracy on RAGAS evaluation.

I'm looking to go deeper in inference systems and am currently building up my CUDA and GPU profiling skills. I'd love to discuss how I can contribute to your team's inference optimization work.

---

## 二、3 分钟版本（电话面试开场）

### 中文版

面试官您好，我是袁曹尧，浙江大学计算机技术硕士，本科是西安交通大学信息与计算科学。我的核心方向是大模型推理系统优化。

**推理优化经验：**

我最核心的项目是在 vLLM 生态中实现 EAGLE-3 推测解码。具体来说，我独立实现了 EAGLE-3 的 proposer 模块，打通了从 draft model 到 hidden states、KV cache、rejection sampler、runner 的完整执行链路，最终 PR #1032 合入了 vLLM-Ascend 社区主线。

在性能方面，spec_tokens=3 时在 Atlas 310P3 上输出吞吐从 9.22 提升到 14.30 tok/s，提升了 55%，TPOT 从 108 毫秒降到 66 毫秒。我还提交了 vLLM-MindSpore 的 PR #1020，为 Qwen2.5 适配了 EAGLE-3，在 Ascend 910B 上 acceptance length 达到 1.63。

**RAG 工程经验：**

另外我独立负责过一个非结构化文档问答系统的后端。我设计了快速问答和 Research 问答双链路架构，抽象了统一的检索接口支持 metadata filters，用 RAGAS 构建了评测流程，准确率达到 90%。

**当前状态和目标：**

我清楚自己的短板——目前没有 CUDA kernel 开发和 GPU profiling 的实操经验，也没有分布式推理的实战。我正在系统学习 CUDA，目标是成为覆盖 GPU 和 NPU 双平台的推理系统工程师。

### English Version

Hi, I'm Yuan Caoyao. I have a Master's in Computer Science from Zhejiang University and a Bachelor's in Information and Computational Science from Xi'an Jiaotong University. My focus is LLM inference system optimization.

**Inference Optimization:**

My core project is implementing EAGLE-3 speculative decoding in the vLLM ecosystem. I independently built the EAGLE-3 proposer module, connecting the full pipeline from draft model through hidden states, KV cache, rejection sampler, to the runner. PR #1032 was merged into vLLM-Ascend's main branch.

Performance-wise, with spec_tokens=3 on Atlas 310P3, output throughput improved from 9.22 to 14.30 tok/s — a 55% increase — and TPOT dropped from 108ms to 66ms. I also submitted PR #1020 to vLLM-MindSpore, adapting EAGLE-3 for Qwen2.5, achieving an acceptance length of 1.63 on Ascend 910B.

**RAG Engineering:**

I also independently built the backend for an unstructured document Q&A system. I designed a dual-path architecture for quick Q&A and deep Research modes, abstracted a unified retrieval interface with metadata filters, and built an evaluation pipeline using RAGAS that achieved 90% accuracy.

**Current Status:**

I'm transparent about my gaps — I don't yet have hands-on CUDA kernel development or GPU profiling experience, and no distributed inference practice. I'm systematically learning CUDA now, aiming to become an inference engineer covering both GPU and NPU platforms.

---

## 三、5 分钟版本（正式面试开场）

### 中文版

面试官您好，我是袁曹尧，浙江大学计算机技术硕士，2025 年 3 月毕业。本科是西安交通大学信息与计算科学专业，数学基础比较扎实。我的核心方向是大模型推理系统优化，同时有 RAG 系统的独立交付经验。

**第一个项目：vLLM 生态推理优化**

这是我最核心的项目。背景是 vLLM-Ascend 社区需要在昇腾 NPU 上支持推测解码来提升推理吞吐。我独立实现了 EAGLE-3 的 proposer 模块。

技术上，我需要先深入理解 vLLM V1 架构中 scheduler、model runner、KV cache manager 的交互机制，然后设计 proposer 接口，打通 draft model 到 hidden states、KV cache、rejection sampler、runner 的完整执行链路。这个过程中需要处理 KV cache 的分配和回收，特别是 rejected token 场景下的 cache 状态管理。

最终 PR #1032 合入了 vLLM-Ascend 主仓。性能数据方面：spec_tokens=3 时在 Atlas 310P3 上输出吞吐从 9.22 提升到 14.30 tok/s，提升 55%；TPOT 从 108 毫秒降到 66 毫秒，降低 39%。在 Ascend 910B 上 num_spec_tokens=2 时 mean acceptance length 达到 1.63，token-1 接受率 70%，token-2 接受率 47%。

这个项目让我对推理系统的核心组件有了深入理解，也积累了开源协作经验——经过多轮 code review，处理了接口重构、测试补充、合并冲突等问题。

之后我还提交了 vLLM-MindSpore 的 PR #1020，为 Qwen2/Qwen2.5 适配 EAGLE-3。这个项目的挑战在于需要同时满足 MindSpore 框架的算子约束和 vLLM 社区的接口规范。

此外我还参与了 KV-select、Sparse Attention 的内部适配工作，对长上下文场景下的性能特征有一定理解。

**第二个项目：非结构化问数系统 RAG 后端**

这个项目我是唯一的后端开发者，独立负责从文档解析到答案生成的完整 RAG 链路。

架构上我设计了快速问答和 Research 问答双链路：快速问答做单次检索加生成，Research 模式支持多步检索和深度推理。我抽象了统一的 retriever 接口，支持 metadata filters 和多后端切换，降低了业务层和检索层的耦合。

质量保障方面，我用 RAGAS 框架构建了端到端评测流程，问答准确率达到 90%。

**我的短板和学习计划：**

我想诚实地说明我的短板：第一，我没有 CUDA kernel 开发经验，这是我目前最大的 gap；第二，没有分布式推理的实操经验；第三，没有 production 规模的 serving 经验。

但我认为在 NPU 上做推理优化积累的方法论——瓶颈分析、方案设计、参数调优、端到端验证——是可以迁移到 GPU 平台的。我目前正在系统学习 CUDA，计划通过实现 softmax、GEMM、attention 等 kernel 来补强这个短板。

我的目标是成为一个覆盖 GPU 和 NPU 双平台的推理系统工程师，能够从算法层面到硬件层面做端到端的推理优化。

### English Version

Hi, I'm Yuan Caoyao. I graduated with a Master's in Computer Science from Zhejiang University in March 2025. My undergraduate degree is in Information and Computational Science from Xi'an Jiaotong University, which gave me a solid math foundation. My core focus is LLM inference system optimization, and I also have independent RAG system delivery experience.

**Project 1: vLLM Ecosystem Inference Optimization**

This is my most significant project. The context was that the vLLM-Ascend community needed speculative decoding support on Ascend NPU to improve inference throughput. I independently implemented the EAGLE-3 proposer module.

Technically, I first needed to deeply understand the interaction between scheduler, model runner, and KV cache manager in vLLM's V1 architecture. Then I designed the proposer interface and connected the full execution pipeline from draft model through hidden states, KV cache, rejection sampler, to the runner. A key challenge was managing KV cache allocation and reclamation, especially cache state management for rejected tokens.

PR #1032 was merged into vLLM-Ascend's main repository. Performance results: with spec_tokens=3 on Atlas 310P3, output throughput improved from 9.22 to 14.30 tok/s (55% increase), TPOT dropped from 108ms to 66ms (39% reduction). On Ascend 910B with num_spec_tokens=2, mean acceptance length reached 1.63, with token-1 acceptance rate of 70% and token-2 at 47%.

This project gave me deep understanding of inference system core components and open-source collaboration experience — going through multiple rounds of code review, handling interface refactoring, test additions, and merge conflicts.

I also submitted PR #1020 to vLLM-MindSpore, adapting EAGLE-3 for Qwen2/Qwen2.5. The challenge there was satisfying both MindSpore's operator constraints and vLLM community's interface standards.

Additionally, I participated in internal adaptation of KV-select and Sparse Attention, gaining understanding of performance characteristics in long-context scenarios.

**Project 2: Unstructured Document Q&A System (RAG Backend)**

I was the sole backend developer, independently responsible for the complete RAG pipeline from document parsing to answer generation.

Architecturally, I designed a dual-path system: quick Q&A for single-retrieval-plus-generation, and Research mode for multi-step retrieval with deep reasoning. I abstracted a unified retriever interface supporting metadata filters and multiple backend switching, decoupling the business layer from the retrieval layer.

For quality assurance, I built an end-to-end evaluation pipeline using RAGAS, achieving 90% Q&A accuracy.

**My Gaps and Learning Plan:**

I want to be transparent about my gaps: First, I have no CUDA kernel development experience — this is my biggest gap. Second, no hands-on distributed inference experience. Third, no production-scale serving experience.

However, I believe the methodology I built doing inference optimization on NPU — bottleneck analysis, solution design, parameter tuning, end-to-end validation — transfers to GPU platforms. I'm currently systematically learning CUDA, planning to implement softmax, GEMM, and attention kernels to close this gap.

My goal is to become an inference system engineer covering both GPU and NPU platforms, capable of end-to-end inference optimization from algorithm level to hardware level.
