# 2-3 个月系统补强计划

## 总体目标

基于已有 vLLM + Ascend NPU 适配经验，系统补强 CUDA/GPU 生态知识，达到大模型推理系统工程师的面试水平。

## 前置说明

- **已掌握（跳过）**：vLLM 架构、KV Cache 概念、Speculative Decoding 原理、NPU 算子适配、RAG 全链路
- **重点补强**：CUDA 编程模型、GPU 算子开发、Triton、FlashAttention/PagedAttention 源码、分布式推理、量化实操
- **每周预计投入**：工作日 2-3h + 周末 6-8h

---

## 第 1 阶段：CUDA 基础与 GPU 架构（第 1-3 周）

### 第 1 周：CUDA 编程模型与内存层次

**学习目标**：
- 理解 CUDA 执行模型（Grid/Block/Thread/Warp）
- 掌握 GPU 内存层次（Global/Shared/Register/L1/L2）
- 能编写基础 CUDA kernel（向量加法、矩阵转置）

**学习资源**：
- 📖 [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — Chapter 1-5
- 📖 [Professional CUDA C Programming](https://www.wiley.com/en-us/Professional+CUDA+C+Programming-p-9781118739327) — Chapter 1-4
- 🎥 [NVIDIA GTC: CUDA Programming](https://www.nvidia.com/en-us/on-demand/)
- 💻 [cuda-samples](https://github.com/NVIDIA/cuda-samples) — 0_Introduction 目录

**练习项目**：
1. 实现 vector_add kernel，对比不同 block size 的性能
2. 实现 matrix_transpose，对比 naive vs shared memory 版本
3. 用 `nvprof` / `nsys` 分析上述 kernel 的内存带宽利用率

**验证标准**：
- [ ] 能独立解释 warp divergence、bank conflict、coalesced access
- [ ] matrix_transpose shared memory 版本达到理论带宽 80%+
- [ ] 能用 NSight Systems 生成 timeline 并解读

---

### 第 2 周：GPU Profiling 与性能优化

**学习目标**：
- 熟练使用 NSight Systems 和 NSight Compute
- 理解 Occupancy、Arithmetic Intensity、Roofline Model
- 掌握常见优化手段（循环展开、向量化访存、指令级并行）

**学习资源**：
- 📖 [NSight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- 📖 [NSight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- 📖 [Roofline Model Paper](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyworksRevised.pdf)
- 🎥 [GTC: Kernel Profiling Guide](https://www.nvidia.com/en-us/on-demand/) — 搜索 "kernel profiling"
- 💻 [Lei Mao's Blog: CUDA Optimization](https://leimao.github.io/)

**练习项目**：
1. 用 NCU 分析 matrix_transpose 的 SOL（Speed of Light）指标
2. 实现 reduction kernel（sum），逐步优化：naive → warp shuffle → 多级 reduction
3. 绘制自己 kernel 的 Roofline 图，标注瓶颈

**验证标准**：
- [ ] 能解读 NCU 报告中的 Memory Throughput、Compute Throughput、Occupancy
- [ ] reduction kernel 达到理论峰值 90%+
- [ ] 能用 Roofline 判断 kernel 是 compute-bound 还是 memory-bound

---

### 第 3 周：CUDA 进阶 — Stream、Event、多 GPU 基础

**学习目标**：
- 理解 CUDA Stream 和异步执行模型
- 掌握 Event 计时和同步机制
- 了解 Unified Memory 和 Multi-GPU 编程基础

**学习资源**：
- 📖 CUDA Programming Guide — Chapter 6 (Streams), Chapter 9 (Multi-Device)
- 📖 [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- 💻 cuda-samples — 6_Performance 目录

**练习项目**：
1. 实现 pipeline：H2D → Kernel → D2H 三阶段流水线，用多 stream 重叠
2. 对比单 stream vs 多 stream 的端到端延迟
3. 实现简单的 multi-GPU vector_add（如有多卡环境）

**验证标准**：
- [ ] 能用 NSight Systems 可视化 stream 重叠
- [ ] pipeline 版本相比串行版本加速 2x+
- [ ] 能解释 stream 间的隐式/显式同步

---

## 第 2 阶段：算子开发与 Triton（第 4-6 周）

### 第 4 周：GEMM 优化 — 从 Naive 到高性能

**学习目标**：
- 理解 GEMM 在 LLM 推理中的核心地位
- 实现多级优化的 GEMM kernel
- 理解 cuBLAS/CUTLASS 的设计思路

**学习资源**：
- 📖 [CUTLASS Documentation](https://github.com/NVIDIA/cutlass)
- 📖 [How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM) — Simon Boehm 经典博客
- 📖 [NVIDIA CUTLASS: Fast Linear Algebra](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)
- 💻 [cuda_gemm_optimization](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE)

**练习项目**：
1. 实现 SGEMM 逐步优化：naive → tiling → shared memory → register blocking → vectorized load
2. 对比每个版本与 cuBLAS 的性能比（目标：达到 cuBLAS 80%+）
3. 用 NCU 分析每个版本的瓶颈

**验证标准**：
- [ ] 最终版本在 (2048, 2048, 2048) 上达到 cuBLAS 80%+ 性能
- [ ] 能解释 tiling size 选择与 shared memory 容量的关系
- [ ] 能解释 register blocking 如何减少 shared memory 访问

---

### 第 5 周：Triton 编程

**学习目标**：
- 掌握 Triton 编程模型（block-level programming）
- 能用 Triton 实现常见算子
- 理解 Triton 与 CUDA 的性能差异和适用场景

**学习资源**：
- 📖 [Triton Official Tutorials](https://triton-lang.org/main/getting-started/tutorials/)
- 📖 [Triton Paper: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)
- 💻 [triton-puzzles](https://github.com/srush/Triton-Puzzles) — 入门练习
- 💻 [unsloth](https://github.com/unslothai/unsloth) — 参考其 Triton kernel 实现

**练习项目**：
1. 用 Triton 实现 softmax、LayerNorm、RMSNorm
2. 用 Triton 实现 fused attention（不用 FlashAttention 算法，先做 naive fused）
3. 对比 Triton 实现与 PyTorch native 的性能

**验证标准**：
- [ ] Triton softmax 性能超过 PyTorch native
- [ ] 能解释 Triton 的 auto-tuning 机制
- [ ] 能对比 Triton vs CUDA：开发效率、性能上限、调试难度

---

### 第 6 周：LLM 核心算子 — Attention 与 Normalization

**学习目标**：
- 深入理解 FlashAttention 算法（tiling + online softmax）
- 理解 PagedAttention 的实现细节
- 实现 RoPE、RMSNorm 等 LLM 常用算子

**学习资源**：
- 📖 [FlashAttention Paper](https://arxiv.org/abs/2205.14135) + [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- 📖 [FlashAttention CUDA 源码](https://github.com/Dao-AILab/flash-attention)
- 📖 [vLLM PagedAttention 源码](https://github.com/vllm-project/vllm/tree/main/csrc)
- 📖 [Online Softmax Trick 推导](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)

**练习项目**：
1. 手推 FlashAttention 的 online softmax 数学推导
2. 阅读 FlashAttention-2 CUDA 源码，画出数据流图
3. 阅读 vLLM PagedAttention kernel 源码，理解 block table 映射
4. 用 Triton 实现简化版 FlashAttention（单头、固定 seq_len）

**验证标准**：
- [ ] 能白板推导 FlashAttention 的 IO 复杂度为何是 O(N²d/M)
- [ ] 能解释 PagedAttention 如何处理 variable-length sequences
- [ ] Triton FlashAttention 实现正确性通过（与 PyTorch 对比误差 < 1e-3）

---

## 第 3 阶段：推理框架源码与分布式（第 7-9 周）

### 第 7 周：vLLM 源码深入 — Scheduler 与 Engine

**学习目标**：
- 深入理解 vLLM 的 Scheduler 调度逻辑
- 理解 Engine 的请求生命周期
- 理解 Prefix Caching 和 Chunked Prefill

**学习资源**：
- 📖 [vLLM 源码](https://github.com/vllm-project/vllm) — `vllm/core/scheduler.py`, `vllm/engine/`
- 📖 [vLLM Paper: Efficient Memory Management for Large Language Model Serving](https://arxiv.org/abs/2309.06180)
- 📖 [SGLang Paper](https://arxiv.org/abs/2312.07104) — 对比学习

**练习项目**：
1. 画出 vLLM 请求从 API 到 token 输出的完整数据流图
2. 阅读 Scheduler 源码，总结 waiting/running/swapped 队列的调度策略
3. 对比 vLLM vs SGLang 的调度设计差异

**验证标准**：
- [ ] 能解释 continuous batching 相比 static batching 的优势
- [ ] 能解释 prefix caching 的 hash 机制和命中率影响因素
- [ ] 能画出 chunked prefill 的执行时序图

---

### 第 8 周：分布式推理 — TP/PP/NCCL

**学习目标**：
- 理解 Tensor Parallelism 和 Pipeline Parallelism 的原理与实现
- 理解 NCCL 通信原语（AllReduce, AllGather, ReduceScatter）
- 理解 vLLM 中的分布式执行路径

**学习资源**：
- 📖 [Megatron-LM Paper](https://arxiv.org/abs/1909.08053)
- 📖 [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- 📖 [vLLM 分布式源码](https://github.com/vllm-project/vllm/tree/main/vllm/distributed)
- 📖 [Efficient Large-Scale Language Model Training on GPU Clusters](https://arxiv.org/abs/2104.04473)
- 💻 [nccl-tests](https://github.com/NVIDIA/nccl-tests)

**练习项目**：
1. 用 PyTorch distributed 实现简单的 Tensor Parallel Linear 层
2. 分析 Megatron-style TP 中 AllReduce 的通信量
3. 阅读 vLLM 的 `tensor_parallel` 模块，画出 TP 推理的通信模式

**验证标准**：
- [ ] 能计算 TP=8 时单次 AllReduce 的通信量和理论延迟
- [ ] 能解释 TP vs PP 的适用场景和 trade-off
- [ ] 能解释 Ring AllReduce 的工作原理和带宽利用率

---

### 第 9 周：TensorRT-LLM 与推理优化技术

**学习目标**：
- 理解 TensorRT-LLM 的架构和优化手段
- 理解 KV Cache 量化、Weight-Only 量化
- 理解 Speculative Decoding 在 TRT-LLM 中的实现

**学习资源**：
- 📖 [TensorRT-LLM 源码](https://github.com/NVIDIA/TensorRT-LLM)
- 📖 [TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/)
- 📖 [FasterTransformer 源码](https://github.com/NVIDIA/FasterTransformer)（已归档但有参考价值）

**练习项目**：
1. 用 TensorRT-LLM 部署一个 7B 模型，对比 vLLM 的吞吐和延迟
2. 分析 TRT-LLM 的 kernel fusion 策略
3. 对比 TRT-LLM vs vLLM 的 Speculative Decoding 实现差异

**验证标准**：
- [ ] 能解释 TRT-LLM 的 build → runtime 两阶段流程
- [ ] 能对比 TRT-LLM vs vLLM 的设计哲学差异
- [ ] 能解释 in-flight batching 的实现原理

---

## 第 4 阶段：量化与系统设计（第 10-12 周）

### 第 10 周：量化技术 — GPTQ/AWQ/SmoothQuant

**学习目标**：
- 理解 PTQ（Post-Training Quantization）的核心思路
- 掌握 GPTQ、AWQ、SmoothQuant 的算法原理
- 理解量化对推理性能和精度的影响

**学习资源**：
- 📖 [GPTQ Paper](https://arxiv.org/abs/2210.17323)
- 📖 [AWQ Paper](https://arxiv.org/abs/2306.00978)
- 📖 [SmoothQuant Paper](https://arxiv.org/abs/2211.10438)
- 💻 [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ)
- 💻 [llm-awq](https://github.com/mit-han-lab/llm-awq)

**练习项目**：
1. 用 AutoGPTQ 量化一个 7B 模型，对比 FP16 vs INT4 的 perplexity 和吞吐
2. 阅读 AWQ 源码，理解 salient weight 的检测和 scale 计算
3. 实现简化版 SmoothQuant：对一个 Linear 层做 per-channel smooth + INT8 量化

**验证标准**：
- [ ] 能解释 GPTQ 的 Hessian-based 逐列量化过程
- [ ] 能解释 AWQ 为何只保护 1% salient weights 就能保持精度
- [ ] 能对比 W4A16 vs W8A8 vs W4A4 的适用场景

---

### 第 11 周：系统设计 — LLM Serving System

**学习目标**：
- 能设计完整的 LLM Serving 系统
- 理解 SLA、吞吐、延迟之间的 trade-off
- 掌握面试中系统设计题的回答框架

**学习资源**：
- 📖 [Orca Paper: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)
- 📖 [Sarathi-Serve Paper](https://arxiv.org/abs/2403.02310)
- 📖 [DistServe Paper](https://arxiv.org/abs/2401.09670)
- 📖 [System Design Interview 框架](https://github.com/donnemartin/system-design-primer)

**练习项目**：
1. 设计题：设计一个支持 100 QPS、P99 < 2s 的 LLM Serving 系统
2. 设计题：设计一个多模型、多租户的推理平台
3. 写出每个设计的 capacity planning 计算过程

**验证标准**：
- [ ] 能在 30 分钟内完成一个系统设计的完整回答
- [ ] 能计算给定 SLA 下需要的 GPU 数量
- [ ] 能讨论 prefill/decode 分离部署的优劣

---

### 第 12 周：综合复习与模拟面试

**学习目标**：
- 串联所有知识点，形成完整知识图谱
- 模拟面试练习，提升表达流畅度
- 查漏补缺

**练习项目**：
1. 完成 3 套模拟面试（见 interview/03_mock_interview_scripts.md）
2. 整理个人技术博客 / GitHub README，展示项目成果
3. 准备 5 分钟自我介绍 + 项目深挖回答

**验证标准**：
- [ ] 模拟面试评分达到 "Strong Hire" 水平
- [ ] 每个项目能讲 10 分钟不卡壳
- [ ] 能应对 2-3 层追问

---

## 每日时间分配建议

| 时间段 | 内容 | 时长 |
|--------|------|------|
| 早晨 | 阅读论文/文档 | 1h |
| 晚间 | 编码练习 | 1.5-2h |
| 周末上午 | 项目开发 | 3-4h |
| 周末下午 | 源码阅读 + 笔记 | 3-4h |

## 关键里程碑

| 时间点 | 里程碑 | 可验证产出 |
|--------|--------|-----------|
| 第 3 周末 | CUDA 基础完成 | GitHub: 5+ 优化 kernel + profiling 报告 |
| 第 6 周末 | 算子开发完成 | GitHub: GEMM 80%+ cuBLAS + Triton FlashAttention |
| 第 9 周末 | 框架源码完成 | 技术博客: vLLM/TRT-LLM 源码分析 |
| 第 12 周末 | 全部完成 | 模拟面试通过 + 简历更新 |
