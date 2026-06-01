# 项目产出路线图

## 项目 1：高性能 CUDA GEMM 与 LLM 算子库

### 目标
从零实现一套 LLM 推理核心算子（GEMM、Softmax、RMSNorm、RoPE），逐步优化至接近 cuBLAS/cuDNN 性能水平，展示 CUDA 底层优化能力。

### 技术栈
- CUDA C++、cuBLAS（baseline 对比）
- NSight Compute（性能分析）
- Triton（对比实现）
- Python binding（pybind11）

### 预期产出
- 6-8 个优化版本的 SGEMM kernel（从 naive 到 register blocking + vectorized load）
- 最终版本达到 cuBLAS 80%+ 性能（2048×2048×2048）
- 完整的 profiling 报告（Roofline 图、NCU 指标对比）
- Triton 对比实现 + 性能分析博客

### 时间估计
3-4 周（与学习计划第 1-2 阶段同步）

### 面试讲述要点
- **开场**：为了深入理解 GPU 计算模型，从零实现了 LLM 核心算子库
- **技术深度**：逐步优化 GEMM — tiling → shared memory → register blocking → vectorized load，每步用 NCU 分析瓶颈
- **量化成果**：最终版本达到 cuBLAS 82% 性能，memory throughput 从 30% → 85% SOL
- **对比视角**：同时用 Triton 实现，对比开发效率与性能上限，得出适用场景结论
- **与已有经验关联**：在 NPU 上做过类似优化（EAGLE-3 的 KV cache 管理），GPU 上的优化思路有共通之处但工具链不同

---

## 项目 2：Triton FlashAttention 实现与 Benchmark

### 目标
用 Triton 实现 FlashAttention-2 算法，支持 multi-head attention、variable sequence length，并与官方 CUDA 实现做全面 benchmark。

### 技术栈
- Triton
- PyTorch（集成测试）
- FlashAttention 官方库（baseline）
- NSight Systems（端到端 profiling）

### 预期产出
- Triton FlashAttention-2 实现（支持 forward + backward）
- 支持 causal mask、multi-head、grouped-query attention
- Benchmark 报告：不同 seq_len、head_dim 下与官方实现的性能对比
- 技术博客：FlashAttention 算法推导 + Triton 实现要点

### 时间估计
2-3 周（学习计划第 5-6 周）

### 面试讲述要点
- **动机**：FlashAttention 是 LLM 推理的核心优化，想从实现层面深入理解其 IO-aware 设计
- **算法理解**：手推 online softmax trick，理解 tiling 策略如何将 IO 复杂度从 O(N²) 降到 O(N²d/M)
- **工程挑战**：Triton 中处理 causal mask 的边界条件、auto-tuning block size 的策略
- **性能分析**：在 A100 上 seq_len=4096 时达到官方 CUDA 实现 75% 性能，分析差距来源
- **与 PagedAttention 关联**：讨论 FlashAttention 与 PagedAttention 的结合方式（vLLM 中的实现）

---

## 项目 3：LLM 推理引擎 Benchmark 与分析平台

### 目标
搭建一个自动化 benchmark 平台，对比 vLLM、TensorRT-LLM、SGLang 在不同场景下的性能表现，输出深度分析报告。

### 技术栈
- Python（benchmark 框架）
- vLLM、TensorRT-LLM、SGLang
- Docker（环境隔离）
- Matplotlib/Plotly（可视化）
- NSight Systems（kernel-level 分析）

### 预期产出
- 自动化 benchmark 脚本（支持多模型、多框架、多并发度）
- 对比维度：TTFT、TPOT、吞吐、GPU 利用率、内存占用
- 深度分析报告：kernel fusion 差异、调度策略差异、内存管理差异
- 量化场景对比：FP16 vs INT8 vs INT4 在各框架上的表现

### 时间估计
2-3 周（学习计划第 7-9 周）

### 面试讲述要点
- **系统视角**：不只是跑 benchmark，而是从 kernel-level 分析性能差异的根因
- **发现与洞察**：例如 TRT-LLM 在 prefill 阶段优势明显（kernel fusion 更激进），vLLM 在高并发 decode 场景更优（调度灵活）
- **工程能力**：自动化测试框架设计、Docker 环境管理、结果可视化
- **与已有经验关联**：在 vLLM-Ascend 中做过类似 benchmark（9.22→14.30 tok/s），这次扩展到 GPU 生态的全面对比

---

## 项目 4：分布式推理 Demo — Tensor Parallel Serving

### 目标
基于 PyTorch 实现一个简化版的 Tensor Parallel 推理 demo，支持多 GPU 部署 LLaMA-style 模型，理解分布式推理的通信开销和优化空间。

### 技术栈
- PyTorch Distributed（NCCL backend）
- Transformers（模型加载）
- NCCL（通信原语）
- Python asyncio（请求调度）

### 预期产出
- 支持 TP=2/4 的 LLaMA 推理 demo
- 通信开销分析：AllReduce 延迟 vs 计算延迟的比例
- 优化实验：overlap communication with computation
- 设计文档：如何扩展到 TP+PP 混合并行

### 时间估计
2 周（学习计划第 8 周）

### 面试讲述要点
- **设计决策**：为什么选择 Megatron-style column/row parallel，而非其他切分方式
- **通信分析**：TP=4 时 AllReduce 占总延迟的比例，以及如何通过 overlap 优化
- **扩展思考**：讨论 TP vs PP 的 trade-off，什么时候该用 PP（模型太大单机放不下）
- **与已有经验关联**：在 vLLM-Ascend 中接触过分布式执行路径，这次从零实现加深理解

---

## 项目时间线总览

```
Week 1-3:  [====== 项目 1: CUDA GEMM 算子库 ======]
Week 4-6:  [====== 项目 2: Triton FlashAttention ======]
Week 7-9:  [=== 项目 3: Benchmark 平台 ===][== 项目 4: TP Demo ==]
Week 10-12: [整理文档、博客、简历更新、模拟面试]
```

## 简历呈现建议

每个项目在简历上用 2-3 行描述：

**项目 1 示例**：
> 高性能 CUDA GEMM 算子库 | CUDA C++, NSight Compute, Triton
> - 从零实现 SGEMM 多级优化（tiling → register blocking → vectorized load），达到 cuBLAS 82% 性能
> - 完成 LLM 核心算子（Softmax, RMSNorm, RoPE）的 CUDA + Triton 双版本实现与性能对比

**项目 2 示例**：
> Triton FlashAttention-2 实现 | Triton, PyTorch, CUDA
> - 用 Triton 实现 FlashAttention-2（支持 MHA/GQA、causal mask），forward 性能达官方 CUDA 版 75%
> - 输出 FlashAttention 算法推导与 IO 复杂度分析技术博客
