# 8-12 周验收标准与行动计划

> 起始日期：2026-06-01
> 前置条件：需要 NVIDIA GPU（建议 RTX 3060+，≥12GB VRAM）或云 GPU（AutoDL/Lambda）

---

## 第 1 周（6/2 - 6/8）：CUDA 基础 + 开源启动

| 维度 | 内容 |
|------|------|
| learning target | CUDA Memory Model（Global/Shared/Register/L1/L2）、Thread→Warp→Block→Grid、Memory Coalescing |
| coding deliverable | `vector_add.cu`、`memory_coalescing_demo.cu`、`shared_memory_transpose.cu` |
| benchmark deliverable | coalesced vs non-coalesced 带宽对比数据 |
| document deliverable | `notes/cuda_memory_model.md` |
| interview deliverable | 能回答 5 道 CUDA Memory Hierarchy 基础题 |
| resume update | 无（尚未产出可写内容） |
| acceptance criteria | 3 个 .cu 文件编译通过 + 正确性验证 + benchmark 数据记录 |
| failure condition | 无法获取 GPU 环境 / 编译失败无法解决 |
| recovery plan | 使用 Colab 或 AutoDL；参考 PMPP 教材 Chapter 4-5 |

---

## 第 2 周（6/9 - 6/15）：Reduction + Nsight + 开源 PR

| 维度 | 内容 |
|------|------|
| learning target | Parallel Reduction、Warp-level Primitives（__shfl_down_sync）、Nsight Compute 基本操作 |
| coding deliverable | `reduction_sum.cu`、`reduction_max.cu`、`warp_reduction.cu` |
| benchmark deliverable | 不同 block size 下 reduction 性能对比 + Nsight Compute 截图 |
| document deliverable | `notes/nsight_compute_guide.md` |
| interview deliverable | 能回答 Warp Divergence、Occupancy 相关追问 |
| resume update | 无 |
| acceptance criteria | reduction 结果正确 + Nsight 能跑通 + 性能数据记录 |
| failure condition | Nsight 安装失败 / 性能数据异常 |
| recovery plan | 使用 nvprof 替代；检查 GPU driver 版本 |

---

## 第 3 周（6/16 - 6/22）：MatMul Tiling + vLLM PR 提交

| 维度 | 内容 |
|------|------|
| learning target | GEMM naive → tiled → vectorized、Shared Memory Bank Conflict |
| coding deliverable | `matmul_naive.cu`、`matmul_tiled.cu`、`matmul_vectorized.cu` |
| benchmark deliverable | 3 个版本 vs cuBLAS 性能对比（目标：tiled 达到 cuBLAS 30-50%） |
| document deliverable | `notes/gemm_optimization.md` |
| interview deliverable | 能画出 tiled GEMM 的 shared memory 使用图 |
| resume update | 无（目标未达到 60%+ 不写） |
| acceptance criteria | tiled 版本正确 + 性能 > naive 5x + 有 Nsight 分析 |
| failure condition | tiled 版本性能不如 naive |
| recovery plan | 检查 bank conflict、检查 tile size 选择、参考 CUTLASS 教程 |

---

## 第 4 周（6/23 - 6/29）：Softmax + LayerNorm + RMSNorm

| 维度 | 内容 |
|------|------|
| learning target | Online Softmax、Fused LayerNorm、RMSNorm、Warp Reduction in Normalization |
| coding deliverable | `softmax_row.cu`、`layernorm.cu`、`rmsnorm.cu` |
| benchmark deliverable | 各 kernel vs PyTorch 实现性能对比 |
| document deliverable | `notes/normalization_kernels.md` |
| interview deliverable | 能解释 online softmax 的数值稳定性 + 为什么 RMSNorm 比 LayerNorm 快 |
| resume update | 可开始写："Implemented CUDA kernels for LLM inference operators (softmax, RMSNorm) with Nsight profiling" |
| acceptance criteria | 3 个 kernel 正确 + 性能达到 PyTorch 80%+ |
| failure condition | 数值精度问题 / 性能远低于 PyTorch |
| recovery plan | 检查 FP32 accumulation、检查 memory access pattern |

---

## 第 5 周（6/30 - 7/6）：RoPE + FlashAttention Toy + Triton 入门

| 维度 | 内容 |
|------|------|
| learning target | Rotary Position Embedding、FlashAttention 算法原理、Triton Programming Model |
| coding deliverable | `rope.cu`、`flash_attention_toy.cu`、`triton_vector_add.py` |
| benchmark deliverable | RoPE vs PyTorch 对比；Triton vs CUDA vector_add 对比 |
| document deliverable | `notes/flash_attention_algorithm.md` |
| interview deliverable | 能解释 FlashAttention 的 tiling 策略和 IO 复杂度 |
| resume update | 更新 CUDA kernel 数量 |
| acceptance criteria | FlashAttention toy 版本正确（对比 PyTorch attention）+ Triton 环境搭建完成 |
| failure condition | FlashAttention 数值不正确 |
| recovery plan | 先实现 naive attention，逐步加 tiling；参考 FlashAttention paper |

---

## 第 6 周（7/7 - 7/13）：Triton Kernels + LLM Benchmark 搭建

| 维度 | 内容 |
|------|------|
| learning target | Triton matmul/softmax/fused_attention、vLLM/SGLang benchmark 方法论 |
| coding deliverable | `triton_matmul.py`、`triton_softmax.py`、`triton_fused_attention.py`、benchmark 脚本 |
| benchmark deliverable | Triton vs CUDA 性能对比；vLLM benchmark 首次运行数据 |
| document deliverable | `notes/triton_vs_cuda.md` |
| interview deliverable | 能对比 Triton 和 CUDA 的 tradeoff |
| resume update | "Implemented Triton kernels (matmul, softmax, fused attention) and benchmarked against CUDA implementations" |
| acceptance criteria | 3 个 Triton kernel 正确 + vLLM benchmark 脚本可运行 |
| failure condition | Triton 编译错误 / vLLM 部署失败 |
| recovery plan | 使用 Triton 官方 tutorial 作为起点；使用小模型（1B）测试 |

---

## 第 7 周（7/14 - 7/20）：LLM Inference Benchmark 完整实验

| 维度 | 内容 |
|------|------|
| learning target | TTFT/TPOT/throughput 测量方法、Continuous Batching 行为分析、Prefill/Decode 分离 |
| coding deliverable | 完整 benchmark suite（多 concurrency、多 input_length） |
| benchmark deliverable | vLLM 在 10+ 配置下的完整性能数据 |
| document deliverable | `notes/llm_serving_performance_analysis.md` |
| interview deliverable | 能用数据回答"vLLM 在什么场景下性能最好/最差" |
| resume update | "Designed LLM inference benchmark suite evaluating vLLM across 50+ configurations" |
| acceptance criteria | 有可视化图表 + 有分析结论 + 数据可复现 |
| failure condition | GPU 内存不足 / 模型加载失败 |
| recovery plan | 使用更小模型；减少 concurrency 上限 |

---

## 第 8 周（7/21 - 7/27）：RAG Eval + 系统设计练习

| 维度 | 内容 |
|------|------|
| learning target | RAGAS 评测深入、Hybrid Search、Milvus/FAISS 性能对比、RAG 系统设计 |
| coding deliverable | RAG eval pipeline（FAISS + BM25 + rerank + RAGAS） |
| benchmark deliverable | 不同检索策略的 recall/precision/latency 对比 |
| document deliverable | `notes/rag_system_design.md` |
| interview deliverable | 能完成 45 分钟 RAG 系统设计面试 |
| resume update | "Built RAG evaluation framework comparing vector/hybrid/rerank strategies with RAGAS metrics" |
| acceptance criteria | 有 3+ 检索策略对比数据 + RAGAS 评分 |
| failure condition | 数据集准备困难 |
| recovery plan | 使用公开 QA 数据集（HotpotQA、Natural Questions） |

---

## 第 9 周（7/28 - 8/3）：Quantization + 开源 PR 第二波

| 维度 | 内容 |
|------|------|
| learning target | INT8/FP8 Quantization、GPTQ/AWQ/SmoothQuant 原理、量化对推理性能的影响 |
| coding deliverable | 量化 benchmark 脚本（对比 FP16 vs INT8 vs INT4） |
| benchmark deliverable | 量化前后 throughput/latency/accuracy 对比 |
| document deliverable | `notes/quantization_deep_dive.md` |
| interview deliverable | 能解释量化的 accuracy-performance tradeoff |
| resume update | 更新 benchmark 覆盖范围 |
| acceptance criteria | 有量化前后对比数据 + 精度损失评估 |
| failure condition | 量化工具安装困难 |
| recovery plan | 使用 vLLM 内置量化支持；使用 HuggingFace 预量化模型 |

---

## 第 10 周（8/4 - 8/10）：分布式推理概念 + Mock Interview 密集练习

| 维度 | 内容 |
|------|------|
| learning target | Tensor Parallelism、Pipeline Parallelism、NCCL 基础、Disaggregated Serving |
| coding deliverable | TP 概念 demo（如有多 GPU）或文档总结 |
| benchmark deliverable | 无（概念学习为主） |
| document deliverable | `notes/distributed_inference.md` |
| interview deliverable | 完成 3 场完整 mock interview（LLM Inference + RAG + Behavioral） |
| resume update | 无（不可写未实操内容） |
| acceptance criteria | 能画出 TP/PP 的数据流图 + mock interview 通过率 > 50% |
| failure condition | 概念理解困难 |
| recovery plan | 阅读 Megatron-LM paper + vLLM TP 源码 |

---

## 第 11 周（8/11 - 8/17）：简历定制 + 投递准备

| 维度 | 内容 |
|------|------|
| learning target | 目标公司技术栈深入了解 |
| coding deliverable | 项目 README 完善、GitHub profile 整理 |
| benchmark deliverable | 所有项目的最终 benchmark 数据汇总 |
| document deliverable | 7 个方向简历最终版 |
| interview deliverable | 完成 5 场 mock interview + 错题本更新 |
| resume update | 所有版本简历定稿 |
| acceptance criteria | 简历无 unsupported claim + 项目有数据支撑 + mock 通过率 > 60% |
| failure condition | 项目数据不够支撑简历 bullet |
| recovery plan | 降低 bullet 表述强度；补充缺失实验 |

---

## 第 12 周（8/18 - 8/24）：投递冲刺 + 持续优化

| 维度 | 内容 |
|------|------|
| learning target | 面试复盘与改进 |
| coding deliverable | 开源 PR 持续提交 |
| benchmark deliverable | 根据面试反馈补充实验 |
| document deliverable | 面试复盘记录 |
| interview deliverable | 每天 1 场 mock interview |
| resume update | 根据面试反馈微调 |
| acceptance criteria | 投递 10+ 岗位 + 获得 3+ 面试机会 |
| failure condition | 无面试邀请 |
| recovery plan | 扩大投递范围；调整简历定位；增加开源贡献可见度 |

---

## 总体验收标准

| 里程碑 | 时间 | 标准 |
|--------|------|------|
| CUDA 基础完成 | 第 4 周末 | 6+ kernel 正确实现 + Nsight 分析 |
| Benchmark Lab 完成 | 第 7 周末 | vLLM 50+ 配置数据 + 可视化 |
| 简历可投递 | 第 8 周末 | 至少 3 个方向简历有数据支撑 |
| Mock Interview 达标 | 第 10 周末 | LLM Inference 面试通过率 > 60% |
| 投递启动 | 第 11 周 | 开始正式投递 |
| 获得面试 | 第 12 周 | 3+ 面试邀请 |
