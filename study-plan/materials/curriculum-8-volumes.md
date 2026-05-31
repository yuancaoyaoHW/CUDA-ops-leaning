# GPU 推理系统学习材料 — 8 卷目录

> 由 4 个领域的学习材料合并而成，覆盖 CUDA 算子、推理框架、后训练与 GPU Profiling。

---

## 卷 1：GPU 计算基础与 CUDA 编程模型

**目标**：建立 GPU 硬件理解和 CUDA 编程基础

| 章节 | 来源文件 | 核心内容 |
|------|---------|---------|
| 1.1 CUDA 执行模型 | cuda-kernels/01-cuda-execution-model.md | thread/block/grid/warp/SM 层次 |
| 1.2 内存层次 | cuda-kernels/02-memory-hierarchy.md | register→shared→L1/L2→HBM，coalescing，bank conflict |
| 1.3 Register/Occupancy/Divergence | cuda-kernels/03-register-occupancy-divergence.md | register pressure，occupancy 计算，warp divergence |
| 1.4 Stream 与 Graph | cuda-kernels/04-stream-graph.md | 多 stream 并发，CUDA Graph 在推理中的应用 |

**习题总数**：80 道 | **复习卡片**：120 张 | **实验任务**：16 个

---

## 卷 2：核心推理算子

**目标**：掌握 LLM 推理中的关键 CUDA/Triton 算子实现

| 章节 | 来源文件 | 核心内容 |
|------|---------|---------|
| 2.1 Tensor Core 与 GEMM | cuda-kernels/05-tensor-core-gemm.md | WMMA/MMA，tiled GEMM，cuBLAS/CUTLASS |
| 2.2 Parallel Reduction | cuda-kernels/06-reduction.md | tree reduction，warp shuffle，multi-block |
| 2.3 Softmax / Fused Softmax | cuda-kernels/07-softmax-fused-softmax.md | online softmax，fused kernel，row-wise |
| 2.4 RMSNorm | cuda-kernels/08-rmsnorm.md | fused RMSNorm+residual，FP16+FP32 混合 |
| 2.5 RoPE | cuda-kernels/09-rope.md | 旋转位置编码，频率设计，NTK scaling |

**习题总数**：100 道 | **复习卡片**：150 张 | **实验任务**：25 个

---

## 卷 3：Attention 与 KV Cache

**目标**：深入理解 attention 机制的实现与优化

| 章节 | 来源文件 | 核心内容 |
|------|---------|---------|
| 3.1 Attention 机制 | cuda-kernels/10-attention.md | MHA/GQA/MQA，prefill vs decode，compute/memory bound |
| 3.2 FlashAttention | cuda-kernels/11-flashattention.md | tiling+online softmax，IO 复杂度，FA-2/FA-3 |
| 3.3 PagedAttention 与 KV Cache | cuda-kernels/12-pagedattention-kvcache.md | 虚拟内存管理，block table，copy-on-write |
| 3.4 MoE Routing | cuda-kernels/13-moe-routing.md | Top-K routing，grouped GEMM，expert parallel |
| 3.5 Triton 对照实现 | cuda-kernels/14-triton-comparison.md | CUDA vs Triton 对比，autotuning |

**习题总数**：100 道 | **复习卡片**：120 张 | **实验任务**：20 个

---

## 卷 4：LLM 推理框架

**目标**：掌握主流推理框架的架构设计与调优

| 章节 | 来源文件 | 核心内容 |
|------|---------|---------|
| 4.1 vLLM 架构 | inference-frameworks/01-vllm-architecture.md | Scheduler/BlockManager/Worker/ModelRunner |
| 4.2 SGLang 架构 | inference-frameworks/02-sglang-architecture.md | RadixAttention，prefix caching，constrained decoding |
| 4.3 Prefill 与 Decode | inference-frameworks/05-prefill-decode.md | 两阶段分析，TTFT/TPOT，PD disaggregation |
| 4.4 TensorRT-LLM | (待生成) | inflight batching，FP8，plugin |
| 4.5 Continuous Batching | (待生成) | 动态调度，iteration-level scheduling |
| 4.6 Speculative Decoding | (待生成) | draft model，N-gram，Medusa |
| 4.7 Tensor/Pipeline Parallel | (待生成) | TP/PP/EP 在推理中的应用 |
| 4.8 量化推理 | (待生成) | INT4/INT8/FP8，GPTQ/AWQ |
| 4.9 生产部署 | (待生成) | benchmark，observability，production checklist |

**习题总数**：60+ 道 | **实验任务**：15+ 个

---

## 卷 5：SFT 与参数高效微调

**目标**：掌握 SFT 训练流程与 LoRA/QLoRA 技术

| 章节 | 来源文件 | 核心内容 |
|------|---------|---------|
| 5.1 SFT / Instruction Tuning | post-training/01-sft-instruction-tuning.md | 数据格式，loss masking，chat template |
| 5.2 LoRA / QLoRA | (待生成) | 低秩分解，NF4，target modules |
| 5.3 Full Fine-tuning / Packing | (待生成) | 学习率策略，多样本拼接 |
| 5.4 Reward Model | (待生成) | Bradley-Terry，pairwise loss |
| 5.5 Evaluation Harness | (待生成) | lm-eval，MT-Bench，自动评估 |

**习题总数**：40+ 道 | **面试题**：40+ 道 | **复习卡片**：60+ 张

---

## 卷 6：RLHF 与偏好对齐

**目标**：掌握 DPO/PPO/GRPO 等对齐方法

| 章节 | 来源文件 | 核心内容 |
|------|---------|---------|
| 6.1 DPO | post-training/05-dpo.md | 隐式 reward，reference model，β 参数 |
| 6.2 PPO | post-training/07-ppo.md | clip objective，GAE，4 模型协作 |
| 6.3 GRPO | post-training/08-grpo.md | 无 Critic，组内相对优势，DeepSeek 实践 |
| 6.4 IPO / KTO / ORPO | (待生成) | 各自动机与 loss 对比 |
| 6.5 RLVR / Agentic RL | (待生成) | verifiable reward，tool use RL |
| 6.6 Reward Hacking | (待生成) | 过优化检测与防御 |
| 6.7 Rollout Engine | (待生成) | vLLM/SGLang backend，async generation |
| 6.8 KL Penalty / Advantage | (待生成) | KL 散度计算，GAE baseline |

**习题总数**：60+ 道 | **面试题**：60+ 道 | **复习卡片**：90+ 张

---

## 卷 7：GPU Profiling 与性能调优

**目标**：掌握 GPU 性能分析工具与调优方法论

| 章节 | 来源文件 | 核心内容 |
|------|---------|---------|
| 7.1 Nsight Systems | gpu-profiling/01-nsight-systems.md | timeline 分析，kernel overlap，CPU-GPU 交互 |
| 7.2 Roofline Analysis | gpu-profiling/05-roofline-analysis.md | arithmetic intensity，bound 判断，优化方向 |
| 7.3 Nsight Compute | (待生成) | kernel 级分析，Speed of Light |
| 7.4 PyTorch Profiler | (待生成) | torch.profiler，TensorBoard |
| 7.5 Memory Bandwidth | (待生成) | HBM/L2 带宽测量 |
| 7.6 Occupancy / Warp Stall | (待生成) | stall reasons，eligible warps |
| 7.7 Tensor Core Utilization | (待生成) | MMA 占比，pipeline utilization |
| 7.8 NCCL / Multi-GPU | (待生成) | all-reduce 带宽，topology |
| 7.9 Benchmark Methodology | (待生成) | warmup，统计显著性，环境控制 |

**习题总数**：40+ 道 | **故障树**：9 个 | **调优 checklist**：9 个

---

## 卷 8：系统设计与面试准备

**目标**：综合运用所有知识，准备系统设计面试

| 章节 | 核心内容 |
|------|---------|
| 8.1 1000 QPS LLM Serving 系统设计 | 架构选型、显存规划、TP/PP 配置、调度策略 |
| 8.2 分布式推理系统设计 | PD 分离、KV routing、故障恢复、扩缩容 |
| 8.3 腾讯 JD 定制问答 | CUDA/CUTLASS/vLLM 深度问题 |
| 8.4 小红书 JD 定制问答 | Mooncake/RBG/dynamic PD 问题 |
| 8.5 美团 JD 定制问答 | LongCat/MoE/fusion/N-gram 问题 |
| 8.6 模型基础兜底 | Transformer 架构、训练细节、scaling law |
| 8.7 Coding Mock | CUDA kernel、系统设计、算法题 |
| 8.8 综合 Mock 评分 | 评分标准、补课清单 |

**来源**：综合卷 1-7 的习题和面试题，结合 week3-8-overview.md 的 JD 定制计划

---

## 材料统计

| 领域 | 已完成文件 | 计划文件 | 完成率 |
|------|-----------|---------|--------|
| CUDA 算子 | 14 | 14 | 100% |
| 推理框架 | 3 | 19 | 16% |
| 后训练 | 4 | 16 | 25% |
| GPU Profiling | 2 | 18 | 11% |
| **总计** | **23** | **67** | **34%** |

### 未完成原因

Sonnet 模型 API 端点 (`api.xylt-space.top`) 持续返回 502 错误，所有 subagent 调用失败。CUDA 算子部分由主 session (Opus) 直接生成完成。其余领域的核心文件已生成，剩余文件待 API 恢复后补充。

### 优先补充顺序

1. **推理框架**：06-continuous-batching, 09-speculative-decoding, 15-scheduler
2. **后训练**：02-lora-qlora, 11-rollout-engine, 13-reward-hacking
3. **GPU Profiling**：02-nsight-compute, 07-memory-bandwidth, 17-benchmark-methodology
