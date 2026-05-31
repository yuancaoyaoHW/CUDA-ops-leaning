# 学习资料索引

本目录包含 56 天学习计划所需的全部参考资料。运行 `bash resources/download_all.sh` 一键下载。

## 目录结构

```
resources/
├── papers/          # 论文 PDF
├── repos/           # GitHub 仓库（git clone）
├── tutorials/       # 官方教程（本地 HTML/MD）
├── manuals/         # 手册和文档
├── download_all.sh  # 一键下载脚本
└── README.md        # 本文件
```

## 论文清单（按周排列）

### Week 1-2: 算子基础
| # | 论文 | 用途 |
|---|------|------|
| 1 | Online normalizer calculation for softmax (Milakov & Gimelshein, 2018) | row_softmax 数值稳定性 |
| 2 | ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (Rajbhandari et al., 2020) | fused RMSNorm 背景 |

### Week 3-4: GEMM / Attention
| # | 论文 | 用途 |
|---|------|------|
| 3 | FlashAttention: Fast and Memory-Efficient Exact Attention (Dao et al., 2022) | attention kernel 核心 |
| 4 | FlashAttention-2: Faster Attention with Better Parallelism (Dao, 2023) | v2 改进 |
| 5 | FlashDecoding (Dao et al., blog post, 2023) | decode attention 并行化 |

### Week 5: KV Cache / Scheduler / Speculative Decoding
| # | 论文 | 用途 |
|---|------|------|
| 6 | Efficient Memory Management for LLM Serving with PagedAttention (Kwon et al., 2023) | vLLM/PagedAttention |
| 7 | Orca: A Distributed Serving System for Transformer-Based Generative Models (Yu et al., 2022) | continuous batching |
| 8 | Fast Inference from Transformers via Speculative Decoding (Leviathan et al., 2023) | speculative decoding |
| 9 | Accelerating LLM Inference with Staged Speculative Decoding (Spector et al., 2023) | staged spec decode |

### Week 6: 推理框架 / 分布式
| # | 论文 | 用途 |
|---|------|------|
| 10 | SGLang: Efficient Execution of Structured Language Model Programs (Zheng et al., 2023) | SGLang/RadixAttention |
| 11 | Mooncake: A KVCache-centric Disaggregated Architecture (2024) | 小红书外部 KV Cache |
| 12 | Ring Attention with Blockwise Transformers for Near-Infinite Context (Liu et al., 2023) | context parallelism |
| 13 | TensorRT-LLM: A TensorRT Toolbox for Optimized LLM Inference (NVIDIA, 2024) | TRT-LLM 架构 |

### Week 7: 量化
| # | 论文 | 用途 |
|---|------|------|
| 14 | GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers (Frantar et al., 2023) | GPTQ 量化 |
| 15 | AWQ: Activation-aware Weight Quantization (Lin et al., 2023) | AWQ 量化 |
| 16 | FP8 Formats for Deep Learning (Micikevicius et al., 2022) | FP8 格式 |
| 17 | DeepSeek-V3 Technical Report (2024) | MoE + FP8 训练 |

### Week 8: 系统设计 / 模型基础
| # | 论文 | 用途 |
|---|------|------|
| 18 | Attention Is All You Need (Vaswani et al., 2017) | Transformer 基础 |
| 19 | RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021) | RoPE |
| 20 | GQA: Training Generalized Multi-Query Transformer Models (Ainslie et al., 2023) | GQA/MQA |
| 21 | LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021) | LoRA/QLoRA |
| 22 | QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023) | QLoRA |

## GitHub 仓库

| 仓库 | 用途 | 周 |
|------|------|-----|
| openai/triton | Triton 编译器源码 + tutorials | W1-W4 |
| NVIDIA/cutlass | CUTLASS GEMM 库 | W3 |
| leimao/CUDA-Reduction | CUDA reduction 优化参考 | W1 |
| leimao/CUDA-GEMM-Optimization | GEMM 优化 V00-V07 | W3 |
| vllm-project/vllm | vLLM 推理框架源码 | W5-W6 |
| sgl-project/sglang | SGLang 框架源码 | W6 |
| NVIDIA/TensorRT-LLM | TensorRT-LLM 源码 | W6 |
| Dao-AILab/flash-attention | FlashAttention 官方实现 | W4 |
| IST-DASLab/gptq | GPTQ 官方实现 | W7 |
| mit-han-lab/llm-awq | AWQ 官方实现 | W7 |
| TimDettmers/bitsandbytes | QLoRA/量化工具 | W7 |

## 官方教程 / 手册

| 资料 | 用途 | 周 |
|------|------|-----|
| Triton Tutorials (fused softmax, matmul, flash attention, layer norm) | 算子实现参考 | W1-W4 |
| NVIDIA CUDA C Programming Guide | CUDA 编程基础 | W1-W3 |
| NVIDIA Nsight Compute Documentation | kernel profiling | W1-W4 |
| NVIDIA Nsight Systems Documentation | system profiling | W1-W4 |
| CUTLASS Documentation | GEMM 设计 | W3 |
| PyTorch CUDA Extension Tutorial | C++/CUDA extension | W2 |
| vLLM Documentation | 框架使用 | W5-W6 |
| SGLang Documentation | 框架使用 | W6 |

## 博客 / 技术文章

| 文章 | 作者 | 用途 |
|------|------|------|
| The FlashAttention CUDA Kernel Line by Line | Stephen Diehl | FlashAttention 实现细节 |
| Inside vLLM: Anatomy of a High-Throughput LLM Inference System | Aleksa Gordić | vLLM 架构 |
| Flash-Decoding for long-context LLM inference | Tri Dao et al. | FlashDecoding |
| How continuous batching enables 23x throughput in LLM inference | Anyscale | continuous batching |
