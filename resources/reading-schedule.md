# 阅读计划：按天分配的论文/资料阅读顺序

每天主线学习之外，安排 30-60 分钟阅读对应论文或资料。标注 ★ 为必读，☆ 为选读/参考。

## Week 1: row_softmax + 算子基础

| Day | 阅读内容 | 时长 | 优先级 |
|-----|---------|------|--------|
| 1 | NVIDIA Nsight Compute 文档 (profiling basics) | 30min | ★ |
| 2 | `01_online_softmax_milakov_2018.pdf` §1-3 (online softmax 算法) | 45min | ★ |
| 3 | Triton Tutorial: Fused Softmax (对照实现) | 30min | ★ |
| 4 | `01_online_softmax_milakov_2018.pdf` §4 (实验结果) | 20min | ☆ |
| 5 | CUDA C Programming Guide: Memory Hierarchy (Ch 5) | 45min | ★ |
| 6 | 闭卷复现，不需要新阅读 | - | - |
| 7 | 周检复习 | - | - |

## Week 2: RMSNorm + CUDA Extension

| Day | 阅读内容 | 时长 | 优先级 |
|-----|---------|------|--------|
| 8 | Triton Tutorial: Layer Normalization | 30min | ★ |
| 9 | `02_zero_rajbhandari_2020.pdf` §3 (ZeRO 分片策略，理解 fused norm 动机) | 40min | ☆ |
| 10 | PyTorch CUDA Extension Tutorial (官方文档) | 45min | ★ |
| 11 | CUDA C Programming Guide: Execution Configuration (Ch 3-4) | 30min | ★ |
| 12 | PyTorch cpp_extension 源码 + pybind11 basics | 45min | ★ |
| 13 | CUTLASS Documentation: GEMM 概念预读 | 30min | ☆ |
| 14 | 阶段检复习 | - | - |

## Week 3: GEMM / cuBLAS / CUTLASS

| Day | 阅读内容 | 时长 | 优先级 |
|-----|---------|------|--------|
| 15 | leimao/CUDA-GEMM-Optimization README + V00 | 45min | ★ |
| 16 | leimao/CUDA-GEMM-Optimization V01-V03 (tiling) | 45min | ★ |
| 17 | CUTLASS Documentation: Efficient GEMM Design | 60min | ★ |
| 18 | CUDA C Programming Guide: Roofline Model 概念 | 30min | ★ |
| 19 | CUTLASS: Epilogue 和 Fusion 文档 | 40min | ★ |
| 20 | Triton Tutorial: Matrix Multiplication | 30min | ★ |
| 21 | 周检复习 | - | - |

## Week 4: Attention Kernel

| Day | 阅读内容 | 时长 | 优先级 |
|-----|---------|------|--------|
| 22 | `03_flashattention_dao_2022.pdf` §1-2 (动机 + 算法) | 60min | ★ |
| 23 | `03_flashattention_dao_2022.pdf` §3 (IO 复杂度证明) | 45min | ★ |
| 24 | `04_flashattention2_dao_2023.pdf` (v2 改进) | 45min | ★ |
| 25 | Triton Tutorial: Flash Attention | 45min | ★ |
| 26 | Stephen Diehl "FlashAttention CUDA Kernel Line by Line" | 60min | ☆ |
| 27 | Flash-Decoding blog (Tri Dao) | 30min | ★ |
| 28 | 阶段检复习 + `06_pagedattention_vllm_kwon_2023.pdf` §1-2 预读 | 40min | ★ |

## Week 5: KV Cache / PagedAttention / Scheduler

| Day | 阅读内容 | 时长 | 优先级 |
|-----|---------|------|--------|
| 29 | `06_pagedattention_vllm_kwon_2023.pdf` §3-4 (PagedAttention 设计) | 60min | ★ |
| 29 | `20_gqa_ainslie_2023.pdf` + `23_mqa_shazeer_2019.pdf` (GQA/MQA) | 45min | ★ |
| 30 | `06_pagedattention_vllm_kwon_2023.pdf` §5 (Block Manager) | 30min | ★ |
| 31 | vLLM 源码: `vllm/core/block_manager.py` | 45min | ★ |
| 32 | `07_orca_yu_2022.pdf` (continuous batching) | 45min | ★ |
| 33 | Anyscale blog "How continuous batching enables 23x throughput" | 30min | ☆ |
| 34 | `08_speculative_decoding_leviathan_2023.pdf` | 60min | ★ |
| 34 | `09_staged_speculative_decoding_spector_2023.pdf` §1-3 | 30min | ☆ |
| 35 | 周检复习 | - | - |

## Week 6: vLLM / SGLang / TensorRT-LLM

| Day | 阅读内容 | 时长 | 优先级 |
|-----|---------|------|--------|
| 36 | vLLM 源码: `vllm/engine/llm_engine.py` + scheduler | 60min | ★ |
| 37 | Aleksa Gordić "Inside vLLM" blog | 45min | ★ |
| 38 | vLLM 源码: `csrc/attention/attention_kernels.cu` | 45min | ★ |
| 39 | `10_sglang_zheng_2023.pdf` §1-4 (RadixAttention) | 60min | ★ |
| 40 | `11_mooncake_kvcache_disaggregated_2024.pdf` (Mooncake/external KV) | 60min | ★ |
| 41 | `13_tensorrt_llm_2024.pdf` + `24_deepseek_v2_mla_2024.pdf` §MoE | 60min | ★ |
| 42 | `12_ring_attention_liu_2023.pdf` (Context Parallelism) | 45min | ★ |
| 43 | 阶段检复习 | - | - |

## Week 7: 量化推理加速

| Day | 阅读内容 | 时长 | 优先级 |
|-----|---------|------|--------|
| 44 | `14_gptq_frantar_2023.pdf` §1-3 (GPTQ 算法) | 60min | ★ |
| 45 | `15_awq_lin_2023.pdf` §1-3 (AWQ 算法) | 45min | ★ |
| 46 | `16_fp8_formats_micikevicius_2022.pdf` (FP8 格式) | 30min | ★ |
| 47 | bitsandbytes 源码: int4 pack/dequant kernel | 45min | ☆ |
| 48 | `17_deepseek_v3_2024.pdf` §FP8 训练 + MoE 部分 | 45min | ★ |
| 49 | 阶段检复习 | - | - |

## Week 8: 项目包装 + 模型基础兜底

| Day | 阅读内容 | 时长 | 优先级 |
|-----|---------|------|--------|
| 50 | 整理 benchmark 图表，不需要新阅读 | - | - |
| 51 | `18_attention_is_all_you_need_2017.pdf` 快速复习 | 30min | ★ |
| 51 | `19_roformer_rope_su_2021.pdf` §1-3 (RoPE) | 30min | ★ |
| 52 | `25_instructgpt_rlhf_ouyang_2022.pdf` §2-3 (RLHF 流程) | 45min | ★ |
| 52 | `26_dpo_rafailov_2023.pdf` §1-3 (DPO vs PPO) | 30min | ★ |
| 53 | `21_lora_hu_2021.pdf` + `22_qlora_dettmers_2023.pdf` 快速复习 | 30min | ★ |
| 53 | `28_rag_lewis_2020.pdf` §1-3 (RAG 架构) | 30min | ★ |
| 54-56 | Mock 准备，按需回顾 | - | - |

## 阅读优先级说明

- ★ 必读：面试高频考点，必须能闭卷口述核心思想
- ☆ 选读：加深理解，时间不够可跳过

## 阅读方法建议

1. **第一遍**（20min）：读 Abstract + Introduction + 看图表，理解 what & why
2. **第二遍**（30min）：读核心算法/方法章节，手推关键公式
3. **第三遍**（10min）：读实验结果，记住关键数字（加速比、精度损失）
4. **面试准备**：每篇论文准备 3 句话总结 — 解决什么问题、核心方法、关键结果
