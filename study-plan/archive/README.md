# 归档说明

`week1-old-draft/` 和 `week2-old-draft/` 是 8 周计划的早期草稿，主题与当前 [inference-acceleration-plan.md](../inference-acceleration-plan.md) 已经不一致：

- 旧 week1 偏 parallel reduction / tiled GEMM / fused RMSNorm 的 CUDA 编程练习
- 旧 week2 偏 FlashAttention math / vLLM scheduler / speculative decoding / quantization

新计划以 LLM Inference Performance Lab 项目为主线，按算子闭环 → GEMM/CUTLASS → Attention → KV/PagedAttention → 框架 → 量化 → 包装的顺序推进。

旧文件保留是因为部分内容（speculative decoding 推导、FlashAttention 数学、parallel reduction 推导、CUDA stream/CUDA graph 知识点）在执行到对应主题（Day 23/24/34）时仍可作为参考资料。**不要再按旧文件的日程顺序执行**。
