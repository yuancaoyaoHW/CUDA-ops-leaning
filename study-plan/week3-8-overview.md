# 第 3-8 周概要计划

本文件与 [inference-acceleration-plan.md](./inference-acceleration-plan.md) 对齐。第 1-2 周补 `row_softmax`、RMSNorm 和 CUDA extension；第 3 周开始进入 GEMM/CUTLASS，并逐步过渡到 attention、KV cache 和推理框架。

## 第 3 周：GEMM / cuBLAS / CUTLASS

### 主题

- PyTorch/cuBLAS GEMM benchmark
- GEMM shape sweep 和 TFLOPS 计算
- CUTLASS profiler 或受限环境说明
- roofline 和 epilogue/fusion 概念

### 每日安排

| Day | 上午（3h） | 下午（2h） | 晚上（1.5h） |
|-----|-----------|-----------|-------------|
| 15 | GEMM benchmark harness | PyTorch/cuBLAS baseline | 记录 shape/dtype |
| 16 | GEMM shape sweep | TFLOPS 表 | 解释小/大 shape 差异 |
| 17 | CUTLASS profiler 入门 | profiler run 或受限说明 | CUTLASS 概念笔记 |
| 18 | arithmetic intensity | roofline 表 | bottleneck 判断 |
| 19 | epilogue/fusion 概念 | bias/activation/fusion note | 面试问答整理 |
| 20 | GEMM tradeoff 复盘 | cuBLAS/CUTLASS/Triton 对比 | 闭卷口述 |
| 21 | 周复习 + 周检 | GEMM/CUTLASS/Nsight mock | 下周计划 |

### 关键交付物

- GEMM benchmark 表
- CUTLASS profiler 笔记或受限说明
- roofline 表

## 第 4 周：Attention Kernel

### 主题

- PyTorch SDPA reference
- online softmax
- FlashAttention forward toy
- prefill vs decode attention

### 每日安排

| Day | 上午（3h） | 下午（2h） | 晚上（1.5h） |
|-----|-----------|-----------|-------------|
| 22 | Attention 数学 | SDPA reference | shape set |
| 23 | online softmax 推导 | 数值稳定证明 | 闭卷推导 |
| 24 | FlashAttention skeleton | block tiling | address mapping |
| 25 | correctness tests | aligned/non-aligned shapes | tolerance note |
| 26 | benchmark | SDPA 对比 | 慢在哪里 |
| 27 | decode attention | prefill vs decode | decode bottleneck note |
| 28 | 周复习 + 阶段检 | attention kernel mock | 下周计划 |

### 关键交付物

- FlashAttention forward toy 或 decode attention toy
- SDPA 对比 benchmark
- attention bottleneck note

## 第 5 周：KV Cache / PagedAttention / Scheduler Toy / 排障

### 主题

- contiguous KV cache
- paged KV block table
- PagedAttention 数据流
- continuous batching 和 chunked prefill
- TTFT/TPOT/ITL/P99/KV usage/prefix hit 排障指标
- Speculative Decoding 和 N-gram cache

### 每日安排

| Day | 上午（3h） | 下午（2h） | 晚上（1.5h） |
|-----|-----------|-----------|-------------|
| 29 | KV cache layout | contiguous KV toy | memory footprint |
| 30 | block table | logical-to-physical mapping | page size tradeoff |
| 31 | PagedAttention 数据流 | access diagram | cache miss/fragmentation |
| 32 | continuous batching toy | prefill/decode queue | scheduling policy |
| 33 | chunked prefill + KV 压缩 | workload simulation | 排障 playbook |
| 34 | speculative decoding | draft-verify toy | N-gram cache |
| 35 | 周复习 + 周检 | KV/scheduler/spec decode mock | producer-consumer drill |

### 关键交付物

- paged KV cache toy
- continuous batching toy scheduler
- latency/throughput workload model 和排障 playbook
- speculative decoding + N-gram cache note

## 第 6 周：vLLM / SGLang / TensorRT-LLM

### 主题

- vLLM engine/scheduler/block manager
- SGLang RadixAttention/prefix caching、Mooncake/external KVCache、RBG、KV Router
- TensorRT-LLM inflight batching、paged KV、plugin/kernel 思路
- 美团 LongCat 相关：MoE routing、TopK fusion、N-gram cache、AllReduce/RMSNorm fusion
- TP/PP/PD/EPD 分离和 dynamic PD

### 每日安排

| Day | 上午（3h） | 下午（2h） | 晚上（1.5h） |
|-----|-----------|-----------|-------------|
| 36 | vLLM architecture | engine/scheduler 图 | 5 分钟串讲 |
| 37 | vLLM scheduler | scheduling decision flow | preemption |
| 38 | vLLM KV/block manager | block lifecycle | PagedAttention 复盘 |
| 39 | SGLang architecture | SGLang vs vLLM | framework 对比表 |
| 40 | RadixAttention | Mooncake/RBG/external KV | 小红书平台题 |
| 41 | TensorRT-LLM | LongCat MoE/fusion/N-gram | 美团平台题 |
| 42 | TP/PP/PD/EPD | KV Router/dynamic PD | 分布式架构图 |
| 43 | 周复习 + 阶段检 | 推理框架 mock | Mooncake/RBG/LongCat drill |

### 关键交付物

- vLLM scheduler/KV block manager 图
- SGLang + Mooncake/RBG/external KV 对比文档
- TensorRT-LLM + LongCat MoE/fusion/N-gram 对比表
- TP/PP/PD/EPD + KV routing 架构图

## 第 7 周：量化推理加速

### 主题

- INT4 weight-only quant
- pack/dequant kernel
- weight-only GEMV
- GPTQ/AWQ/FP8 tradeoff
- bit packing 和 quant math coding

### 每日安排

| Day | 上午（3h） | 下午（2h） | 晚上（1.5h） |
|-----|-----------|-----------|-------------|
| 44 | INT4 weight-only | quant math note | scale/zero-point |
| 45 | INT4 pack/dequant | kernel tests | mask/dtype |
| 46 | dequant benchmark | GB/s + overhead | speedup 分析 |
| 47 | weight-only GEMV | toy 或设计说明 | decode 场景 |
| 48 | GPTQ/AWQ/FP8 | tradeoff 文档 | 面试问答 |
| 49 | 周复习 + 阶段检 | quant + bitops drill | LongCat fusion/cache mock |

### 关键交付物

- INT4 dequant 或 weight-only GEMV benchmark
- GPTQ/AWQ/FP8 对比文档
- quant bit packing drill

## 第 8 周：项目包装 + JD 定制 Mock

### 主题

- benchmark 图表
- 腾讯/小红书/美团 JD 定制准备
- LLM serving 系统设计
- 分布式推理设计
- 模型基础兜底和 coding mock
- 最终 mock

### 每日安排

| Day | 上午 | 下午 | 晚上 |
|-----|------|------|------|
| 50 | benchmark 图表 | README 更新 | baseline/metric/badcase 表 |
| 51 | 腾讯 JD 问答 | CUDA/CUTLASS/vLLM mock | 模型基础 |
| 52 | 小红书 JD 问答 | Mooncake/RBG/dynamic PD mock | 模型基础 |
| 53 | 美团 JD 问答 | LongCat/MoE/fusion mock | 模型基础 |
| 54 | 1000 QPS serving design | p50/p99/throughput | 复盘 |
| 55 | TP/PP/PD/KV routing design | 故障与扩缩容 | 复盘 |
| 56 | coding + 系统设计 mock | 终版评分 | 投递 next action |

### 关键交付物

- README benchmark 图
- 三家公司定制问答
- 2 个系统设计稿
- coding、模型基础、项目深挖最终 mock 评分和补课清单
