# 8 周大模型推理框架/加速学习计划

目标：以腾讯混元大模型推理加速、小红书大模型推理框架研发、美团大模型推理岗位为第一目标，补强 GPU 加速库、算子开发、推理框架和性能分析能力。

## 主线定位

本计划不再按泛 AI Infra 覆盖推进，而是围绕一个可展示的学习项目组织：

> LLM Inference Performance Lab：从 Triton/CUDA 算子到 KV cache、PagedAttention、continuous batching、量化推理和 Nsight 性能分析的推理加速实验室。

## 能力权重

| 方向 | 权重 | 目标 |
|------|------|------|
| GPU 加速库/算子 | 45% | Triton、CUDA extension、cuBLAS/PyTorch GEMM、CUTLASS、FlashAttention/Decode Attention、INT4 dequant |
| 推理框架/Serving | 30% | vLLM/SGLang/TensorRT-LLM 的 scheduler、KV cache、PagedAttention、continuous batching、PD 分离 |
| 性能分析 | 15% | benchmark、Nsight Systems/Compute、latency/throughput、roofline 和瓶颈判断 |
| 量化压缩 | 7% | INT4/FP8、weight-only、GPTQ/AWQ、推理落地 tradeoff |
| 面试包装 | 3% 前期，后期拉高 | README、benchmark 图、源码走读图、STAR、mock interview |

## JD 能力标签

每天在 `progress.yaml` 里记录 `jd_tags`，用于自动分析覆盖度：

- `kernel`: CUDA/Triton/CUTLASS/算子实现
- `framework`: vLLM/SGLang/TensorRT-LLM/ONNX Runtime
- `serving`: KV cache、scheduler、batching、PD 分离、容错
- `perf`: benchmark、Nsight、吞吐/延迟、瓶颈分析
- `quant`: INT4/FP8/GPTQ/AWQ/压缩
- `docs`: 源码走读、系统图、README、技术文档
- `interview`: mock、系统设计、项目 STAR、岗位问答

## 面经补强项

基于腾讯、小红书、美团 AI Infra/推理岗位面经和公开技术材料，计划额外纳入 5 个面试风险项：

| 风险项 | 加入计划的位置 |
|--------|----------------|
| 算法/C++ 基础薄弱 | 每周周检固定做 2 个 drill：TopK、LRU、merge k sorted arrays、最长公共子串、RAII、smart pointer、cache locality |
| 项目指标讲不清 | 每周 STAR 增加 baseline、metric、improvement、failure、badcase 字段 |
| 线上排障经验不足 | W5 增加 TTFT/TPOT/ITL/P99/GPU util/KV usage/prefix hit playbook |
| 小红书框架特色 | W6 增加 SGLang、Mooncake/external KVCache、RBG、KV Router、dynamic PD |
| 美团系统特色 | W6/W8 增加 LongCat、MoE routing、TopK fusion、N-gram cache、PDL、AllReduce/RMSNorm fusion |

## 8 周安排

完整 56 天计划见 [inference-acceleration-plan.md](./inference-acceleration-plan.md)。
`progress.yaml` 是进度分析的数据源。旧的 `week1/`、`week2/` 已迁移到 [`archive/`](./archive/)，仅作主题参考（speculative decoding 推导、FlashAttention 数学等），不要按旧日程执行。

## 硬件与执行约束

- GPU: RTX 4060 Laptop 8GB / WSL2 / Triton 3.7 / PyTorch 2.12 / CUDA 12.6
- Day 1 必须验证 `ncu --set full` 在 WSL2 下的可用性，备选方案见 [inference-acceleration-plan.md](./inference-acceleration-plan.md)
- W5-W6 框架学习以源码阅读 + toy simulation 为主，避免本地跑 7B serving
- Day 21 / Day 28 / Day 49 为 buffer day，用于消化前一周延迟任务
- 每周周检写一段 STAR 素材到 [`../notes/star-weekly.md`](../notes/star-weekly.md)，含 baseline/target/final/badcase
- 每周周检完成 1 道算法题 + 1 道 C++/系统基础题（drill 表见 [inference-acceleration-plan.md](./inference-acceleration-plan.md)）

| 周 | 主线 | 必交付 |
|----|------|--------|
| W1 | Triton 基础算子闭环 | `row_softmax` 完整闭环：reference、implementation、tests、benchmark、profile、note |
| W2 | RMSNorm + CUDA extension | fused RMSNorm + 最小 C++/CUDA extension |
| W3 | GEMM / cuBLAS / CUTLASS | GEMM benchmark、CUTLASS profiler 笔记、roofline 表 |
| W4 | Attention kernel | FlashAttention forward toy 或 decode attention toy |
| W5 | KV cache / PagedAttention / 排障 | paged KV cache toy、scheduler toy、spec decode/N-gram、线上排障 playbook |
| W6 | vLLM/SGLang/TensorRT-LLM | scheduler、Mooncake/RBG/external KV、LongCat/MoE/fusion、PD/EPD 分离文档 |
| W7 | 量化推理加速 | INT4 dequant 或 weight-only GEMV benchmark |
| W8 | 项目包装 + mock | README、benchmark 图、三家公司 JD 问答、系统设计、coding/model basics |

## 算子完成标准

每个算子按 6 项成熟度评分：

| 项 | 要求 |
|----|------|
| Reference | 有 PyTorch reference |
| Implementation | 有最小 Triton/CUDA/CUTLASS 实现 |
| Tests | 覆盖 aligned/non-aligned、dtype、mask、边界和 shape error |
| Benchmark | 有 warmup/sync，内存型报 GB/s，GEMM 报 TFLOPS |
| Profile | 有 Nsight Systems 或 Nsight Compute 结论 |
| Note | 记录 correctness、performance、bottleneck、next experiment |

少于 6 项时不要说“算子完成”。例如 tests 通过但没有 benchmark，只能说 correctness stage complete。

## GPU 加速库覆盖标准

进度系统会跟踪这些能力是否有证据：

| 能力 | 证据 |
|------|------|
| Triton | 至少 3 个 kernel 完整闭环 |
| CUDA extension | 至少 1 个可 pytest 验证的 C++/CUDA op |
| cuBLAS/PyTorch GEMM | GEMM benchmark 表，能解释 shape 和 dtype 对性能的影响 |
| CUTLASS | profiler 或最小 GEMM 实验笔记 |
| Nsight Systems | 至少 1 个 timeline/launch overhead 分析 |
| Nsight Compute | 至少 2 个单 kernel profile 结论 |
| TensorRT-LLM concepts | inflight batching、paged KV、plugin/kernel 思路对比文档 |

## 进度分析

命令：

```bash
python study-plan/progress.py
python study-plan/progress.py week
python study-plan/progress.py history
python study-plan/progress.py analyze
python study-plan/dashboard.py --build
```

`progress.py analyze` 会输出：

- JD 标签覆盖度
- 算子成熟度
- GPU 加速库覆盖度
- 当前风险项
- 下一步建议

## 验收节奏

| 阶段 | 最低通过标准 |
|------|--------------|
| W2 结束 | 2 个 LLM 推理相关算子有 tests + benchmark；至少 1 个有 profile |
| W4 结束 | 能讲清 softmax/RMSNorm/GEMM/attention 的性能瓶颈；有 FlashAttention 或 decode attention toy |
| W6 结束 | 能讲清 vLLM/SGLang/TensorRT-LLM 的 scheduler、KV cache、PagedAttention、Mooncake/RBG、LongCat/MoE、PD/EPD 分离 |
| W8 结束 | README、benchmark 图、系统设计稿、腾讯/小红书/美团定制问答、coding drill 和模型基础兜底齐全 |
