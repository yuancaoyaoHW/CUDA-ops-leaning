# 大模型推理框架/加速 56 天计划

第一目标岗位：腾讯混元大模型推理加速、小红书大模型推理框架研发、美团大模型推理。

核心短板：GPU 加速库和算子能力薄弱。因此前 4 周把算子、GPU 库、benchmark、Nsight 放到主线；第 5-6 周集中推理框架；第 7-8 周做量化、包装和面试。

## 硬件环境与约束

- GPU: NVIDIA GeForce RTX 4060 Laptop (8GB VRAM, SM 89)
- 环境: WSL2 + PyTorch 2.12 + Triton 3.7 + CUDA 12.6
- **Nsight Compute**: WSL2 下硬件计数器支持需验证（Day 1 必做）。若 `ncu --set full` 无法获取完整指标，备选方案为 `triton.testing.do_bench` + `torch.cuda.Event` 做 latency/throughput 分析，profile 部分改为读官方 case study + 分析 kernel launch 参数。
- **显存限制**: W5-W6 框架学习以源码阅读 + toy simulation 为主，使用 synthetic workload 或 ≤1.5B 模型验证概念，不要试图本地跑完整 7B serving。

## Buffer Day 策略

56 天中安排 3 个 buffer day（Day 21、Day 28、Day 49 各为周检+buffer），用于消化前一周的延迟任务。如果没有延迟，buffer day 用于加深当周最薄弱环节或提前预习下周内容。

## 面经调研后的补强项

联网调研腾讯、小红书、美团相关 AI Infra/推理岗位面经和公开技术材料后，当前计划需要额外覆盖 6 类内容：

1. **算法与 C++ 基础**：merge k sorted arrays、最长公共子串、TopK/LRU、线程安全队列、RAII/move/smart pointer，避免只会系统题不会编码。
2. **项目上线追问**：每个项目都要准备 baseline、核心指标、收益、badcase、失败实验、上线约束，面试官常沿着“你具体做了什么、指标怎么量、哪里没做好”追问。
3. **线上推理排障 playbook**：TTFT、TPOT、ITL、P50/P99、GPU util、SM occupancy、HBM bandwidth、KV cache 使用率、prefix hit rate、queue length。
4. **小红书特色框架题**：SGLang、Mooncake/external KVCache、RBG、KV Router、PD/EPD 分离、dynamic PD、滚动升级和请求迁移。
5. **美团特色系统题**：MoE routing、TopK router fusion、N-gram cache、PDL、AllReduce + Residual Add + RMSNorm fusion、Softmax + TopK + Scaling fusion。
6. **模型基础兜底**：Transformer、BERT vs GPT、RoPE/GQA/MQA/MLA、SFT/RLHF/DPO/PPO、LoRA/QLoRA、RAG。目标不是转算法岗，而是避免框架/系统面里基础题失分。

## 每周 STAR 素材积累

每周周检时花 30 分钟写一段 STAR 素材（`notes/star-weekly.md`）：
- **Situation**: 本周在做什么
- **Task**: 遇到了什么具体问题/挑战
- **Action**: 怎么分析和解决的
- **Result**: 结果如何，学到了什么
- **Metric**: baseline、目标指标、最终数值、提升比例或失败原因
- **Badcase**: 哪些 shape/workload/场景没有优化好，下一步怎么验证

W8 只需要组装这些素材，不需要回忆。

## 每周算法/C++ drill

每周周检固定加入 2 道编码/基础题，控制在 60-90 分钟，不抢主线：

| 周 | 算法题 | C++/系统基础 |
|----|--------|--------------|
| W1 | TopK / heap | RAII、move semantics |
| W2 | LRU cache / thread-safe queue | smart pointer、copy/move control |
| W3 | merge k sorted arrays | memory layout、cache locality |
| W4 | longest common substring / DP | CUDA stream、event、CUDA graph 基础 |
| W5 | producer-consumer / BFS/DFS | queueing metrics、backpressure |
| W6 | 线上排障 case 复盘 | networking/RDMA/collective basics |
| W7 | quant math coding / bit operations | packed int4 layout |
| W8 | mock coding | 项目深挖问答 |

## 56 天安排

| Day | 主题 | 产物 | JD 标签 |
|-----|------|------|---------|
| 1 | 仓库校准 + Nsight Compute WSL2 验证 | 算子成熟度审计表 + ncu 可用性结论 | `perf`, `docs` |
| 2 | `row_softmax` PyTorch reference + Triton skeleton | reference + minimal implementation | `kernel` |
| 3 | `row_softmax` correctness tests | aligned/non-aligned/dtype/mask tests | `kernel` |
| 4 | `row_softmax` benchmark | PyTorch 对比 + GB/s | `kernel`, `perf` |
| 5 | `row_softmax` Nsight Compute | 单 kernel profile + bottleneck note | `kernel`, `perf` |
| 6 | softmax 闭卷复现 + note | addressing/mask/numerical stability note | `kernel`, `docs` |
| 7 | Week 1 周检 + STAR 素材 | 手写 softmax + TopK/RAII drill + STAR metrics | `interview` |
| 8 | RMSNorm reference + Triton 实现 | RMSNorm kernel + PyTorch reference | `kernel` |
| 9 | fused add RMSNorm | residual add + RMSNorm fusion | `kernel` |
| 10 | RMSNorm tests | dtype、hidden size、shape error tests | `kernel` |
| 11 | RMSNorm benchmark | unfused PyTorch vs fused Triton | `kernel`, `perf` |
| 12 | 最小 PyTorch C++/CUDA extension | one CUDA op demo | `kernel` |
| 13 | CUDA extension pytest + note | build/run/verify note | `kernel`, `docs` |
| 14 | Week 2 阶段检 + STAR 素材 | 算子闭环 mock + LRU/smart pointer drill + STAR metrics | `interview` |
| 15 | GEMM baseline | PyTorch/cuBLAS matmul benchmark harness | `kernel`, `perf` |
| 16 | GEMM shape sweep | M/N/K + dtype TFLOPS 表 | `kernel`, `perf` |
| 17 | CUTLASS profiler 入门 | profiler run 或受限说明 + note | `kernel`, `perf` |
| 18 | GEMM roofline | arithmetic intensity + roofline 表 | `perf`, `docs` |
| 19 | epilogue/fusion 概念 | bias/activation/fusion note | `kernel`, `docs` |
| 20 | GEMM 面试复盘 | cuBLAS/CUTLASS/Triton tradeoff | `interview`, `docs` |
| 21 | Week 3 周检 + buffer + STAR | GEMM/CUTLASS/Nsight mock + merge-k/C++ cache drill + STAR metrics | `interview` |
| 22 | Attention 数学与 PyTorch SDPA | SDPA reference + shape set | `kernel`, `docs` |
| 23 | online softmax 推导 | numerical stable online update note | `kernel`, `docs` |
| 24 | FlashAttention forward toy skeleton | block tiling skeleton | `kernel` |
| 25 | FlashAttention correctness | small shape correctness tests | `kernel` |
| 26 | FlashAttention benchmark | SDPA 对比，说明慢在哪里 | `kernel`, `perf` |
| 27 | decode attention / prefill vs decode | decode attention 特性文档 | `serving`, `perf` |
| 28 | Week 4 阶段检 + buffer + STAR | attention kernel mock + DP/CUDA stream drill + STAR metrics | `interview` |
| 29 | KV cache layout + GQA/MQA | contiguous KV cache toy + GQA/MQA 对比 note | `serving` |
| 30 | paged KV block table | logical-to-physical block mapping demo | `serving` |
| 31 | PagedAttention 数据流 | block table + attention access diagram | `serving`, `docs` |
| 32 | continuous batching toy scheduler | prefill/decode queue simulation | `serving` |
| 33 | chunked prefill + KV cache 压缩 + 排障指标 | TTFT/TPOT/ITL/P99/KV usage/prefix hit playbook | `serving`, `perf` |
| 34 | Speculative Decoding + N-gram cache | draft-verify toy + acceptance rate + N-gram cache note | `serving`, `perf` |
| 35 | Week 5 周检 + STAR | KV/scheduler/spec decode mock + producer-consumer drill + STAR metrics | `interview` |
| 36 | vLLM architecture | engine/scheduler/block manager 图 | `framework`, `docs` |
| 37 | vLLM scheduler 深挖 | scheduling decision flow | `framework`, `serving` |
| 38 | vLLM KV/block manager | block lifecycle + preemption note | `framework`, `serving` |
| 39 | SGLang architecture | SGLang vs vLLM overview | `framework`, `docs` |
| 40 | RadixAttention + Mooncake/RBG/external KVCache | 小红书 external KV/KV Router/RBG 对比文档 | `framework`, `serving` |
| 41 | TensorRT-LLM + Meituan LongCat MoE/fusion | inflight batching + MoE/TopK/fusion/N-gram note | `framework`, `docs` |
| 42 | 分布式推理基础 + dynamic PD | TP/PP/PD/EPD/KV routing 架构图 | `serving`, `framework` |
| 43 | Week 6 阶段检 + STAR | 框架 mock + Mooncake/RBG/LongCat drill + STAR metrics | `interview` |
| 44 | INT4 weight-only quant 基础 | quantization math note | `quant`, `docs` |
| 45 | INT4 pack/dequant Triton | pack/dequant kernel + tests | `quant`, `kernel` |
| 46 | INT4 dequant benchmark | GB/s + speedup/overhead 分析 | `quant`, `perf` |
| 47 | weight-only GEMV | GEMV toy 或设计说明 | `quant`, `kernel` |
| 48 | GPTQ vs AWQ vs FP8 | 推理落地 tradeoff 文档 | `quant`, `docs` |
| 49 | Week 7 阶段检 + buffer + STAR | quant + bitops drill + LongCat fusion/cache mock + STAR metrics | `interview` |
| 50 | 项目 README + benchmark 图表 | baseline/metric/improvement/failure/badcase 表 + 图表 | `docs`, `perf` |
| 51 | 腾讯 JD 定制准备 | CUDA/CUTLASS/TensorRT-LLM/vLLM/性能排障问答 | `interview` |
| 52 | 小红书 JD 定制准备 | SGLang/Mooncake/RBG/KV Router/dynamic PD 问答 | `interview` |
| 53 | 美团 JD 定制准备 | LongCat/MoE/N-gram/fusion/压缩部署问答 | `interview` |
| 54 | LLM serving 系统设计 | 1000 QPS serving design | `serving`, `interview` |
| 55 | 分布式推理设计深化 | TP/PP/PD/KV routing + failure recovery design | `serving`, `interview` |
| 56 | 最终 mock + 投递准备 | coding + 系统设计 + 项目深挖终版评分 | `interview`, `docs` |

## 相比旧版的主要变更

1. **Day 1 加入 Nsight Compute WSL2 验证**——避免 Day 5 才发现工具不可用
2. **Day 21/28/49 改为周检+buffer**——消化延迟任务，降低连锁风险
3. **Day 34 加入 Speculative Decoding**——腾讯/小红书面试高频话题
4. **Day 29 加入 GQA/MQA/KV 压缩**——长上下文场景必备知识
5. **Day 41 加入 MoE 推理调度**——Expert Parallelism 是当前热点
6. **Day 42 加入分布式推理基础**——TP/PP/PD 分离提前到 W6（原 Day 55 太晚）
7. **Day 55 改为分布式推理设计深化**——在 Day 42 基础上做系统设计级别的展开
8. **Day 50 合并 README + benchmark 图表**——原来分两天，压缩为一天释放空间
9. **每周周检加入 STAR 素材积累**——面试叙事从第一周就开始构建
10. **W6 从 7 天压缩为 8 天（含周检）**——加入分布式推理和 MoE，但通过合并 W8 内容保持总量 56 天
11. **每周加入算法/C++ drill**——覆盖腾讯/小红书/美团面经里的 coding 与基础题
12. **W5 加入线上推理排障指标**——把 TTFT/TPOT/ITL/P99/KV usage/prefix hit 纳入项目叙事
13. **W6 加入 Mooncake/RBG/external KV 和 LongCat/fusion/N-gram**——对齐小红书、美团公开技术栈
14. **W8 加入模型基础兜底**——Transformer、RoPE/GQA/MQA、LoRA/QLoRA、SFT/RLHF/DPO/RAG

## 每日记录模板

```yaml
status: not_started
daily_check: 0
jd_tags: [kernel, perf]
tasks:
  morning: false
  afternoon: false
  evening: false
artifacts:
  implementation: false
  tests: false
  benchmark: false
  profile: false
  note: false
verification: ''
weaknesses: ''
next_fix: ''
notes: ''
```

## 周检题型

- Week 1: `tl.program_id`、offset、mask、softmax 数值稳定、launch-bound vs memory-bound。
- Week 2: RMSNorm fusion、CUDA extension build path、PyTorch reference、dtype tolerance。
- Week 3: GEMM arithmetic intensity、cuBLAS/CUTLASS/Triton tradeoff、TFLOPS 计算。
- Week 4: FlashAttention online softmax、tiling、prefill vs decode attention。
- Week 5: KV cache layout、PagedAttention block table、continuous batching、chunked prefill、speculative decoding、N-gram cache、线上排障指标。
- Week 6: vLLM/SGLang/TensorRT-LLM scheduler、Mooncake/RBG/external KV、PD/EPD 分离、MoE EP、LongCat fusion、TP/PP 通信。
- Week 7: INT4/FP8、weight-only、GPTQ/AWQ、量化 kernel overhead、int4 bit packing。
- Week 8: 三家公司 JD 定制问答、系统设计、项目 STAR、coding、模型基础兜底、性能排障。
