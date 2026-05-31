# 系统设计材料参考链接索引

## 学习目标

这份索引用来补齐当前系统设计与面试材料中出现的项目、论文、技术报告和博客链接。读完后应能：

1. 快速定位每个术语背后的一手资料。
2. 区分项目文档、论文、技术报告、厂商博客和通用概念。
3. 按面试主题选择优先阅读材料，而不是在搜索结果里反复跳转。
4. 用链接回查文档中的关键说法，例如 PagedAttention、Mooncake、FlashAttention、LongCat、RoPE scaling、N-gram speculation。

## 前置知识

- 已读或准备阅读本目录 01-08 的系统设计和面试材料。
- 了解 LLM 推理中的 prefill、decode、KV cache、batching、GPU kernel、MoE、long context 等基础词汇。
- 能区分“概念来源论文”和“工程项目文档”：前者解释为什么，后者解释线上怎么用。

## 收录范围

本索引覆盖当前 01-08 文档中明确出现或直接依赖的对象。优先采用官方文档、项目主页、GitHub、arXiv、OpenReview、ACL Anthology、NVIDIA/Redis/Prometheus/Meituan/vLLM 官方博客等一手来源。

没有单独公认来源的内部化表达会单独说明。例如本文档中的 `RBG` 是按 request-based grouping 讲解的一种路由分组思路，不是当前材料可确认的独立公开论文名。

## 项目与框架

| 名称 | 出现场景 | 推荐链接 | 备注 |
|---|---|---|---|
| vLLM | PagedAttention、continuous batching、prefix caching、OpenAI-compatible server、Scheduler/Worker/ModelRunner | [官方文档](https://docs.vllm.ai/), [OpenAI-compatible server](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/), [PagedAttention 论文](https://arxiv.org/abs/2309.06180) | 03 腾讯问答中的源码组件和 04 框架选型都围绕 vLLM 展开。 |
| SGLang | 小红书框架选型、PD 分离、prefix/cache 生态 | [官网](https://www.sglang.io/), [GitHub](https://github.com/sgl-project/sglang) | 适合作为 vLLM 之外的高吞吐 serving 框架对照。 |
| NVIDIA TensorRT-LLM | NVIDIA GPU 上的推理框架选型 | [官方文档](https://docs.nvidia.com/tensorrt-llm/), [开发者页](https://developer.nvidia.com/tensorrt-llm) | 用于回答 vLLM vs TensorRT-LLM 的工程 tradeoff。 |
| Mooncake | PD 分离、KV-cache-centric disaggregated serving、KV transfer/store | [论文](https://arxiv.org/abs/2407.00079), [GitHub](https://github.com/kvcache-ai/Mooncake), [文档站](https://kvcache-ai.github.io/Mooncake/) | 02 和 04 的核心分布式推理案例。 |
| Mooncake Store with vLLM | 分布式 KV cache pool、agentic workload、PD + distributed KV store | 见“博客与工程文章”中的 vLLM x Mooncake 条目 | 当前材料中 Mooncake/RBG/KV routing 的延伸阅读。 |
| CUTLASS | 腾讯 CUDA/CUTLASS 问答、GEMM tile、epilogue、自定义 kernel | [CUTLASS 3.x GEMM API](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html), [CUTLASS 3.x NVIDIA 技术博客](https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/) | 面试中重点看 mainloop、collective epilogue、tile 层次。 |
| CUDA | CUDA kernel、grid/block/thread、shared memory、warp、stream | [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) | 07 coding mock 的基础参考。 |
| cuBLAS | GEMM baseline、性能对照 | [cuBLAS 文档](https://docs.nvidia.com/cuda/cublas/index.html) | 手写 GEMM 或 CUTLASS kernel 的常见 baseline。 |
| NCCL | TP/PP/EP 通信、NCCL timeout、all-reduce/all-to-all | [NCCL 文档](https://docs.nvidia.com/deeplearning/nccl/) | 01-04 中多 GPU 通信和故障排查都会用到。 |
| PyTorch custom ops | PyTorch reference、C++/CUDA extension、operator test | [Custom C++ and CUDA Operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html), [torch.cuda 文档](https://docs.pytorch.org/docs/stable/cuda.html) | 07/08 的 coding 与验证题可用作实现路径参考。 |
| Triton | Triton kernel、autotune、Triton/CUDA 对比 | [Triton 文档](https://triton-lang.org/main/index.html), [OpenAI Triton 发布博客](https://openai.com/index/triton/) | 美团 fusion 题和 mock 评分中的框架能力参考。 |
| Tensor Core / WMMA / MMA | 手写 GEMM、CUTLASS、Tensor Core 编程、warp-level matrix op | [CUDA C++ Programming Guide: Warp Matrix Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-matrix-functions), [Programming Tensor Cores in CUDA 9](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/) | 03/05/07 中讲 GEMM tiling、MMA 指令和 Tensor Core 利用率时需要直接引用。 |
| Prometheus / Alertmanager | metrics、alerting、SLA dashboard | [Prometheus alerting overview](https://prometheus.io/docs/alerting/latest/overview/), [Alertmanager](https://prometheus.io/docs/alerting/latest/alertmanager/) | 01 的监控告警方案。 |
| Redis | 分布式 rate limiter、token bucket、Lua 原子扣减 | [Redis rate limiter docs](https://redis.io/docs/latest/develop/use-cases/rate-limiter/), [Token bucket with Redis](https://redis.io/docs/latest/develop/use-cases/rate-limiter/nodejs/) | 07/08 的 rate limiter 题。 |
| Kubernetes Autoscaling | autoscaling、HPA/VPA、replica 扩缩容 | [Autoscaling workloads](https://kubernetes.io/docs/concepts/workloads/autoscaling), [Autoscaling API](https://kubernetes.io/docs/reference/kubernetes-api/autoscaling/) | 当前材料没有绑定 Kubernetes，但 autoscaling 回答可以引用其控制面抽象。 |
| SentencePiece | tokenizer、BPE/Unigram、raw text tokenization | [论文](https://arxiv.org/abs/1808.06226), [GitHub](https://github.com/google/sentencepiece) | 06 模型基础的 tokenization 题。 |
| Meta Llama / LLaMA | 70B 模型规模、GQA/KV cache 估算 | [Llama 2 论文](https://arxiv.org/abs/2307.09288), [Llama 2 70B model card](https://huggingface.co/meta-llama/Llama-2-70b) | 当前材料使用 LLaMA-70B 类配置做容量估算。 |
| LongCat / LongCat-Flash | 美团 JD、LongCat long context、MoE、N-gram 相关讨论 | [LongCat-Flash 技术报告](https://arxiv.org/abs/2509.01322), [Hugging Face LongCatFlash 文档](https://huggingface.co/docs/transformers/model_doc/longcat_flash), [LongCat 组织页](https://huggingface.co/meituan-longcat) | 05 美团材料的主要业务背景。 |
| LoRA | adapter、多租户、cache key、训练-推理一体化 | [LoRA 论文](https://arxiv.org/abs/2106.09685), [Microsoft LoRA repo](https://github.com/microsoft/LoRA) | 材料里主要用于解释 adapter 版本隔离和多租户 cache key。 |

## 论文与技术报告

| 名称 | 出现场景 | 推荐链接 | 阅读重点 |
|---|---|---|---|
| Attention Is All You Need | Transformer 架构、attention、positional encoding | [arXiv](https://arxiv.org/abs/1706.03762) | 06 基础兜底题的根论文。 |
| Efficient Memory Management for Large Language Model Serving with PagedAttention | vLLM、PagedAttention、block table、copy-on-write、swap | [arXiv](https://arxiv.org/abs/2309.06180) | KV cache 分页、block table、prefix/beam 共享。 |
| FlashAttention | FlashAttention、online softmax、IO-aware attention | [arXiv](https://arxiv.org/abs/2205.14135) | 03/05/06 中要讲清楚减少 HBM IO，不是近似 attention。 |
| FlashAttention-3 | H100/Hopper 上的低精度和异步优化延伸 | [arXiv](https://arxiv.org/abs/2407.08608) | 可作为追问材料，不是当前面试答案的必需 baseline。 |
| Fast Inference from Transformers via Speculative Decoding | speculative decoding、draft/target verify、acceptance rate | [PMLR 页面](https://proceedings.mlr.press/v202/leviathan23a.html), [PDF](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf) | 04 speculative decoding 工程题。 |
| N-Gram Trie Speculative Decoding for Faster LLM In-Context Inference | N-gram speculative decoding、trie、training-free draft | [OpenReview PDF](https://openreview.net/pdf?id=MVfpYw1pxX) | 05 的 N-gram speculation 追问可读。 |
| Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving | Mooncake、PD 分离、KV-centric scheduler、分布式 KV cache | [arXiv](https://arxiv.org/abs/2407.00079) | 02/04 的核心系统设计来源。 |
| DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving | PD 分离、TTFT/TPOT SLO、prefill/decode 资源解耦 | [arXiv](https://arxiv.org/abs/2401.09670), [USENIX OSDI PDF](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf) | 02/04 中解释为什么 prefill 和 decode 要拆池。 |
| Splitwise: Efficient generative LLM inference using phase splitting | prefill/decode phase splitting、异构机器池、state transfer | [arXiv](https://arxiv.org/abs/2311.18677), [Microsoft Research Blog](https://www.microsoft.com/en-us/research/blog/splitwise-improves-gpu-usage-by-splitting-llm-inference-phases/) | Dynamic PD 和成本优化的背景材料。 |
| DynaServe: Unified and Elastic Execution for Dynamic Disaggregated LLM Serving | dynamic disaggregation、elastic tandem execution、动态 workload | [arXiv](https://arxiv.org/abs/2504.09285) | 用来解释 04 中 Dynamic PD disaggregation 的“动态”含义。 |
| Ring Attention with Blockwise Transformers for Near-Infinite Context | ring attention、sequence parallel、long context | [arXiv](https://arxiv.org/abs/2310.01889) | 02 long context 中分布式 attention 的依据。 |
| Efficient Streaming Language Models with Attention Sinks | StreamingLLM、sliding window、attention sink/sink token | [arXiv](https://arxiv.org/abs/2309.17453) | 04/06 中 sliding window 和 sink token 的直接来源。 |
| SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills | chunked prefill、decode-maximal batching、prefill/decode 混排 | [arXiv](https://arxiv.org/abs/2308.16369) | 01/02 中 chunked prefill 调度的论文背景。 |
| Training Compute-Optimal Large Language Models | Chinchilla、Scaling Law、compute-optimal | [arXiv](https://arxiv.org/abs/2203.15556) | 06 scaling law 题，注意它是训练 compute-optimal，不是 serving-optimal。 |
| Scaling Laws for Neural Language Models | scaling law、模型规模/数据/计算量的 power-law 关系 | [arXiv](https://arxiv.org/abs/2001.08361) | 06 中 Chinchilla 之前的基础 scaling law 来源。 |
| Emergent Abilities of Large Language Models | emergent abilities、scale threshold、BIG-Bench 现象 | [arXiv](https://arxiv.org/abs/2206.07682), [Are Emergent Abilities a Mirage?](https://arxiv.org/abs/2304.15004) | 面试中建议同时说明“现象”和“度量选择争议”。 |
| RoFormer: Enhanced Transformer with Rotary Position Embedding | RoPE、rotary positional embedding | [arXiv](https://arxiv.org/abs/2104.09864) | 06 RoPE/RoPE scaling 的基础。 |
| Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation | ALiBi、长度外推、linear bias | [arXiv](https://arxiv.org/abs/2108.12409) | 06 ALiBi 对比题。 |
| YaRN: Efficient Context Window Extension of Large Language Models | RoPE scaling、context extension、long context fine-tuning | [arXiv](https://arxiv.org/abs/2309.00071) | 06 long context 追问材料。 |
| Fast Transformer Decoding: One Write-Head is All You Need | MQA、decode KV cache 缩减 | [arXiv](https://arxiv.org/abs/1911.02150) | 01/06 中 GQA/MQA 对 KV cache 的影响。 |
| GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints | GQA、KV heads、MHA 到 GQA 的折中 | [arXiv](https://arxiv.org/abs/2305.13245), [ACL Anthology PDF](https://aclanthology.org/2023.emnlp-main.298.pdf) | 01/06 的 GQA KV cache 估算依据。 |
| Switch Transformers | MoE、routing、load balancing、capacity | [arXiv](https://arxiv.org/abs/2101.03961) | 05/06 MoE 基础和 load balancing loss 的背景。 |
| LongCat-Flash Technical Report | LongCat、Meituan MoE、dynamic activation | [arXiv](https://arxiv.org/abs/2509.01322), [HF paper page](https://huggingface.co/papers/2509.01322) | 05 美团 JD 背景。 |
| SentencePiece | BPE/Unigram tokenizer、raw text tokenization | [arXiv](https://arxiv.org/abs/1808.06226), [ACL Anthology PDF](https://aclanthology.org/D18-2012.pdf) | 06 tokenization 基础。 |
| Neural Machine Translation of Rare Words with Subword Units | BPE/subword units 在 NMT 中的经典来源 | [arXiv](https://arxiv.org/abs/1508.07909), [ACL Anthology PDF](https://aclanthology.org/P16-1162.pdf) | 06 BPE 题的来源。 |
| Layer Normalization | LayerNorm、pre-norm/post-norm、训练稳定性 | [arXiv](https://arxiv.org/abs/1607.06450) | 03/05/06 中 normalization 与 fusion 题的基础。 |
| Root Mean Square Layer Normalization | RMSNorm、去掉 mean-centering、LLM 常用归一化 | [arXiv](https://arxiv.org/abs/1910.07467), [PyTorch RMSNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html) | 03/05/06 中 RMSNorm kernel fusion 和模型基础题。 |
| Gaussian Error Linear Units | GELU activation、Transformer FFN activation | [arXiv](https://arxiv.org/abs/1606.08415) | 06 FFN/activation 题。 |
| GLU Variants Improve Transformer | SwiGLU、GEGLU、gated FFN | [arXiv](https://arxiv.org/abs/2002.05202) | 05/06 中 gated FFN、SwiGLU 和 fusion 题。 |
| Decoupled Weight Decay Regularization | AdamW、decoupled weight decay | [arXiv](https://arxiv.org/abs/1711.05101) | 06 AdamW 题。 |
| LoRA: Low-Rank Adaptation of Large Language Models | LoRA adapter、参数高效微调、多租户 serving | [arXiv](https://arxiv.org/abs/2106.09685) | adapter cache key、训练-推理一体化。 |

## 硬件与厂商资料

| 名称 | 出现场景 | 推荐链接 | 备注 |
|---|---|---|---|
| NVIDIA A100 | 01 硬件选型、A100 vs H100/H200 | [NVIDIA A100 产品页](https://www.nvidia.com/en-us/data-center/a100/), [A100 datasheet PDF](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf) | 文档中使用约 2 TB/s HBM 和 600 GB/s NVLink 量级。 |
| NVIDIA H100 | 01 硬件选型、H100 decode bandwidth、NVLink | [NVIDIA H100 产品页](https://www.nvidia.com/en-us/data-center/h100/), [Hopper architecture](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/) | 官方页给出约 3 TB/s HBM 和 900 GB/s GPU-to-GPU NVLink 量级。 |
| NVIDIA H200 | 01 硬件选型、KV cache 容量、long context | [NVIDIA H200 产品页](https://www.nvidia.com/en-us/data-center/h200/) | 官方页给出 141 GB HBM3e 和 4.8 TB/s 带宽，并提供 datasheet 入口。 |
| NVLink / NVSwitch | TP/PP、单节点多 GPU 通信 | [NVIDIA NVLink and NVLink Switch](https://www.nvidia.com/en-us/data-center/nvlink/), [NVSwitch technical overview PDF](https://images.nvidia.com/content/pdf/nvswitch-technical-overview.pdf) | 回答 TP/PP 时强调节点内 NVLink/NVSwitch 和跨节点 RDMA 的差异。 |
| GPUDirect RDMA / InfiniBand / PCIe | 跨节点 KV transfer、PD 分离、GPU-NIC 直连、PCIe staging | [GPUDirect RDMA 文档](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html), [GPUDirect Developer Page](https://developer.nvidia.com/gpudirect), [CUDA Best Practices: Data Transfer](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#data-transfer-between-host-and-device) | 02 中讨论跨节点 prefill/decode 和 KV cache transfer 时需要区分节点内 NVLink 与跨节点 RDMA/IB。 |
| Nsight Compute | 单 kernel 指标、SM/HBM/tensor core/occupancy | [Nsight Compute 文档](https://docs.nvidia.com/nsight-compute/) | 03 profiling 闭环中的 kernel 级工具。 |
| Nsight Systems | timeline、launch gap、CPU/GPU overlap | [Nsight Systems get started](https://developer.nvidia.com/nsight-systems/get-started) | 03 中用于系统 timeline 和调度排查。 |

## 博客与工程文章

| 名称 | 出现场景 | 推荐链接 | 为什么值得读 |
|---|---|---|---|
| vLLM x Mooncake: Serving Agentic Workloads at Scale | Mooncake Store、分布式 KV cache、PD disaggregation | [vLLM Blog](https://vllm.ai/blog/2026-05-06-mooncake-store) | 把 Mooncake 从论文概念落到 vLLM connector、RDMA KV transfer 和 agentic traces。 |
| CUTLASS 3.x abstractions for GEMM kernel design | CUTLASS tile、collective mainloop、epilogue | [NVIDIA Technical Blog](https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/) | 03 腾讯 CUTLASS 追问的工程化解释。 |
| Introducing Triton | Triton 作为 Python-like GPU kernel 语言 | [OpenAI Blog](https://openai.com/index/triton/) | 05/08 中解释 Triton 为什么适合快速写 kernel。 |
| LongCat-Flash-Lite N-gram 模型 | N-gram embedding、推测解码协同、轻量 MoE | [美团技术团队博客](https://tech.meituan.com/2026/02/10/longcat-flash-lite.html), [LongCat-Flash-Lite 页面](https://www.longcatai.org/models/flash-lite/) | 05 的 N-gram speculation 和美团 JD 背景可以结合阅读。 |
| LongCat-Flash-Thinking-2601 技术报告发布 | 美团 LongCat 系列、agentic/generalization 背景 | [美团技术团队博客](https://tech.meituan.com/2026/02/02/longcat-flash-thinking-2601-techreport.html) | 面试准备中可用于了解 LongCat 系列方向，但不是 05 的核心必读。 |
| Redis rate limiter guide | token bucket、fixed/sliding window、Lua 原子性 | [Redis Docs](https://redis.io/docs/latest/develop/use-cases/rate-limiter/) | 07 rate limiter 题的工程背景。 |

## 概念与命名说明

| 术语 | 当前材料中的含义 | 链接建议 |
|---|---|---|
| RBG | 当前材料按 `request-based grouping` 使用，表示按 system prompt、模板、租户、SLA、长度等请求属性分组路由；未把它定义成独立公开系统名。 | 可结合 [Mooncake 论文](https://arxiv.org/abs/2407.00079) 的 KV locality 调度和 [vLLM prefix caching](https://docs.vllm.ai/en/stable/design/prefix_caching/) 理解。 |
| Dynamic PD disaggregation | 当前材料指根据 workload、SLO、GPU 利用率和 KV transfer 成本，动态调整 prefill/decode 的拆分、配比、路由或协作方式；它是一类工程模式，不等同于单一系统名。 | 可结合 [DistServe](https://arxiv.org/abs/2401.09670)、[Splitwise](https://arxiv.org/abs/2311.18677)、[DynaServe](https://arxiv.org/abs/2504.09285) 和 [Mooncake](https://arxiv.org/abs/2407.00079) 理解。 |
| Continuous batching | decode step 之间持续加入/移除请求的 serving 调度思想。 | 可读 [vLLM 论文](https://arxiv.org/abs/2309.06180) 和 [vLLM Optimization and Tuning](https://docs.vllm.ai/en/latest/configuration/optimization/)。 |
| Chunked prefill | 将长 prompt prefill 拆成 token chunk，避免长请求阻塞 decode。 | 可读 [vLLM Optimization and Tuning](https://docs.vllm.ai/en/latest/configuration/optimization/) 和 [SARATHI](https://arxiv.org/abs/2308.16369)。 |
| Prefix caching / context caching | 缓存共享前缀的 KV blocks，跳过重复 prefill。 | [vLLM automatic prefix caching](https://docs.vllm.ai/en/stable/design/prefix_caching/) 是当前材料最直接参考。 |
| Sliding window / attention sink | sliding window 限制活跃 KV 长度，attention sink 保留早期高注意力 token 以稳定流式推理。 | [StreamingLLM](https://arxiv.org/abs/2309.17453) 是当前材料中 sink token 的主要来源。 |
| Spot / preemptible | 低成本但可被回收的计算资源，适合可重试 prefill/offline，不适合无 checkpoint 的高优先级 decode。 | 可结合云厂商文档理解；当前材料未绑定单一云厂商。 |

## 建议阅读顺序

1. 系统设计主线：vLLM/PagedAttention -> DistServe/Splitwise/Mooncake -> vLLM x Mooncake blog -> Redis/Prometheus/Kubernetes。
2. CUDA/Kernel 主线：CUDA guide -> Tensor Core/WMMA -> FlashAttention -> CUTLASS docs/blog -> Nsight Compute/Systems -> PyTorch custom ops/Triton。
3. 模型基础主线：Attention Is All You Need -> LayerNorm/RMSNorm/GELU/SwiGLU -> RoPE/RoFormer -> ALiBi/YaRN/Ring Attention/StreamingLLM -> Scaling Laws/Chinchilla -> SentencePiece/BPE -> Switch Transformer。
4. 公司 JD 主线：腾讯优先 CUTLASS/vLLM/FlashAttention；小红书优先 Mooncake/vLLM/SGLang/speculative decoding；美团优先 LongCat/MoE/N-gram speculation。

## 维护规则

- 新材料中出现新的公开项目、论文、技术博客时，把链接补到本索引。
- 优先使用一手来源；第三方解读只放在“博客与工程文章”且标明用途。
- 如果术语只是本文档自定义的回答框架，不要强行找同名论文；在“概念与命名说明”里解释即可。
