# 面试专项资料：小红书 / 美团 / 腾讯技术栈

本文件整理三家公司公开的技术博客、论文和系统设计资料，用于 W6-W8 面试准备。

---

## 小红书：大模型推理框架研发

### 核心技术栈
- SGLang (RadixAttention, prefix caching)
- Mooncake (external KVCache, disaggregated architecture)
- RBG (Request-Based Grouping)
- KV Router
- PD/EPD 分离 (Prefill-Decode separation)
- Dynamic PD (动态 prefill-decode 调度)

### 必读资料

| 资料 | 链接 | 重点 |
|------|------|------|
| Mooncake 论文 | `papers/11_mooncake_kvcache_disaggregated_2024.pdf` | KVCache 分离架构、transfer engine |
| SGLang 论文 | `papers/10_sglang_zheng_2023.pdf` | RadixAttention、prefix sharing |
| SGLang GitHub | `repos/sglang/` | scheduler、radix_cache 源码 |
| Mooncake GitHub | https://github.com/kvcache-ai/Mooncake | 开源实现 |

### 面试高频问题
1. RadixAttention 和 vLLM PagedAttention 的区别？各自优劣？
2. Mooncake 的 external KVCache 怎么实现？和 local KV cache 的 tradeoff？
3. PD 分离的动机？什么时候 prefill 和 decode 应该分开？
4. Dynamic PD 怎么做？如何决定 prefill/decode 的资源分配？
5. KV Router 的路由策略？prefix hit rate 怎么优化？
6. 滚动升级时如何做请求迁移？KV cache 怎么处理？

### 技术博客/公开分享
- 小红书技术博客: https://tech.xiaohongshu.com/
- "大模型推理加速实践" (搜索小红书技术分享)
- SGLang blog: https://lmsys.org/blog/

---

## 美团：大模型推理

### 核心技术栈
- LongCat (长上下文推理优化)
- MoE routing + TopK router fusion
- N-gram cache (投机解码加速)
- PDL (Prefill-Decode-Length aware scheduling)
- AllReduce + Residual Add + RMSNorm fusion
- Softmax + TopK + Scaling fusion

### 必读资料

| 资料 | 链接 | 重点 |
|------|------|------|
| DeepSeek-V3 报告 | `papers/17_deepseek_v3_2024.pdf` | MoE 设计、FP8 训练 |
| DeepSeek-V2 报告 | `papers/24_deepseek_v2_mla_2024.pdf` | MLA、MoE EP |
| Speculative Decoding | `papers/08_speculative_decoding_leviathan_2023.pdf` | N-gram cache 基础 |

### 面试高频问题
1. MoE 的 Expert Parallelism 怎么做？All-to-All 通信开销怎么优化？
2. TopK router fusion 是什么？为什么要把 softmax + topk + scaling 融合？
3. N-gram cache 怎么加速 speculative decoding？命中率怎么提升？
4. AllReduce + Residual Add + RMSNorm 为什么能融合？融合后减少多少 kernel launch？
5. 长上下文场景下 KV cache 的内存压力怎么解决？
6. PDL scheduling 和普通 continuous batching 的区别？

### 技术博客/公开分享
- 美团技术博客: https://tech.meituan.com/
- "美团大模型推理优化实践" (搜索美团技术分享)
- LongCat 相关: 搜索 "美团 LongCat" 或 "Meituan long context inference"

---

## 腾讯混元：大模型推理加速

### 核心技术栈
- CUDA/Triton/CUTLASS 算子开发
- vLLM/TensorRT-LLM 框架
- Nsight Compute/Systems profiling
- 量化推理 (INT4/FP8)
- 分布式推理 (TP/PP)

### 必读资料

| 资料 | 链接 | 重点 |
|------|------|------|
| FlashAttention 1&2 | `papers/03_*.pdf`, `papers/04_*.pdf` | attention kernel 优化 |
| CUTLASS | `repos/cutlass/` 或项目根目录 `cutlass/` | GEMM 优化 |
| GPTQ/AWQ | `papers/14_*.pdf`, `papers/15_*.pdf` | 量化方案 |
| TensorRT-LLM | `papers/13_*.pdf` | 推理框架 |

### 面试高频问题
1. 写一个 Triton softmax kernel，解释 mask 和数值稳定性
2. GEMM 的 roofline 分析？什么时候 compute-bound，什么时候 memory-bound？
3. cuBLAS vs CUTLASS vs Triton 各自适合什么场景？
4. FlashAttention 的 IO 复杂度推导？为什么比标准 attention 快？
5. INT4 weight-only quantization 的 pack/dequant 怎么实现？
6. Nsight Compute 看哪些指标判断 kernel 瓶颈？

### 技术博客/公开分享
- 腾讯技术博客: https://cloud.tencent.com/developer/
- 混元大模型: https://hunyuan.tencent.com/
- "腾讯混元推理加速" (搜索腾讯技术分享)

---

## 通用面试准备资料

### 线上推理排障 Playbook

| 指标 | 含义 | 正常范围 | 异常排查 |
|------|------|---------|---------|
| TTFT | Time To First Token | < 500ms (短 prompt) | prefill 慢 → 检查 prompt 长度、chunked prefill |
| TPOT | Time Per Output Token | < 50ms | decode 慢 → 检查 batch size、KV cache 压力 |
| ITL | Inter-Token Latency | < 50ms | 同 TPOT，关注 P99 |
| P50/P99 | 延迟分位数 | P99 < 3x P50 | 长尾 → 检查 preemption、GC、调度 |
| GPU Util | GPU 利用率 | > 80% | 低 → batch 太小或 memory-bound |
| SM Occupancy | SM 占用率 | > 50% | 低 → register/shared memory 压力 |
| HBM Bandwidth | 显存带宽利用率 | > 60% peak | 低 → kernel 未充分利用 |
| KV Cache Usage | KV cache 使用率 | < 90% | 高 → 需要 preemption 或扩容 |
| Prefix Hit Rate | 前缀缓存命中率 | > 30% (多轮对话) | 低 → 检查 radix cache 策略 |
| Queue Length | 等待队列长度 | < 10 | 高 → 吞吐不足，需要扩容 |

### 系统设计模板

**题目**: 设计一个支持 1000 QPS 的 LLM serving 系统

**回答框架**:
1. 需求澄清: 模型大小、延迟 SLA、输入/输出长度分布
2. 单机架构: engine + scheduler + KV cache manager
3. 优化手段: continuous batching, chunked prefill, speculative decoding
4. 分布式: TP/PP 切分、PD 分离、负载均衡
5. 容错: 请求重试、KV cache 迁移、graceful degradation
6. 监控: TTFT/TPOT/P99、GPU util、KV usage、queue length

---

## 下载补充资料脚本

以下资料需要手动搜索或访问（无法直接 wget）：

```bash
# 小红书/美团/腾讯技术博客需要浏览器访问
# 建议收藏以下搜索关键词：

# 小红书
# - "小红书 大模型推理 SGLang"
# - "Mooncake KVCache disaggregated"
# - "小红书 PD分离 推理优化"

# 美团
# - "美团 大模型推理 LongCat"
# - "美团 MoE routing fusion"
# - "美团 N-gram cache speculative decoding"

# 腾讯
# - "腾讯混元 推理加速 CUDA"
# - "Tencent Hunyuan inference optimization"
```
