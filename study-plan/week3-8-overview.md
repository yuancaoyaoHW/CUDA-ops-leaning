# 第 3-8 周概要计划

前两周的详细日计划已经写好。第 3-8 周的详细日计划将在第 2 周结束后根据实际进度和阶段检暴露的薄弱点调整。以下是每周的方向和关键任务。

---

## 第 3 周：推理系统深入 + 项目启动

### 主题
- SGLang 源码（RadixAttention / prefix caching）
- PD 分离（Prefill-Decode Disaggregation）
- Attention 变体（GQA, MQA, MLA）
- 启动主项目（Mini Inference Engine 或 开源 PR）

### 每日安排

| Day | 上午（3h） | 下午（2h） | 晚上（1.5h） |
|-----|-----------|-----------|-------------|
| 15 | SGLang RadixAttention 源码 | 写 prefix caching 对比文档 | LeetCode (Trie 变体) |
| 16 | PD 分离原理 + DistServe 论文 | KV Cache 压缩 (FP8 KV) | 4D Parallelism 组合策略 |
| 17 | GQA/MQA 实现 + 为什么省内存 | MLA (Multi-head Latent Attention) | LeetCode (编辑距离) |
| 18 | 项目选型：Mini Engine vs PR | 搭建项目骨架 | 分布式推理：TP inference |
| 19 | 项目开发：continuous batching scheduler | 项目开发：KV cache manager | Nsight profiling 实操 |
| 20 | 项目开发：model loading + forward | 项目开发：sampling + streaming | LeetCode |
| 21 | 周复习 + 周检 | 项目 demo 验证 | 下周规划 |

### 关键交付物
- `docs/prefix-caching-comparison.md`（vLLM vs SGLang）
- `docs/pd-disaggregation.md`
- 项目骨架代码可运行

### 周检题库方向
- RadixAttention vs hash-based prefix caching
- PD 分离的 tradeoff
- GQA 为什么能减少 KV cache 大小
- MLA 的核心 idea
- 4D parallelism 怎么组合

---

## 第 4 周：项目产出 + 开源贡献

### 主题
- 完成主项目的核心功能
- 尝试给 vLLM/SGLang 提 PR
- 量化实操（用 AutoGPTQ/LLM-Compressor 量化模型）
- 分布式推理实操

### 每日安排

| Day | 上午（3h） | 下午（2h） | 晚上（1.5h） |
|-----|-----------|-----------|-------------|
| 22 | 项目：实现 paged attention | 项目：benchmark 脚本 | 量化实操：AWQ 量化 1.5B 模型 |
| 23 | 项目：实现 continuous batching | 项目：对比 vLLM 性能 | 量化实操：对比精度和速度 |
| 24 | 开源 PR：找 good-first-issue | 开源 PR：理解代码 + 写方案 | LeetCode |
| 25 | 开源 PR：实现 + 测试 | 开源 PR：提交 | 分布式：TP 推理实验 |
| 26 | 项目完善：error handling | 项目完善：metrics/monitoring | LeetCode |
| 27 | 项目 README + benchmark 图表 | 项目 demo 录屏 | 论文阅读：DistServe |
| 28 | 周复习 + 周检 + 阶段检 #2 | 阶段检 #2 | 下周规划 |

### 关键交付物
- 可运行的 mini inference engine（或已提交的 PR）
- 量化实验报告（AWQ vs 原始模型的 perplexity + throughput）
- 项目 README 有 benchmark 数据

### 阶段检 #2 目标
- 得分 ≥ 65（比 #1 提升 5+）
- 项目深挖环节能讲清楚自己的实现

---

## 第 5 周：项目完善 + 技术输出

### 主题
- 项目 benchmark 完善（多维度对比）
- 写 2-3 篇技术博客
- Nsight Compute profiling 深入
- 开始系统设计练习

### 每日安排

| Day | 上午（3h） | 下午（2h） | 晚上（1.5h） |
|-----|-----------|-----------|-------------|
| 29 | 项目 benchmark：不同 seq_len/batch | 生成 benchmark 图表 | 博客 1：FlashAttention 原理 |
| 30 | Nsight Compute 深入分析 | 写 roofline 分析报告 | 博客 1 完成 |
| 31 | 项目优化：找到瓶颈并改进 | 更新 benchmark | 博客 2：vLLM 调度原理 |
| 32 | 系统设计练习 #1 | 系统设计练习 #2 | 博客 2 完成 |
| 33 | 系统设计练习 #3 | GitHub repo 整理 | LeetCode |
| 34 | 博客 3：量化技术对比 | 博客 3 完成 | 论文串讲练习 |
| 35 | 周复习 + 周检 | 检查所有交付物 | 下周规划 |

### 关键交付物
- GitHub repo 有完整 README + benchmark 图表 + roofline 分析
- 2-3 篇技术博客（知乎/掘金）
- 3 个系统设计练习的完整文档

### 系统设计练习题
1. 设计支持 1000 QPS 的 LLM serving（考虑 batching、KV cache、负载均衡）
2. 8xA100 部署 70B 模型方案（TP vs PP vs TP+PP）
3. 设计多模型推理平台（路由、资源隔离、弹性扩缩）

---

## 第 6 周：面试冲刺 - 系统设计 + 论文

### 主题
- 系统设计专项训练
- 论文 5 分钟串讲练习
- 八股题系统复习
- 第一次完整 mock interview

### 每日安排

| Day | 上午（3h） | 下午（2h） | 晚上（1.5h） |
|-----|-----------|-----------|-------------|
| 36 | 系统设计：LLM serving 全链路 | 论文串讲：FlashAttention 1&2 | 八股复习：CUDA 基础 |
| 37 | 系统设计：分布式训练集群 | 论文串讲：PagedAttention | 八股复习：分布式 |
| 38 | 系统设计：模型压缩部署 | 论文串讲：Speculative Decoding | 八股复习：量化 |
| 39 | Mock interview #1（找朋友或 AI） | 复盘 + 补课 | LeetCode |
| 40 | 论文串讲：Orca + DistServe | 论文串讲：DeepSeek-V3 | 八股复习：推理系统 |
| 41 | Mock interview #2 | 复盘 + 补课 | 项目故事 STAR 练习 |
| 42 | 周复习 + 周检 + 阶段检 #3 | 阶段检 #3 | 下周规划 |

### 必读论文清单（能 5 分钟讲清楚）
1. FlashAttention 1 & 2
2. PagedAttention (vLLM)
3. Speculative Decoding (Leviathan et al.)
4. GPTQ / AWQ / SmoothQuant
5. Orca (Continuous Batching)
6. DistServe (PD 分离)
7. Megatron-LM (3D Parallelism)
8. DeepSeek-V3 (MoE + FP8)

### 八股题清单
```
CUDA:
- 线程层次 grid/block/warp/thread
- Shared memory bank conflict
- Warp divergence
- Memory coalescing
- Occupancy 和性能关系
- Tensor Core 使用条件
- CUDA stream 和 event

推理系统:
- KV Cache 管理
- Continuous batching
- Chunked prefill
- Prefix caching
- Speculative decoding
- FlashAttention/FlashDecoding

分布式:
- TP/PP/DP/EP/CP
- Ring AllReduce
- NCCL
- ZeRO-1/2/3/FSDP
- 1F1B pipeline

量化:
- GPTQ vs AWQ
- FP8 E4M3 vs E5M2
- Per-tensor vs per-channel vs per-group
- Weight-only vs weight+activation
```

### 阶段检 #3 目标：≥ 70

---

## 第 7 周：Mock Interview 密集 + 项目包装

### 主题
- 每天 1 次 mock interview
- 项目包装（README、demo、数据）
- 简历撰写
- 薄弱点补课

### 每日安排

| Day | 上午（3h） | 下午（2h） | 晚上（1.5h） |
|-----|-----------|-----------|-------------|
| 43 | Mock #3 + 复盘 | 项目 README 最终版 | 简历初稿 |
| 44 | Mock #4 + 复盘 | 补课（根据 mock 暴露的问题） | 简历修改 |
| 45 | Mock #5 + 复盘 | 项目 demo 录屏 | LeetCode 模拟 |
| 46 | Mock #6 + 复盘 | 补课 | 简历定稿 |
| 47 | Mock #7 + 复盘 | 技术博客最终检查 | 投递准备 |
| 48 | 全天系统设计练习 | 全天系统设计练习 | 复盘 |
| 49 | 周检 + 阶段检 #4 | 阶段检 #4 | 投递计划 |

### Mock Interview 重点
- 每次 mock 后记录：哪些问题答得好、哪些卡壳、哪些完全不会
- 卡壳的问题第二天上午补课
- 目标：到 Day 49 时 mock 得分稳定 ≥ 70

### 阶段检 #4 目标：≥ 75

---

## 第 8 周：投递 + 查漏补缺

### 主题
- 开始投递（内推 + 官网）
- 针对性准备（根据 JD 调整重点）
- 保持手感（每天 1 题 LeetCode + 1 个概念复习）

### 每日安排

| Day | 上午 | 下午 | 晚上 |
|-----|------|------|------|
| 50 | 投递 3-5 家 | 针对字节 JD 准备 | LeetCode |
| 51 | Mock（字节风格） | 补课 | 投递 |
| 52 | 针对腾讯 JD 准备 | Mock（腾讯风格） | 投递 |
| 53 | 针对小红书 JD 准备 | Mock（小红书风格） | 投递 |
| 54 | 针对美团 JD 准备 | Mock（美团风格） | 投递 |
| 55 | 全天复习薄弱点 | 全天复习薄弱点 | 休息 |
| 56 | 最终 mock | 复盘总结 | 准备面试 |

### 投递策略
- 优先内推（找人脉、脉脉、牛客）
- 同时官网投递
- 每家公司投递前针对性准备 1-2 小时（读 JD、准备相关项目故事）
- 面试后立即复盘记录

---

## 关键资源汇总

### 必读源码
- vLLM: scheduler.py, block_manager.py, attention_kernels.cu
- SGLang: radix_cache.py, scheduler.py
- DeepGEMM: FP8 GEMM 实现

### 必读论文
- FlashAttention 1 & 2 (Dao et al.)
- PagedAttention / vLLM (Kwon et al.)
- Orca (Yu et al.)
- Speculative Decoding (Leviathan et al.)
- GPTQ (Frantar et al.) / AWQ (Lin et al.)
- Megatron-LM (Shoeybi et al.)
- DistServe (Zhong et al.)
- DeepSeek-V3 Technical Report

### 必做项目
- Kernel Lab: reduction, softmax, GEMM, FlashAttention, RMSNorm, INT4 dequant
- Mini Inference Engine 或 开源 PR
- Benchmark + Roofline 分析报告

### 学习社区
- 牛客网 AI Infra 面经
- 知乎 AI Infra 专栏
- GitHub: vLLM, SGLang, Megatron-LM, CUTLASS
- 博客园: cnblogs.com/xmwblogs (AI infra 面试收录)
