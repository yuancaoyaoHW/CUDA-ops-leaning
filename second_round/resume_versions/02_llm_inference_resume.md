# LLM Inference 方向简历版本

> 面向：大模型推理优化岗（推理引擎开发、Speculative Decoding、Serving 优化）
> 目标公司：字节 AML/Seed、阿里通义推理、百度文心推理、DeepSeek、Moonshot、MiniMax

---

## Headline

LLM 推理优化工程师 | EAGLE-3 Speculative Decoding 实现者 | vLLM 社区贡献者

---

## Summary

在 vLLM 生态中独立实现 EAGLE-3 推测解码并合入社区主线，具备从推理算法理解到硬件适配验证的完整能力。在昇腾 NPU 上实现 55% 吞吐提升、39% 延迟降低，对 speculative decoding 的 draft-verify 机制、KV cache 管理、acceptance rate 调优有实战经验。目标是成为覆盖 GPU/NPU 双平台的推理系统工程师。

---

## Skills Section（按优先级排列）

```
推理优化: EAGLE-3 Speculative Decoding, Draft-Verify Pipeline, Rejection Sampling, KV Cache Management
vLLM 生态: vLLM V1 架构 (Scheduler/ModelRunner/KVCacheManager), vLLM-Ascend, vLLM-MindSpore
性能验证: MT-Bench Benchmarking, Throughput/TPOT/Acceptance Rate 多维度评测, 参数调优 (num_spec_tokens)
硬件平台: Ascend 910B, Atlas 3000/310P3
开源贡献: vLLM-Ascend PR #1032 (merged), vLLM-MindSpore PR #1020
编程语言: Python, C++
工具链: Linux, Docker, Git
```

---

## Project Ordering

1. **vLLM 生态推理优化与昇腾 NPU 适配**（核心项目，重点展开）
2. **非结构化问数系统 RAG 后端**（辅助，体现工程广度）

---

## Verified Bullet Points（可安全使用）

### 项目 1：vLLM 推理优化（重点展开）

1. ✅ 独立实现 EAGLE-3 Speculative Decoding proposer 并合入 vLLM-Ascend 社区主线（PR #1032），为 vLLM V1 架构新增推测解码能力
2. ✅ 在 Atlas 310P3 上验证推测解码效果：输出吞吐 +55%（9.22→14.30 tok/s），TPOT -39%（108→66 ms），spec_tokens=3
3. ✅ 设计并验证 token acceptance 策略：num_spec_tokens=2 时 mean acceptance length 1.63，token-1/token-2 接受率 70%/47%
4. ✅ 打通 draft model → hidden states → KV cache → rejection sampler → runner 完整执行链路，实现端到端推测解码 pipeline
5. ✅ 提交 vLLM-MindSpore PR #1020，为 Qwen2.5-7B 适配 EAGLE-3，验证跨框架推测解码可行性
6. ✅ 分析 vLLM V1 架构中 scheduler、model runner、KV cache manager 的交互机制，基于架构理解设计 proposer 接口 [合理推断]
7. ✅ 在推测解码实现中处理 KV cache 分配与回收逻辑，管理 rejected token 场景下的 cache 状态 [合理推断]
8. ✅ 基于 MT-Bench 前 80 轮设计 benchmark 方案，覆盖吞吐、延迟、acceptance rate 多维度指标 [合理推断]
9. ✅ 参与 KV-select、Sparse Attention 内部适配，理解长上下文场景下 attention 计算与 KV cache 访问的性能特征

### 项目 2：RAG 后端（简述）

1. ✅ 独立负责 RAG 系统后端，基于 RAGAS 评测准确率达 90%
2. ✅ 设计双链路架构（快速问答 + Research），抽象统一检索接口

---

## Risky Bullet Points to Avoid

| 声明 | 原因 | 标记 |
|------|------|------|
| "熟悉 CUDA 编程" | 零 kernel 经验 | 🚫 Do Not Use |
| "GPU 性能优化 / Nsight profiling" | 无产出 | 🚫 Do Not Use |
| "分布式推理（TP/PP）" | 无实操 | 🚫 Do Not Use |
| "Continuous Batching 实现" | 未参与实现，仅理解 | ⚠️ 只能写"理解" |
| "PagedAttention 优化" | 未参与实现 | ⚠️ 只能写"理解原理" |
| "量化推理（FP8/INT4）" | 无实践 | 🚫 Do Not Use |

---

## Missing Evidence List

| 缺失项 | 对该方向的影响 | 补充方式 |
|--------|---------------|----------|
| CUDA kernel 开发 | 多数推理岗要求 | 实现 softmax/attention kernel |
| GPU profiling | 性能工程必备 | Nsight Compute 分析报告 |
| 分布式推理 | 大模型 serving 必备 | 学习 TP/PP 并在 vLLM 中验证 |
| Production serving 经验 | 区分学术与工业 | 部署有真实流量的服务 |
| Continuous Batching 实现细节 | 推理引擎核心 | 阅读 vLLM scheduler 源码并总结 |
