# CUDA / Performance 方向简历版本

> 面向：GPU 算子开发 / 性能工程岗
> 目标公司：NVIDIA、OpenAI Inference、DeepSeek、Moonshot、MiniMax、字节 AML、AMD
> ⚠️ 注意：候选人当前零 CUDA 经验，此版本侧重"异构硬件推理优化"视角切入，诚实标注短板

---

## Headline

推理系统性能工程师 | 异构硬件推测解码优化 | vLLM 社区贡献者 | CUDA 学习中

---

## Summary

在非 NVIDIA 硬件（昇腾 NPU）上实现推测解码全链路优化，输出吞吐提升 55%，TPOT 降低 39%。具备从推理算法到硬件适配的性能优化方法论：瓶颈分析 → 方案设计 → 参数调优 → 端到端验证。理解 KV cache 内存管理、token acceptance 策略等与 GPU 性能优化相通的核心问题。正在系统学习 CUDA kernel 开发与 GPU profiling。

---

## Skills Section（按优先级排列）

```
推理优化: Speculative Decoding (EAGLE-3), KV Cache 内存管理, Token Acceptance 策略, Draft-Verify Pipeline
性能方法论: 瓶颈分析 (memory-bound 识别), Benchmark 设计 (MT-Bench), 多维度指标评测 (Throughput/TPOT/Acceptance Rate)
硬件适配: Ascend 910B/Atlas 310P3 (NPU), 异构硬件推理优化, 跨平台性能验证
vLLM 生态: V1 架构理解, Scheduler/ModelRunner/KVCacheManager 交互
开源贡献: vLLM-Ascend PR #1032 (merged), vLLM-MindSpore PR #1020
编程语言: Python, C++
学习中: CUDA Kernel 开发, GPU Profiling (Nsight Compute)
```

---

## Project Ordering

1. **vLLM 推理优化 — 性能视角**（核心，突出性能方法论）
2. **非结构化问数系统 RAG 后端**（辅助，简述）

---

## Verified Bullet Points（可安全使用）

### 项目 1：推理性能优化

1. ✅ 在非 NVIDIA 硬件（昇腾 NPU）上实现推测解码全链路优化：输出吞吐 +55%（9.22→14.30 tok/s），TPOT -39%（108→66 ms），验证跨硬件平台推理加速可行性
2. ✅ 独立实现 EAGLE-3 proposer（vLLM-Ascend PR #1032 merged），设计 draft model 与 target model 的 hidden states 传递机制和 KV cache 分配策略
3. ✅ 设计并验证 speculative decoding 的 token acceptance 策略：num_spec_tokens=2 时 mean acceptance length 1.63，token-1/token-2 接受率 70%/47%
4. ✅ 在推测解码实现中处理 KV cache 分配与回收逻辑，管理 rejected token 场景下的 cache 状态 [合理推断]
5. ✅ 基于 MT-Bench 设计多维度 benchmark 方案，覆盖吞吐（tok/s）、延迟（TPOT）、acceptance rate 等指标 [合理推断]
6. ✅ 分析 decode 阶段 memory-bound 特征，选择推测解码方案提升计算利用率 [合理推断]
7. ✅ 参与 KV-select、Sparse Attention 适配，理解长上下文场景下 attention 计算与 KV cache 访问的性能瓶颈

### 项目 2：RAG（简述）

1. ✅ 独立负责 RAG 系统后端，RAGAS 评测准确率 90%

---

## Risky Bullet Points to Avoid

| 声明 | 原因 | 标记 |
|------|------|------|
| "精通 CUDA 编程" | 正在学习阶段 | 🚫 Do Not Use |
| "开发高性能 GPU kernel" | 无生产级 kernel | 🚫 Do Not Use |
| "使用 Nsight 优化 kernel 性能" | 无 profiling 产出 | 🚫 Do Not Use |
| "TensorRT-LLM 部署优化" | 无使用经验 | 🚫 Do Not Use |
| "分布式推理系统设计" | 无 TP/PP 实操 | 🚫 Do Not Use |
| "FP8/INT4 量化优化" | 无量化实践 | 🚫 Do Not Use |
| "GPU 内存优化" | 无 GPU 侧实操 | 🚫 Do Not Use |
| "Roofline model 分析" | 无实际产出 | 🚫 Do Not Use |

---

## Missing Evidence List

| 缺失项 | 严重程度 | 补充方式 | 预计时间 |
|--------|----------|----------|----------|
| CUDA kernel 开发 | 致命 — 此方向硬性要求 | 实现 softmax/GEMM/attention kernel | 4-6 周 |
| Nsight Compute profiling | 致命 | 产出 kernel 性能分析报告 | 2-3 周 |
| GPU memory hierarchy 实操 | 高 | shared memory tiling、bank conflict 实验 | 2-4 周 |
| Roofline model 分析 | 高 | 对自实现 kernel 做 roofline 分析 | 1-2 周 |
| Triton kernel 开发 | 中 | 实现 fused softmax/attention | 2-3 周 |
| 分布式推理 | 中 | TP/PP 学习与验证 | 3-4 周 |

---

## 投递策略建议

⚠️ **此方向当前匹配度较低（<40%）**，建议：
- 完成 CUDA 基础学习后再投递纯 GPU 性能岗
- 当前可投递"推理优化"岗（不硬性要求 CUDA kernel 开发的）
- 面试时诚实说明 CUDA 学习进度，强调 NPU 优化经验的可迁移性
- 突出"性能优化方法论"而非"具体 GPU 技能"
