# Ascend / NPU 方向简历版本

> 面向：华为昇腾生态岗（NPU 推理引擎、CANN 算子、MindSpore 推理、盘古大模型推理）
> 目标公司：华为、昇腾生态合作伙伴、使用 NPU 的 AI 公司
> ✅ 此方向匹配度最高（>70%）

---

## Headline

昇腾 NPU 推理优化工程师 | vLLM-Ascend 社区贡献者 | EAGLE-3 推测解码实现者

---

## Summary

在昇腾 NPU 生态中深度参与大模型推理优化，独立实现 EAGLE-3 推测解码并合入 vLLM-Ascend 社区主线（PR #1032），在 Atlas 310P3 上实现 55% 吞吐提升。同时为 vLLM-MindSpore 适配 EAGLE-3（PR #1020），具备跨框架（vLLM-Ascend / vLLM-MindSpore）的推理优化能力。熟悉 Ascend 910B 硬件特性和 NPU 推理部署流程。

---

## Skills Section（按优先级排列）

```
昇腾生态: Ascend 910B, Atlas 3000/310P3, vLLM-Ascend, vLLM-MindSpore, NPU 推理优化
推理优化: EAGLE-3 Speculative Decoding, KV Cache Management, Draft-Verify Pipeline, Rejection Sampling
vLLM 生态: V1 架构 (Scheduler/ModelRunner/KVCacheManager), Proposer 接口设计, 社区协作
AI 框架: MindSpore, PyTorch
性能验证: MT-Bench Benchmarking, Throughput/TPOT/Acceptance Rate 评测
开源贡献: vLLM-Ascend PR #1032 (merged), vLLM-MindSpore PR #1020
编程语言: Python, C++
工具链: Linux, Docker, Git
```

---

## Project Ordering

1. **vLLM-Ascend 推理优化（EAGLE-3 推测解码）**（核心项目）
2. **vLLM-MindSpore EAGLE-3 适配**（补充项目，体现跨框架能力）
3. **非结构化问数系统 RAG 后端**（辅助，体现工程广度）

---

## Verified Bullet Points（可安全使用）

### 项目 1：vLLM-Ascend 推理优化

1. ✅ 独立实现 EAGLE-3 Speculative Decoding proposer 并合入 vLLM-Ascend 社区主线（PR #1032），为 vLLM V1 架构在昇腾 NPU 上新增推测解码能力
2. ✅ 在 Atlas 310P3 上验证推测解码效果：输出吞吐 +55%（9.22→14.30 tok/s），TPOT -39%（108→66 ms），spec_tokens=3
3. ✅ 打通 draft model → hidden states → KV cache → rejection sampler → runner 完整执行链路，适配 Ascend NPU 的算子能力和内存管理特性
4. ✅ Ascend 910B 上 num_spec_tokens=2 时 mean acceptance length 达 1.63，token-1/token-2 接受率 70%/47%
5. ✅ 分析 Ascend NPU 的算子能力与 GPU 实现的差异，设计适配方案桥接 NPU 特性与 vLLM 核心接口 [合理推断]
6. ✅ 按社区 reviewer 反馈完成多轮迭代：proposer 接口重构、runner 执行链路收敛、单元测试补充与合并冲突处理 [合理推断]
7. ✅ 参与 KV-select、Sparse Attention、Suffix Decoding 内部适配，理解长上下文场景下 NPU 上的性能特征

### 项目 2：vLLM-MindSpore 适配

1. ✅ 提交 vLLM-MindSpore PR #1020，为 Qwen2/Qwen2.5 适配 EAGLE-3 推测解码
2. ✅ 在 MindSpore 框架约束下设计适配层，桥接 MindSpore 的 KV cache 管理与 vLLM 社区接口规范 [合理推断]
3. ✅ 打通 draft model、hidden states、KV cache、rejection sampler 与 runner 执行链路

### 项目 3：RAG 后端（简述）

1. ✅ 独立负责 RAG 系统后端，RAGAS 评测准确率 90%
2. ✅ 设计双链路架构，抽象统一检索接口

---

## Risky Bullet Points to Avoid

| 声明 | 原因 | 标记 |
|------|------|------|
| "CANN 算子开发" | 无 CANN 底层算子开发经验 | 🚫 Do Not Use |
| "NPU 算子性能调优" | 无算子级 profiling | 🚫 Do Not Use |
| "Ascend 集群部署" | 无多机部署经验 | 🚫 Do Not Use |
| "MindSpore 框架开发" | 使用级别，非框架开发 | ⚠️ 只能写"使用" |
| "大规模 NPU 推理服务" | 无 production 规模 | 🚫 Do Not Use |
| "华为内部工具链" | 非华为内部员工 | 🚫 Do Not Use |

---

## Missing Evidence List

| 缺失项 | 对该方向的影响 | 补充方式 |
|--------|---------------|----------|
| CANN 算子开发 | 华为内部岗位可能要求 | 学习 CANN 算子开发流程 |
| NPU profiling 工具使用 | 性能优化深度 | 学习 Ascend Profiler |
| 多机 NPU 部署 | 大规模推理要求 | 在多卡环境验证 |
| 量化推理（W8A8 等） | NPU 推理常见需求 | 学习 NPU 量化方案 |
| 更多模型适配经验 | 体现通用性 | 适配更多模型（LLaMA 等） |

---

## 投递策略建议

✅ **此方向当前匹配度最高（>70%）**，建议：
- 立即投递华为昇腾推理引擎相关岗位
- 强调 vLLM-Ascend PR 合入的社区贡献
- 面试时重点讲 NPU 与 GPU 的差异以及适配挑战
- 准备 CANN 基础知识（即使没有深度经验，展示学习意愿）
- 准备回答"为什么选择昇腾生态"的问题
