# 开源贡献优先级排序与执行计划

## 总体策略

基于候选人当前能力（vLLM-Ascend PR 合入、EAGLE-3 经验、零 CUDA）和目标（补强简历、了解公司技术栈），按 **ROI（简历价值 × 可行性 / 时间投入）** 排序。

---

## Tier 1：立即行动（本周开始）

### 1. vLLM — speculators 库文档/benchmark PR
- **理由**: 已有合入记录，社区认可，spec decode 方向最匹配
- **具体行动**: 为 speculators 库补充 EAGLE-3 使用文档或 benchmark 数据
- **预期时间**: 3-5 天
- **简历价值**: ⭐⭐⭐⭐（展示持续贡献）
- **面试价值**: ⭐⭐⭐⭐⭐（EAGLE 3.1 刚发布，时效性强）
- **风险**: 低

### 2. SGLang — Ascend 平台文档/适配 PR
- **理由**: 有 Ascend 贡献指南，候选人 NPU 经验直接对口
- **具体行动**: 改进 SGLang Ascend 部署文档或适配新功能
- **预期时间**: 3-5 天
- **简历价值**: ⭐⭐⭐⭐（展示跨框架能力）
- **面试价值**: ⭐⭐⭐⭐⭐（Together AI/Fireworks 关注 SGLang）
- **风险**: 低

### 3. vLLM-Ascend — 持续贡献
- **理由**: 已有基础，继续深化
- **具体行动**: 新功能适配或性能优化
- **预期时间**: 1-2 周
- **简历价值**: ⭐⭐⭐⭐⭐（核心贡献）
- **面试价值**: ⭐⭐⭐⭐⭐
- **风险**: 极低

---

## Tier 2：2-4 周内启动（CUDA 基础建立后）

### 4. FlashInfer — benchmark PR
- **理由**: vLLM/SGLang 核心依赖，benchmark 门槛较低
- **具体行动**: 在 flashinfer-bench 贡献新 benchmark 场景
- **预期时间**: 3-5 天
- **简历价值**: ⭐⭐⭐（展示对 attention kernel 的理解）
- **面试价值**: ⭐⭐⭐⭐
- **风险**: 低-中

### 5. TensorRT-LLM — EAGLE-3 bug 复现/文档
- **理由**: NVIDIA 生态，EAGLE-3 经验可迁移
- **具体行动**: 复现 EAGLE-3 相关 issue，提交 bug report 或文档 PR
- **预期时间**: 3-5 天
- **简历价值**: ⭐⭐⭐（NVIDIA 岗位加分）
- **面试价值**: ⭐⭐⭐⭐
- **风险**: 中（审核严格）

### 6. Triton — 教程/kernel PR
- **理由**: 学习 Triton 的同时产出贡献
- **具体行动**: 实现 RMSNorm/RoPE Triton kernel，整理为教程 PR
- **预期时间**: 1 周
- **简历价值**: ⭐⭐⭐⭐（展示 Triton 能力）
- **面试价值**: ⭐⭐⭐⭐
- **风险**: 中

---

## Tier 3：1-2 月后（CUDA 能力建立后）

### 7. vLLM 主仓库 — spec decode 优化 PR
- **理由**: 从 vLLM-Ascend 升级到主仓库贡献
- **具体行动**: 基于 profiling 发现的 spec decode 优化点
- **预期时间**: 2-3 周
- **简历价值**: ⭐⭐⭐⭐⭐（主仓库 PR 极有价值）
- **面试价值**: ⭐⭐⭐⭐⭐
- **风险**: 中-高（审核严格）

### 8. SGLang — EAGLE proposer 实现
- **理由**: 将 EAGLE 经验完整迁移到 SGLang
- **具体行动**: 在 SGLang 中实现 EAGLE-3 proposer
- **预期时间**: 2-4 周
- **简历价值**: ⭐⭐⭐⭐⭐
- **面试价值**: ⭐⭐⭐⭐⭐
- **风险**: 中

### 9. FlashInfer — kernel 贡献
- **理由**: 展示 CUDA kernel 能力
- **具体行动**: 贡献新 attention pattern 或优化现有 kernel
- **预期时间**: 2-3 周
- **简历价值**: ⭐⭐⭐⭐⭐
- **面试价值**: ⭐⭐⭐⭐⭐
- **风险**: 中-高

---

## Tier 4：长期目标（3+ 月）

### 10. TGI — 性能对比报告
- **理由**: 展示对多框架的深入理解
- **预期时间**: 1 周
- **简历价值**: ⭐⭐⭐
- **面试价值**: ⭐⭐⭐

### 11. DeepSpeed — 分布式推理贡献
- **理由**: 补强分布式经验
- **预期时间**: 2-3 周
- **简历价值**: ⭐⭐⭐
- **面试价值**: ⭐⭐⭐⭐

---

## 执行时间线

```
Week 1-2:
├── [Tier 1] vLLM speculators 文档/benchmark PR
├── [Tier 1] SGLang Ascend 文档 PR
└── [Tier 1] vLLM-Ascend 持续贡献

Week 3-4:
├── [Tier 2] FlashInfer benchmark PR
├── [Tier 2] TensorRT-LLM EAGLE-3 bug 复现
└── [Tier 2] Triton RMSNorm/RoPE kernel

Week 5-8:
├── [Tier 3] vLLM 主仓库 spec decode 优化
├── [Tier 3] SGLang EAGLE proposer
└── [Tier 3] FlashInfer kernel 贡献

Month 3+:
├── [Tier 4] TGI 性能对比
└── [Tier 4] DeepSpeed 分布式推理
```

---

## 简历展示策略

### 贡献完成后的简历 Bullet

**Tier 1 完成后：**
- "Active contributor to vLLM ecosystem: implemented EAGLE-3 speculative decoding proposer for Ascend NPU (PR #1032 merged), achieving 55% throughput improvement"
- "Contributed to SGLang Ascend platform adaptation, expanding multi-framework serving capabilities"

**Tier 2 完成后：**
- "Contributed performance benchmarks to FlashInfer attention kernel library, systematically evaluating decode attention across batch/seq configurations"
- "Implemented RMSNorm and RoPE kernels in Triton, achieving 90%+ of CUDA native performance"

**Tier 3 完成后：**
- "Contributed speculative decoding optimization to vLLM main repository, improving verification efficiency by X%"
- "Implemented EAGLE-3 proposer in SGLang framework, enabling cross-framework speculative decoding"

---

## 面试叙事

### 故事线：从 Ascend 适配到全栈推理优化

1. **起点**: vLLM-Ascend EAGLE-3 PR 合入 — 证明开源协作能力
2. **扩展**: SGLang Ascend 适配 — 证明跨框架理解
3. **深入**: FlashInfer/Triton kernel — 证明底层能力
4. **升级**: vLLM 主仓库贡献 — 证明核心贡献能力

### 面试中如何讲述

> "我的开源贡献路径是从 vLLM-Ascend 开始的。在实现 EAGLE-3 proposer 的过程中，我深入理解了 speculative decoding 的 verification 机制和 KV cache 管理。之后我将这个经验迁移到 SGLang，并开始学习底层 kernel 优化。现在我在 FlashInfer 和 Triton 上也有贡献，展示了从系统层到算子层的全栈能力。"

---

## 注意事项

1. **不要把"计划贡献"写成"已贡献"** — 简历中只写已合入的 PR
2. **质量优先于数量** — 一个有深度的 PR 比十个 typo fix 有价值
3. **保持持续性** — 每周至少有一个 commit/PR 活动
4. **建立社区关系** — 在 Discord/Discussion 中积极参与讨论
5. **记录学习过程** — 每个贡献都可以写成博客或面试素材
