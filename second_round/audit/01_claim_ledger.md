# Claim Ledger — 简历声明事实审计总表

> 审计日期：2026-06-01
> 审计范围：所有简历 bullet、项目包装、面试故事中的事实性声明

## 审计标准

| evidence_status | 含义 |
|-----------------|------|
| verified | 简历原文直接支撑，可安全使用 |
| reasonable_inference | 基于已有事实的合理推断，标注后可用 |
| needs_evidence | 需要候选人补充具体数据或确认 |
| unsupported | 无事实支撑，不可使用 |
| fabricated | 虚构内容，必须删除 |

---

## 完整 Claim Ledger

| # | claim | source_file | source_type | evidence_status | can_use_in_resume | needs_user_data | rewrite_suggestion |
|---|-------|-------------|-------------|-----------------|-------------------|-----------------|-------------------|
| 1 | 独立实现 EAGLE-3 Speculative Decoding proposer 并合入 vLLM-Ascend 社区主线（PR #1032） | 12_resume_facts_extracted.md | resume_original | verified | ✅ 是 | 否 | — |
| 2 | 输出吞吐由 9.22 tok/s 提升至 14.30 tok/s（+55%） | 12_resume_facts_extracted.md | resume_original | verified | ✅ 是 | 否 | — |
| 3 | TPOT 由 108.18 ms 降至 65.76 ms（-39%） | 12_resume_facts_extracted.md | resume_original | verified | ✅ 是 | 否 | — |
| 4 | num_spec_tokens=2 时 mean acceptance length 1.63 | 12_resume_facts_extracted.md | resume_original | verified | ✅ 是 | 否 | — |
| 5 | token-1/token-2 接受率 70%/47% | 12_resume_facts_extracted.md | resume_original | verified | ✅ 是 | 否 | — |
| 6 | vLLM-MindSpore PR #1020 已提交 | 12_resume_facts_extracted.md | resume_original | verified | ✅ 是 | 否 | 注意：状态为"已提交"非"已合入" |
| 7 | 独立负责非结构化问数系统后端与 RAG 链路 | 12_resume_facts_extracted.md | resume_original | verified | ✅ 是 | 否 | — |
| 8 | RAGAS 评测问答准确率 90% | 12_resume_facts_extracted.md | resume_original | verified | ✅ 是 | 否 | — |
| 9 | 设计快速问答与 Research 问答双链路 | 12_resume_facts_extracted.md | resume_original | verified | ✅ 是 | 否 | — |
| 10 | 抽象检索与 metadata filters 查询接口 | 12_resume_facts_extracted.md | resume_original | verified | ✅ 是 | 否 | — |
| 11 | 参与 KV-select、Sparse Attention、Suffix Decoding 内部适配 | 12_resume_facts_extracted.md | resume_original | verified | ✅ 是 | 否 | — |
| 12 | 打通 draft model → hidden states → KV cache → rejection sampler → runner 执行链路 | 12_resume_facts_extracted.md | resume_original | verified | ✅ 是 | 否 | — |
| 13 | 浙江大学硕士（计算机技术）2021.09-2025.03 | 12_resume_facts_extracted.md | resume_original | verified | ✅ 是 | 否 | — |
| 14 | 西安交通大学学士（信息与计算科学）2016.09-2020.07 | 12_resume_facts_extracted.md | resume_original | verified | ✅ 是 | 否 | — |
| 15 | 深入理解 vLLM V1 架构中 scheduler、model runner、KV cache manager 的交互机制 | 14_resume_to_ai_infra_positioning.md | report_generated | reasonable_inference | ✅ 是（标注推断） | 否 | 建议改为"分析并理解 vLLM V1 架构…" |
| 16 | 定位并修复推测解码中 hidden states 传递、rejection sampling 边界条件等问题 | 14_resume_to_ai_infra_positioning.md | report_generated | reasonable_inference | ✅ 是（标注推断） | 是：确认具体修复了哪些问题 | — |
| 17 | 按社区 CI 标准补充单元测试，覆盖 proposer 接口、KV cache 分配与 token 验证链路 | 14_resume_to_ai_infra_positioning.md | report_generated | reasonable_inference | ✅ 是（标注推断） | 否 | PR 合入通常需要测试 |
| 18 | 经过 3 轮 review 后 PR 成功合入 | 04_behavioral_stories.md | interview_generated | needs_evidence | ⚠️ 需确认 | 是：确认具体 review 轮次 | 改为"经过多轮 review" |
| 19 | Research 链路精简后检索效率提升 | 04_behavioral_stories.md | interview_generated | needs_evidence | ⚠️ 需确认 | 是：补充具体提升数据 | 删除或补充数据 |
| 20 | 适配方案被认可为后续其他模型接入的参考模板 | 04_behavioral_stories.md | interview_generated | needs_evidence | ⚠️ 需确认 | 是：确认是否有社区反馈 | 改为"为后续模型接入提供了参考" |
| 21 | 基于 RAGAS 框架构建端到端评测流程，覆盖 faithfulness、answer relevancy、context precision | 17_resume_bullets_rag_infra_optimized.md | report_generated | needs_evidence | ⚠️ 需确认 | 是：确认具体使用了哪些 RAGAS 维度 | 删除具体维度名或确认后保留 |
| 22 | 设计基于语义边界的 chunk 切分策略 | 17_resume_bullets_rag_infra_optimized.md | report_generated | needs_evidence | ⚠️ 需确认 | 是：确认具体切分策略 | — |
| 23 | 快速问答支持秒级响应 | 17_resume_bullets_rag_infra_optimized.md | report_generated | needs_evidence | ⚠️ 需确认 | 是：确认具体延迟数据 | 删除"秒级"或补充数据 |
| 24 | 支持增量更新 | 06_resume_bullets_by_role.md | report_generated | needs_evidence | ⚠️ 需确认 | 是：确认是否实现了增量更新 | — |
| 25 | 熟悉 CUDA 编程 | — | — | unsupported | 🚫 否 | — | 不可使用 |
| 26 | 分布式推理经验（TP/PP） | — | — | unsupported | 🚫 否 | — | 不可使用 |
| 27 | GPU 性能优化 / Nsight profiling | — | — | unsupported | 🚫 否 | — | 不可使用 |
| 28 | TensorRT-LLM 部署经验 | — | — | unsupported | 🚫 否 | — | 不可使用 |
| 29 | Kubernetes GPU serving | — | — | unsupported | 🚫 否 | — | 不可使用 |
| 30 | 大规模向量数据库部署经验 | — | — | unsupported | 🚫 否 | — | 不可使用 |
| 31 | 百万级文档索引优化 | — | — | unsupported | 🚫 否 | — | 不可使用 |
| 32 | Fine-tune embedding 模型 | — | — | unsupported | 🚫 否 | — | 不可使用 |
| 33 | 多模态 RAG | — | — | unsupported | 🚫 否 | — | 不可使用 |
| 34 | FP8/INT4 量化优化 | — | — | unsupported | 🚫 否 | — | 不可使用 |
| 35 | 最终版本达到 cuBLAS 82% 性能 | plans/02_project_roadmap.md | plan_generated | unsupported | 🚫 否 | — | 这是未来目标，非已有成果 |
| 36 | memory throughput 从 30% → 85% SOL | plans/02_project_roadmap.md | plan_generated | unsupported | 🚫 否 | — | 这是未来目标，非已有成果 |

---

## 统计汇总

| evidence_status | 数量 | 占比 |
|-----------------|------|------|
| verified | 14 | 39% |
| reasonable_inference | 3 | 8% |
| needs_evidence | 7 | 19% |
| unsupported | 12 | 33% |
| fabricated | 0 | 0% |

**结论**：候选人材料中无虚构内容。14 条核心事实经验证可安全使用，3 条合理推断标注后可用，7 条需要候选人补充数据确认，12 条为明确不可使用的技能声明（已正确标注为红线）。
