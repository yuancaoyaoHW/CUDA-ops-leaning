# 第二轮深加工主报告

> 生成日期：2026-06-01
> 团队：4 teammates (claim-audit-resume-agent, project-implementation-agent, mock-interview-agent, opensource-company-agent)

---

## 1. 执行摘要

本轮对第一轮所有结论进行了二次审计、多版本简历改写、项目骨架设计、高压模拟面试生成、开源贡献机会挖掘和线上故障演练 playbook 生成。

### 核心结论

| 维度 | 结论 |
|------|------|
| 现在能投什么岗位 | 华为昇腾生态岗、vLLM 生态开发岗、RAG 后端岗（中小公司） |
| 2-3 个月后能冲什么岗位 | 阿里云 PAI / 火山引擎推理优化、Together AI / Fireworks AI、国内大模型公司推理岗 |
| 哪些方向风险高 | NVIDIA CUDA 岗（零 kernel 经验）、GPU Cluster/K8s 岗（零 production 经验）、Staff-level 岗（经验不足） |
| 哪些内容不能写进简历 | "熟悉 CUDA 编程"、"分布式推理经验"、"GPU 性能优化"、"TensorRT-LLM 部署"、"K8s GPU serving" |
| 哪些项目优先做 | CUDA Ops Lab（P0）> LLM Inference Benchmark Lab（P0）> Triton Kernels Lab（P1）> RAG Eval Lab（P1） |
| 哪些面试题优先练 | vLLM 架构深度追问、EAGLE-3 原理与 tradeoff、PagedAttention 实现细节、KV cache 管理、RAG 规模化 |

---

## 2. 事实审计结果

### Claim Ledger 统计

| 类别 | 数量 | 说明 |
|------|------|------|
| verified（可安全使用） | 14 条 | 简历原文直接支撑 |
| reasonable_inference（标注后可用） | 3 条 | 基于 PR 合入的合理推断 |
| needs_evidence（需确认） | 7 条 | 需要候选人补充数据 |
| unsupported（不可使用） | 5 条 | 无事实支撑，禁止使用 |
| fabricated | 0 条 | 未发现虚构内容 |

### 关键发现
- 第一轮材料整体质量良好，未发现虚构内容
- 5 条 unsupported 声明已明确标注禁止使用（CUDA、分布式、GPU profiling 等）
- 7 条 needs_evidence 需要候选人确认具体数据后才能使用

---

## 3. 简历多版本产出

已生成 7 个方向的简历版本：

| 版本 | 匹配度 | 投递建议 |
|------|--------|---------|
| AI Infra | 中 | 需补 CUDA 项目后投递 |
| LLM Inference | 高 | 可立即投递昇腾生态岗 |
| CUDA / Performance | 低 | 需完成 CUDA Lab 后投递 |
| RAG Infrastructure | 高 | 可立即投递 |
| ML Platform | 中 | 需补 K8s 经验 |
| GPU Infrastructure | 低 | 需大量补强 |
| Ascend / NPU | 最高 | 立即投递华为生态 |

---

## 4. 项目骨架产出

### 5 个项目已设计完整骨架

| 项目 | 状态 | 预计完成时间 | 简历价值 |
|------|------|------------|---------|
| CUDA Ops Learning Lab | README + 结构设计完成 | 4 周 | ⭐⭐⭐⭐⭐ |
| Triton Kernels Lab | README + 结构设计完成 | 2 周 | ⭐⭐⭐⭐ |
| LLM Inference Benchmark Lab | README + 结构设计完成 | 3 周 | ⭐⭐⭐⭐⭐ |
| RAG Infra Eval Lab | README + 结构设计完成 | 2 周 | ⭐⭐⭐⭐ |
| GPU Serving Observability | README + 结构设计完成 | 2 周 | ⭐⭐⭐ |

---

## 5. 模拟面试产出

### 8 场高压模拟面试已生成

| 场次 | 方向 | 候选人当前通过概率 |
|------|------|------------------|
| 1 | LLM Inference 系统设计 | 60%（有 vLLM 经验支撑） |
| 2 | CUDA Kernel 优化 | 15%（零 CUDA 经验） |
| 3 | Triton Kernel 优化 | 10%（零 Triton 经验） |
| 4 | RAG Infrastructure 设计 | 55%（有 RAG 经验但缺规模化） |
| 5 | GPU Cluster / K8s 设计 | 10%（零 K8s GPU 经验） |
| 6 | Staff-level AI Infra 综合 | 20%（经验广度不足） |
| 7 | Production Debugging | 30%（有问题定位经验） |
| 8 | Behavioral + Project Deep Dive | 70%（有真实项目故事） |

### 错题本已生成
- 覆盖 concept gap、weak phrasing、missing metric、missing tradeoff 等分类

### 待完成
- 强答案库（10_strong_answer_bank.md）
- 高压追问库（11_high_pressure_followup_bank.md）
- Hire/No-hire 评分卡（12_hire_no_hire_scorecards.md）

---

## 6. 开源贡献机会

### Tier 1（立即行动）
1. **vLLM speculators 文档/benchmark PR** — 3-5 天，低风险
2. **SGLang Ascend 适配 PR** — 3-5 天，低风险
3. **vLLM-Ascend 持续贡献** — 1-2 周

### Tier 2（学习后行动）
4. **FlashInfer benchmark PR** — 需学习 CUDA 基础
5. **Triton tutorial PR** — 需完成 Triton Lab
6. **TensorRT-LLM 文档 PR** — 需了解 TRT-LLM 架构

---

## 7. 公司画像产出

已生成 23 家公司的技术画像，按优先级分为：
- **P0 立即投递**：华为昇腾、DeepSeek、Moonshot
- **P1 补材料后投递**：阿里云 PAI、火山引擎、Together AI、Fireworks AI
- **P2 长期目标**：NVIDIA、Anthropic、OpenAI、CoreWeave

---

## 8. 线上故障演练

已生成 14 个事故场景的完整 playbook，覆盖：
- LLM Serving 事故（TTFT/TPOT/KV cache/batching）
- CUDA 性能事故（kernel regression/utilization）
- RAG 质量事故（recall drop/reranker latency）
- GPU 集群事故（NCCL timeout/node NotReady）

---

## 9. 下一步行动

1. **本周**：确认 claim ledger 中 7 条 needs_evidence 的具体数据
2. **本周**：开始 CUDA Ops Lab 第一个 kernel（vector_add）
3. **本周**：向 vLLM speculators 提交文档 PR
4. **第 2 周**：完成 CUDA memory model 学习 + 3 个基础 kernel
5. **第 3-4 周**：完成 matmul tiling + softmax + LLM inference benchmark 搭建
6. **持续**：每周对照 `01_action_items_next_12_weeks.md` 验收

---

## 10. 文件产出统计

| 目录 | 文件数 | 状态 |
|------|--------|------|
| audit/ | 3 | ✅ 完成 |
| resume_versions/ | 10 | ✅ 完成 |
| projects/ | 8 | ✅ 完成 |
| benchmarks/ | 5 | ✅ 完成 |
| mock_interview/ | 9/12 | ⚠️ 缺 3 文件 |
| opensource/ | 7 | ✅ 完成 |
| company_dossiers/ | 5 | ✅ 完成 |
| incidents/ | 5 | ✅ 完成 |
| final/ | 5 | ✅ 完成 |
| **总计** | **57/60** | **95%** |
