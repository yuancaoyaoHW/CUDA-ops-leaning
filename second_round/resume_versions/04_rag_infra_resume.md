# RAG Infrastructure 方向简历版本

> 面向：RAG / 知识库后端岗
> 目标公司：各大厂 AI 应用团队、RAG 创业公司、知识库产品、LangChain 生态公司

---

## Headline

RAG 后端工程师 | 独立交付端到端问答系统 | RAGAS 评测 90% 准确率

---

## Summary

具备 RAG 系统独立交付能力，从文档解析到答案生成全链路设计与实现，基于 RAGAS 构建评测闭环达 90% 准确率。设计双链路架构支持快速问答与深度 Research 两种模式。同时具备大模型推理系统优化背景（vLLM 生态 PR 合入），理解从推理层到应用层的完整技术栈，能从推理性能角度优化 RAG 系统的生成环节。

---

## Skills Section（按优先级排列）

```
RAG 系统: LangChain, 文档解析, Chunk 策略设计, Embedding 索引, 向量检索, Metadata Filters, 双链路架构设计
评测体系: RAGAS (端到端评测流程), 问答准确率验证, 检索质量评估
后端工程: Python, RESTful API, 接口抽象设计, 状态流转管理, 异常处理
推理系统: vLLM, Speculative Decoding (EAGLE-3), KV Cache (理解推理层优化)
开源贡献: vLLM-Ascend PR #1032 (merged)
工具链: Linux, Docker, Git
```

---

## Project Ordering

1. **非结构化问数系统 RAG 后端**（主打项目，重点展开）
2. **vLLM 推理优化**（辅助，体现技术深度和全栈理解）

---

## Verified Bullet Points（可安全使用）

### 项目 1：非结构化问数系统（RAG）— 重点展开

1. ✅ 独立负责非结构化问数系统后端与 RAG 链路设计，基于 Python + LangChain 实现文档解析 → chunk 构建 → embedding 索引 → 向量检索 → 上下文组装 → 答案生成完整 pipeline
2. ✅ 设计快速问答与 Research 问答双链路架构，支持不同复杂度查询的差异化处理策略
3. ✅ 基于 RAGAS 框架构建端到端评测流程，问答准确率达 90%
4. ✅ 抽象统一检索接口，支持 metadata filters 查询与多后端切换，实现业务层与检索层解耦
5. ✅ 重构 Research 链路状态流转逻辑，为每个阶段添加成功/失败分支处理，提升系统稳定性 [合理推断 — 基于简历提到重构]
6. ✅ 设计文档解析模块，支持非结构化文档的结构化提取与清洗 [合理推断]

### 项目 2：vLLM 推理优化（RAG 方向侧重）

1. ✅ 在 vLLM 生态中实现 EAGLE-3 推测解码（PR #1032 merged），具备大模型推理系统的深度理解，能从推理层优化 RAG 系统的生成环节性能
2. ✅ 在昇腾 NPU 上验证推测解码效果：吞吐 +55%，延迟 -39%，具备推理性能 benchmark 设计与验证能力

---

## Risky Bullet Points to Avoid

| 声明 | 原因 | 标记 |
|------|------|------|
| "大规模向量数据库部署经验" | 无 Milvus/OpenSearch 大规模部署 | 🚫 Do Not Use |
| "Kubernetes 部署 RAG 服务" | 无 K8s 经验 | 🚫 Do Not Use |
| "百万级文档索引优化" | 无大规模数据量证据 | 🚫 Do Not Use |
| "Fine-tune embedding 模型" | 无 fine-tuning 记录 | 🚫 Do Not Use |
| "多模态 RAG" | 无多模态处理经验 | 🚫 Do Not Use |
| "秒级响应" | 未确认具体延迟数据 | ⚠️ 需确认后使用 |
| "支持增量更新" | 未确认是否实现 | ⚠️ 需确认后使用 |
| "覆盖 faithfulness、answer relevancy、context precision" | 未确认具体维度 | ⚠️ 需确认后使用 |

---

## Missing Evidence List

| 缺失项 | 对该方向的影响 | 补充方式 |
|--------|---------------|----------|
| 具体文档规模数据 | 面试官会问规模 | 补充文档数量、索引大小 |
| 具体延迟数据（P50/P99） | 体现性能意识 | 补充实测延迟 |
| 具体 chunk 策略选择 | 体现技术决策能力 | 确认使用了哪种策略 |
| Reranking 实现 | RAG 岗位常见要求 | 如有实现则补充 |
| 向量数据库选型 | 体现技术选型能力 | 补充使用了哪个向量库 |
| 监控与可观测性 | Production 必备 | 添加 monitoring 能力 |
| 负载测试数据 | 体现工程成熟度 | 补充 QPS 数据 |

---

## 差异化优势（相比纯 RAG 候选人）

| 维度 | 纯 RAG 候选人 | 本简历优势 |
|------|---------------|------------|
| 推理层理解 | 黑盒调用 LLM API | 理解推理系统内部机制，能优化生成环节 |
| 性能优化 | 依赖框架默认配置 | 有 benchmark 设计和性能调优方法论 |
| 开源经验 | 使用开源工具 | 贡献开源社区（PR merged） |
| 系统深度 | 应用层 | 应用层 + 推理层双重理解 |
