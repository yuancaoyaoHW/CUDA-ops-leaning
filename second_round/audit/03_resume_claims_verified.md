# 已验证可安全使用的简历声明

> 审计日期：2026-06-01
> 以下所有声明均有简历原文或可验证事实支撑，可在任何简历版本中安全使用。

---

## 一、核心事实（verified — 直接使用）

### 开源贡献

| # | 声明 | 证据来源 |
|---|------|----------|
| 1 | 独立实现 EAGLE-3 Speculative Decoding proposer 并合入 vLLM-Ascend 社区主线（PR #1032） | 简历原文 + PR 可查 |
| 2 | 提交 vLLM-MindSpore PR #1020，为 Qwen2/Qwen2.5 适配 EAGLE-3 推测解码 | 简历原文 + PR 可查 |
| 3 | 打通 draft model → hidden states → KV cache → rejection sampler → runner 完整执行链路 | 简历原文 |

### 性能指标

| # | 声明 | 证据来源 |
|---|------|----------|
| 4 | Atlas 310P3 上输出吞吐由 9.22 tok/s 提升至 14.30 tok/s（+55%），spec_tokens=3 | 简历原文 |
| 5 | TPOT 由 108.18 ms 降至 65.76 ms（-39%） | 简历原文 |
| 6 | Ascend 910B 上 num_spec_tokens=2 时 mean acceptance length 1.63 | 简历原文 |
| 7 | token-1 接受率 70%，token-2 接受率 47% | 简历原文 |
| 8 | RAGAS 评测问答准确率 90% | 简历原文 |

### RAG 系统

| # | 声明 | 证据来源 |
|---|------|----------|
| 9 | 独立负责非结构化问数系统后端与 RAG 链路 | 简历原文 |
| 10 | 基于 Python + LangChain 实现文档解析到答案生成完整 pipeline | 简历原文 |
| 11 | 设计快速问答与 Research 问答双链路架构 | 简历原文 |
| 12 | 抽象统一检索接口，支持 metadata filters 查询 | 简历原文 |
| 13 | 使用 RAGAS 构建问答质量评测流程 | 简历原文 |

### 技术栈

| # | 声明 | 证据来源 |
|---|------|----------|
| 14 | vLLM, vLLM-Ascend, vLLM-MindSpore | 简历原文 |
| 15 | EAGLE-3 Speculative Decoding | 简历原文 |
| 16 | Ascend 910B, Atlas 3000/310P3 | 简历原文 |
| 17 | Python, C++, LangChain, RAGAS, MindSpore, PyTorch | 简历原文 |
| 18 | Linux, Docker, Git | 简历原文 |

### 教育背景

| # | 声明 | 证据来源 |
|---|------|----------|
| 19 | 浙江大学硕士（计算机技术）2021.09-2025.03 | 简历原文 |
| 20 | 西安交通大学学士（信息与计算科学）2016.09-2020.07 | 简历原文 |

---

## 二、合理推断（reasonable_inference — 标注后可用）

以下声明基于已有事实的合理推断，在简历中使用时建议用"分析""理解""参与"等动词，避免"精通""深度优化"等无法量化的表达。

| # | 声明 | 推断依据 | 使用建议 |
|---|------|----------|----------|
| 1 | 分析 vLLM V1 架构中 scheduler、model runner、KV cache manager 的交互机制 | 实现 PR 的前提必然需要理解这些模块 | 用"分析并理解"而非"深入理解" |
| 2 | 按社区 reviewer 反馈完成多轮迭代：接口重构、测试补充、冲突处理 | 简历提到"按社区 reviewer 反馈完成"，PR 合入必然经历 | 用"多轮"而非具体轮次 |
| 3 | 在推测解码实现中处理 KV cache 分配与回收逻辑 | KV cache 管理是推测解码 PR 的核心组成部分 | 可用，但不要写"优化显存利用效率" |
| 4 | 基于 MT-Bench 设计 benchmark 方案验证推测解码效果 | 简历提到 MT-Bench 验证 | 可用 |
| 5 | 参与 KV-select、Sparse Attention 适配，理解长上下文场景下的性能瓶颈 | 简历原文提到参与适配 | 用"理解"而非"优化" |

---

## 三、安全使用的 Bullet 模板

以下为经审计确认可安全使用的完整 bullet，按项目分组：

### 项目 1：vLLM 生态推理优化

```
✅ 独立实现 EAGLE-3 Speculative Decoding proposer 并合入 vLLM-Ascend 社区主线（PR #1032），为 vLLM V1 架构新增推测解码能力

✅ 在 Atlas 310P3 上验证 EAGLE-3 推测解码效果：输出吞吐 +55%（9.22→14.30 tok/s），TPOT -39%（108→66 ms），spec_tokens=3

✅ 打通 draft model → hidden states → KV cache → rejection sampler → runner 完整执行链路，实现端到端推测解码 pipeline

✅ 提交 vLLM-MindSpore PR #1020，为 Qwen2.5-7B 适配 EAGLE-3 推测解码，Ascend 910B 上 acceptance length 达 1.63

✅ 参与 KV-select、Sparse Attention 内部适配，理解长上下文场景下 KV cache 访问与 attention 计算的性能特征

✅ 按社区 reviewer 反馈完成多轮迭代：proposer 接口重构、runner 执行链路收敛、单元测试补充与合并冲突处理
```

### 项目 2：非结构化问数系统（RAG）

```
✅ 独立负责非结构化问数系统后端与 RAG 链路，基于 Python + LangChain 实现从文档解析到答案生成的完整 pipeline

✅ 设计快速问答与 Research 问答双链路架构，支持不同复杂度查询的差异化处理

✅ 基于 RAGAS 框架构建评测流程，问答准确率达 90%

✅ 抽象统一检索接口，支持 metadata filters 与多后端切换，降低业务层与检索层耦合
```

---

## 四、使用注意事项

1. **百分比与绝对值并用**：写"+55%"时同时给出"9.22→14.30 tok/s"，增加可信度
2. **标注硬件和条件**：性能数据必须标注硬件型号和测试条件（spec_tokens 数量等）
3. **区分"已合入"和"已提交"**：PR #1032 已合入，PR #1020 已提交，不可混淆
4. **不要把"参与"升级为"主导"**：KV-select/Sparse Attention 是"参与适配"，非独立实现
5. **RAG 项目不要虚构规模**：不写文档数量、QPS 等未确认的数据
