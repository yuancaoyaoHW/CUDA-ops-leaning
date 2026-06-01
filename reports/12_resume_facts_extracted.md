# 简历结构化事实提炼

## 基本信息

| 字段 | 内容 |
|------|------|
| 姓名 | 袁曹尧 |
| 邮箱 | 2749322671@qq.com |
| 电话 | (+86) 153-3248-1217 |
| GitHub | github.com/yuancaoyaoHW |
| Gitee | gitee.com/yuancaoyao_HW |
| 求职方向 | 大模型推理系统 / NPU 推理优化 / RAG 后端工程 |

## 教育背景

| 学校 | 学院 | 学位 | 专业 | 时间 |
|------|------|------|------|------|
| 浙江大学 | 计算机科学与技术学院 | 硕士 | 计算机技术 | 2021.09 - 2025.03 |
| 西安交通大学 | 数学与统计学院 | 学士 | 信息与计算科学 | 2016.09 - 2020.07 |

## 技术栈分类

### 推理系统
- vLLM, vLLM-Ascend, vLLM-MindSpore
- EAGLE-3 Speculative Decoding
- KV Cache 管理
- Decode 调度
- Benchmark 验证

### RAG / 后端
- Python, LangChain, RAGAS
- 文档解析, Chunk 切分, Embedding 索引
- 向量检索, Metadata Filters
- Research Workflow

### AI 框架 / 硬件
- MindSpore, PyTorch
- Ascend NPU (910B), Atlas 3000/310P3
- C++, Linux, Docker, Git

## 项目经历摘要

| 项目 | 角色 | 时间 | 关键指标 |
|------|------|------|----------|
| vLLM 生态大模型推理优化与昇腾 NPU 适配 | 核心开发 | 2025.03 - 2025.10 | 吞吐 9.22→14.30 tok/s；TPOT 108.18→65.76 ms；acceptance length 1.63 |
| 非结构化问数系统（RAG 后端） | 独立负责 | 2025.11 - 2026.05 | 问答准确率 90%（RAGAS 评测） |

## 开源贡献

| 仓库 | PR 编号 | 状态 | 内容 |
|------|---------|------|------|
| vLLM-Ascend | #1032 | 已合入 | 为 vLLM V1 实现 EAGLE-3 proposer |
| vLLM-MindSpore | #1020 | 已提交 | 为 Qwen2/Qwen2.5 适配 EAGLE-3 推测解码，打通 draft model、hidden states、KV cache、rejection sampler 与 runner 执行链路 |

## 可量化成果

1. **吞吐提升 55%**：Atlas 3000/310P3 上 MT-Bench 场景，spec=3 时输出吞吐由 9.22 tok/s 提升至 14.30 tok/s
2. **TPOT 降低 39%**：同场景下 TPOT 由 108.18 ms 降至 65.76 ms
3. **推测解码接受率**：Ascend 910B 上 num_spec_tokens=2 时 mean acceptance length 1.63，token-1/token-2 接受率 70%/47%
4. **RAG 问答准确率 90%**：基于 RAGAS 评测框架验证
5. **开源 PR 合入**：vLLM-Ascend #1032 合入社区主线
