# AI Infra 岗位定位策略

## 核心叙事（Elevator Pitch）

> 浙大计算机硕士，专注大模型推理系统优化。在 vLLM 生态中实现 EAGLE-3 推测解码并合入社区主线（PR #1032），在昇腾 NPU 上实现 55% 吞吐提升。同时具备 RAG 系统独立交付能力，覆盖从推理优化到应用落地的完整链路。

**30 秒版本**：我做 vLLM 推测解码的 NPU 适配，PR 合入了社区，吞吐提升 55%。也独立做过 RAG 后端，准确率 90%。我想继续在推理系统方向深入。

---

## 简历 Bullet 优化建议

### 已有事实（直接强化表达）

| 原始表达 | 优化建议 | 理由 |
|----------|----------|------|
| "贡献并合入 vLLM-Ascend PR #1032" | "独立实现 EAGLE-3 Speculative Decoding proposer 并合入 vLLM-Ascend 社区主线（PR #1032），为 vLLM V1 架构新增推测解码能力" | 突出独立性和架构级贡献 |
| "输出吞吐由 9.22 tok/s 提升至 14.30 tok/s" | "在 Atlas 310P3 上验证 EAGLE-3 推测解码效果：输出吞吐 +55%（9.22→14.30 tok/s），TPOT -39%（108→66 ms）" | 百分比更直观，合并指标 |
| "参与 KV-select、Sparse Attention、Suffix Decoding 内部适配" | "参与 KV-select / Sparse Attention 适配，优化长上下文场景下 KV cache 访问效率与 attention 计算开销" | 聚焦价值而非罗列名词 |
| "问答准确率达到 90%" | "基于 RAGAS 框架构建端到端评测流程，检索命中率与答案准确率达 90%" | 体现评测工程能力 |

### 可强化表达（基于事实合理推断）

- **系统理解深度**：可补充"深入理解 vLLM V1 架构中 scheduler、model runner、KV cache manager 的交互机制"——这是做 PR 的前提
- **问题定位能力**：可补充"定位并修复推测解码中 hidden states 传递、rejection sampling 边界条件等问题"——PR review 过程中必然涉及
- **工程规范**：可补充"按社区 CI 标准补充单元测试，覆盖 proposer 接口、KV cache 分配与 token 验证链路"

### 需补充证据（当前简历无法支撑）

- ❌ 不要写"熟悉 CUDA 编程"——无实际 kernel 代码
- ❌ 不要写"分布式推理经验"——无 TP/PP 实操记录
- ❌ 不要写"GPU 性能优化"——无 profiling 产出
- ✅ 如果补强后有成果，可以加入（参见 gap 分析中的补强建议）

---

## 面试故事线设计（STAR 格式）

### 故事 1：EAGLE-3 推测解码 PR 合入

| 维度 | 内容 |
|------|------|
| **Situation** | vLLM-Ascend 社区需要在昇腾 NPU 上支持推测解码以提升推理吞吐，但 V1 架构尚无 EAGLE-3 proposer 实现 |
| **Task** | 独立实现 EAGLE-3 proposer，打通 draft model → hidden states → KV cache → rejection sampler 完整链路，并通过社区 review |
| **Action** | 1) 分析 vLLM V1 架构中 scheduler 与 model runner 的交互机制；2) 实现 proposer 接口，处理 KV cache 分配与 hidden states 传递；3) 在 Ascend 910B 上验证 acceptance rate；4) 根据 reviewer 反馈迭代接口设计、补充测试、解决合并冲突 |
| **Result** | PR #1032 合入主线；num_spec_tokens=2 时 acceptance length 1.63；Atlas 310P3 上吞吐提升 55%，TPOT 降低 39% |

**面试追问准备**：
- Q: 为什么选择 EAGLE-3 而不是其他推测解码方案？
- Q: KV cache 在推测解码中如何管理？rejected token 的 cache 如何回收？
- Q: acceptance rate 不高时怎么调优？num_spec_tokens 如何选择？
- Q: NPU 适配和 GPU 实现的主要差异是什么？

### 故事 2：RAG 系统独立交付

| 维度 | 内容 |
|------|------|
| **Situation** | 业务需要对非结构化文档进行智能问答，要求支持快速问答和深度 Research 两种模式 |
| **Task** | 独立负责后端 RAG 链路设计与实现，从文档解析到答案生成全流程 |
| **Action** | 1) 设计双链路架构（快速问答 vs Research）；2) 抽象 retriever 接口支持 metadata filters；3) 重构 Research 总结模块解决状态流转问题；4) 用 RAGAS 构建评测闭环 |
| **Result** | 问答准确率 90%；系统稳定上线；retriever 抽象支持多后端切换 |

**面试追问准备**：
- Q: chunk 切分策略怎么选择的？
- Q: Research 模式和快速问答的区别是什么？
- Q: 如何处理检索结果不相关的情况？
- Q: RAGAS 评测具体评估哪些维度？

---

## 目标公司/团队类型匹配

### 高匹配（核心经验直接对口）

| 公司 | 团队/方向 | 匹配理由 |
|------|-----------|----------|
| **华为** | 昇腾 AI 推理引擎 / MindSpore 推理 | NPU 适配经验直接对口，有 PR 合入记录 |
| **华为** | 盘古大模型推理优化 | vLLM 生态经验 + 昇腾硬件理解 |
| 字节跳动 | AML 推理优化 / Seed 推理 | vLLM 经验对口，推测解码是热点方向 |
| 阿里 | 通义大模型推理 / PAI 平台 | 推理系统经验匹配 |
| 百度 | 文心推理优化 | 推理系统 + RAG 双重匹配 |

### 中等匹配（需补强部分技能）

| 公司 | 团队/方向 | 需补强 |
|------|-----------|--------|
| NVIDIA | TensorRT-LLM | CUDA kernel、GPU profiling |
| Meta | GenAI Inference | CUDA、分布式推理 |
| 字节跳动 | CUDA 算子团队 | CUDA kernel 开发 |
| Moonshot / 智谱 / DeepSeek | 推理引擎 | CUDA、Triton、分布式 |
| MiniMax | 推理优化 | GPU 侧经验 |

### 可切入（RAG/应用层方向）

| 公司 | 团队/方向 | 匹配点 |
|------|-----------|--------|
| 各大厂 | RAG / 知识库 / AI 应用后端 | RAG 独立交付经验 |
| Langchain 生态公司 | 后端工程 | LangChain + 评测经验 |

### 定位建议

**主攻方向**：大模型推理系统工程师（偏 vLLM 生态 / 推测解码 / NPU 适配）
**备选方向**：RAG 后端工程师（独立交付能力强，但天花板相对低）
**长期目标**：补齐 CUDA + GPU profiling 后，可覆盖更广泛的 GPU 推理优化岗位
