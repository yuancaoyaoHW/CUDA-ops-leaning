# 行为面试故事库（STAR 格式）

基于简历事实构造，适用于 "Tell me about a time when..." 类问题。

---

## 故事 1：技术难题攻克 — EAGLE-3 Proposer 在 Ascend NPU 上的适配

**适用问题**：遇到过最难的技术挑战？如何解决一个没有先例的问题？

**Situation**：
vLLM-Ascend 社区需要在昇腾 910B NPU 上实现 EAGLE-3 推测解码，但 EAGLE-3 的 proposer 逻辑依赖 CUDA-specific 的 KV cache 管理和 hidden states 传递机制，NPU 上没有现成实现可参考。

**Task**：
作为该功能的主要开发者，我需要将 EAGLE-3 proposer 完整移植到 Ascend NPU，打通 draft model → hidden states → KV cache → rejection sampler → runner 的完整执行链路。

**Action**：
1. 深入阅读 vLLM 主仓 EAGLE-3 的 GPU 实现，理解 proposer 接口、KV cache 分配策略和 token acceptance 逻辑
2. 分析 Ascend NPU 的算子能力和内存管理差异，设计适配方案
3. 逐模块实现：先打通 draft model 前向推理，再实现 KV cache 的分配与复用，最后对接 rejection sampler
4. 在 MT-Bench 前 80 轮上进行端到端验证，用 num_spec_tokens=2/3 对比吞吐和 acceptance rate
5. 根据社区 reviewer 反馈迭代：调整 proposer 接口、收敛 runner 执行链路、补充单元测试

**Result**：
- PR #1032 成功合入 vLLM-Ascend 主仓
- Ascend 910B 上 num_spec_tokens=2 时 mean acceptance length 达 1.63，token-1/token-2 接受率 70%/47%
- spec=3 时输出吞吐从 9.22 tok/s 提升至 14.30 tok/s（+55%），TPOT 从 108ms 降至 65ms

**加分要点**：展示了从零到一的系统级实现能力、跨硬件平台适配经验、开源协作能力。

---

## 故事 2：开源协作与 Code Review — 从本地验证到 PR 合入

**适用问题**：如何处理 code review 中的分歧？如何与远程团队协作？

**Situation**：
我提交的 vLLM-Ascend PR 在 review 过程中收到了多轮反馈，涉及接口设计、测试覆盖、代码风格和与主仓的兼容性问题。

**Task**：
需要在保证功能正确性的前提下，按社区规范完成所有修改，推动 PR 从 draft 状态到最终合入。

**Action**：
1. 逐条分析 reviewer 的反馈，区分"必须修改"和"建议优化"
2. 对于接口适配问题：重新设计 proposer 的抽象层，使其与 vLLM V1 的 worker 接口对齐
3. 对于测试覆盖：补充了 proposer 初始化、KV cache 分配、token acceptance 的单元测试
4. 对于冲突处理：在主仓频繁更新期间，多次 rebase 并解决合并冲突
5. 主动在 PR 中说明设计决策的 trade-off，减少后续 review 轮次

**Result**：
- 经过 3 轮 review 后 PR 成功合入
- 建立了与社区 maintainer 的良好协作关系
- 后续 vLLM-MindSpore PR #1020 的 review 流程更顺畅

**加分要点**：展示了沟通能力、对代码质量的重视、持续改进的态度。

---

## 故事 3：独立负责项目 — 非结构化问数系统 RAG 后端

**适用问题**：独立负责过什么项目？如何从零搭建一个系统？

**Situation**：
团队需要一个非结构化文档问答系统，支持快速问答和深度 Research 两种模式，但没有现成的后端架构，需要从零设计和实现。

**Task**：
作为唯一的后端开发者，独立负责从文档解析、chunk 构建、索引入库、向量检索、上下文组装到答案生成的完整 RAG 链路。

**Action**：
1. 技术选型：基于 Python + LangChain 构建，选择适合非结构化文档的 chunk 策略
2. 架构设计：设计快速问答（单次检索+生成）和 Research 问答（多轮检索+总结）双链路
3. 抽象层设计：封装统一的 retriever 接口，支持 metadata filters（文档来源、业务字段、时间条件），为不同向量检索后端预留扩展
4. 质量保障：使用 RAGAS 构建评测流程，围绕检索命中率和答案准确性进行系统评估
5. 工程完善：完善日志定位、失败分支处理和结果回写逻辑

**Result**：
- 系统上线运行，RAGAS 评测问答准确率达到 90%
- Research 链路精简后检索效率提升，总结模块重构后稳定性显著改善
- 为后续接入新的向量检索后端提供了清晰的扩展点

**加分要点**：展示了系统设计能力、独立交付能力、质量意识。

---

## 故事 4：性能优化 — 推测解码吞吐提升 55%

**适用问题**：做过什么性能优化？如何定位和解决性能瓶颈？

**Situation**：
在 Atlas 3000/310P3 上部署 Qwen2.5-7B 模型，baseline 的输出吞吐仅 9.22 tok/s，TPOT 为 108ms，无法满足在线服务的延迟要求。

**Task**：
通过推测解码技术提升推理吞吐，同时保证输出质量不下降。

**Action**：
1. 分析瓶颈：decode 阶段是 memory-bound，单 token 生成的计算利用率低
2. 方案选择：采用 EAGLE-3 推测解码，用小的 draft model 并行生成多个候选 token
3. 参数调优：在 MT-Bench 前 80 轮上系统测试 num_spec_tokens=1/2/3 的效果
4. 端到端验证：对比不同配置下的吞吐、TPOT、acceptance rate，确认最优配置
5. 质量验证：确认推测解码不影响输出分布（rejection sampling 保证精确性）

**Result**：
- spec=3 时输出吞吐从 9.22 tok/s 提升至 14.30 tok/s（+55%）
- TPOT 从 108.18ms 降至 65.76ms（-39%）
- num_spec_tokens=2 时 mean acceptance length 1.63，token-1 接受率 70%

**加分要点**：展示了性能分析方法论、数据驱动决策、端到端验证习惯。

---

## 故事 5：问题定位与调试 — Research 链路状态流转 Bug

**适用问题**：调试过最复杂的 bug？如何在复杂系统中定位问题？

**Situation**：
RAG 系统的 Research 问答链路在特定场景下会返回空结果或重复内容，用户反馈问答质量不稳定。

**Task**：
定位 Research 链路中的状态流转问题，修复异常处理和结果汇总逻辑。

**Action**：
1. 复现问题：构造触发空结果和重复内容的测试用例
2. 日志分析：在检索、总结、状态流转各环节添加详细日志，追踪数据流
3. 根因定位：发现问题出在多轮检索的状态管理——当某一轮检索失败时，状态未正确回退，导致后续轮次使用了错误的上下文
4. 修复方案：重构状态流转逻辑，为每个阶段添加明确的成功/失败分支处理，失败时正确回退状态并记录原因
5. 防御性改进：完善结果汇总的去重逻辑，添加异常情况的兜底返回

**Result**：
- 修复后 Research 链路稳定性显著提升，空结果和重复内容问题消除
- 日志改进降低了后续问题的排查成本
- 建立了 Research 链路的回归测试用例集

**加分要点**：展示了系统性调试方法、防御性编程意识、从 bug 中改进流程的习惯。

---

## 故事 6：跨团队沟通 — 推动 vLLM-MindSpore 适配方案对齐

**适用问题**：如何与不同团队协调？如何推动跨团队的技术决策？

**Situation**：
vLLM-MindSpore 的 EAGLE-3 适配需要同时考虑 MindSpore 框架的算子约束和 vLLM 社区的接口规范，两边的设计理念存在差异。

**Task**：
在 MindSpore 框架团队和 vLLM 社区之间找到技术方案的平衡点，确保 PR 既能通过社区 review 又能在 MindSpore 上正确运行。

**Action**：
1. 梳理两边的约束：vLLM 社区要求接口与主仓对齐，MindSpore 的 KV cache 管理和 hidden states 传递有框架特定的实现方式
2. 设计适配层：在不修改 vLLM 核心接口的前提下，通过适配层桥接 MindSpore 的算子调用
3. 主动沟通：在 PR 描述中详细说明设计决策和 trade-off，提前回答可能的 review 问题
4. 为 Qwen2/Qwen2.5 两个模型系列分别验证，确保适配方案的通用性

**Result**：
- PR #1020 提交并进入 review 流程
- 打通了 draft model、hidden states、KV cache、rejection sampler 与 runner 的完整执行链路
- 适配方案被认可为后续其他模型接入的参考模板

**加分要点**：展示了技术沟通能力、在约束条件下做设计决策的能力、文档化习惯。
