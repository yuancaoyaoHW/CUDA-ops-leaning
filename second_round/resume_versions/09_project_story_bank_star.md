# 面试项目故事库（STAR 结构）

> 每个故事包含：Situation, Task, Action, Result, Technical Depth, Tradeoff, Failure/Limitation, Interviewer Follow-up
> 覆盖不同项目和能力维度

---

## 故事 1：从零实现推测解码 — 系统级技术挑战

**能力维度**：技术攻坚、系统设计、独立解决问题

### STAR

**Situation**：
vLLM-Ascend 社区需要在昇腾 NPU 上支持推测解码以提升推理吞吐。EAGLE-3 的 proposer 逻辑依赖 CUDA-specific 的 KV cache 管理和 hidden states 传递机制，NPU 上没有现成实现可参考，需要从零设计适配方案。

**Task**：
作为该功能的主要开发者，独立实现 EAGLE-3 proposer，打通 draft model → hidden states → KV cache → rejection sampler → runner 的完整执行链路，并通过社区 code review 合入主线。

**Action**：
1. 深入阅读 vLLM 主仓 EAGLE-3 的 GPU 实现，理解 proposer 接口设计、KV cache 分配策略和 token acceptance 逻辑
2. 分析 Ascend NPU 的算子能力和内存管理与 GPU 的差异，确定需要适配的关键点
3. 逐模块实现：先打通 draft model 前向推理，再实现 KV cache 的分配与复用，最后对接 rejection sampler
4. 在 MT-Bench 前 80 轮上进行端到端验证，系统测试 num_spec_tokens=1/2/3 的效果
5. 根据社区 reviewer 反馈迭代：调整 proposer 接口、收敛 runner 执行链路、补充单元测试

**Result**：
- PR #1032 成功合入 vLLM-Ascend 主仓
- spec=3 时输出吞吐从 9.22 tok/s 提升至 14.30 tok/s（+55%），TPOT 从 108ms 降至 66ms（-39%）
- Ascend 910B 上 num_spec_tokens=2 时 mean acceptance length 达 1.63

### Technical Depth

- vLLM V1 架构中 scheduler 负责 batch 调度，model runner 执行前向推理，KV cache manager 管理显存分配
- 推测解码的核心挑战：draft model 生成的 token 需要 target model 验证，rejected token 的 KV cache 需要正确回收
- Hidden states 传递：EAGLE-3 用 target model 的 hidden states 作为 draft model 的输入，需要在两个 model 之间正确传递中间状态
- Rejection sampling 保证输出分布与 target model 一致，不影响生成质量

### Tradeoff

- num_spec_tokens 的选择：tokens 越多，单次验证的计算开销越大，但如果 acceptance rate 高则吞吐提升越大。实测 spec=3 时吞吐最优，spec=4 时因 acceptance rate 下降反而不如 spec=3
- Draft model 大小：更大的 draft model acceptance rate 更高，但推理开销也更大，需要找到平衡点

### Failure/Limitation

- 初始实现中 KV cache 的分配逻辑有 bug，导致 rejected token 的 cache 未正确释放，造成内存泄漏。通过添加详细日志定位到问题
- Acceptance rate（1.63/2）不算特别高，说明 draft model 的预测质量还有提升空间
- 仅在单卡上验证，未测试多卡场景下推测解码的表现

### Interviewer Follow-up

- Q: 为什么选择 EAGLE-3 而不是 Medusa 或其他方案？
  - A: EAGLE-3 用 target model 的 hidden states 作为 draft model 输入，比独立 draft model 的方案 acceptance rate 更高，且不需要额外训练 Medusa heads
- Q: KV cache 在推测解码中如何管理？rejected token 的 cache 怎么处理？
  - A: 为 draft tokens 预分配 KV cache slots，验证后 accepted tokens 的 cache 保留，rejected tokens 的 slots 标记为可回收
- Q: NPU 适配和 GPU 实现的主要差异是什么？
  - A: 主要差异在算子层面——NPU 的 attention 算子接口和内存管理 API 与 CUDA 不同，需要通过适配层桥接

---

## 故事 2：开源协作与 Code Review — 推动 PR 合入

**能力维度**：沟通协作、代码质量、持续改进

### STAR

**Situation**：
提交的 vLLM-Ascend PR 在 review 过程中收到了多轮反馈，涉及接口设计、测试覆盖、代码风格和与主仓的兼容性问题。社区 maintainer 对代码质量要求严格。

**Task**：
在保证功能正确性的前提下，按社区规范完成所有修改，推动 PR 从 draft 状态到最终合入。

**Action**：
1. 逐条分析 reviewer 的反馈，区分"必须修改"和"建议优化"
2. 对于接口适配问题：重新设计 proposer 的抽象层，使其与 vLLM V1 的 worker 接口对齐
3. 对于测试覆盖：补充了 proposer 初始化、KV cache 分配、token acceptance 的单元测试
4. 对于冲突处理：在主仓频繁更新期间，多次 rebase 并解决合并冲突
5. 主动在 PR 中说明设计决策的 trade-off，减少后续 review 轮次

**Result**：
- PR 经过多轮 review 后成功合入
- 建立了与社区 maintainer 的良好协作关系
- 后续 vLLM-MindSpore PR #1020 的 review 流程更顺畅

### Technical Depth

- vLLM 社区的 CI 要求：代码风格检查、单元测试通过、与主仓接口兼容
- Proposer 接口设计需要考虑通用性——不仅支持 EAGLE-3，还要为其他推测解码方案预留扩展点
- Rebase 过程中主仓的 scheduler 逻辑有更新，需要理解新逻辑并调整适配

### Tradeoff

- 接口通用性 vs 实现简洁性：reviewer 要求 proposer 接口更通用，但这增加了实现复杂度。最终选择了适度抽象，满足当前需求同时预留扩展点
- 测试覆盖率 vs 开发速度：补充完整测试延长了 PR 周期，但提高了代码可维护性

### Failure/Limitation

- 初始提交的接口设计过于耦合 EAGLE-3 的具体实现，reviewer 指出需要更通用的抽象。这说明我在接口设计时对扩展性考虑不足
- 合并冲突处理耗时较多，说明对主仓的更新节奏把握不够

### Interviewer Follow-up

- Q: 如果 reviewer 的意见你不同意怎么办？
  - A: 先理解 reviewer 的出发点，如果是合理的架构考虑就接受修改；如果有不同看法，在 PR 中详细说明理由和 trade-off，用数据或代码示例支撑
- Q: 如何保证 rebase 后功能不被破坏？
  - A: 每次 rebase 后跑完整的单元测试和端到端验证，确认性能指标没有退化

---

## 故事 3：RAG 系统独立交付 — 从零到上线

**能力维度**：独立交付、系统设计、质量保障

### STAR

**Situation**：
团队需要一个非结构化文档问答系统，支持快速问答和深度 Research 两种模式。没有现成的后端架构，需要从零设计和实现。我是唯一的后端开发者。

**Task**：
独立负责从文档解析、chunk 构建、索引入库、向量检索、上下文组装到答案生成的完整 RAG 链路设计与实现。

**Action**：
1. 技术选型：基于 Python + LangChain 构建，选择适合非结构化文档的 chunk 策略
2. 架构设计：设计快速问答（单次检索+生成）和 Research 问答（多轮检索+总结）双链路
3. 抽象层设计：封装统一的 retriever 接口，支持 metadata filters，为不同向量检索后端预留扩展
4. 质量保障：使用 RAGAS 构建评测流程，围绕检索命中率和答案准确性进行系统评估
5. 工程完善：完善日志定位、失败分支处理和结果回写逻辑

**Result**：
- 系统上线运行，RAGAS 评测问答准确率达到 90%
- 双链路架构满足不同复杂度查询的需求
- 统一检索接口为后续接入新的向量检索后端提供了清晰的扩展点

### Technical Depth

- Chunk 策略选择：需要平衡检索精度（chunk 太大则噪声多）和上下文完整性（chunk 太小则信息不完整）
- Metadata filters：支持按文档来源、业务字段、时间条件过滤，减少无关检索结果
- RAGAS 评测：覆盖检索质量和生成质量两个维度，用数据驱动优化方向

### Tradeoff

- 快速问答 vs Research 模式：快速问答牺牲深度换取响应速度，Research 模式牺牲速度换取答案质量。通过路由策略让用户选择
- 接口抽象程度：过度抽象增加开发成本，不够抽象则后续切换后端困难。选择了适度抽象——统一 retriever 接口但不抽象 embedding 层

### Failure/Limitation

- Research 链路初期存在状态流转 bug，某些场景下返回空结果。根因是多轮检索的状态管理不完善
- 未做负载测试，不清楚系统在高并发下的表现
- 未实现 reranking，检索质量还有提升空间

### Interviewer Follow-up

- Q: chunk 切分策略怎么选择的？
  - A: 根据文档类型选择——结构化文档按段落切分，非结构化文档用 recursive character splitter，设置 overlap 保证上下文连续性
- Q: 如何处理检索结果不相关的情况？
  - A: 通过 metadata filters 缩小检索范围，同时在生成阶段让 LLM 判断检索结果是否足以回答问题，不足时返回"无法回答"而非强行生成
- Q: 90% 准确率是怎么算的？
  - A: 基于 RAGAS 框架，用标注好的 QA 对作为 ground truth，评估生成答案与标准答案的语义一致性

---

## 故事 4：性能调优方法论 — 推测解码参数优化

**能力维度**：性能优化、数据驱动决策、实验设计

### STAR

**Situation**：
在 Atlas 310P3 上部署 Qwen2.5-7B 模型，baseline 的输出吞吐仅 9.22 tok/s，TPOT 为 108ms。需要通过推测解码提升性能，但最优配置未知。

**Task**：
设计系统的 benchmark 方案，找到推测解码的最优参数配置，在保证输出质量的前提下最大化吞吐提升。

**Action**：
1. 分析瓶颈：decode 阶段是 memory-bound，单 token 生成的计算利用率低，推测解码通过并行验证多个 token 来提升利用率
2. 设计 benchmark：基于 MT-Bench 前 80 轮，覆盖不同对话长度和复杂度
3. 参数扫描：系统测试 num_spec_tokens=1/2/3/4，记录每个配置下的吞吐、TPOT、acceptance rate
4. 分析结果：spec=3 时吞吐最优，spec=4 时因 acceptance rate 下降导致额外计算开销超过收益
5. 质量验证：确认 rejection sampling 保证输出分布不变

**Result**：
- 确定最优配置 spec_tokens=3：吞吐 +55%，TPOT -39%
- 建立了完整的 benchmark 方法论：指标定义 → 实验设计 → 数据收集 → 分析决策
- num_spec_tokens=2 时 acceptance length 1.63，为后续优化提供了 baseline

### Technical Depth

- Memory-bound 分析：decode 阶段每次只生成一个 token，但需要加载整个 KV cache，计算/访存比很低
- Acceptance rate 与 spec_tokens 的关系：tokens 越多，后面的 token 被 reject 的概率越高（累积效应）
- Rejection sampling 原理：用 target model 的概率分布修正 draft model 的输出，保证最终分布精确

### Tradeoff

- spec_tokens 数量：更多 tokens 意味着更高的潜在加速比，但也意味着更多的 draft model 计算开销和更低的 acceptance rate
- Benchmark 覆盖度 vs 时间成本：MT-Bench 前 80 轮是一个折中——覆盖了多种对话场景，但不是全量测试

### Failure/Limitation

- 仅在单一模型（Qwen2.5-7B）上验证，不同模型的最优 spec_tokens 可能不同
- 未测试不同 batch size 下的表现——batch size 增大时推测解码的收益可能下降
- 未做 ablation study 分析 draft model 质量对 acceptance rate 的影响

### Interviewer Follow-up

- Q: 为什么 spec=4 反而不如 spec=3？
  - A: token-3 和 token-4 的 acceptance rate 显著下降，额外的 draft model 计算开销超过了偶尔多 accept 一个 token 的收益
- Q: 如果要在不同模型上部署，怎么确定最优 spec_tokens？
  - A: 需要对每个模型做类似的参数扫描，或者实现 adaptive draft length——根据运行时 acceptance rate 动态调整

---

## 故事 5：Bug 定位与系统稳定性 — Research 链路状态流转问题

**能力维度**：调试能力、系统性思维、防御性编程

### STAR

**Situation**：
RAG 系统的 Research 问答链路在特定场景下会返回空结果或重复内容，用户反馈问答质量不稳定。问题难以复现，因为只在特定的多轮检索场景下触发。

**Task**：
定位 Research 链路中的状态流转问题，修复异常处理逻辑，确保系统稳定性。

**Action**：
1. 复现问题：构造触发空结果和重复内容的测试用例，确定触发条件
2. 日志分析：在检索、总结、状态流转各环节添加详细日志，追踪数据流
3. 根因定位：发现问题出在多轮检索的状态管理——当某一轮检索失败时，状态未正确回退，导致后续轮次使用了错误的上下文
4. 修复方案：重构状态流转逻辑，为每个阶段添加明确的成功/失败分支处理，失败时正确回退状态并记录原因
5. 防御性改进：完善结果汇总的去重逻辑，添加异常情况的兜底返回

**Result**：
- 修复后空结果和重复内容问题消除
- 日志改进降低了后续问题的排查成本
- 建立了 Research 链路的回归测试用例集

### Technical Depth

- 状态机设计：Research 链路本质是一个多步状态机，每步的输出是下一步的输入。状态回退需要保证幂等性
- 去重逻辑：多轮检索可能返回重复文档，需要在上下文组装阶段去重，同时保留最相关的版本

### Tradeoff

- 日志详细度 vs 性能：详细日志有助于调试但增加 I/O 开销。选择在关键节点记录结构化日志，非关键路径用 debug level
- 兜底策略：当检索完全失败时，是返回"无法回答"还是用已有部分结果生成？选择了返回部分结果 + 标注置信度

### Failure/Limitation

- 初始排查方向错误——以为是检索层的问题，花了时间排查向量检索，后来才发现是状态管理的问题
- 修复后未做压力测试，不确定高并发下状态管理是否仍然正确

### Interviewer Follow-up

- Q: 你是怎么确定问题在状态管理而不是检索层的？
  - A: 通过日志发现检索层每次都返回了结果，但在状态流转后结果被丢弃了。对比正常和异常 case 的日志，定位到状态回退逻辑的缺失
- Q: 如何防止类似问题再次发生？
  - A: 建立了回归测试覆盖各种异常场景（检索失败、部分失败、超时），并在状态流转的每个节点添加了断言检查

---

## 故事 6：跨框架适配 — vLLM-MindSpore EAGLE-3 实现

**能力维度**：跨团队协调、约束条件下的设计、技术文档

### STAR

**Situation**：
vLLM-MindSpore 的 EAGLE-3 适配需要同时考虑 MindSpore 框架的算子约束和 vLLM 社区的接口规范。两边的设计理念存在差异——MindSpore 的 KV cache 管理和 hidden states 传递有框架特定的实现方式，而 vLLM 社区要求接口与主仓对齐。

**Task**：
在两套约束之间找到技术方案的平衡点，确保 PR 既能通过社区 review 又能在 MindSpore 上正确运行。

**Action**：
1. 梳理两边的约束：vLLM 社区要求 proposer 接口与主仓对齐，MindSpore 的算子调用方式和内存管理 API 不同
2. 设计适配层：在不修改 vLLM 核心接口的前提下，通过适配层桥接 MindSpore 的算子调用
3. 主动沟通：在 PR 描述中详细说明设计决策和 trade-off，提前回答可能的 review 问题
4. 为 Qwen2/Qwen2.5 两个模型系列分别验证，确保适配方案的通用性

**Result**：
- PR #1020 提交并进入 review 流程
- 打通了完整的推测解码执行链路
- 适配方案为后续模型接入提供了参考

### Technical Depth

- 适配层设计：将 MindSpore 特定的算子调用封装在适配层内，对外暴露与 vLLM 主仓一致的接口
- KV cache 差异：MindSpore 的 tensor 管理方式与 PyTorch 不同，需要在适配层做格式转换
- 模型通用性：Qwen2 和 Qwen2.5 的 attention 实现有细微差异，适配方案需要处理这些差异

### Tradeoff

- 适配层厚度：适配层越薄，性能越好但对 MindSpore 变更越敏感；适配层越厚，隔离性越好但引入额外开销
- 接口对齐程度：完全对齐 vLLM 主仓接口可能需要在 MindSpore 侧做较多 workaround

### Failure/Limitation

- PR #1020 目前仍在 review 中，尚未合入——说明方案可能还需要进一步迭代
- 仅适配了 Qwen 系列，未验证其他模型架构（如 LLaMA）的通用性
- 未做性能对比——不清楚适配层引入了多少额外开销

### Interviewer Follow-up

- Q: 如果两边的要求冲突怎么办？
  - A: 优先满足 vLLM 社区的接口规范（因为这是合入的前提），在 MindSpore 侧通过适配层解决差异。如果适配层无法解决，则与 MindSpore 团队沟通是否可以调整底层实现
- Q: PR 还没合入，你觉得可能的原因是什么？
  - A: 可能是接口设计还需要进一步通用化，或者测试覆盖不够完整。我在持续关注 review 反馈并准备迭代

---

## 故事使用指南

| 面试问题类型 | 推荐故事 |
|-------------|----------|
| 最难的技术挑战 | 故事 1 |
| 如何处理 code review / 团队协作 | 故事 2 |
| 独立负责的项目 / 从零搭建系统 | 故事 3 |
| 性能优化经验 | 故事 4 |
| 调试复杂 bug | 故事 5 |
| 跨团队沟通 / 约束条件下的设计 | 故事 6 |
| 失败经历 / 从错误中学习 | 故事 1（KV cache bug）或故事 5（排查方向错误） |
| 时间压力下的决策 | 故事 2（rebase 冲突）或故事 5（用户反馈紧急） |
