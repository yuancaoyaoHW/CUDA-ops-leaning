# 模拟面试 8：45 分钟 Behavioral + Project Deep Dive

## 面试官 Profile

- **公司**：某 AI 公司（对标 Databricks / Scale AI）
- **级别**：Hiring Manager，Engineering Director，管理 3 个团队共 30 人
- **风格**：温和但深入，会从一个简单问题开始不断追问细节。关注候选人的思考过程、决策能力和成长潜力。
- **偏好**：STAR 格式的回答，有具体数字和 tradeoff 分析。讨厌泛泛而谈。

---

## Opening Question

> "请用 3 分钟介绍你最有技术深度的一个项目。我想听到：问题是什么、你做了什么、结果如何、以及你学到了什么。"

---

## Part 1：EAGLE-3 Project Deep Dive（20 分钟）

### Q1：项目背景
> "为什么选择做 EAGLE-3？是你自己发起的还是被分配的？"

**Strong Answer**：
"这是我主动选择的方向。背景是 vLLM-Ascend 社区需要在昇腾 NPU 上支持推测解码来提升推理吞吐。我分析了几个选项：
- Medusa：需要额外训练 head，对 NPU 适配复杂度高
- EAGLE-2：tree-based，实现复杂度中等
- EAGLE-3：利用 target model 的 hidden states，acceptance rate 更高

我选择 EAGLE-3 是因为：1）它的 acceptance rate 最高（意味着 speedup 最大）；2）它的 proposer 逻辑相对独立，适合作为 plugin 接入 vLLM 架构；3）社区还没有人做这个，有 first-mover 的价值。"

**Weak Answer**：
"导师让我做推测解码，我就选了 EAGLE-3 因为它是最新的。"

---

### Q2：技术挑战
> "实现过程中最难的部分是什么？卡了多久？怎么解决的？"

**Strong Answer**：
"最难的是 KV cache 的跨模型管理。EAGLE-3 的 draft model 需要使用 target model 的 hidden states 作为输入，同时维护自己的 KV cache。在 NPU 上，内存管理和 GPU 不同——没有 CUDA 的 unified memory，需要显式管理数据搬运。

具体卡点：
1. Draft model 的 KV cache 分配时机——需要在 target model forward 之后、verification 之前分配
2. Rejected tokens 的 KV cache 回收——需要精确追踪哪些 block 可以释放
3. NPU 的 async 执行模型和 CUDA 不同，同步点的选择影响性能

卡了大约 2 周在 KV cache 管理上。解决方法：
1. 仔细阅读 vLLM 主仓的 GPU 实现，理解 block manager 的状态机
2. 在 NPU 上用简化版本先跑通（不做 block 复用），验证正确性
3. 逐步添加优化（block 复用、lazy allocation），每步都做正确性验证
4. 最终方案：为 draft model 维护独立的 block table，与 target model 共享物理 block pool"

---

### Q3：数据与结果
> "55% throughput 提升和 39% TPOT 降低。这些数字是怎么测的？测试条件是什么？你对这些数字有多大信心？"

**Strong Answer**：
"测试条件：
- 硬件：Atlas 3000/310P3（Ascend 910B NPU）
- 模型：Qwen2.5-7B + 对应的 EAGLE-3 draft model
- 数据集：MT-Bench 前 80 轮对话
- 配置：num_spec_tokens=3，batch_size=1

具体数字：
- Baseline（无推测解码）：throughput 9.22 tok/s，TPOT 108.18ms
- EAGLE-3 spec=3：throughput 14.30 tok/s（+55%），TPOT 65.76ms（-39%）
- EAGLE-3 spec=2：mean acceptance length 1.63，token-1 接受率 70%，token-2 接受率 47%

对数字的信心：
- 中等信心。MT-Bench 80 轮是有限的测试集，不同 prompt 的 acceptance rate 差异很大
- 局限性：batch_size=1 的测试，大 batch 下 speedup 会降低
- 没有测试不同模型大小（7B vs 13B vs 70B）的效果差异
- 没有做长时间稳定性测试（只跑了 benchmark，不是 24h stress test）"

---

### Q4：设计决策
> "如果让你重新做这个项目，有什么会做不同的？"

**Strong Answer**：
"三个方面：

1. **测试策略**：一开始就建立更完善的测试框架，而不是最后补。我应该先写 acceptance rate 的 unit test，再写 end-to-end throughput test，这样每次改动都能快速验证。

2. **性能分析**：应该更早做 profiling，理解 NPU 上的瓶颈在哪。我花了太多时间在功能正确性上，后来发现有些性能问题是因为不必要的同步点。

3. **Draft length 自适应**：当前实现用固定的 num_spec_tokens。更好的方案是根据 acceptance rate 动态调整——如果最近的 acceptance rate 高，增加 draft length；如果低，减少。这样可以在不同 prompt 上都接近最优。

另外，如果有更多时间，我会：
- 支持 tree-based speculation（而不是 chain），进一步提升 acceptance rate
- 做 batch_size > 1 的优化（当前主要优化了 batch=1 的场景）
- 与 prefix caching 集成（共享 prefix 的请求可以共享 draft model 的 KV cache）"

---

### Q5：协作与 Code Review
> "PR review 过程中收到了什么反馈？有没有你不同意的反馈？怎么处理的？"

**Strong Answer**：
"主要反馈分三类：

1. **接口设计**（必须修改）：
   - Reviewer 要求 proposer 接口与 vLLM V1 的 worker 接口对齐
   - 我最初的设计是独立的 proposer class，reviewer 认为应该作为 worker 的 plugin
   - 我同意这个反馈——统一接口降低了后续维护成本

2. **测试覆盖**（必须修改）：
   - Reviewer 要求补充 proposer 初始化、KV cache 分配的单元测试
   - 我最初只有 end-to-end test，确实不够
   - 补充后发现了一个 edge case bug（draft length > remaining KV cache blocks 时的处理）

3. **代码风格**（建议优化）：
   - 一些命名和注释的建议
   - 大部分我接受了，有一个我不同意：reviewer 建议把一个 50 行的函数拆成 3 个小函数，但我认为这个函数的逻辑是连贯的，拆开反而降低可读性
   - 我在 PR comment 中解释了我的理由，reviewer 最终接受了

处理原则：
- 对于正确性和架构问题：无条件接受
- 对于风格问题：如果有充分理由可以讨论，但不要 block PR
- 主动在 PR description 中说明设计决策，减少 review 轮次"

---

## Part 2：Behavioral Questions（15 分钟）

### Q6：Conflict Resolution
> "Tell me about a time when you disagreed with a technical decision. What happened?"

**Strong Answer（STAR）**：

**Situation**：在 vLLM-MindSpore 的适配中，MindSpore 框架团队建议直接修改 vLLM 核心接口来适配 MindSpore 的算子调用方式。

**Task**：我需要在不破坏 vLLM 社区兼容性的前提下完成适配。

**Action**：
1. 我分析了两种方案的 tradeoff：
   - 修改核心接口：MindSpore 适配简单，但会导致与 vLLM 主仓的 merge conflict
   - 添加适配层：多一层抽象，但保持与主仓兼容
2. 我写了一个 design doc 对比两种方案，量化了维护成本
3. 在 PR 中详细说明了选择适配层的理由
4. 与 MindSpore 团队沟通，解释为什么长期来看适配层更好

**Result**：最终采用了适配层方案。虽然初始实现多了约 100 行代码，但后续 vLLM 主仓更新时我们的 rebase 非常顺畅，验证了这个决策的正确性。

---

### Q7：Failure & Learning
> "Tell me about a time when something you built didn't work as expected. What did you learn?"

**Strong Answer（STAR）**：

**Situation**：RAG 系统的 Research 链路上线后，用户反馈"有时候回答很好，有时候完全答非所问"。

**Task**：定位并修复 Research 链路的不稳定性问题。

**Action**：
1. 收集 bad case，发现问题集中在多轮检索的场景
2. 添加详细日志追踪每轮的 query、检索结果、状态变化
3. 发现 root cause：当某轮检索返回空结果时，状态没有正确回退，后续轮次使用了错误的上下文
4. 重构状态管理：为每个阶段添加明确的成功/失败分支，失败时回退到上一个 valid state
5. 添加回归测试用例，覆盖各种失败场景

**Result**：修复后 Research 链路稳定性显著提升，空结果问题消除。

**Learning**：
1. 状态机设计要先画状态转移图，明确每个状态的进入和退出条件
2. 多轮/多步骤的系统，每一步都要有 fallback 策略
3. 上线前应该做 chaos testing（故意让某些步骤失败，验证系统行为）

---

### Q8：独立工作 vs 团队协作
> "你的 RAG 项目是独立负责的。你怎么确保自己没有走偏？没有 code review 的情况下怎么保证质量？"

**Strong Answer**：
"独立负责不意味着完全孤立：

1. **设计阶段**：和团队讨论了架构方案，确认技术选型方向
2. **开发阶段**：
   - 自己做 code review：写完代码后隔一天再看，用 fresh eyes 发现问题
   - 写测试先行：关键逻辑先写测试，再写实现
   - 小步提交：每个功能点独立提交，便于回退
3. **验证阶段**：
   - RAGAS 评测提供了客观的质量指标
   - 找同事做 user testing，收集使用反馈
4. **文档化**：
   - 记录设计决策和 tradeoff，便于后续维护
   - 写 README 和 API 文档，降低他人接手成本

反思：如果重来，我会更早引入 code review 机制——即使是独立项目，找一个人做 weekly review 也比完全自己看好。"

---

### Q9：Growth & Self-awareness
> "你觉得自己最大的技术短板是什么？你在怎么补？"

**Strong Answer**：
"三个主要短板，按优先级排序：

1. **CUDA kernel 开发**（最大 gap）：
   - 现状：零实战经验，只有理论知识
   - 补强计划：正在系统学习，已完成 vector add 和 reduction 的实现
   - 目标：3 个月内能独立写 GEMM 和 attention kernel
   - 为什么重要：LLM inference 的核心瓶颈在 kernel 层面，不懂 kernel 就无法做深度优化

2. **Production 运维经验**（第二大 gap）：
   - 现状：没有 on-call 经验，没有管理过大规模线上服务
   - 补强计划：学习 SRE 方法论，在下一份工作中主动参与 on-call rotation
   - 为什么重要：AI Infra 岗位需要保证服务可靠性

3. **分布式系统实战**（第三大 gap）：
   - 现状：理解 TP/PP 概念，但没有实际部署过多机推理
   - 补强计划：在学习 CUDA 的同时，用 NCCL 实现简单的 AllReduce
   - 为什么重要：大模型推理必然涉及分布式

我的学习方法：不只是看文档，而是动手实现。比如学 CUDA 不是只看 programming guide，而是实际写 kernel、用 NCU profiling、对比和 cuBLAS 的性能差距。"

---

### Q10：Why This Role
> "为什么想做 AI Infra？你觉得自己和其他候选人相比，独特的优势是什么？"

**Strong Answer**：
"为什么 AI Infra：
- 我喜欢系统级的优化工作——不是训练模型，而是让模型跑得更快、更便宜、更可靠
- 推测解码的工作让我体会到：一个好的系统优化可以在不改变模型的情况下提升 55% 性能
- AI Infra 是 AI 落地的关键瓶颈——模型再好，serving 不行也没用

我的独特优势：
1. **跨硬件视角**：我在 NPU 上做过推理优化，理解不同硬件的 tradeoff，这在异构计算时代是稀缺能力
2. **开源社区经验**：有 vLLM 生态的 PR 合入经验，理解大型开源项目的协作模式
3. **端到端能力**：从算法（EAGLE-3）到系统（RAG pipeline）都有实际交付经验
4. **学习速度**：从零开始理解 EAGLE-3 到 PR 合入，整个过程约 2 个月

我的劣势也很明确：CUDA 经验为零、production 经验有限。但我认为这些是可以通过时间和实践补强的，而系统思维和学习能力是更难培养的。"

---

## Pressure Follow-up（故意挑战候选人）

> "你说你的 PR 提升了 55% throughput。但这是在 NPU 上，用的是 7B 模型，batch=1。在实际 production 中（GPU, 70B, batch=32），这个优化可能完全没有收益。你怎么看？"

**期望应对**：
- 承认局限性：完全正确。我的 benchmark 条件和 production 差距很大
- 但不完全否定价值：
  1. 算法层面的理解是通用的——rejection sampling、KV cache 管理、draft model 设计
  2. 在 latency-sensitive 低并发场景（如 interactive chat），speculative decoding 仍然有价值
  3. 我的贡献是在 NPU 生态中从零到一打通了这个能力，这本身有工程价值
- 展示成熟度：
  - 我不会在面试中夸大这个数字的适用范围
  - 如果面试官问"这在 production 中有多大收益"，我会诚实说"需要在目标环境重新 benchmark"
  - 这也是我想加入 GPU 生态团队的原因——在更主流的硬件上验证和扩展这些优化

---

## Hire/No-Hire Evaluation

| 维度 | 权重 | 评分标准 |
|------|------|---------|
| 项目深度 | 25% | 对自己项目的掌控力，能否回答任何深度的追问 |
| 决策能力 | 20% | 技术决策的 reasoning，tradeoff 分析 |
| 自我认知 | 20% | 对自己优劣势的准确评估 |
| 成长潜力 | 20% | 学习能力、成长速度、正确的 mindset |
| 沟通能力 | 15% | STAR 格式、结构化表达、简洁有力 |

---

## Scorecard

| 维度 | 候选人预期得分 | 说明 |
|------|--------------|------|
| 项目深度 | 7/10 | 对 EAGLE-3 有深入理解，能回答大部分追问 |
| 决策能力 | 6/10 | 有 tradeoff 思维，但缺乏大规模系统的决策经验 |
| 自我认知 | 8/10 | 对自己的优劣势有准确认知，不夸大不回避 |
| 成长潜力 | 7/10 | 学习速度快，有正确的 mindset，有明确的成长计划 |
| 沟通能力 | 7/10 | 表达清晰，能用 STAR 格式，有具体数字 |
| **总分** | **7.0/10** | **Hire — 有潜力的 Senior Engineer** |

### 决策依据
- **Hire 信号**：诚实、有深度、有成长潜力、自我认知准确
- **担忧**：项目规模小、production 经验不足、CUDA gap
- **建议**：适合 Senior Engineer 岗位，需要 mentor 和 production 环境来加速成长
- **与技术面对比**：Behavioral 面是候选人最强的环节——诚实、有条理、有反思能力
