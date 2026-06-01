# 模拟面试 4：45 分钟 RAG Infrastructure 设计

## 面试官 Profile

- **公司**：某企业搜索/知识库公司（对标 Glean / Notion AI / Cohere）
- **级别**：Tech Lead，前 Google Search Quality 团队，专注 retrieval at scale
- **风格**：注重工程细节和数据驱动决策。会追问具体数字、具体 failure case、具体优化过程。
- **偏好**：喜欢候选人能从用户体验出发讨论技术决策，而不是纯技术视角。

---

## Opening Question

> "介绍一下你独立负责的 RAG 系统。从架构开始，然后我会深入追问每个组件的设计决策。特别想听：你遇到的最大技术挑战是什么？你是怎么解决的？"

---

## Candidate Expected Answer（基于候选人真实经验）

"我独立负责了一个非结构化文档问答系统的 RAG 后端，支持快速问答和深度 Research 两种模式。

**架构概览**：
```
用户 Query → Query 理解/改写
           → Retriever（向量检索 + metadata filter）
           → Context Assembly（排序 + 截断）
           → LLM Generation（流式输出）
           → 结果回写 + 日志
```

**技术选型**：
- 框架：Python + LangChain
- Embedding：选择了适合中文的 embedding 模型
- 向量存储：支持 metadata filter 的向量数据库
- LLM：支持多种后端切换

**两种模式的差异**：
- 快速问答：单次检索 → top-k 结果 → 直接生成答案
- Research 模式：多轮检索 → 每轮根据中间结果调整 query → 汇总多轮结果 → 生成综合答案

**最大技术挑战**：Research 链路的状态流转 bug。多轮检索中，当某一轮检索失败时，状态未正确回退，导致后续轮次使用错误上下文，产生空结果或重复内容。

**解决过程**：
1. 构造触发问题的测试用例
2. 在各环节添加详细日志追踪数据流
3. 定位到状态管理的回退逻辑缺失
4. 重构状态流转，为每个阶段添加成功/失败分支
5. 添加结果去重和兜底返回

**量化成果**：RAGAS 评测准确率达到 90%。"

---

## Weak Answer（常见错误）

"我用 LangChain 搭了一个 RAG 系统，调用 embedding API 把文档向量化，存到向量数据库里，然后检索 top-5 拼到 prompt 里让 LLM 回答。RAGAS 评测 90%。"

**为什么弱**：
- 没有架构细节
- 没有设计决策的 why
- 没有讨论 failure case
- 没有规模数据（多少文档、QPS、latency）
- 听起来像 tutorial 项目而非 production 系统

---

## Strong Answer（工程深度 + 数据 + tradeoff）

"让我从系统全貌开始，然后深入每个关键决策：

**1. 系统规模与约束**
- 文档量：数千篇非结构化文档（PDF、Word、网页）
- Chunk 数量：约数万个 chunk
- 用户并发：内部使用，约 10-20 并发
- 延迟要求：快速问答 < 5s，Research < 30s
- 准确率目标：RAGAS 综合 > 85%

**2. Document Ingestion Pipeline**
```
原始文档 → 格式解析（PDF/Word/HTML）
         → 内容清洗（去除 header/footer/页码）
         → Chunk 切分（递归分割 + overlap）
         → Metadata 提取（来源、时间、业务字段）
         → Embedding 生成
         → 向量入库
```

Chunk 策略决策：
- 选择递归分割：先按段落，段落过长再按句子
- Chunk size：约 512 tokens，基于实验确定
- Overlap：10-15%，保证边界信息不丢失
- 特殊处理：表格保持完整不切分，代码块按函数切分

**3. Retrieval 设计**
- 基础检索：向量相似度 top-k（k=10-20）
- Metadata filter：支持按文档来源、业务字段、时间范围过滤
- 抽象层设计：统一 retriever 接口，支持切换不同向量后端

为什么没用 hybrid search（BM25 + vector）：
- 当时文档量不大，纯向量检索的 recall 已经足够
- BM25 对中文需要分词，增加复杂度
- 如果 recall 不够再加 BM25 作为 fallback

**4. Research 模式设计**
```
Query → 初始检索 → 分析结果是否充分
     → 如果不充分：生成 follow-up query → 再次检索
     → 重复 2-3 轮
     → 汇总所有检索结果 → 去重 → 生成综合答案
```

状态管理：
- 每轮维护：当前 query、已检索结果、已覆盖的信息点
- 失败处理：某轮失败时回退到上一轮状态，记录失败原因
- 终止条件：信息充分 / 达到最大轮次 / 连续失败

**5. 评测体系**
- 使用 RAGAS 框架
- 核心指标：Faithfulness（答案是否基于检索内容）、Answer Relevancy（答案是否相关）
- 评测集：手工构造 + 自动生成，覆盖不同难度
- 迭代过程：发现 faithfulness 低 → 优化 prompt 约束；relevancy 低 → 优化检索策略

**6. 工程完善**
- 日志：每个环节记录输入输出，便于问题定位
- 错误处理：LLM 调用超时重试、检索为空时兜底
- 可观测性：记录每次查询的检索耗时、生成耗时、chunk 命中情况"

---

## Follow-up Chain（5 层递进追问）

### Follow-up 1：Chunk 策略深入
> "你说 chunk size 512 tokens 是基于实验确定的。具体做了什么实验？256 和 1024 的效果差多少？"

**期望回答**：
- 实验方法：固定评测集，分别用 256/512/1024 tokens 的 chunk size，对比 retrieval recall 和最终答案质量
- 256 tokens：recall 高（更精确匹配），但 context 不完整，答案经常缺少关键信息
- 1024 tokens：context 完整，但噪声多，faithfulness 下降（LLM 容易被无关信息干扰）
- 512 tokens：平衡点，recall 和 faithfulness 都在可接受范围
- **候选人诚实标注**：实际上我的实验不够系统化，主要是基于几十个 case 的定性观察，没有做严格的 ablation study

### Follow-up 2：检索质量诊断
> "RAGAS 90% 准确率。那 10% 的失败 case 是什么类型？你分析过吗？"

**期望回答**：
- 失败类型分类：
  1. 检索未命中（约 40% 的失败）：query 和 document 的表述差异大
  2. 检索命中但 context 不足（约 30%）：答案需要跨多个 chunk 的信息
  3. LLM 幻觉（约 20%）：检索到了正确信息但 LLM 生成了错误答案
  4. 问题超出文档范围（约 10%）：文档中没有相关信息
- 针对性优化：
  - 类型 1：考虑加 query expansion 或 hybrid search
  - 类型 2：增加 top-k 或优化 chunk overlap
  - 类型 3：优化 prompt，增加"仅基于提供的信息回答"的约束
  - 类型 4：添加"无法回答"的判断逻辑

### Follow-up 3：规模化挑战
> "如果文档从几千篇增长到 100 万篇，你的架构需要怎么改？"

**期望回答**：
- 向量索引：从 brute-force 切换到 ANN（HNSW 或 IVF）
  - HNSW：recall 95%+ 但内存占用大（每个向量需要额外存储图结构）
  - IVF-PQ：内存效率高但 recall 可能降到 90%
  - 选择依据：如果内存够用 HNSW，否则 IVF-PQ + reranking 补偿
- 分片策略：按 tenant 或 document category 分片，减少单次检索的搜索空间
- Embedding 计算：需要 GPU 加速，batch processing pipeline
- 增量更新：新文档实时入库 vs 定期全量重建
  - HNSW 支持增量插入但性能会退化，需要定期 rebuild
- 缓存：热门 query 的检索结果缓存，减少重复计算
- **候选人诚实标注**：我没有处理过百万级文档的经验，以上是基于对向量数据库的理解推测的

### Follow-up 4：Embedding 模型选择
> "你的 embedding 模型是怎么选的？评估过几个？在你的 domain 上 fine-tune 过吗？"

**期望回答**：
- 选择过程：对比了几个主流中文 embedding 模型
- 评估方法：用标注的 query-document pair 计算 recall@k
- 没有 fine-tune：
  - 原因：数据量不够大，fine-tune 可能 overfit
  - 如果要 fine-tune：需要构造 hard negative，用 contrastive learning
  - 预期收益：domain-specific fine-tune 通常能提升 5-15% recall
- Trade-off：更大的 embedding 模型 recall 更高，但推理更慢、存储更大
  - 768 维 vs 1024 维：recall 差 2-3%，但存储和计算差 33%

### Follow-up 5：实时性与一致性
> "文档更新后，用户多久能搜到新内容？你怎么保证一致性？"

**期望回答**：
- 当前实现：文档更新后触发重新 embedding + 入库，延迟取决于 pipeline 处理时间
- 一致性问题：
  - 旧版本 chunk 需要删除，新版本 chunk 需要插入
  - 如果中间有查询，可能看到部分新部分旧的结果
- 解决方案：
  - 方案 A：版本化——新旧版本共存，切换时原子替换
  - 方案 B：标记删除——先标记旧 chunk 为 deleted，插入新 chunk，最后清理
  - 方案 C：双写——写入新索引，切换流量，删除旧索引
- **候选人诚实标注**：我的系统规模小，没有严格的一致性保证，更新时直接删旧插新

---

## Pressure Follow-up（故意挑战候选人）

> "你说 RAGAS 准确率 90%。但你的测试集有多少条？是你自己构造的还是真实用户 query？如果只有 50 条测试数据，90% 的置信区间是多少？这个数字有统计意义吗？"

**期望应对**：
- 承认测试集规模有限：确实不是大规模评测
- 统计分析：50 条数据，90% 准确率，95% 置信区间约 [78%, 97%]（二项分布）
- 这意味着真实准确率可能在 78%-97% 之间，确实不够精确
- 改进方向：
  1. 扩大测试集到 200+ 条
  2. 分层采样：不同难度、不同类型的 query 都要覆盖
  3. 加入 adversarial case（故意构造难的 query）
  4. 持续收集线上 bad case 加入测试集
- 诚实承认：这是我需要改进的地方，评测体系还不够严谨

---

## Debugging Scenario

> "用户反馈：'系统有时候回答得很好，有时候完全答非所问。同一个问题问两次，答案可能完全不同。' 你怎么排查？"

**排查思路**：

1. **复现问题**（5min）：
   - 收集用户的具体 query 和对应的 bad case
   - 自己重复查询，确认是否能复现不一致

2. **检查检索结果**（10min）：
   - 对同一 query 多次检索，看返回的 chunk 是否一致
   - 如果不一致：可能是向量数据库的 ANN 算法有随机性（如 HNSW 的 ef_search 参数）
   - 如果一致：问题在 LLM generation 阶段

3. **检查 LLM 输出**（5min）：
   - Temperature > 0 会导致输出不确定性
   - 检查是否设置了 temperature=0 for deterministic output
   - 检查 context window 是否溢出导致 truncation 不一致

4. **可能的 root cause**：
   - **最可能**：LLM temperature > 0，导致相同 context 生成不同答案
   - **次可能**：检索结果排序不稳定（相似度相近的 chunk 排序随机）
   - **较少见**：embedding 模型有 dropout（推理时应关闭）
   - **边缘情况**：并发请求导致 context assembly 的 race condition

5. **修复方案**：
   - 设置 temperature=0（或很低如 0.1）for factual QA
   - 检索结果加入 deterministic tie-breaking（如按 chunk ID 排序）
   - 添加 response caching：相同 query + 相同 context → 返回缓存结果
   - 日志记录每次的 retrieved chunks 和 final prompt，便于对比

---

## System Design Extension（扩展到更大规模）

> "现在要把你的 RAG 系统做成 SaaS 产品，支持 1000 个企业客户，每个客户有自己的文档库（1万-100万篇），总 QPS 10000。重新设计。"

**设计要点**：

1. **多租户架构**：
   - 逻辑隔离：每个 tenant 有独立的 namespace，检索时强制 filter
   - 物理隔离（大客户）：独立的向量索引实例
   - 混合方案：小客户共享集群 + metadata filter，大客户独立部署

2. **数据管道**：
   - 异步 ingestion：文档上传 → 消息队列 → worker 处理 → 入库
   - 处理能力：需要 GPU cluster 做 embedding（1000 客户 × 平均 10 万文档 = 1 亿文档）
   - 增量更新：CDC（Change Data Capture）监听文档变更，实时更新索引

3. **检索层**：
   - 分布式向量索引：按 tenant 分片，热门 tenant 多副本
   - 缓存层：Redis 缓存热门 query 的检索结果（TTL 基于文档更新频率）
   - 降级策略：向量检索超时时 fallback 到 BM25

4. **生成层**：
   - LLM serving cluster：共享 GPU pool，按 tenant 配额
   - 流式输出：WebSocket/SSE
   - 成本控制：per-tenant token quota，超额降级到小模型

5. **可观测性**：
   - Per-tenant metrics：QPS, latency, retrieval recall, answer quality
   - 自动质量监控：抽样评测 + 用户反馈收集
   - 告警：单 tenant latency spike, retrieval recall 下降, error rate 上升

6. **成本优化**：
   - Embedding 缓存：相同文档不重复计算
   - 向量压缩：PQ（Product Quantization）减少存储
   - LLM 调用优化：短 query 用小模型，复杂 query 用大模型
   - 预估成本：1 亿文档 × 768 维 × 4 bytes = 300GB 向量存储

---

## Hire/No-Hire Evaluation

| 维度 | 权重 | 评分标准 |
|------|------|---------|
| RAG 全链路理解 | 25% | 从 ingestion 到 generation 的完整理解 |
| 检索质量优化 | 20% | 能否诊断和改进 retrieval quality |
| 工程实践 | 20% | 代码质量、错误处理、可观测性 |
| 规模化思维 | 20% | 能否讨论 scaling 到百万级的方案 |
| 评测体系 | 15% | 评测方法论、数据驱动决策 |

---

## Scorecard

| 维度 | 候选人预期得分 | 说明 |
|------|--------------|------|
| RAG 全链路理解 | 7/10 | 有实际经验，能描述完整链路和设计决策 |
| 检索质量优化 | 5/10 | 有基本优化意识，但缺乏 hybrid search、reranking 经验 |
| 工程实践 | 6/10 | 有日志、错误处理，但缺乏 monitoring/alerting |
| 规模化思维 | 4/10 | 能讨论方向但缺乏大规模系统经验 |
| 评测体系 | 6/10 | 使用 RAGAS，有质量意识，但评测集规模和严谨度不足 |
| **总分** | **5.6/10** | **Lean Hire — 有实际经验但规模和深度有限** |

### 决策依据
- **Hire 信号**：独立负责过完整 RAG 系统，有端到端交付能力，有质量意识
- **No Hire 信号**：系统规模小，缺乏大规模检索优化经验，评测不够严谨
- **建议**：适合中小公司的 RAG 工程师岗位（Hire），大厂的 Search/Retrieval 岗位偏弱（Lean No Hire）
- **成长方向**：需要补强 hybrid search、reranking、大规模向量索引优化
