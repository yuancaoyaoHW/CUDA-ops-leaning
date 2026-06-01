# 模拟面试 6：60 分钟 Staff-level AI Infra 综合面

## 面试官 Profile

- **公司**：某大厂 AI Platform 部门（对标 Meta FAIR Infra / Google DeepMind Platform）
- **级别**：AI Infra Director，管理 50+ 人团队，负责从训练到推理的全栈基础设施
- **风格**：战略视角，关注 cross-cutting concerns。不会深入单一技术细节，而是考察全局观、优先级判断、技术领导力。
- **偏好**：喜欢候选人能从业务需求出发推导技术决策，而不是从技术出发找业务场景。

---

## Opening Question

> "假设你加入我们团队，负责一个新的 AI Infra 项目：为公司内部 50 个 AI 团队提供统一的模型推理平台。目前各团队各自部署，GPU 利用率平均只有 30%，成本每月 $2M。你的目标是：6 个月内将 GPU 利用率提升到 70%，成本降低 40%，同时不降低任何团队的 SLA。你怎么做？"

---

## Candidate Expected Answer（基于候选人真实经验）

"这是一个平台化的问题，需要从技术和组织两个维度来解决。

**第一步：现状分析（第 1-2 周）**
- 调研 50 个团队的使用模式：模型大小、QPS、latency 要求、GPU 型号
- 分析 GPU 利用率低的原因：
  - 独占部署但流量低（最可能）
  - 模型太小没用满 GPU memory
  - Peak/off-peak 差异大但没有 auto-scaling
- 量化机会：哪些团队可以合并？哪些可以用更小的 GPU？

**第二步：技术方案设计（第 3-6 周）**
- 统一推理平台：提供标准化的 serving API，各团队迁移上来
- Multi-model serving：多个小模型共享 GPU（MPS 或 time-sharing）
- Auto-scaling：根据流量自动扩缩容
- Spot instance：非关键服务使用 spot GPU

**第三步：迁移执行（第 2-5 月）**
- 分批迁移：先迁移低风险团队，验证平台稳定性
- 提供兼容层：保持 API 兼容，降低迁移成本
- SLA 保证：迁移前后 A/B 对比 latency 和 availability

**第四步：持续优化（第 5-6 月）**
- 监控 GPU 利用率趋势
- 识别新的优化机会（量化、模型蒸馏、batch 优化）
- 建立 cost attribution 机制，让各团队有成本意识

基于我的经验，我在 vLLM 上的工作让我理解推理服务的核心优化点（continuous batching、speculative decoding、KV cache 管理），这些是提升 GPU 利用率的关键技术。"

---

## Weak Answer（常见错误）

"把所有团队的模型都迁移到 vLLM 上，开 continuous batching，GPU 利用率就上去了。然后用 Kubernetes auto-scaling 处理流量波动。"

**为什么弱**：
- 忽略了组织维度（50 个团队的协调）
- 假设所有模型都适合 vLLM（有些可能是 CV 模型、小模型）
- 没有分阶段计划
- 没有风险评估
- 没有 cost attribution

---

## Strong Answer（工程深度 + 数据 + tradeoff）

"让我用一个结构化的方法来解决这个问题：

**Phase 0: Discovery & Measurement（Week 1-3）**

首先需要数据驱动的现状分析：

```
数据收集：
- 每个团队：model_name, model_size, gpu_type, gpu_count, avg_qps, peak_qps, 
  p99_latency, gpu_utilization_avg, gpu_utilization_peak
- 集群级：total_gpus, total_cost, utilization_distribution
```

预期发现（基于行业经验）：
- 30% 的 GPU 被 < 1 QPS 的服务占用（严重浪费）
- 50% 的服务 peak/off-peak 比 > 10x（需要 auto-scaling）
- 20% 的服务用了过大的 GPU（7B 模型占了 A100-80G）
- 多个团队部署了相同模型的不同版本

**Phase 1: Quick Wins（Week 3-8）— 目标：利用率 30% → 45%，成本 -15%**

1. **Right-sizing**：
   - 7B 模型从 A100-80G 迁移到 A100-40G 或 L4
   - 预期节省：20% 的 GPU 可以降级 → 成本 -10%

2. **Auto-scaling**：
   - 为所有服务添加基于 QPS 的 auto-scaling
   - Off-peak 缩容到 min replicas
   - 预期节省：peak/off-peak 比 > 5x 的服务，off-peak 释放 60% GPU

3. **Consolidation**：
   - 合并相同模型的不同部署（统一版本管理）
   - 多个 < 1 QPS 的服务合并到共享 GPU

**Phase 2: Platform Build（Week 6-16）— 目标：利用率 45% → 60%，成本 -30%**

4. **Multi-model GPU Sharing**：
   - 小模型（< 10GB）使用 NVIDIA MPS 共享 GPU
   - 中模型（10-40GB）使用 time-sharing（快速切换）
   - 大模型（> 40GB）独占 GPU 但用 continuous batching 提升利用率
   
   技术实现：
   ```
   GPU Sharing Controller:
   - 监控每个 model 的 memory usage 和 compute usage
   - Bin-packing: 将多个小模型 pack 到一张 GPU
   - 约束: 总 memory < GPU memory × 90%, 总 compute < 80%
   - 隔离: memory 硬隔离（OOM 不影响其他模型），compute 软隔离
   ```

5. **Unified Serving Layer**：
   ```
   Client → API Gateway (auth, rate limit, routing)
          → Model Router (model_id → backend)
          → Serving Backend (vLLM / TensorRT-LLM / Triton)
          → GPU Pool
   ```
   
   关键设计决策：
   - 不强制所有团队用同一个 serving framework
   - 提供标准 API（OpenAI compatible），后端可以是 vLLM/TRT-LLM/custom
   - 平台负责 GPU 分配和 scaling，团队负责模型和 API 逻辑

6. **Spot Instance Integration**：
   - 非 SLA-critical 服务（内部工具、batch processing）使用 spot GPU
   - Spot 被回收时自动 failover 到 on-demand
   - 预期节省：30% 的 workload 可以用 spot → 成本 -20% for those

**Phase 3: Advanced Optimization（Week 14-24）— 目标：利用率 60% → 70%，成本 -40%**

7. **Intelligent Scheduling**：
   - 预测性 scaling：基于历史流量模式预分配 GPU
   - Cross-team resource sharing：Team A off-peak 的 GPU 给 Team B 的 burst
   - Priority-based preemption：低优先级任务可被高优先级抢占

8. **Model Optimization as a Service**：
   - 提供自动量化服务：FP16 → INT8/FP8
   - 提供 speculative decoding 集成（我的 EAGLE-3 经验可以直接应用）
   - 提供 prefix caching for common system prompts
   - 预期收益：量化可以减少 50% GPU memory → 同样 GPU 服务更多请求

9. **Cost Attribution & Chargeback**：
   - 每个团队看到自己的 GPU 使用量和成本
   - 提供优化建议（"你的模型可以量化，预计节省 40%"）
   - 激励机制：节省的成本部分返还给团队

**组织维度**：

- **Stakeholder management**：
  - 与 50 个团队的 tech lead 沟通，理解他们的顾虑
  - 最大阻力：担心迁移影响 SLA、担心失去对 GPU 的控制
  - 解决：提供 SLA guarantee + 自助 dashboard + 紧急回退机制

- **Migration strategy**：
  - 先找 2-3 个 friendly team 做 pilot
  - 用 pilot 的成功案例说服其他团队
  - 提供 migration toolkit：一键迁移脚本 + 兼容性测试

- **Team structure**：
  - Platform team（我负责）：5-8 人，负责核心平台
  - Embedded SRE：每个大团队配 1 个 SRE 协助迁移
  - On-call rotation：7×24 on-call 保证平台可用性

**风险与 mitigation**：

| 风险 | 概率 | 影响 | Mitigation |
|------|------|------|-----------|
| 迁移导致 SLA 违反 | 中 | 高 | 灰度迁移 + 快速回退 |
| GPU sharing 导致 noisy neighbor | 中 | 中 | Memory 硬隔离 + compute quota |
| 团队抵触不愿迁移 | 高 | 中 | 先做 quick wins 建立信任 |
| Spot 回收导致服务中断 | 低 | 高 | 自动 failover + 只用于非关键服务 |

**成功指标**：
- GPU 利用率：30% → 70%（+133%）
- 月成本：$2M → $1.2M（-40%）
- 平台 availability：99.9%
- 迁移覆盖率：> 80% 的团队
- 用户满意度：NPS > 50"

---

## Follow-up Chain（5 层递进追问）

### Follow-up 1：优先级决策
> "你有 8 个人的团队，6 个月时间。Phase 1-3 中如果只能做一半，你砍哪些？为什么？"

**期望回答**：
- 保留（ROI 最高）：
  1. Auto-scaling（实现简单，收益大）
  2. Right-sizing（低风险，立即见效）
  3. Multi-model GPU sharing（核心平台能力）
  4. Cost attribution（改变团队行为）
- 砍掉（可以后做）：
  - Spot integration（需要 failover 机制，复杂度高）
  - Intelligent scheduling（需要大量数据积累）
  - Model optimization as a service（nice-to-have）
- 决策原则：先做低风险高收益的，建立信任后再做复杂的

### Follow-up 2：技术选型争议
> "Team A 坚持用 TensorRT-LLM（性能最好），Team B 坚持用 vLLM（灵活性好），Team C 用自研框架。你怎么处理？"

**期望回答**：
- 不强制统一 serving framework——这是平台的核心设计原则
- 平台提供的是 GPU 管理 + API gateway + monitoring，不是 serving engine
- 各团队可以选择自己的 serving backend，只要符合平台的 API 标准
- 但平台提供"推荐路径"：对于标准 LLM serving，推荐 vLLM（社区活跃、功能全面）
- 对于极致性能需求，支持 TensorRT-LLM
- 对于自研框架：提供 adapter 接口，但不保证所有平台功能都支持
- Trade-off：灵活性 vs 维护成本。支持多框架增加平台复杂度，但强制统一会导致团队抵触

### Follow-up 3：Reliability vs Cost
> "你说要用 spot instance 降低成本。但 spot 被回收时服务会中断。99.9% SLA 怎么保证？"

**期望回答**：
- Spot 只用于非 SLA-critical 服务（batch processing, internal tools, dev/staging）
- SLA-critical 服务用 on-demand + reserved instance
- 混合策略：
  - Baseline capacity：reserved instance（最便宜的长期承诺）
  - Normal load：on-demand
  - Burst：spot（可以接受偶尔中断）
- Spot 的 failover 机制：
  - 2 分钟 warning → graceful drain
  - 自动切换到 on-demand backup
  - 如果 backup 不够 → 降级（减少 batch size, 增加 latency）
- 数学：如果 spot 可用性 95%，加上 failover 后服务可用性 > 99.9%

### Follow-up 4：从推理到 RAG 到集群的全栈视角
> "你的平台不只是推理。有些团队需要 RAG pipeline（embedding + retrieval + generation），有些需要 fine-tuning。怎么扩展平台？"

**期望回答**：
- 平台分层：
  ```
  Layer 4: AI Applications (RAG, Agents, Fine-tuning)
  Layer 3: Model Serving (Inference API)
  Layer 2: Compute Orchestration (GPU scheduling, scaling)
  Layer 1: Infrastructure (GPU nodes, network, storage)
  ```
- 我的平台核心是 Layer 2-3，Layer 4 由各团队自己构建
- 但平台可以提供 Layer 4 的 building blocks：
  - RAG：提供 embedding serving + vector DB hosting
  - Fine-tuning：提供 training job scheduler + checkpoint management
  - Agents：提供 tool calling gateway + conversation state management
- 我的 RAG 经验可以直接应用：
  - 设计统一的 embedding serving（多模型共享 GPU）
  - 提供 RAG evaluation pipeline as a service
- 我的 speculative decoding 经验：
  - 作为平台的 opt-in 优化，对所有 LLM serving 团队可用

### Follow-up 5：技术领导力
> "你是 Staff Engineer。除了技术方案，你怎么 influence 50 个团队的 tech lead 接受你的平台？有人说'我们自己管 GPU 挺好的，不需要你的平台'。"

**期望回答**：
- 不靠 mandate，靠 value demonstration：
  1. 先帮 2-3 个团队解决痛点（如 GPU 利用率低导致预算被砍）
  2. 用数据说话：展示迁移后的成本节省和性能提升
  3. 提供 self-service：团队可以自己迁移，不需要等平台团队
- 处理阻力：
  - "我们自己管挺好" → 展示他们的 GPU 利用率数据，计算浪费的成本
  - "迁移有风险" → 提供灰度迁移 + 一键回退
  - "平台不够灵活" → 收集需求，快速迭代
- 建立 trust：
  - 透明的 SLA dashboard
  - 快速的 incident response
  - 定期的 office hours 和 feedback session
- **候选人诚实标注**：我没有 Staff-level 的 influence 经验，以上是基于开源社区协作的理解推测的

---

## Pressure Follow-up（故意挑战候选人）

> "你的简历上最大的项目是一个 PR 和一个小规模 RAG 系统。Staff Engineer 需要 lead 跨团队的大型项目、做技术决策、mentor 其他工程师。你有什么证据证明你能做到这些？"

**期望应对**：
- 承认 scope 差距：确实，我目前的项目规模和 Staff 的要求有差距
- 但展示 Staff 潜质的信号：
  1. 独立性：RAG 系统从零到一独立负责，包括技术选型、架构设计、质量保证
  2. 跨团队协作：vLLM 开源社区的 PR review 过程，需要与不同背景的 reviewer 对齐
  3. 技术深度：对 speculative decoding 有从原理到实现的完整理解
  4. 学习能力：从 NPU 到 GPU 生态的快速适应
- 诚实定位：我目前更适合 Senior Engineer 级别，Staff 需要更多的 scope 和 impact 积累
- 成长计划：通过参与更大规模的项目、mentor junior engineers、做技术 RFC 来积累 Staff 经验

---

## Debugging Scenario

> "平台上线 3 个月后，某天早上 9 点，5 个团队同时报告 latency spike。你的 on-call 收到告警。排查发现：GPU utilization 正常，network 正常，但所有服务的 TTFT 都增加了 3x。怎么排查？"

**排查思路**：

1. **共性分析**（2min）：
   - 5 个团队同时出问题 → 不是单个服务的问题，是平台级问题
   - GPU/network 正常 → 不是硬件问题
   - TTFT 增加但 TPOT 正常 → prefill 阶段变慢

2. **平台组件排查**（5min）：
   - API Gateway latency：是否 gateway 本身变慢？
   - Model Router latency：路由决策是否变慢？
   - GPU scheduler latency：是否 GPU 分配变慢？
   - 检查平台的 control plane 组件

3. **可能的 root cause**：
   - **最可能**：早上 9 点是流量高峰，auto-scaling 触发但新 instance 还在 warmup（模型加载中），导致现有 instance 过载
   - **次可能**：平台的 metadata service（如 etcd）负载过高，导致 scheduling 决策变慢
   - **较少见**：共享存储（模型权重存储）带宽打满，多个服务同时加载模型

4. **验证**：
   - 检查 auto-scaling 事件日志：是否有大量 scale-up 同时发生
   - 检查模型加载时间：是否比正常慢
   - 检查存储带宽：是否接近上限

5. **修复**：
   - 短期：手动增加 instance，跳过 warmup 直接接流量（牺牲首几个请求的质量）
   - 中期：预测性 scaling，在 9 点前提前 scale up
   - 长期：模型权重缓存在 local NVMe，减少加载时间

---

## System Design Extension

> "3 年后，公司的 AI workload 增长 10x。从 500 GPU 到 5000 GPU，从 50 个团队到 200 个团队。平台需要怎么演进？"

**演进方向**：

1. **组织**：平台团队从 8 人扩展到 30+ 人，分为 scheduling、serving、storage、observability 子团队
2. **技术**：
   - 调度器从单体演进为分布式（类似 Borg → Omega）
   - 引入 ML-based scheduling（预测流量、优化 placement）
   - 支持异构硬件（A100 + H100 + B200 + TPU）
3. **产品**：
   - 从 internal platform 演进为 platform-as-a-service
   - 提供 self-service portal、cost dashboard、optimization recommendations
   - 可能对外提供服务（成为 GPU cloud）

---

## Hire/No-Hire Evaluation

| 维度 | 权重 | 评分标准 |
|------|------|---------|
| 全局视角 | 20% | 能否从业务需求推导技术方案 |
| 技术广度 | 20% | 对推理、RAG、集群、成本的综合理解 |
| 技术深度 | 20% | 至少一个方向有深入理解 |
| 执行规划 | 20% | 分阶段计划、风险评估、优先级判断 |
| 领导力 | 20% | 跨团队 influence、stakeholder management |

---

## Scorecard

| 维度 | 候选人预期得分 | 说明 |
|------|--------------|------|
| 全局视角 | 5/10 | 能理解问题但缺乏大型项目经验 |
| 技术广度 | 5/10 | 推理和 RAG 有经验，集群和成本偏弱 |
| 技术深度 | 6/10 | speculative decoding 方向有深度 |
| 执行规划 | 4/10 | 能给出框架但细节和风险评估不足 |
| 领导力 | 3/10 | 缺乏跨团队 influence 经验 |
| **总分** | **4.6/10** | **No Hire for Staff level, Lean Hire for Senior** |

### 决策依据
- **Staff No Hire 原因**：缺乏大规模项目 ownership、跨团队 influence 经验、production 运维经验
- **Senior Lean Hire 原因**：有技术深度（speculative decoding）、有独立交付能力（RAG）、有学习潜力
- **建议**：适合 Senior Engineer 岗位，在大平台团队中积累 2-3 年经验后可以冲击 Staff
