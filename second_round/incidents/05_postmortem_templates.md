# 事后复盘模板 (Postmortem Templates)

---

## 通用 Postmortem 模板

```markdown
# Postmortem: [事故标题]

## 基本信息
- **日期**: YYYY-MM-DD
- **严重级别**: P0/P1/P2/P3
- **影响时长**: X 小时 Y 分钟
- **影响范围**: X 个用户 / Y 个请求 / Z% 流量
- **On-call**: [姓名]
- **状态**: 已解决 / 监控中

## Executive Summary
一句话总结事故原因和影响。

## Timeline (UTC+8)
| 时间 | 事件 |
|------|------|
| HH:MM | 监控告警触发 |
| HH:MM | On-call 响应 |
| HH:MM | 初步定位原因 |
| HH:MM | 实施缓解措施 |
| HH:MM | 服务恢复 |
| HH:MM | 确认完全恢复 |

## Root Cause
详细描述根本原因。包括：
- 直接原因（触发条件）
- 根本原因（为什么会有这个触发条件）
- 贡献因素（加剧问题的因素）

## Impact
- **用户影响**: 具体描述用户感知到的问题
- **业务影响**: 请求失败数、SLA 违反、收入损失
- **数据影响**: 是否有数据丢失或损坏

## Detection
- **如何发现**: 监控告警 / 用户反馈 / 人工巡检
- **检测延迟**: 从问题发生到被发现的时间
- **检测 gap**: 为什么没有更早发现

## Mitigation
- **短期缓解**: 立即采取的措施
- **恢复步骤**: 如何恢复服务
- **验证方法**: 如何确认恢复

## Lessons Learned
### What went well
- 

### What went wrong
- 

### Where we got lucky
- 

## Action Items
| # | 行动项 | 负责人 | 优先级 | 截止日期 | 状态 |
|---|--------|--------|--------|----------|------|
| 1 | | | P0/P1/P2 | | TODO/DONE |
| 2 | | | | | |
| 3 | | | | | |

## References
- 相关 PR/Issue 链接
- 监控 dashboard 链接
- 相关文档链接
```

---

## LLM Serving 事故 Postmortem 模板

```markdown
# Postmortem: [LLM Serving 事故标题]

## 基本信息
- **日期**: YYYY-MM-DD
- **严重级别**: P0/P1/P2
- **影响时长**: 
- **影响指标**: TTFT/TPOT/Throughput/Error Rate
- **影响模型**: 
- **影响 GPU**: 

## 性能指标对比
| 指标 | 正常值 | 事故期间 | 恢复后 |
|------|--------|----------|--------|
| TTFT p50 | ms | ms | ms |
| TTFT p99 | ms | ms | ms |
| TPOT p50 | ms | ms | ms |
| Throughput | tok/s | tok/s | tok/s |
| Error Rate | % | % | % |
| GPU Cache Usage | % | % | % |

## Root Cause Analysis
### 直接原因
[描述触发事故的直接原因]

### 根本原因
[描述为什么会有这个问题存在]

### 5 Whys 分析
1. Why: [现象]
2. Why: [第一层原因]
3. Why: [第二层原因]
4. Why: [第三层原因]
5. Why: [根本原因]

## Serving 系统状态快照
```
事故期间:
- Running requests: X
- Waiting requests: Y
- GPU cache usage: Z%
- Batch size: W
- Preemption count: N
```

## Action Items
| # | 行动项 | 类型 | 优先级 | 状态 |
|---|--------|------|--------|------|
| 1 | 添加 [指标] 监控告警 | 检测 | P0 | |
| 2 | 实现 [缓解措施] 自动化 | 缓解 | P1 | |
| 3 | 修复 [根本原因] | 修复 | P1 | |
| 4 | 添加 [测试] 防止回归 | 预防 | P2 | |
```

---

## CUDA/GPU Performance 事故 Postmortem 模板

```markdown
# Postmortem: [CUDA Performance 事故标题]

## 基本信息
- **日期**: YYYY-MM-DD
- **影响 Kernel**: [kernel 名称]
- **影响硬件**: [GPU 型号]
- **影响版本**: [软件版本变更]

## 性能对比
| Kernel | Before | After | Regression |
|--------|--------|-------|-----------|
| [name] | X ms | Y ms | +Z% |

## Profiling 数据
### Nsight Compute 对比
| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Execution Time | | | |
| SM Throughput | | | |
| DRAM Throughput | | | |
| Occupancy | | | |
| Registers/Thread | | | |
| Shared Memory | | | |
| L1 Hit Rate | | | |
| L2 Hit Rate | | | |

### Roofline Position
- Before: [memory-bound / compute-bound], X% of peak
- After: [memory-bound / compute-bound], Y% of peak

## Root Cause
[详细描述 kernel 性能回归的原因]

## Fix
[描述修复方案和验证结果]

## Prevention
- [ ] Kernel benchmark CI（每次更新自动对比）
- [ ] 版本锁定策略
- [ ] 性能回归阈值告警
```

---

## GPU Cluster 事故 Postmortem 模板

```markdown
# Postmortem: [GPU Cluster 事故标题]

## 基本信息
- **日期**: YYYY-MM-DD
- **影响节点**: [node list]
- **影响 GPU**: [GPU IDs]
- **故障类型**: 硬件/Driver/网络/K8s

## 集群状态快照
```
事故期间:
- Total nodes: X
- Healthy nodes: Y
- Affected nodes: Z
- Total GPUs: A
- Available GPUs: B
- Affected workloads: C
```

## 硬件诊断
| Node | GPU | Error Type | Error Count | Action |
|------|-----|-----------|-------------|--------|
| | | ECC/XID/Thermal | | Replace/Reset |

## Network 诊断
| Link | Status | Bandwidth | Error Rate |
|------|--------|-----------|-----------|
| IB port X | | Gbps | |

## Recovery Steps
1. [步骤 1]
2. [步骤 2]
3. [步骤 3]

## MTTR Analysis
- Detection time: X min
- Triage time: Y min
- Mitigation time: Z min
- Recovery time: W min
- **Total MTTR**: X+Y+Z+W min

## Prevention
- [ ] 硬件健康检查自动化
- [ ] 故障节点自动 cordon
- [ ] 冗余容量规划
- [ ] 定期维护窗口
```

---

## RAG Quality 事故 Postmortem 模板

```markdown
# Postmortem: [RAG Quality 事故标题]

## 基本信息
- **日期**: YYYY-MM-DD
- **影响组件**: Retrieval / Reranker / Generation
- **影响指标**: Recall / Precision / Faithfulness / Latency

## 质量指标对比
| 指标 | 正常值 | 事故期间 | 恢复后 |
|------|--------|----------|--------|
| Recall@10 | % | % | % |
| Precision@10 | % | % | % |
| Faithfulness | | | |
| Answer Relevancy | | | |
| E2E Latency p99 | ms | ms | ms |

## 影响分析
- **受影响 Query 类型**: [描述哪类 query 受影响最大]
- **用户感知**: [用户看到的具体问题]
- **业务影响**: [对业务的具体影响]

## Root Cause
[详细描述质量下降的原因]

## 数据分析
- Index size: X documents
- Affected documents: Y
- Query distribution change: [描述]
- Embedding model version: [版本]

## Prevention
- [ ] Golden set 持续评测（每小时）
- [ ] Embedding drift 检测
- [ ] 索引完整性检查
- [ ] 数据质量 gate
```

---

## 面试中如何讲述 Postmortem

### STAR 框架

**Situation**: 描述事故背景和影响
> "我们的 LLM serving 系统在某天下午 TTFT p99 突然从 300ms 涨到 3s，影响了 30% 的用户请求。"

**Task**: 你的角色和目标
> "作为 on-call 工程师，我需要在 15 分钟内定位问题并恢复服务。"

**Action**: 具体的排查和修复步骤
> "我首先检查了 GPU cache usage 发现达到 95%，然后检查 request log 发现某个客户开始发送 32K token 的长文档。短期：对超长请求限流；长期：实现 chunked prefill。"

**Result**: 量化的结果
> "5 分钟内恢复服务，TTFT p99 降回 500ms。后续实现的 chunked prefill 使得即使有长文档也能保持 TTFT < 1s。"

### 面试中的关键要素

1. **展示系统性思维**: 不是碰运气找到问题，而是有方法论
2. **展示指标意识**: 用具体数字描述问题和结果
3. **展示工具使用**: 提到具体的 profiling/monitoring 工具
4. **展示预防思维**: 不只是修复，还要防止再次发生
5. **展示协作能力**: 如何与团队协作解决问题

### 常见追问及回答

**Q: 如何确保修复不引入新问题？**
> "我们有 staging 环境做 canary 部署，同时有 kernel benchmark CI 检测性能回归。修复上线后持续监控 24 小时确认稳定。"

**Q: 如何平衡修复速度和修复质量？**
> "分两步：先用最快的方式缓解（如限流、回滚），确保用户体验恢复；然后用 1-2 周时间做根本修复，经过完整测试后上线。"

**Q: 如何防止类似问题再次发生？**
> "三个层面：1) 监控告警（更早发现）；2) 自动缓解（如自动限流）；3) 架构改进（如 chunked prefill 从根本上解决长文档问题）。"
