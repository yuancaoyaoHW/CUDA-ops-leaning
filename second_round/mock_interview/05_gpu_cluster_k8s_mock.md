# 模拟面试 5：45 分钟 GPU Cluster / Kubernetes 设计

## 面试官 Profile

- **公司**：某 GPU Cloud 公司（对标 CoreWeave / Lambda / RunPod）
- **级别**：SRE Lead，前 Google Borg 团队，管理过 10000+ GPU 集群
- **风格**：关注可靠性和运维效率，对"理论上可以"的回答不满意，要求"production 中怎么做"。
- **偏好**：看重 failure mode 思维、capacity planning 能力、自动化程度。

---

## Opening Question

> "设计一个 GPU 集群调度系统，支持 500 张 A100/H100 混合部署，同时运行 LLM inference 服务和训练任务。要求：inference 服务 99.9% 可用性，训练任务可以被抢占。请从资源抽象开始设计。"

---

## Candidate Expected Answer（基于候选人真实经验）

**候选人诚实回应**：
"我需要坦诚说明，我没有 GPU 集群管理和 Kubernetes 的实战经验。我的工作主要在单机或小规模环境下做推理优化。但我可以基于对分布式系统和推理服务的理解来讨论这个设计。

高层架构思路：
1. 资源抽象：将 GPU 作为可调度资源，区分 inference（高优先级、低延迟）和 training（低优先级、可抢占）
2. 调度策略：inference 任务优先分配，training 任务填充空闲资源
3. 隔离机制：inference 和 training 不共享同一张 GPU（避免干扰）
4. 容错：inference 服务多副本，单点故障自动切换

但具体的 Kubernetes operator 实现、NVIDIA device plugin 配置、NCCL 拓扑感知调度等，我目前不熟悉。"

---

## Weak Answer（常见错误）

"用 Kubernetes 部署就行了，装上 NVIDIA device plugin，然后用 resource request 申请 GPU。inference 用 Deployment，training 用 Job。设置 priority class 让 inference 优先。"

**为什么弱**：
- 没有考虑 GPU 拓扑（NVLink 连接）
- 没有考虑 multi-GPU 任务的 affinity
- 没有考虑 GPU 碎片问题
- 没有考虑 memory 隔离
- 没有讨论 failure detection 和 recovery

---

## Strong Answer（工程深度 + 数据 + tradeoff）

"让我分层设计：

**1. 资源抽象层**

```yaml
# GPU 资源模型
GPUNode:
  id: node-001
  gpus:
    - id: gpu-0, model: A100-80G, memory: 80GB, nvlink_peers: [gpu-1, gpu-2, gpu-3]
    - id: gpu-1, model: A100-80G, memory: 80GB, nvlink_peers: [gpu-0, gpu-2, gpu-3]
    ...
  network:
    ib_bandwidth: 200Gbps
    pcie_gen: 4
  topology:
    nvlink_domain: [gpu-0, gpu-1, gpu-2, gpu-3]  # 4 GPU NVLink group
    pcie_domain: [gpu-4, gpu-5, gpu-6, gpu-7]
```

关键抽象：
- **GPU Slice**：单张 GPU 或 MIG partition
- **GPU Group**：NVLink 连接的 GPU 组（TP 任务需要）
- **GPU Pool**：按用途划分（inference pool / training pool / shared pool）

**2. 调度策略**

```
调度决策流程：
1. 新任务到达 → 检查 priority class
2. Inference 任务（P0）：
   - 从 inference pool 分配
   - 如果 inference pool 满 → 从 shared pool 分配
   - 如果 shared pool 满 → 抢占 training 任务
3. Training 任务（P1）：
   - 从 training pool 分配
   - 如果 training pool 满 → 从 shared pool 分配
   - 如果 shared pool 满 → 排队等待
```

拓扑感知调度：
- TP=4 的 inference 任务必须分配到同一 NVLink domain 的 4 张 GPU
- 跨 NVLink domain 的 TP 会导致 AllReduce 走 PCIe，latency 增加 5-10x
- 训练任务的 DP 可以跨节点（走 InfiniBand），但 TP 必须在节点内

碎片整理：
- 问题：inference 任务释放后留下零散的空闲 GPU，无法满足大任务
- 方案：定期 compaction——将小任务迁移到同一节点，腾出连续 GPU
- 迁移成本：inference 服务需要 graceful drain（30s），training 需要 checkpoint

**3. Inference 服务管理**

```
Inference Service Spec:
  model: llama-70b
  tp_degree: 4
  replicas: 3  # 99.9% 可用性需要至少 3 副本
  resources:
    gpu: 4 × A100-80G (NVLink connected)
    memory: 256GB
    network: 25Gbps
  sla:
    availability: 99.9%
    ttft_p99: 300ms
    tpot_p99: 50ms
  scaling:
    min_replicas: 2
    max_replicas: 8
    metric: queue_depth > 100 for 60s
```

高可用设计：
- 3 副本分布在不同故障域（不同机架/电源）
- Health check：每 5s 发送 probe request，3 次失败标记 unhealthy
- Failover：unhealthy 副本的流量自动切换到 healthy 副本
- 自动恢复：unhealthy 副本重启，重新加载模型（约 60s）

**4. Training 任务管理**

抢占机制：
- 当 inference 需要更多 GPU 时，选择抢占哪个 training 任务
- 选择标准：优先抢占运行时间最短的（减少浪费）、checkpoint 最近的
- 抢占流程：发送 SIGTERM → 等待 checkpoint（最多 60s）→ 强制 kill → 释放 GPU
- 恢复：被抢占的任务进入队列，GPU 可用时从 checkpoint 恢复

Checkpoint 策略：
- 定期 checkpoint（每 30min 或每 1000 steps）
- 抢占时触发 emergency checkpoint
- Checkpoint 存储：分布式文件系统（如 Lustre / S3）

**5. 监控与告警**

GPU 健康监控：
- 温度：> 85°C 预警，> 90°C 降频，> 95°C 下线
- ECC 错误：单 bit 错误计数，double bit 错误立即下线
- NVLink 错误：CRC error rate > threshold → 标记 degraded
- 功耗：异常功耗可能预示硬件故障

集群级监控：
- GPU 利用率分布（目标 > 80%）
- 碎片率（空闲 GPU 中无法分配给大任务的比例）
- 调度延迟（从任务提交到开始运行的时间）
- 抢占频率（过高说明 capacity 不足）

**6. 容量规划**

500 张 GPU 的分配：
- Inference pool：200 张（40%）— 保证 SLA
- Training pool：200 张（40%）— 长期任务
- Shared pool：100 张（20%）— 弹性缓冲
- 依据：inference 需要 headroom 应对 burst，training 可以排队

成本优化：
- Off-peak 时间（夜间）：inference 缩容，释放 GPU 给 training
- Spot-like 机制：training 任务可以使用 inference 的空闲 GPU，但随时可能被抢占"

---

## Follow-up Chain（5 层递进追问）

### Follow-up 1：GPU 碎片问题
> "你有 500 张 GPU，分布在 63 个 8-GPU 节点上。现在有一个需要 32 张 GPU（4 节点）的训练任务，但空闲 GPU 分散在 20 个节点上。怎么办？"

**期望回答**：
- 短期：等待——当其他任务结束释放连续 GPU 时再调度
- 中期：compaction——将小任务迁移到少数节点，腾出连续节点
- 长期：预留——为大任务预留连续节点，不分配给小任务
- Bin-packing 算法：优先将小任务 pack 到已有任务的节点，保持空节点连续
- Trade-off：compaction 有迁移成本（inference 需要 drain，training 需要 checkpoint）
- **候选人诚实标注**：我没有实际处理过 GPU 碎片问题

### Follow-up 2：NCCL 故障处理
> "训练任务运行中，一个节点的 NVLink 出现间歇性错误（不是完全断开，而是偶尔超时）。怎么检测？怎么决定是否下线？"

**期望回答**：
- 检测方法：
  - NCCL 的 timeout 日志（`NCCL WARN Timeout`）
  - `nvidia-smi nvlink -e` 查看 NVLink error counter
  - 训练 step time 的 variance 增大（正常 100ms ± 5ms，异常 100ms ± 50ms）
- 决策标准：
  - 如果 error rate < 0.1%：继续运行，记录告警
  - 如果 error rate 0.1-1%：标记 degraded，不分配新任务
  - 如果 error rate > 1% 或影响训练收敛：下线，迁移任务
- 下线流程：
  1. 通知训练任务做 checkpoint
  2. 将该节点标记为 maintenance
  3. 重新调度任务到健康节点
  4. 运维人员检查硬件

### Follow-up 3：Multi-tenancy 隔离
> "多个团队共享这 500 张 GPU。Team A 的 inference 服务突然 burst，把 shared pool 全占了，Team B 的训练任务无法启动。怎么防止？"

**期望回答**：
- Quota 机制：每个 team 有 guaranteed quota 和 burst quota
  - Team A guaranteed: 100 GPU, burst: 150 GPU
  - Team B guaranteed: 100 GPU, burst: 150 GPU
- Guaranteed quota 不可被抢占，burst quota 可以被其他 team 的 guaranteed 抢占
- Fair-share scheduling：当资源紧张时，按 quota 比例分配
- Preemption priority：guaranteed > burst > best-effort
- 实现：类似 Kubernetes ResourceQuota + PriorityClass

### Follow-up 4：Auto-scaling Inference
> "Inference 服务的 auto-scaling 策略是什么？scale up 的 trigger 是什么？cold start 怎么处理？"

**期望回答**：
- Scale up trigger：
  - Queue depth > threshold for 60s（反应式）
  - 预测性：基于历史流量模式预测（如每天 9am 流量上升）
- Scale up 流程：
  1. 分配 GPU（从 shared pool 或抢占 training）
  2. 加载模型到 GPU memory（30-60s for 70B model）
  3. Warmup（运行几个 dummy request）
  4. 加入 load balancer
- Cold start 优化：
  - 预加载：保持 1-2 个 standby instance（模型已加载但不接流量）
  - 模型缓存：将模型权重缓存在 host memory 或 NVMe，加载更快
  - Checkpoint sharding：并行从多个 source 加载模型分片
- Scale down：
  - 流量下降后等待 cooldown period（5min）再缩容
  - Graceful drain：停止接收新请求，等待现有请求完成

### Follow-up 5：灾难恢复
> "整个机房断电 5 分钟后恢复。500 张 GPU 同时重启。你的恢复策略是什么？"

**期望回答**：
- 恢复优先级：
  1. P0：Inference 服务（影响用户）→ 优先恢复
  2. P1：Training checkpoint server（防止数据丢失）
  3. P2：Training 任务（从 checkpoint 恢复）
  4. P3：Monitoring/logging（恢复可观测性）
- 避免 thundering herd：
  - 不要 500 张 GPU 同时加载模型（会打爆存储带宽）
  - 分批恢复：每批 50 张，间隔 30s
  - 优先恢复 inference 的最小副本数（如 2/3 副本）
- Training 恢复：
  - 从最近的 checkpoint 恢复（最多丢失 30min 训练）
  - 验证 checkpoint 完整性
  - 重新建立 NCCL 通信组
- 事后：
  - 检查所有 GPU 健康状态（断电可能导致硬件损坏）
  - 验证 NVLink/InfiniBand 连接正常
  - 运行 GPU stress test 确认无隐性故障

---

## Pressure Follow-up（故意挑战候选人）

> "你说你没有 Kubernetes 和 GPU 集群经验。那你凭什么觉得你能做这个岗位？这个岗位 80% 的工作是集群运维和调度优化，不是写推理代码。"

**期望应对**：
- 承认经验 gap：确实，GPU 集群管理是我的短板
- 但展示可迁移的能力：
  1. 我理解 GPU 硬件特性（memory hierarchy, NVLink, TP/PP），这是调度优化的基础
  2. 我有分布式系统的理论基础（一致性、容错、调度算法）
  3. 我在 vLLM 上的工作让我理解 inference 服务的需求（latency SLA, memory management）
  4. Kubernetes 的学习曲线是可控的，核心概念（Pod, Service, Operator）可以快速掌握
- 诚实定位：这个岗位如果要求 Day 1 就能独立运维集群，我确实不够格。但如果允许 2-3 个月 ramp-up，我有信心胜任。

---

## Debugging Scenario

> "集群中 8 张 A100 的一个训练任务，step time 从正常的 200ms 突然变成 800ms，但 GPU utilization 仍然显示 90%+。没有代码变更。怎么排查？"

**排查思路**：

1. **GPU utilization 高但 step time 慢 → 通信瓶颈**：
   - GPU 在做计算（utilization 高），但在等待通信（step time 长）
   - 检查 NCCL AllReduce 时间：正常应该 < 10ms，如果 > 100ms 说明网络问题

2. **检查网络**：
   - `nvidia-smi nvlink -e`：NVLink error counter 是否增加
   - InfiniBand 带宽测试：`ib_write_bw` 是否正常
   - 检查交换机日志：是否有端口 flapping

3. **检查 thermal throttling**：
   - `nvidia-smi -q -d PERFORMANCE`：是否有 throttle reason
   - GPU 温度是否 > 83°C（A100 throttle 温度）
   - 可能原因：机房空调故障、风扇故障

4. **检查 PCIe 带宽**：
   - 如果 NVLink 正常但 PCIe 慢 → Host-Device 数据传输瓶颈
   - 可能原因：PCIe link degradation（x16 降到 x8）
   - 检查：`nvidia-smi -q -d PCIE`

5. **最可能的 root cause**：
   - NVLink 间歇性错误导致 NCCL 重试
   - 或者某张 GPU thermal throttle 导致整个 TP group 等待最慢的 GPU

---

## System Design Extension

> "扩展到 10000 张 GPU，跨 3 个数据中心。怎么设计？"

**设计要点**：
- 分层调度：Global scheduler → DC-level scheduler → Node-level scheduler
- 跨 DC 策略：inference 每个 DC 独立部署（低延迟），training 可以跨 DC（高带宽需求用 DC 内）
- 网络：DC 内 InfiniBand 400Gbps，DC 间 100Gbps 专线
- 容灾：每个 DC 独立可用，单 DC 故障时流量切换到其他 DC
- 一致性：集群状态用 etcd 集群（跨 DC Raft consensus）

---

## Hire/No-Hire Evaluation

| 维度 | 权重 | 评分标准 |
|------|------|---------|
| 集群调度设计 | 25% | 资源抽象、调度算法、拓扑感知 |
| 可靠性设计 | 25% | 容错、failover、灾难恢复 |
| GPU 硬件理解 | 20% | NVLink、memory、thermal 等 |
| 运维自动化 | 15% | 监控、告警、自动修复 |
| 实战经验 | 15% | 是否有 production 集群管理经验 |

---

## Scorecard

| 维度 | 候选人预期得分 | 说明 |
|------|--------------|------|
| 集群调度设计 | 3/10 | 能给出基本框架但缺乏细节和实战 |
| 可靠性设计 | 3/10 | 理解概念但无 production 经验 |
| GPU 硬件理解 | 5/10 | 对 GPU 特性有基本理解（来自推理优化经验） |
| 运维自动化 | 2/10 | 无 monitoring/alerting 设计经验 |
| 实战经验 | 1/10 | 无 GPU 集群管理经验 |
| **总分** | **2.8/10** | **No Hire — 经验严重不足** |

### 决策依据
- **No Hire 原因**：GPU cluster 岗位需要大量运维经验，候选人完全没有
- **候选人亮点**：对 GPU 硬件有基本理解，系统设计思维不为零
- **建议**：不适合 SRE/Platform 岗位。如果是 ML Infra 岗位中偶尔涉及集群的，可以考虑
- **成长路径**：需要实际操作 Kubernetes + GPU 集群的经验，建议先在小规模环境练习
