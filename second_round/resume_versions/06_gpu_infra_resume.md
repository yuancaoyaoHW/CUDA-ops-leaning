# GPU Infrastructure 方向简历版本

> 面向：GPU 集群 / 调度岗
> 目标公司：CoreWeave、Lambda、各大厂 GPU 集群团队
> ⚠️ 注意：候选人缺乏 GPU 集群管理、K8s、调度系统经验，此方向匹配度低

---

## Headline

AI 推理系统工程师 | 异构硬件推理优化 | vLLM Serving 系统理解

---

## Summary

在异构硬件（昇腾 NPU）上实现推理优化，理解推理 serving 系统中的资源管理问题（KV cache 分配/回收、decode scheduling）。在 vLLM 生态中实现推测解码并合入社区主线，对推理服务的性能瓶颈分析和 benchmark 验证有实战经验。具备 Docker 容器化部署和 Linux 系统调试能力。

---

## Skills Section（按优先级排列）

```
推理 Serving: vLLM (V1 架构), KV Cache Management, Decode Scheduling, Continuous Batching
资源管理: KV Cache 分配/回收策略, 推测解码中的内存管理, Benchmark 驱动的资源配置
硬件平台: Ascend 910B, Atlas 3000/310P3, 异构硬件推理部署
性能验证: MT-Bench Benchmarking, Throughput/Latency 多维度评测
容器化: Docker, Linux 系统调试
开源贡献: vLLM-Ascend PR #1032 (merged)
编程语言: Python, C++
```

---

## Project Ordering

1. **vLLM 推理优化**（突出资源管理和 serving 系统理解）
2. **RAG 后端**（体现系统工程能力）

---

## Verified Bullet Points（可安全使用）

### 项目 1：vLLM 推理优化（GPU Infra 视角）

1. ✅ 在 vLLM 生态中实现推测解码，理解 serving 系统中 scheduler、model runner、KV cache manager 的资源协调机制
2. ✅ 在推测解码实现中设计 KV cache 分配与回收策略，处理 rejected token 场景下的内存状态管理 [合理推断]
3. ✅ 在 Atlas 310P3 上验证推理性能：吞吐 +55%，延迟 -39%，具备硬件级性能验证能力
4. ✅ 使用 Docker 容器化部署推理服务，具备 Linux 环境下的系统调试能力
5. ✅ 独立实现 EAGLE-3 proposer 并合入社区主线（PR #1032），理解开源推理引擎的架构设计

### 项目 2：RAG 后端

1. ✅ 独立负责后端系统设计与实现，具备从零搭建服务的工程能力
2. ✅ 设计接口抽象层支持多后端切换，体现系统可扩展性设计

---

## Risky Bullet Points to Avoid

| 声明 | 原因 | 标记 |
|------|------|------|
| "GPU 集群管理经验" | 无集群运维经验 | 🚫 Do Not Use |
| "K8s GPU 调度" | 无 K8s 经验 | 🚫 Do Not Use |
| "NVIDIA GPU 优化" | 无 GPU 侧实操 | 🚫 Do Not Use |
| "多节点分布式部署" | 无多节点经验 | 🚫 Do Not Use |
| "SLA 监控与告警" | 无 production 运维经验 | 🚫 Do Not Use |
| "Auto-scaling / 弹性伸缩" | 无实操 | 🚫 Do Not Use |
| "网络拓扑优化（NVLink/InfiniBand）" | 无相关经验 | 🚫 Do Not Use |
| "GPU 虚拟化（MIG/MPS）" | 无相关经验 | 🚫 Do Not Use |

---

## Missing Evidence List

| 缺失项 | 严重程度 | 补充方式 |
|--------|----------|----------|
| K8s 集群管理 | 致命 — 此方向硬性要求 | 学习 K8s + GPU operator |
| GPU 调度策略 | 致命 | 学习 GPU 调度器（Volcano/Kueue） |
| 多节点部署 | 高 | 在多机环境部署推理服务 |
| 监控与可观测性 | 高 | Prometheus + Grafana + GPU metrics |
| 网络优化 | 中 | 学习 NCCL、InfiniBand 基础 |
| 故障处理 | 中 | 积累 incident 处理经验 |
| 容量规划 | 中 | 学习 GPU 资源规划方法 |

---

## 投递策略建议

⚠️ **此方向当前匹配度很低（<30%）**，不建议当前投递。建议：
- 此方向需要 6+ 个月的系统学习和实践
- 如果对 GPU Infra 感兴趣，先从"推理 serving 部署"切入
- 当前更适合投递"推理优化"岗位而非"集群管理"岗位
