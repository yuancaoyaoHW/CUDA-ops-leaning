# ML Platform 方向简历版本

> 面向：机器学习平台岗（模型训练/推理平台、MLOps、模型管理）
> 目标公司：阿里 PAI、字节火山引擎、腾讯 Angel、百度 PaddlePaddle 平台、各大厂 ML Platform 团队
> ⚠️ 注意：候选人缺乏 K8s、分布式训练、MLOps pipeline 经验，此方向匹配度中等

---

## Headline

AI 系统工程师 | 推理 Serving 优化 + RAG 应用交付 | vLLM 社区贡献者

---

## Summary

具备推理系统优化与 AI 应用交付双重经验。在 vLLM 生态中实现推测解码并合入社区主线，理解推理 serving 系统的核心组件（scheduler、KV cache manager、model runner）。独立交付 RAG 系统后端，具备从模型推理到应用落地的全链路工程能力。对 ML 平台中的模型 serving、性能评测、质量保障环节有实战理解。

---

## Skills Section（按优先级排列）

```
推理 Serving: vLLM (V1 架构), Speculative Decoding, KV Cache Management, Continuous Batching, Decode Scheduling
模型评测: MT-Bench Benchmarking, RAGAS 评测框架, 多维度指标设计 (Throughput/Latency/Accuracy)
AI 应用工程: LangChain, RAG Pipeline, 文档解析, 向量检索, 接口抽象设计
AI 框架: MindSpore, PyTorch
硬件平台: Ascend 910B, Atlas 3000/310P3
开源贡献: vLLM-Ascend PR #1032 (merged), vLLM-MindSpore PR #1020
编程语言: Python, C++
工具链: Linux, Docker, Git
```

---

## Project Ordering

1. **vLLM 推理优化**（体现对 serving 系统的理解）
2. **RAG 后端**（体现应用层工程能力和评测能力）

---

## Verified Bullet Points（可安全使用）

### 项目 1：vLLM 推理优化（ML Platform 视角）

1. ✅ 在 vLLM 生态中实现推测解码功能并验证生产级性能指标（吞吐 +55%、延迟 -39%、acceptance rate），具备推理 serving 系统理解
2. ✅ 独立实现 EAGLE-3 proposer 并合入社区主线（PR #1032），理解 vLLM V1 中 scheduler、model runner、KV cache manager 的交互机制
3. ✅ 设计端到端 benchmark 方案（基于 MT-Bench），覆盖吞吐、延迟、acceptance rate 多维度指标 [合理推断]
4. ✅ 按社区规范完成多轮 code review 迭代，补充单元测试，处理合并冲突 [合理推断]
5. ✅ 使用 Docker 容器化部署推理服务，具备 Linux 环境下的系统调试能力

### 项目 2：RAG 后端（评测与工程视角）

1. ✅ 独立负责 RAG 系统后端，基于 Python + LangChain 实现完整 pipeline
2. ✅ 基于 RAGAS 框架构建端到端评测流程，问答准确率达 90%，体现数据驱动的质量保障方法
3. ✅ 抽象统一检索接口，支持 metadata filters 与多后端切换，体现平台化设计思维
4. ✅ 设计双链路架构（快速问答 + Research），支持不同场景的差异化处理策略

---

## Risky Bullet Points to Avoid

| 声明 | 原因 | 标记 |
|------|------|------|
| "K8s 部署 ML 服务" | 无 K8s 经验 | 🚫 Do Not Use |
| "分布式训练经验" | 无分布式训练实操 | 🚫 Do Not Use |
| "MLOps pipeline 设计" | 无 CI/CD for ML 经验 | 🚫 Do Not Use |
| "模型版本管理" | 无 MLflow/Weights&Biases 经验 | 🚫 Do Not Use |
| "GPU 集群调度" | 无集群管理经验 | 🚫 Do Not Use |
| "A/B Testing 框架" | 无实验平台经验 | 🚫 Do Not Use |
| "Auto-scaling 策略" | 无弹性伸缩实操 | 🚫 Do Not Use |

---

## Missing Evidence List

| 缺失项 | 对该方向的影响 | 补充方式 |
|--------|---------------|----------|
| K8s 部署能力 | ML Platform 基础要求 | 学习 K8s + 部署推理服务 |
| CI/CD for ML | MLOps 核心 | 搭建模型部署 pipeline |
| 分布式训练 | 训练平台核心 | 学习 DeepSpeed/FSDP |
| 监控与可观测性 | 平台运维必备 | 添加 Prometheus/Grafana |
| 模型版本管理 | 模型管理核心 | 学习 MLflow |
| Auto-scaling | Serving 平台核心 | 实现基于 QPS 的弹性伸缩 |
| Production 流量处理 | 区分学术与工业 | 部署有真实流量的服务 |

---

## 投递策略建议

⚠️ **此方向当前匹配度中等（40-50%）**，建议：
- 投递侧重"推理 serving"的 ML Platform 岗位（而非训练平台或 MLOps）
- 强调对 vLLM serving 系统的理解和评测能力
- 避免投递要求 K8s/分布式训练/MLOps 为硬性条件的岗位
- 面试时突出"从推理到应用的全链路理解"作为差异化
