# Hire / No Hire 评分卡 (Scorecards)

候选人背景：浙大硕士+西交本科，vLLM-Ascend PR#1032（EAGLE-3 proposer，吞吐+55%），vLLM-MindSpore PR#1020，RAG后端独立负责（RAGAS 90%），零CUDA、零Production规模、零分布式推理、硬件生态单一（Ascend NPU）

---

## 1. LLM Inference 系统设计

### 评分维度

| 维度 | 权重 | 说明 |
|------|------|------|
| 架构设计能力 | 25% | 能否设计完整的 serving 系统（routing、scheduling、scaling） |
| 性能优化深度 | 25% | 对 prefill/decode 特性、KV cache 管理、batching 策略的理解 |
| 规模化思维 | 20% | 多租户、autoscaling、容量规划、成本优化 |
| 故障处理 | 15% | preemption、failover、graceful degradation |
| 指标与 tradeoff | 15% | TTFT/TPOT/throughput 的量化分析和 tradeoff 决策 |

### 判定标准

- **Strong Hire (4.5+):** 能独立设计千卡级 LLM serving 系统，深入理解 PagedAttention/continuous batching/speculative decoding 的实现细节和 tradeoff，能给出具体的容量规划数字，有 production 故障处理经验
- **Hire (3.5-4.4):** 理解核心概念并能正确应用，能设计基本架构，知道关键优化方向，能给出合理的性能估算
- **Lean Hire (3.0-3.4):** 概念理解正确但缺乏深度，设计偏教科书式，缺少 production 经验的体现，数字估算有偏差
- **Lean No Hire (2.0-2.9):** 概念模糊，设计有明显漏洞，无法给出性能数字，不理解关键 tradeoff
- **No Hire (<2.0):** 基本概念错误，无法完成系统设计

### 候选人当前预估

**总分：3.2 (Lean Hire)**

- 架构设计能力：3.5（有 vLLM 源码级理解，能描述核心组件）
- 性能优化深度：3.5（EAGLE-3 实现证明对 speculative decoding 有深入理解）
- 规模化思维：2.8（缺乏大规模部署经验，容量规划偏理论）
- 故障处理：2.5（零 production 经验，只能给出教科书式回答）
- 指标与 tradeoff：3.5（自己的 PR 有具体数字，但缺乏多场景对比）

### 达到 Hire 需要补强

1. 准备 2-3 个完整的系统设计案例（1000 QPS serving、多租户平台、disaggregated serving），每个能讲 30 分钟
2. 背熟关键数字：各模型的 KV cache 大小、单卡 throughput、网络带宽需求
3. 准备 production 故障处理的 story（可基于 vLLM 社区 issue 构造）
4. 练习从 EAGLE-3 经验延伸到更大规模场景的叙述

---

## 2. CUDA Kernel 优化

### 评分维度

| 维度 | 权重 | 说明 |
|------|------|------|
| Memory hierarchy 理解 | 25% | HBM/L2/SMEM/Register 的带宽、延迟、优化策略 |
| Kernel 优化技巧 | 25% | coalescing、bank conflict、occupancy、tiling |
| 性能分析能力 | 20% | roofline model、Nsight 工具使用、瓶颈定位 |
| 实现能力 | 20% | 能否写出正确且高效的 CUDA kernel |
| GPU 架构理解 | 10% | warp scheduler、Tensor Core、SM 结构 |

### 判定标准

- **Strong Hire (4.5+):** 能手写高效 CUDA kernel（GEMM tiling、FlashAttention-level），熟练使用 Nsight 定位瓶颈，理解 Tensor Core 编程，有优化 kernel 达到理论峰值 70%+ 的经验
- **Hire (3.5-4.4):** 理解 memory hierarchy 和常见优化技巧，能写基本 kernel（reduction、transpose），能用 roofline 分析性能，知道 Tensor Core 使用条件
- **Lean Hire (3.0-3.4):** 概念正确但缺乏实践，能描述优化方向但没写过复杂 kernel，roofline 分析偏理论
- **Lean No Hire (2.0-2.9):** 概念模糊，不理解 coalescing 或 bank conflict 的具体影响，无法写出正确 kernel
- **No Hire (<2.0):** 不了解 GPU 编程模型

### 候选人当前预估

**总分：2.3 (Lean No Hire)**

- Memory hierarchy 理解：2.5（理论知识有，但无实际优化经验）
- Kernel 优化技巧：2.0（零 CUDA 编程经验，只能背概念）
- 性能分析能力：2.5（了解 roofline 理论，未用过 Nsight）
- 实现能力：2.0（无法手写 CUDA kernel）
- GPU 架构理解：2.5（理论了解 warp/SM 结构）

### 达到 Hire 需要补强

1. **必须动手写 kernel：** 至少完成 reduction、softmax、GEMM tiling、transpose 四个经典 kernel
2. 用 Nsight Compute 分析自己写的 kernel，练习从 metrics 定位瓶颈
3. 实现一个 FlashAttention 的简化版本，理解 tiling + online softmax
4. 在 A100/H100 上跑 benchmark，积累真实性能数字
5. 学习 CUTLASS 的 tiling hierarchy，理解 warp-level MMA

---

## 3. Triton Kernel 优化

### 评分维度

| 维度 | 权重 | 说明 |
|------|------|------|
| Triton 编程模型理解 | 25% | block-level 编程、auto-tuning、与 CUDA 的区别 |
| Kernel 实现能力 | 25% | 能否用 Triton 写出高效 kernel |
| 性能调优 | 20% | block size 选择、num_warps、num_stages 调优 |
| 与 PyTorch 集成 | 15% | custom op、autograd integration、compilation |
| 适用场景判断 | 15% | 何时用 Triton vs CUDA vs PyTorch native |

### 判定标准

- **Strong Hire (4.5+):** 能用 Triton 实现 production-level kernel（fused attention、quantized GEMM），理解 Triton compiler 的优化 pass，能做到接近 CUDA 手写性能的 90%+
- **Hire (3.5-4.4):** 能用 Triton 写常见 kernel（matmul、softmax、layernorm），理解 auto-tuning 机制，知道 Triton 的限制
- **Lean Hire (3.0-3.4):** 了解 Triton 编程模型，能写简单 kernel，但调优经验不足
- **Lean No Hire (2.0-2.9):** 只了解 Triton 概念，无法写出正确 kernel
- **No Hire (<2.0):** 不了解 Triton

### 候选人当前预估

**总分：2.0 (No Hire 边缘)**

- Triton 编程模型理解：2.0（可能了解概念，无实践）
- Kernel 实现能力：1.5（零 Triton 编程经验）
- 性能调优：2.0（无调优经验）
- 与 PyTorch 集成：2.5（有 PyTorch 使用经验，但未做 custom op）
- 适用场景判断：2.5（理论上能分析）

### 达到 Hire 需要补强

1. 完成 Triton 官方 tutorial 的所有 kernel（vector_add、softmax、matmul、flash_attention）
2. 用 Triton 重写自己 CUDA 学习中的 kernel，对比性能
3. 理解 `tl.load`/`tl.store` 的 mask 机制和 boundary handling
4. 练习 auto-tuning config 的设计（`triton.autotune` decorator）
5. 实现一个 fused kernel（如 fused_add_layernorm）并集成到 PyTorch model

---

## 4. RAG Infrastructure 设计

### 评分维度

| 维度 | 权重 | 说明 |
|------|------|------|
| 检索系统设计 | 25% | chunking、embedding、indexing、hybrid search |
| 质量优化 | 25% | rerank、query transformation、evaluation |
| 系统工程 | 20% | 延迟优化、scale、monitoring、freshness |
| 生产经验 | 20% | 故障处理、A/B testing、迭代改进 |
| 技术选型 | 10% | 组件选择的 tradeoff 分析 |

### 判定标准

- **Strong Hire (4.5+):** 设计过百万级文档的 RAG 系统，有完整的质量评估体系，能处理 multi-hop/multi-modal，有 production 故障处理经验，能量化每个优化的收益
- **Hire (3.5-4.4):** 独立设计并上线过 RAG 系统，有评估指标驱动的优化经验，理解 scale 挑战，能做合理的技术选型
- **Lean Hire (3.0-3.4):** 有 RAG 实现经验但规模小，评估体系不完整，scale 经验不足
- **Lean No Hire (2.0-2.9):** 只做过 demo 级 RAG，无评估体系，不理解 production 挑战
- **No Hire (<2.0):** 不了解 RAG 架构

### 候选人当前预估

**总分：3.6 (Hire)**

- 检索系统设计：3.8（独立实现完整 pipeline，hybrid search + rerank）
- 质量优化：4.0（RAGAS 90%，有系统化评估和迭代）
- 系统工程：3.2（规模较小 50K docs，延迟优化有但 scale 经验不足）
- 生产经验：3.5（独立负责后端，有上线经验）
- 技术选型：3.5（能说清选择理由和 tradeoff）

### 达到 Strong Hire 需要补强

1. 准备 scale 到百万级文档的架构演进方案
2. 补充 multi-hop reasoning 和 multi-modal RAG 的设计思路
3. 准备 2-3 个 production 故障处理的具体 story
4. 量化每个优化步骤的具体收益（数字要精确）
5. 了解 Graph RAG、ColBERT 等前沿方案

---

## 5. GPU Cluster / K8s 设计

### 评分维度

| 维度 | 权重 | 说明 |
|------|------|------|
| GPU 集群架构 | 25% | 网络拓扑、存储、调度、资源管理 |
| K8s 运维 | 25% | GPU operator、device plugin、scheduling、monitoring |
| 故障处理 | 20% | GPU 故障检测、自动恢复、drain/replace |
| 性能优化 | 15% | NCCL 调优、网络优化、存储 IO |
| 容量规划 | 15% | 采购策略、利用率优化、成本管理 |

### 判定标准

- **Strong Hire (4.5+):** 管理过 1000+ GPU 集群，熟悉 InfiniBand/RoCE 网络调优，有 GPU 故障自动化处理经验，能做 TCO 分析和采购决策
- **Hire (3.5-4.4):** 理解 GPU 集群架构，有 K8s GPU 调度经验，能设计故障处理流程，了解 NCCL 通信优化
- **Lean Hire (3.0-3.4):** 了解基本概念，有小规模 K8s 经验，GPU 集群知识偏理论
- **Lean No Hire (2.0-2.9):** 只有单机 GPU 使用经验，不了解集群管理
- **No Hire (<2.0):** 无 GPU 和 K8s 经验

### 候选人当前预估

**总分：2.2 (Lean No Hire)**

- GPU 集群架构：2.5（理论了解网络拓扑，无实际经验）
- K8s 运维：2.0（可能有基本 K8s 使用经验，无 GPU 调度经验）
- 故障处理：2.0（零集群运维经验）
- 性能优化：2.5（了解 NCCL 概念，无调优经验）
- 容量规划：2.0（无规模化经验）

### 达到 Hire 需要补强

1. 学习 GPU 集群网络拓扑（fat-tree、rail-optimized）和 NCCL 通信模式
2. 在本地搭建 K8s + GPU operator 环境，练习 GPU 调度和 device plugin
3. 了解 InfiniBand vs RoCE 的区别和调优参数
4. 学习 GPU 故障检测工具（DCGM、XID error 分类）
5. 准备一个 100-GPU 集群的设计方案（网络、存储、调度、监控）

---

## 6. Staff-level AI Infra 综合

### 评分维度

| 维度 | 权重 | 说明 |
|------|------|------|
| 技术广度 | 20% | 跨领域知识（inference、training、data、infra） |
| 技术深度 | 20% | 至少一个领域有深入贡献 |
| 系统思维 | 20% | 端到端设计、跨团队协作、技术决策 |
| 影响力 | 20% | 开源贡献、技术方案推动、mentor |
| 业务理解 | 20% | 成本意识、优先级判断、ROI 分析 |

### 判定标准

- **Strong Hire (4.5+):** 主导过大型 AI Infra 项目（影响 >100 GPU），有跨团队技术决策经验，开源社区有影响力，能做 cost-performance tradeoff 分析
- **Hire (3.5-4.4):** 在 AI Infra 某个方向有深入贡献，理解端到端系统，有技术方案设计和推动经验，能做合理的优先级判断
- **Lean Hire (3.0-3.4):** 有 AI Infra 相关经验但深度/广度不足，缺乏大规模系统经验，技术决策能力待验证
- **Lean No Hire (2.0-2.9):** 经验局限于单一方向且规模小，缺乏系统思维
- **No Hire (<2.0):** 无 AI Infra 相关经验

### 候选人当前预估

**总分：2.8 (Lean No Hire)**

- 技术广度：3.0（inference + RAG 有经验，training/cluster 缺失）
- 技术深度：3.5（EAGLE-3 PR 证明在 speculative decoding 方向有深度）
- 系统思维：2.8（独立负责 RAG 后端，但规模有限）
- 影响力：3.0（vLLM 社区 PR 合入，但只有 1 个）
- 业务理解：2.5（缺乏成本优化和 ROI 分析经验）

### 达到 Hire 需要补强

1. 扩展技术广度：补充 training infra（分布式训练、checkpoint）和 cluster 管理知识
2. 增加开源贡献：再提 1-2 个有影响力的 PR，或写技术博客
3. 准备跨团队协作的 story（即使是学校项目也要体现系统思维）
4. 学习成本分析：能算出 GPU 小时成本、cost/token、ROI
5. 练习从业务需求出发做技术决策的叙述方式

---

## 7. Production Debugging

### 评分维度

| 维度 | 权重 | 说明 |
|------|------|------|
| 排查方法论 | 25% | 系统化的 debug 流程（观察→假设→验证→修复） |
| 工具使用 | 25% | profiler、monitoring、logging、tracing |
| GPU/CUDA 调试 | 20% | GPU 特有问题（OOM、ECC、NCCL、降频） |
| 故障恢复 | 15% | rollback、failover、graceful degradation |
| 经验积累 | 15% | 常见 failure pattern 的识别和快速定位 |

### 判定标准

- **Strong Hire (4.5+):** 有丰富的 production debugging 经验（>10 次 oncall incident），能快速定位 GPU/网络/系统问题，有完善的 runbook 和自动化恢复方案
- **Hire (3.5-4.4):** 有系统化的排查方法论，了解常见 failure pattern，能使用 profiling 工具定位问题，有一定的 production 经验
- **Lean Hire (3.0-3.4):** 排查思路正确但缺乏实战，工具使用偏理论，故障恢复方案偏教科书
- **Lean No Hire (2.0-2.9):** 排查思路不清晰，不了解 GPU 特有问题，无 production 经验
- **No Hire (<2.0):** 无法描述基本的 debug 流程

### 候选人当前预估

**总分：2.5 (Lean No Hire)**

- 排查方法论：3.0（CS 基础好，能给出逻辑清晰的排查步骤）
- 工具使用：2.5（了解工具名称和用途，未实际使用过 Nsight/DCGM）
- GPU/CUDA 调试：2.0（零 CUDA debugging 经验）
- 故障恢复：2.5（能描述方案但无实战）
- 经验积累：2.5（RAG 项目可能有一些 debug 经验，但非 GPU 相关）

### 达到 Hire 需要补强

1. 构造并练习 10 个 debugging scenario（TTFT spike、OOM、NCCL timeout 等）
2. 在本地环境用 Nsight Systems/Compute 分析真实 kernel
3. 学习 GPU 故障分类（XID error codes、ECC error 类型）
4. 准备 RAG 项目中的 debugging story（延迟排查、质量下降定位）
5. 练习 structured debugging 叙述：观察现象→缩小范围→定位根因→修复验证→预防措施

---

## 8. Behavioral + Project Deep Dive

### 评分维度

| 维度 | 权重 | 说明 |
|------|------|------|
| 技术深度展示 | 25% | 能否深入讲解自己的项目技术细节 |
| 问题解决能力 | 25% | 遇到困难时的思考和解决过程 |
| 沟通与协作 | 20% | 表达清晰度、与团队/社区的协作 |
| 学习能力 | 15% | 快速学习新技术的能力和方法 |
| 自我认知 | 15% | 对自身优劣势的清醒认识、成长规划 |

### 判定标准

- **Strong Hire (4.5+):** 项目讲解深入且有洞察，能清晰描述技术决策的 why，展示出强大的学习能力和成长潜力，沟通高效有条理
- **Hire (3.5-4.4):** 项目理解深入，能回答追问，展示出解决问题的能力，沟通清晰，有明确的成长方向
- **Lean Hire (3.0-3.4):** 项目能讲清楚但深度不够，追问时有些犹豫，沟通基本清晰
- **Lean No Hire (2.0-2.9):** 项目讲解浮于表面，无法回答深入追问，沟通不够清晰
- **No Hire (<2.0):** 无法清晰描述自己的项目

### 候选人当前预估

**总分：3.5 (Hire 边缘)**

- 技术深度展示：3.8（EAGLE-3 PR 有深度，能讲清楚实现细节）
- 问题解决能力：3.5（PR 合入过程体现了解决问题的能力）
- 沟通与协作：3.5（开源社区协作经验，PR review 过程）
- 学习能力：3.5（从零到 PR 合入体现学习能力）
- 自我认知：3.0（需要准备对自身短板的坦诚回答和补强计划）

### 达到 Strong Hire 需要补强

1. 准备 EAGLE-3 项目的 15 分钟深度讲解（背景→挑战→方案→结果→反思）
2. 准备 3 个 "遇到困难如何解决" 的 STAR story
3. 准备对 "零 CUDA 经验" 的正面回应（展示学习计划和进展）
4. 练习用 whiteboard 画架构图讲解系统设计
5. 准备 "为什么选择 AI Infra 方向" 的 compelling narrative

---

## 综合评估总结

| 面试场次 | 当前预估分 | 判定 | 最大短板 |
|----------|-----------|------|----------|
| LLM Inference 系统设计 | 3.2 | Lean Hire | 缺乏 production 规模经验 |
| CUDA Kernel 优化 | 2.3 | Lean No Hire | 零 CUDA 编程经验 |
| Triton Kernel 优化 | 2.0 | No Hire 边缘 | 零 Triton 经验 |
| RAG Infrastructure | 3.6 | Hire | 规模偏小，缺 multi-hop |
| GPU Cluster / K8s | 2.2 | Lean No Hire | 零集群管理经验 |
| Staff-level AI Infra | 2.8 | Lean No Hire | 广度和规模不足 |
| Production Debugging | 2.5 | Lean No Hire | 零 production debugging |
| Behavioral + Project | 3.5 | Hire 边缘 | 需要更好的 narrative |

### 整体判定：Lean No Hire → 目标 Hire

**核心差距：**
1. CUDA/Triton 实操能力为零（最致命短板）
2. Production 规模经验缺失（无法讲出真实故障处理 story）
3. 分布式/集群经验缺失

**核心优势：**
1. vLLM 社区贡献（EAGLE-3 PR 合入）证明了学习能力和代码能力
2. RAG 独立负责且有量化指标（RAGAS 90%）
3. 学校背景好（浙大+西交），基础扎实

**8-12 周补强优先级：**
1. **Week 1-4:** CUDA kernel 实操（reduction → softmax → GEMM → FlashAttention 简化版）
2. **Week 3-6:** Triton kernel（跟着 tutorial 写完所有 example）
3. **Week 5-8:** Production debugging scenario 练习 + Nsight 工具实操
4. **Week 7-10:** 系统设计 mock interview（每周 2 场）
5. **Week 9-12:** Behavioral story 打磨 + 全真模拟面试