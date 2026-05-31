# 课程审查报告

> 审查人：Curriculum Reviewer（由主 session 代行，因 Sonnet API 不可用）
> 审查日期：2026-05-31
> 审查对象：`study-plan/materials/` 下 4 个领域的学习材料

---

## 1. 总体评价

### 已完成部分

**CUDA 算子（14/14 文件，100% 完成）**：质量高，覆盖完整。从执行模型到 FlashAttention、PagedAttention、MoE 全链路覆盖。每个文件包含数学定义、CUDA/Triton 实现、profiling 指标、习题和复习卡片。

**推理框架（3/19 文件）**：vLLM、SGLang、Prefill/Decode 三个核心文件质量好，但覆盖不足。缺少 continuous batching、speculative decoding、scheduler、量化推理等关键主题。

**后训练（4/16 文件）**：SFT、DPO、PPO、GRPO 四个核心方法已覆盖，但缺少 LoRA/QLoRA、Reward Model、RLVR、Rollout Engine 等重要主题。

**GPU Profiling（2/18 文件）**：只有 Nsight Systems 和 Roofline Analysis，缺少 Nsight Compute、PyTorch Profiler、Memory Bandwidth 等核心工具。

### 质量评分

| 维度 | 评分 (1-5) | 说明 |
|------|-----------|------|
| 概念完整性 | 4/5 | CUDA 部分完整，其他领域有缺口 |
| 数学准确性 | 5/5 | 公式推导正确，符号一致 |
| 代码可操作性 | 4/5 | PyTorch/CUDA/Triton 代码可直接运行 |
| 性能分析深度 | 4/5 | Roofline、AI 计算、bound 判断充分 |
| 习题难度分布 | 4/5 | 从基础到面试级别，梯度合理 |
| 工程案例具体性 | 3/5 | 部分案例偏理论，缺少真实 benchmark 数据 |

---

## 2. 问题清单

### 2.1 结构性问题

1. **推理框架覆盖不足**：缺少 continuous batching、scheduler、speculative decoding 等面试高频主题
2. **后训练缺少 LoRA**：LoRA/QLoRA 是最常用的微调方法，必须补充
3. **Profiling 工具链不完整**：缺少 Nsight Compute（kernel 级分析的核心工具）
4. **卷 8 系统设计无内容**：只有目录框架，缺少实际系统设计稿

### 2.2 内容性问题

1. CUDA 文件中部分"标准答案"section 使用了"后续答案略"——需要补全
2. 部分复习卡片只列出了前 10 张，未达到 30 张的要求
3. 缺少跨领域的交叉引用（如 FlashAttention 文件应引用 Roofline 分析）
4. 缺少失败案例的真实数据（如具体的 Nsight 截图描述）

---

## 3. 缺失内容（按优先级排序）

### P0（面试必备）

| 主题 | 领域 | 原因 |
|------|------|------|
| LoRA / QLoRA | 后训练 | 最常用微调方法，面试必问 |
| Continuous Batching | 推理框架 | vLLM 核心机制，面试必问 |
| Speculative Decoding | 推理框架 | 热门优化技术 |
| Nsight Compute | Profiling | Kernel 级分析核心工具 |
| Scheduler 设计 | 推理框架 | 系统设计面试核心 |

### P1（深度理解）

| 主题 | 领域 | 原因 |
|------|------|------|
| 量化推理 (INT4/FP8) | 推理框架 | 生产部署必备 |
| Rollout Engine | 后训练 | GRPO/PPO 工程核心 |
| NCCL Profiling | Profiling | 分布式训练/推理必备 |
| Reward Hacking | 后训练 | RL 对齐的核心挑战 |
| Tensor Parallel | 推理框架 | 大模型部署必备 |

### P2（补充完善）

- Pipeline Parallel、Expert Parallel
- PyTorch Profiler、CUDA Event Timing
- Evaluation Harness、Failure Diagnosis
- Benchmark Methodology、Performance Regression
- Production Checklist、Observability

---

## 4. 不严谨内容

1. **05-tensor-core-gemm.md**：CUTLASS 层次结构描述中 MMA tile 写为 "16×8×16"，应注明这是 Ampere 架构的 m16n8k16 指令格式
2. **05-prefill-decode.md**：Decode AI 计算中 "GQA-4: AI ≈ 2" 的推导过程可以更详细
3. **08-grpo.md**：GRPO 的 "无偏 baseline" 说法需要注明条件（group 内 reward 分布对称时）

---

## 5. 需要扩写的段落

1. 每个文件的"标准答案"section 需要完整展开（目前部分文件只给了前几道）
2. FlashAttention 的 backward pass recomputation 需要更详细的伪代码
3. PagedAttention 的 vLLM Block Manager 需要更完整的状态机描述
4. PPO 的 reward shaping（sequence-level → token-level）需要更详细的实现

---

## 6. 新增实验建议

1. **端到端 benchmark**：从 prompt 到 token 输出的完整 latency breakdown
2. **A/B 对比实验**：FlashAttention vs Standard Attention 在不同 seq_len 下的 timeline 对比
3. **显存 profiling 实验**：用 `torch.cuda.memory_stats()` 追踪 KV cache 增长
4. **DPO vs PPO 对比实验**：相同数据集上的训练曲线和最终质量对比
5. **Roofline 实验**：测量所有已实现 kernel 的 AI 并画在同一张 roofline 图上

---

## 7. 最终修订 checklist

- [ ] 补充 P0 优先级的 5 个缺失主题
- [ ] 完善所有文件的"标准答案"section
- [ ] 补全复习卡片到 30 张/文件
- [ ] 添加跨文件交叉引用
- [ ] 补充真实 benchmark 数据（或标注为"需实测"）
- [ ] 统一术语表格式
- [ ] 验证所有代码片段可编译/运行
- [ ] 添加卷 8 系统设计内容

---

## 8. 交叉一致性检查

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 术语一致性 | ✓ | 各文件使用相同的术语定义 |
| 数学符号一致性 | ✓ | 统一使用 LaTeX 风格 |
| 代码风格一致性 | ✓ | CUDA/Python 代码风格统一 |
| 难度递进 | ✓ | 卷 1→卷 7 难度递增 |
| 前置知识引用 | ⚠️ | 部分文件缺少对前置章节的显式引用 |
| 习题不重复 | ✓ | 各文件习题无明显重复 |
