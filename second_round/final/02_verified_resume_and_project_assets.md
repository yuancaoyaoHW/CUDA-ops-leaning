# 已验证简历与项目资产清单

> 生成日期：2026-06-01
> 基于 claim-audit-resume-agent 审计结果

---

## 一、已验证可安全使用的简历 Bullet

以下内容有简历原文直接支撑，可在任何版本简历中使用：

### 推理优化方向

1. ✅ 独立实现 EAGLE-3 Speculative Decoding proposer 并合入 vLLM-Ascend 社区主线（PR #1032），为 vLLM V1 架构新增推测解码能力
2. ✅ 在 Atlas 3000/310P3 上验证 EAGLE-3 推测解码效果：输出吞吐 +55%（9.22→14.30 tok/s），TPOT -39%（108.18→65.76 ms）
3. ✅ Ascend 910B 上 num_spec_tokens=2 时 mean acceptance length 达 1.63，token-1/token-2 接受率 70%/47%
4. ✅ 提交 vLLM-MindSpore PR #1020，为 Qwen2/Qwen2.5 适配 EAGLE-3 推测解码，打通 draft model → hidden states → KV cache → rejection sampler → runner 执行链路
5. ✅ 参与 KV-select、Sparse Attention、Suffix Decoding 内部适配，优化长上下文场景下 KV cache 访问效率
6. ✅ 按社区 reviewer 反馈完成 proposer 接口适配、runner 执行链路收敛、测试补充与冲突处理

### RAG 方向

7. ✅ 独立负责非结构化问数系统后端与 RAG 链路，实现文档解析、chunk 构建、索引入库、向量检索、上下文组装与答案生成
8. ✅ 设计快速问答与 Research 问答双链路
9. ✅ 抽象检索与 metadata filters 查询接口，支持按文档来源、业务字段与时间条件过滤检索
10. ✅ 使用 RAGAS 构建问答质量评测流程，问答准确率达到 90%
11. ✅ 完善日志定位、失败分支和结果回写逻辑

### 教育背景

12. ✅ 浙江大学硕士（计算机技术）2021.09-2025.03
13. ✅ 西安交通大学学士（信息与计算科学）2016.09-2020.07

---

## 二、标注后可用的合理推断

以下内容基于 PR 合入事实的合理推断，使用时需标注"推断"：

14. ⚠️ 分析并理解 vLLM V1 架构中 scheduler、model runner、KV cache manager 的交互机制 → 来源：做 PR 的前提
15. ⚠️ 定位并修复推测解码中 hidden states 传递、rejection sampling 边界条件等问题 → 来源：PR review 过程
16. ⚠️ 按社区 CI 标准补充单元测试，覆盖 proposer 接口、KV cache 分配与 token 验证链路 → 来源：PR 合入通常需要测试

---

## 三、需要候选人确认后才能使用

17. ❓ "经过 3 轮 review" → 改为"经过多轮 review"（除非确认具体轮次）
18. ❓ Research 链路精简后检索效率提升 → 需补充具体数据
19. ❓ 适配方案被认可为后续模型接入的参考模板 → 需确认社区反馈
20. ❓ RAGAS 评测覆盖 faithfulness、answer relevancy、context precision → 需确认具体维度
21. ❓ 设计基于语义边界的 chunk 切分策略 → 需确认具体策略
22. ❓ 快速问答支持秒级响应 → 需确认具体延迟数据
23. ❓ 支持增量更新 → 需确认是否实现

---

## 四、禁止使用（🚫）

24. 🚫 "熟悉 CUDA 编程" — 无实际 kernel 代码
25. 🚫 "分布式推理经验（TP/PP）" — 无实操记录
26. 🚫 "GPU 性能优化 / Nsight profiling" — 无 profiling 产出
27. 🚫 "TensorRT-LLM 部署经验" — 无实操记录
28. 🚫 "Kubernetes GPU serving" — 无实操记录

---

## 五、项目资产与简历映射

| 项目 | 完成后可写的 Bullet | 当前状态 |
|------|-------------------|---------|
| CUDA Ops Lab | "Implemented CUDA kernels for LLM inference (GEMM, softmax, RMSNorm, FlashAttention) achieving X% of peak bandwidth" | 未开始 |
| Triton Kernels Lab | "Developed Triton kernels and benchmarked against CUDA implementations" | 未开始 |
| LLM Inference Benchmark | "Designed LLM inference benchmark suite evaluating vLLM/SGLang across N configurations" | 未开始 |
| RAG Eval Lab | "Built RAG evaluation framework comparing vector/hybrid/rerank with RAGAS metrics" | 未开始 |
| GPU Observability | "Implemented GPU serving observability with Prometheus/Grafana" | 未开始 |

---

## 六、开源贡献资产

| 贡献 | 状态 | 简历价值 |
|------|------|---------|
| vLLM-Ascend PR #1032 | ✅ 已合入 | 可直接使用 |
| vLLM-MindSpore PR #1020 | ✅ 已提交 | 标注"已提交" |
| vLLM speculators 文档 PR | 计划中 | 完成后可用 |
| SGLang Ascend 适配 PR | 计划中 | 完成后可用 |
