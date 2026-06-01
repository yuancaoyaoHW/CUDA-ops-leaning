# AI Infra 方向简历版本

> 面向：AI 基础设施平台岗（推理 serving、模型部署、AI 平台工程）
> 目标公司：华为昇腾推理引擎、CoreWeave、Baseten、阿里云 PAI、火山引擎、Together AI、Fireworks AI

---

## Headline

大模型推理系统工程师 | vLLM 生态贡献者 | Speculative Decoding + RAG 全链路

---

## Summary

专注大模型推理系统优化，在 vLLM 生态中独立实现 EAGLE-3 推测解码并合入社区主线（PR #1032），在昇腾 NPU 上实现 55% 吞吐提升。具备从推理算法设计到端到端 benchmark 验证的全链路能力。同时有 RAG 系统独立交付经验，覆盖从推理优化到应用落地的完整技术栈。正在系统补强 CUDA 与 GPU profiling 能力。

---

## Skills Section（按优先级排列）

```
推理系统: vLLM (V1 架构), Speculative Decoding (EAGLE-3), KV Cache Management, Continuous Batching, Decode Scheduling
开源贡献: vLLM-Ascend PR #1032 (merged), vLLM-MindSpore PR #1020
硬件平台: Ascend 910B, Atlas 3000/310P3, NPU 推理优化
性能验证: MT-Bench Benchmarking, Throughput/Latency/Acceptance Rate 多维度评测
RAG 系统: LangChain, RAGAS, 文档解析, 向量检索, Metadata Filters
编程语言: Python, C++
工具链: Linux, Docker, Git
```

---

## Project Ordering

1. **vLLM 生态推理优化与昇腾 NPU 适配**（主打）
2. **非结构化问数系统 RAG 后端**（辅助，体现全栈能力）

---

## Verified Bullet Points（可安全使用）

### 项目 1：vLLM 生态推理优化

1. ✅ 独立实现 EAGLE-3 Speculative Decoding proposer 并合入 vLLM-Ascend 社区主线（PR #1032），为 vLLM V1 架构新增推测解码能力
2. ✅ 在 Atlas 310P3 上验证推测解码效果：输出吞吐 +55%（9.22→14.30 tok/s），TPOT -39%（108→66 ms），spec_tokens=3
3. ✅ 打通 draft model → hidden states → KV cache → rejection sampler → runner 完整执行链路，实现端到端推测解码 pipeline
4. ✅ 提交 vLLM-MindSpore PR #1020，为 Qwen2.5-7B 适配 EAGLE-3，Ascend 910B 上 acceptance length 达 1.63
5. ✅ 分析 vLLM V1 架构中 scheduler、model runner、KV cache manager 的交互机制，基于架构理解设计 proposer 接口 [合理推断]
6. ✅ 按社区 reviewer 反馈完成多轮迭代：proposer 接口重构、runner 执行链路收敛、单元测试补充与合并冲突处理 [合理推断]
7. ✅ 参与 KV-select、Sparse Attention 内部适配，理解长上下文场景下 KV cache 访问与 attention 计算的性能特征

### 项目 2：RAG 后端

1. ✅ 独立负责非结构化问数系统后端与 RAG 链路，基于 Python + LangChain 实现从文档解析到答案生成的完整 pipeline
2. ✅ 设计快速问答与 Research 问答双链路架构，支持不同复杂度查询的差异化处理
3. ✅ 基于 RAGAS 框架构建评测流程，问答准确率达 90%
4. ✅ 抽象统一检索接口，支持 metadata filters 与多后端切换，降低业务层与检索层耦合

---

## Risky Bullet Points to Avoid

| 声明 | 原因 | 标记 |
|------|------|------|
| "熟悉 CUDA 编程" | 零 CUDA kernel 经验 | 🚫 Do Not Use |
| "分布式推理经验" | 无 TP/PP 实操 | 🚫 Do Not Use |
| "GPU 性能优化" | 无 profiling 产出 | 🚫 Do Not Use |
| "TensorRT-LLM 部署" | 无使用经验 | 🚫 Do Not Use |
| "K8s GPU serving" | 无 K8s 部署经验 | 🚫 Do Not Use |
| "量化模型部署" | 无量化实践 | 🚫 Do Not Use |
| "经过 3 轮 review" | 具体轮次未验证 | ⚠️ 改为"多轮" |

---

## Missing Evidence List

补充以下内容后可增强简历竞争力：

| 缺失项 | 补充方式 | 优先级 |
|--------|----------|--------|
| CUDA kernel 开发经验 | 完成 softmax/GEMM kernel 实现 | P0 |
| GPU profiling 产出 | 用 Nsight Compute 分析自实现 kernel | P0 |
| Production serving 经验 | 部署一个有真实流量的推理服务 | P1 |
| 分布式推理理解 | 学习 TP/PP 并在 vLLM 中验证 | P1 |
| K8s 部署能力 | 用 K8s 部署推理服务 | P2 |
| 具体 RAG 规模数据 | 补充文档数、QPS、延迟数据 | P1 |
