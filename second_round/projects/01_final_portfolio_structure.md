# 最终项目组合结构

## 总览

```
Portfolio
├── 项目 A: CUDA Ops Learning Lab        ← 补强核心缺口（CUDA kernel 编写）
├── 项目 B: Triton Kernels Lab           ← 补强 Triton 编程能力
├── 项目 C: LLM Inference Benchmark Lab  ← 展示性能工程方法论
├── 项目 D: RAG Infrastructure Eval Lab  ← 强化已有 RAG 经验
└── 项目 E: GPU Serving Observability    ← 展示生产化能力
```

---

## 项目定位与权重

| 项目 | 核心价值 | 面试权重 | 简历权重 | 优先级 |
|------|---------|---------|---------|--------|
| A: CUDA Ops Lab | 证明底层 GPU 编程能力 | 35% | 30% | P0 |
| B: Triton Kernels Lab | 展示现代 kernel 开发能力 | 15% | 15% | P1 |
| C: LLM Inference Benchmark | 展示性能分析方法论 | 25% | 25% | P0 |
| D: RAG Infra Eval Lab | 强化系统设计 + 已有经验 | 15% | 20% | P1 |
| E: GPU Serving Observability | 展示生产化思维 | 10% | 10% | P2 |

---

## 目录结构总览

```
CUDA-ops-leaning/
├── kernels/                          # 已有，不覆盖
├── second_round/
│   ├── projects/                     # 项目规划文档（本目录）
│   └── benchmarks/                   # Benchmark 协议文档
├── cuda-ops-lab/                     # 项目 A 代码（新建）
│   ├── kernels/
│   │   ├── 01_vector_add/
│   │   ├── 02_reduction/
│   │   ├── 03_matmul/
│   │   ├── 04_softmax/
│   │   ├── 05_layernorm/
│   │   ├── 06_rmsnorm/
│   │   ├── 07_rope/
│   │   └── 08_flash_attention_toy/
│   ├── tests/
│   ├── benchmarks/
│   ├── profiling/
│   ├── CMakeLists.txt
│   ├── Makefile
│   └── README.md
├── triton-kernels-lab/               # 项目 B 代码（新建）
│   ├── kernels/
│   ├── benchmarks/
│   ├── tests/
│   └── README.md
├── llm-inference-benchmark-lab/      # 项目 C 代码（新建）
│   ├── benchmark/
│   ├── workloads/
│   ├── results/
│   ├── plots/
│   ├── docs/
│   └── README.md
├── rag-infra-eval-lab/               # 项目 D 代码（新建）
│   ├── src/
│   ├── evaluation/
│   ├── benchmarks/
│   ├── docs/
│   └── README.md
└── gpu-serving-observability/        # 项目 E 代码（新建）
    ├── monitoring/
    ├── simulation/
    ├── dashboards/
    └── README.md
```

---

## 项目间依赖关系

```
项目 A (CUDA Ops) ──────────────────────────────────────┐
    │                                                    │
    ├── 产出 kernel 性能数据 ──→ 项目 C (Benchmark)      │
    │                                                    │
    └── 产出 CUDA kernel ──→ 项目 B (Triton 对比)        │
                                                         │
项目 C (Benchmark) ──→ 项目 E (Observability 指标设计)   │
                                                         │
项目 D (RAG Eval) ──→ 项目 E (延迟监控设计)              │
                                                         │
所有项目 ──→ 简历 bullets + 面试素材 ─────────────────────┘
```

---

## 执行时间线

| 周次 | 项目 A | 项目 B | 项目 C | 项目 D | 项目 E |
|------|--------|--------|--------|--------|--------|
| W1-2 | vector_add, reduction | — | — | — | — |
| W3-4 | matmul (核心) | — | 框架搭建 | — | — |
| W5-6 | softmax, layernorm, rmsnorm | vector_add, softmax | 基础实验 | 复盘 + 评测集 | — |
| W7-8 | rope, flash_attention_toy | matmul, layernorm | 深度分析 | 混合检索实验 | — |
| W9-10 | 整理 + profiling 报告 | fused_attention, quant | 报告 + 可视化 | RAGAS 评测 | 设计 + 实现 |
| W11-12 | README 完善 | 对比报告 | 最终报告 | 评测报告 | 完善 + 文档 |

---

## 验收标准总表

| 项目 | Done Criteria |
|------|--------------|
| A | 8 个算子全部实现 + 正确性测试通过 + Nsight profiling 报告 + benchmark 数据 |
| B | 6 个 Triton kernel + 与 CUDA 版本性能对比表 |
| C | 完整 benchmark 报告 + 可视化图表 + 优化建议文档 |
| D | RAGAS 评测完成 + 检索策略对比 + 延迟 benchmark |
| E | Grafana dashboard JSON + Prometheus 配置 + autoscaling 模拟脚本 |

---

## 差异化策略

### 与其他候选人的区别
1. **从算子到系统的全栈覆盖** — 不只是调 API，能写 kernel 也能做系统设计
2. **有 benchmark 数据支撑** — 每个项目都有量化结果
3. **有开源贡献背景** — vLLM-Ascend EAGLE-3 推测解码
4. **双平台经验** — Ascend + NVIDIA（通过本项目补强）
5. **方法论驱动** — Nsight profiling + roofline analysis + RAGAS 评测

### GitHub Profile 展示

```
📌 Pinned Repositories:
1. cuda-ops-lab — "CUDA kernel implementations for LLM inference (GEMM, FlashAttention, RMSNorm)"
2. triton-kernels-lab — "Triton kernel implementations with CUDA performance comparison"
3. llm-inference-benchmark-lab — "Systematic LLM serving benchmark suite (vLLM/SGLang)"
4. rag-infra-eval-lab — "RAG infrastructure with comprehensive RAGAS evaluation"
```
