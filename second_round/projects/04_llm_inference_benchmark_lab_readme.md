# LLM Inference Benchmark Lab

> 系统化评测 LLM 推理框架性能，建立完整的 benchmark 方法论、实验矩阵和可视化报告。

[![vLLM](https://img.shields.io/badge/vLLM-latest-blue.svg)](https://github.com/vllm-project/vllm)
[![SGLang](https://img.shields.io/badge/SGLang-latest-green.svg)](https://github.com/sgl-project/sglang)

## Motivation

LLM 推理性能优化不是跑个脚本看数字，而是需要系统化的方法论：单变量控制、统计显著性、latency breakdown、roofline 分析。本项目建立完整的 benchmark 框架，覆盖 vLLM/SGLang 在不同 batch/seq/concurrency 配置下的性能表现，产出可复现的实验报告和优化建议。

## Key Results

> ⚠️ 以下为目标值，需完成实验后填入实际数据

- 目标：覆盖 200+ 配置组合的完整实验矩阵
- 目标：识别各框架在不同 workload 下的最优配置
- 目标：产出 latency breakdown 和 roofline 分析
- 已有基础：Atlas 3000 上 vLLM 55% throughput 提升经验（9.22→14.30 tok/s）

## Directory Structure

```
llm-inference-benchmark-lab/
├── benchmark/
│   ├── core/
│   │   ├── runner.py              # 统一 benchmark runner
│   │   ├── metrics.py             # 指标计算 (TTFT, TPOT, throughput)
│   │   ├── workload.py            # Workload 生成器
│   │   ├── reporter.py            # JSON/CSV 报告生成
│   │   └── config.py              # 实验配置管理
│   ├── frameworks/
│   │   ├── base.py                # Framework adapter 基类
│   │   ├── vllm_bench.py          # vLLM adapter
│   │   └── sglang_bench.py        # SGLang adapter
│   ├── experiments/
│   │   ├── batch_sweep.py         # Batch size 扫描
│   │   ├── seq_len_sweep.py       # Sequence length 扫描
│   │   ├── concurrency_sweep.py   # 并发度扫描
│   │   ├── framework_compare.py   # 框架对比
│   │   ├── param_tuning.py        # 框架参数调优
│   │   └── quantization_impact.py # 量化影响
│   └── analysis/
│       ├── latency_breakdown.py   # 延迟分解分析
│       ├── roofline_analysis.py   # Roofline 绘图
│       └── statistical.py         # 统计分析工具
├── workloads/
│   ├── synthetic.py               # 合成 workload 生成器
│   ├── sharegpt_sample.json       # ShareGPT 采样数据（小规模）
│   └── README.md                  # Workload 说明
├── results/                       # 实验结果 (JSON + CSV)
├── plots/                         # 可视化图表
├── docs/
│   ├── methodology.md             # 方法论文档
│   ├── benchmark_report.md        # 完整报告
│   └── optimization_guide.md      # 优化建议
└── README.md
```

---

## 实验矩阵

### 测试维度

| 维度 | 取值 |
|------|------|
| Framework | vLLM, SGLang |
| Model | Qwen2.5-7B, Llama-3-8B（或可用小模型） |
| Concurrency | 1, 4, 8, 16, 32, 64 |
| Batch Size | 1, 4, 8, 16, 32 |
| Input Length | 128, 512, 1024, 2048, 4096 |
| Output Length | 64, 128, 256, 512 |

### 指标体系

| 类别 | 指标 | 单位 | 说明 |
|------|------|------|------|
| 延迟 | TTFT | ms | Time To First Token |
| 延迟 | TPOT | ms | Time Per Output Token |
| 延迟 | E2E p50/p95/p99 | ms | 端到端延迟百分位 |
| 吞吐 | tokens/s | tok/s | 输出 token 速率 |
| 吞吐 | QPS | req/s | 请求处理速率 |
| 资源 | GPU Utilization | % | SM 活跃率 |
| 资源 | Memory Usage | GB | GPU memory 占用 |

---

## 核心实验设计

### 实验 1: Batch Size Sweep

```
固定: model=Qwen2.5-7B, input_len=512, output_len=256, framework=vLLM
变量: batch_size=[1, 4, 8, 16, 32]
指标: throughput, TPOT, GPU utilization, memory
预期: throughput 先线性增长后饱和，memory 线性增长
```

### 实验 2: Sequence Length Sweep

```
固定: model=Qwen2.5-7B, batch=8, output_len=256, framework=vLLM
变量: input_len=[128, 512, 1024, 2048, 4096]
指标: TTFT, prefill throughput, memory
预期: TTFT 近似线性增长（compute-bound）
```

### 实验 3: Concurrency Sweep

```
固定: model=Qwen2.5-7B, workload=synthetic(512→256)
变量: concurrency=[1, 4, 8, 16, 32, 64]
指标: throughput, QPS, p50/p95/p99 latency
预期: throughput 增长后饱和，tail latency 恶化
```

### 实验 4: Framework Compare

```
固定: model=Qwen2.5-7B, concurrency=16, workload=synthetic
对比: vLLM (default) vs vLLM (optimized) vs SGLang (default) vs SGLang (optimized)
指标: 全部指标
```

### 实验 5: Parameter Tuning (vLLM)

```
参数空间:
- max_num_seqs: [64, 128, 256]
- gpu_memory_utilization: [0.85, 0.9, 0.95]
- enable_chunked_prefill: [true, false]
目标: 找到最优配置组合
```

---

## Setup

### 环境要求
- NVIDIA GPU (16GB+ VRAM，推荐 A30/A100)
- Python 3.10+
- vLLM, SGLang (latest)

### 安装

```bash
# 创建环境
conda create -n llm-bench python=3.10 && conda activate llm-bench

# 安装框架
pip install vllm
pip install "sglang[all]"

# 安装项目依赖
pip install -e .

# 下载小模型用于测试（不下载大模型权重）
# 使用 Qwen2.5-1.5B 做快速验证，Qwen2.5-7B 做正式实验
```

### 运行实验

```bash
# 单个实验
python benchmark/experiments/batch_sweep.py --model Qwen/Qwen2.5-7B --gpu 0

# 完整实验矩阵
python benchmark/experiments/run_all.py --config configs/full_matrix.yaml

# 生成报告
python benchmark/analysis/generate_report.py --results results/ --output docs/benchmark_report.md
```

---

## Correctness Test Design

```python
def validate_benchmark_result(result):
    """验证 benchmark 结果的合理性"""
    assert result["ttft_ms"] > 0, "TTFT must be positive"
    assert result["tpot_ms"] > 0, "TPOT must be positive"
    assert result["throughput_tps"] > 0, "Throughput must be positive"
    # TTFT 应该随 input_len 增长
    # Throughput 应该随 batch_size 增长（到饱和点）
    # Memory 不应超过 GPU 总量
    assert result["memory_gb"] < GPU_TOTAL_MEMORY_GB
```

## Benchmark Design Protocol

1. **预热**: 前 10 个 request 不计入统计
2. **重复**: 每个配置至少 100 个 request
3. **统计**: 报告 median, p50, p95, p99, std
4. **隔离**: 每次实验重启 serving 进程
5. **公平**: 相同硬件、相同模型权重、相同 workload

## Expected Metrics

| 实验 | 关键发现（预期） |
|------|----------------|
| Batch Sweep | batch=X 时 throughput 饱和，此时 GPU util ~Y% |
| Seq Len Sweep | TTFT 与 seq_len 近似线性，拐点在 seq=Z |
| Concurrency | 最大可支持 concurrency=N（p99 < 5s SLA） |
| Framework Compare | SGLang 在 prefix-heavy workload 下优于 vLLM X% |
| Param Tuning | 最优配置比默认配置 throughput 提升 X% |

## Profiling Method

```bash
# Nsight Systems: 端到端 timeline
nsys profile --trace=cuda,nvtx -o vllm_profile \
    python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B

# 分析 decode iteration 中各 kernel 时间占比
nsys stats vllm_profile.nsys-rep
```

## Resume Bullet

*（完成后使用）*
- "Designed comprehensive LLM inference benchmark suite evaluating vLLM/SGLang across 200+ configurations (batch/seq/concurrency), identifying optimal serving parameters for different workload patterns"
- "Achieved X% throughput improvement through systematic profiling and parameter tuning on [GPU]"

## Interview Talking Points

### 方法论表达

- "我的 benchmark 方法论：单变量 sweep 找拐点 → 多变量找最优 → Nsight 做 breakdown → roofline 定位瓶颈"
- "我发现 batch=X 是 throughput 饱和点，因为此时 GPU memory 成为瓶颈"
- "SGLang 在 prefix-heavy workload 下比 vLLM 快 X%，因为 RadixAttention 复用了 KV cache"

### 深度追问

1. "Batch size 增大为什么 throughput 先升后降？" → memory 限制 + scheduling overhead
2. "TTFT 和 TPOT 分别受什么影响？" → prefill compute vs decode memory bandwidth
3. "怎么判断当前是 compute-bound 还是 memory-bound？" → roofline + Nsight metrics
4. "vLLM 参数怎么调？" → 具体实验数据 + tradeoff 分析
