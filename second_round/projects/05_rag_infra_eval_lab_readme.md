# RAG Infrastructure Evaluation Lab

> 生产级 RAG 系统评测实验室，覆盖检索策略对比、RAGAS 质量评测、延迟 benchmark 和成本分析。

[![RAGAS](https://img.shields.io/badge/RAGAS-latest-purple.svg)](https://docs.ragas.io/)
[![FAISS](https://img.shields.io/badge/FAISS-latest-blue.svg)](https://github.com/facebookresearch/faiss)

## Motivation

RAG 系统的质量评测需要系统化方法：不仅要测 retrieval precision/recall，还要测 generation faithfulness/relevancy，更要测端到端延迟和成本。本项目建立完整的 RAG 评测体系，覆盖向量检索、混合检索、重排序、metadata filter 等策略的对比实验，使用 RAGAS 框架进行质量评测。

## Key Results

> ⚠️ 以下为目标值，需完成实验后填入实际数据

- 已有基础：RAGAS 评测准确率 90%，faithfulness 0.87
- 目标：混合检索比纯向量检索准确率提升 5-10%
- 目标：Cross-encoder reranking 提升 answer relevancy 3-5%
- 目标：p95 retrieval latency < 200ms（10K 文档）

## Directory Structure

```
rag-infra-eval-lab/
├── src/
│   ├── retrieval/
│   │   ├── vector_search.py       # FAISS / Milvus 向量检索
│   │   ├── bm25_search.py         # BM25 关键词检索
│   │   ├── hybrid_search.py       # 混合检索 (vector + BM25)
│   │   ├── reranker.py            # Cross-encoder 重排序
│   │   └── metadata_filter.py     # Metadata 过滤
│   ├── ingestion/
│   │   ├── chunker.py             # 分块策略
│   │   └── embedder.py            # Embedding 生成
│   └── utils/
│       ├── metrics.py             # 自定义指标计算
│       └── data_loader.py         # 评测数据加载
├── evaluation/
│   ├── ragas_eval.py              # RAGAS 评测主脚本
│   ├── retrieval_eval.py          # 检索质量评测
│   ├── latency_bench.py           # 延迟 benchmark
│   ├── ablation_study.py          # 消融实验
│   └── eval_datasets/
│       ├── qa_pairs_500.json      # 500 QA 评测对
│       └── README.md              # 数据集说明
├── benchmarks/
│   ├── vector_db_bench.py         # 向量数据库性能对比
│   ├── scaling_bench.py           # 文档规模扩展性测试
│   └── cost_analysis.py           # 成本分析
├── configs/
│   ├── faiss_config.yaml
│   ├── milvus_config.yaml
│   └── eval_config.yaml
├── docs/
│   ├── system_design.md           # RAG 系统设计文档
│   ├── evaluation_report.md       # 评测报告
│   └── optimization_guide.md      # 优化建议
├── docker-compose.yml             # Milvus + 依赖服务
└── README.md
```

---

## 评测维度

### 1. Vector Search（FAISS, Milvus）

| 实验 | 配置 | 指标 |
|------|------|------|
| Index 类型对比 | FAISS Flat / IVF / HNSW | recall@K, latency, memory |
| 文档规模扩展 | 1K / 10K / 50K / 100K chunks | latency vs scale |
| Embedding 维度 | 768 / 1024 / 1536 | precision, index size |

### 2. Hybrid Search（BM25 + Vector）

| 实验 | 配置 | 指标 |
|------|------|------|
| 权重对比 | α=[0.3, 0.5, 0.7, 0.9] (vector weight) | precision@5, recall@5 |
| 融合策略 | RRF / weighted sum / learned | MRR@5 |
| vs 纯向量 | hybrid vs vector-only | Δ precision, Δ recall |

### 3. Rerank（Cross-encoder）

| 实验 | 配置 | 指标 |
|------|------|------|
| Reranker 对比 | BGE-reranker / cross-encoder / Cohere | precision@5, latency |
| Top-K 选择 | 初始 top-20 → rerank → top-5 | quality vs latency tradeoff |
| 有无 rerank | with vs without | Δ answer_relevancy |

### 4. Metadata Filter

| 实验 | 配置 | 指标 |
|------|------|------|
| Filter 类型 | 时间 / 来源 / 类别 | precision 提升 |
| Pre-filter vs Post-filter | 先过滤再检索 vs 先检索再过滤 | latency, recall |

### 5. RAGAS Evaluation

| 指标 | 定义 | 目标 |
|------|------|------|
| Faithfulness | 答案是否基于检索内容 | > 0.85 |
| Answer Relevancy | 答案是否回答了问题 | > 0.90 |
| Context Precision | 检索内容是否相关 | > 0.80 |
| Context Recall | 是否检索到所有相关内容 | > 0.85 |

### 6. Latency Benchmark

| 阶段 | 目标 p95 |
|------|----------|
| Query Embedding | < 20ms |
| Vector Search | < 50ms |
| BM25 Search | < 30ms |
| Reranking | < 100ms |
| LLM Generation | < 2000ms |
| **E2E Total** | **< 3000ms** |

---

## Setup

```bash
# 安装依赖
pip install faiss-gpu ragas langchain sentence-transformers rank_bm25
pip install pymilvus  # 如使用 Milvus

# 启动 Milvus（可选，用于大规模测试）
docker-compose up -d

# 准备评测数据
python evaluation/prepare_eval_data.py

# 运行评测
python evaluation/ragas_eval.py --config configs/eval_config.yaml

# 运行延迟 benchmark
python benchmarks/latency_bench.py --docs 10000
```

## Correctness Test Design

```python
def test_hybrid_search_improves_over_vector():
    """混合检索应该比纯向量检索效果更好"""
    vector_results = vector_search(queries, top_k=5)
    hybrid_results = hybrid_search(queries, top_k=5, alpha=0.7)
    
    vector_precision = compute_precision(vector_results, ground_truth)
    hybrid_precision = compute_precision(hybrid_results, ground_truth)
    
    assert hybrid_precision >= vector_precision, \
        f"Hybrid ({hybrid_precision:.3f}) should >= Vector ({vector_precision:.3f})"
```

## Benchmark Design

### 检索策略 A/B 测试

```python
strategies = [
    {"name": "vector_only", "fn": vector_search},
    {"name": "bm25_only", "fn": bm25_search},
    {"name": "hybrid_0.7", "fn": lambda q: hybrid_search(q, alpha=0.7)},
    {"name": "hybrid_rerank", "fn": lambda q: rerank(hybrid_search(q, alpha=0.7))},
]

for strategy in strategies:
    results = evaluate_strategy(strategy, eval_dataset)
    # 输出: precision@5, recall@5, MRR@5, latency_p50, latency_p95
```

## Expected Metrics

| 策略 | 目标 Precision@5 | 目标 Recall@5 | 目标 Latency p95 |
|------|-----------------|--------------|-----------------|
| Vector Only | 0.70 | 0.75 | < 50ms |
| BM25 Only | 0.60 | 0.65 | < 30ms |
| Hybrid (0.7V+0.3B) | 0.75 | 0.80 | < 80ms |
| Hybrid + Rerank | 0.82 | 0.82 | < 180ms |

## Profiling Method

```bash
# 延迟分解
python benchmarks/latency_bench.py --breakdown --output results/latency_breakdown.json

# 内存分析
python benchmarks/scaling_bench.py --docs 1000,10000,50000,100000
```

## Resume Bullet

*（完成后使用）*
- "Built RAG evaluation framework achieving 90% QA accuracy (RAGAS), with hybrid retrieval improving precision by X% over vector-only search"
- "Benchmarked RAG infrastructure at scale: p95 retrieval latency < 200ms with 10K+ documents, supporting vector/BM25/rerank pipeline"

## Interview Talking Points

### 系统设计回答

- "我的 RAG 系统采用混合检索架构：向量检索负责语义匹配，BM25 负责精确匹配，cross-encoder 做最终重排序"
- "评测我用 RAGAS 框架，覆盖 faithfulness/relevancy/precision/recall 四个维度"
- "通过消融实验，我发现 reranker 对 answer relevancy 贡献最大（+X%），而 hybrid search 对 recall 贡献最大（+Y%）"

### 深度追问

1. "Chunk size 怎么选？" → 实验数据：512 + overlap=50 在 precision/recall 平衡最好
2. "向量检索 miss 了怎么办？" → hybrid retrieval + reranking + query expansion
3. "文档更新怎么处理？" → incremental indexing + versioning + stale detection
4. "怎么保证 faithfulness？" → citation tracking + RAGAS 持续监控 + 阈值告警
