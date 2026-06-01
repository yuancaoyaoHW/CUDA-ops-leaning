# RAG Evaluation Benchmark Protocol

## 目的

定义 RAG 系统质量评测的标准流程，确保检索质量、生成质量和系统性能的评测结果可复现、有统计意义。

---

## 1. 评测数据集标准

### 数据集构成

| 组成部分 | 要求 | 数量 |
|---------|------|------|
| 文档集 | 多格式、多主题、有 ground truth | 1K-10K 文档 |
| QA 评测对 | 含 question + ground_truth answer + source docs | 500 对 |
| 问题类型覆盖 | 事实型/解释型/对比型/多跳/否定型 | 按比例分配 |

### 问题类型分布

| 类型 | 占比 | 示例 |
|------|------|------|
| 事实型 (What/Who/When) | 40% | "X 的发布日期是什么？" |
| 解释型 (Why/How) | 30% | "为什么要用 PagedAttention？" |
| 对比型 (Compare) | 15% | "vLLM 和 SGLang 的区别？" |
| 多跳推理 (Multi-hop) | 10% | "A 用了什么技术，该技术的作者是谁？" |
| 否定型 (Not found) | 5% | "文档中没有的信息" |

### 数据集质量要求

- Ground truth 由人工标注或从权威来源提取
- 每个 QA 对标注对应的 source document(s)
- 定期更新，避免数据泄露到 LLM 训练集

---

## 2. 检索质量评测

### 指标定义

```python
def precision_at_k(retrieved_docs, relevant_docs, k):
    """前 K 个检索结果中相关文档的比例"""
    top_k = retrieved_docs[:k]
    relevant_in_top_k = len(set(top_k) & set(relevant_docs))
    return relevant_in_top_k / k

def recall_at_k(retrieved_docs, relevant_docs, k):
    """前 K 个检索结果覆盖了多少相关文档"""
    top_k = retrieved_docs[:k]
    relevant_in_top_k = len(set(top_k) & set(relevant_docs))
    return relevant_in_top_k / len(relevant_docs)

def mrr_at_k(retrieved_docs, relevant_docs, k):
    """Mean Reciprocal Rank"""
    for i, doc in enumerate(retrieved_docs[:k]):
        if doc in relevant_docs:
            return 1.0 / (i + 1)
    return 0.0
```

### 评测流程

```python
def evaluate_retrieval(retrieval_fn, eval_dataset, k=5):
    precisions, recalls, mrrs = [], [], []
    
    for qa in eval_dataset:
        retrieved = retrieval_fn(qa["question"], top_k=k)
        relevant = qa["source_docs"]
        
        precisions.append(precision_at_k(retrieved, relevant, k))
        recalls.append(recall_at_k(retrieved, relevant, k))
        mrrs.append(mrr_at_k(retrieved, relevant, k))
    
    return {
        "precision@k": np.mean(precisions),
        "recall@k": np.mean(recalls),
        "mrr@k": np.mean(mrrs),
        "num_queries": len(eval_dataset),
    }
```

---

## 3. 生成质量评测（RAGAS）

### 指标定义

| 指标 | 评测内容 | 计算方式 |
|------|---------|---------|
| Faithfulness | 答案是否基于检索内容 | LLM 判断每个 claim 是否有 context 支撑 |
| Answer Relevancy | 答案是否回答了问题 | 从答案反向生成问题，计算与原问题的相似度 |
| Context Precision | 检索内容中相关部分的排名 | 相关 context 是否排在前面 |
| Context Recall | 是否检索到所有需要的信息 | ground truth 中的信息是否都能在 context 中找到 |

### 评测流程

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness, answer_relevancy,
    context_precision, context_recall
)
from datasets import Dataset

def run_ragas_evaluation(eval_dataset, retrieval_fn, generation_fn):
    """完整 RAGAS 评测流程"""
    records = []
    for qa in eval_dataset:
        contexts = retrieval_fn(qa["question"], top_k=5)
        answer = generation_fn(qa["question"], contexts)
        records.append({
            "question": qa["question"],
            "answer": answer,
            "contexts": [c.text for c in contexts],
            "ground_truth": qa["ground_truth"],
        })
    
    dataset = Dataset.from_list(records)
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy,
                 context_precision, context_recall],
    )
    return scores
```

### 评测配置

```yaml
# configs/eval_config.yaml
ragas:
  llm: "gpt-4"  # 或本地模型
  embeddings: "text-embedding-3-small"
  metrics:
    - faithfulness
    - answer_relevancy
    - context_precision
    - context_recall
  num_samples: 500
  batch_size: 10
```

---

## 4. 策略对比实验设计

### A/B 测试框架

```python
def compare_strategies(strategies, eval_dataset, generation_fn):
    """对比多个检索策略"""
    results = {}
    for name, retrieval_fn in strategies.items():
        # 检索质量
        retrieval_metrics = evaluate_retrieval(retrieval_fn, eval_dataset)
        # 生成质量（RAGAS）
        ragas_metrics = run_ragas_evaluation(eval_dataset, retrieval_fn, generation_fn)
        # 延迟
        latency_metrics = benchmark_latency(retrieval_fn, eval_dataset)
        
        results[name] = {
            **retrieval_metrics,
            **ragas_metrics,
            **latency_metrics,
        }
    return results
```

### 消融实验设计

```python
ablation_configs = [
    {"name": "full", "vector": True, "bm25": True, "rerank": True, "filter": True},
    {"name": "no_filter", "vector": True, "bm25": True, "rerank": True, "filter": False},
    {"name": "no_rerank", "vector": True, "bm25": True, "rerank": False, "filter": True},
    {"name": "no_bm25", "vector": True, "bm25": False, "rerank": True, "filter": True},
    {"name": "vector_only", "vector": True, "bm25": False, "rerank": False, "filter": False},
]
```

---

## 5. 延迟 Benchmark

### 测量方法

```python
import time

def benchmark_latency(retrieval_fn, queries, num_runs=3):
    """测量检索延迟"""
    latencies = []
    
    for _ in range(num_runs):
        for query in queries:
            start = time.perf_counter()
            retrieval_fn(query)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms
    
    return {
        "latency_p50_ms": np.percentile(latencies, 50),
        "latency_p95_ms": np.percentile(latencies, 95),
        "latency_p99_ms": np.percentile(latencies, 99),
        "latency_mean_ms": np.mean(latencies),
    }
```

### E2E Latency Breakdown

```python
def benchmark_e2e_breakdown(query, retrieval_fn, generation_fn):
    """端到端延迟分解"""
    t0 = time.perf_counter()
    
    # Query embedding
    query_embedding = embed(query)
    t1 = time.perf_counter()
    
    # Vector search
    vector_results = vector_search(query_embedding)
    t2 = time.perf_counter()
    
    # BM25 search
    bm25_results = bm25_search(query)
    t3 = time.perf_counter()
    
    # Reranking
    reranked = rerank(vector_results + bm25_results, query)
    t4 = time.perf_counter()
    
    # LLM generation
    answer = generate(query, reranked[:5])
    t5 = time.perf_counter()
    
    return {
        "embedding_ms": (t1 - t0) * 1000,
        "vector_search_ms": (t2 - t1) * 1000,
        "bm25_search_ms": (t3 - t2) * 1000,
        "reranking_ms": (t4 - t3) * 1000,
        "generation_ms": (t5 - t4) * 1000,
        "total_ms": (t5 - t0) * 1000,
    }
```

---

## 6. 规模扩展性测试

### 文档规模 Sweep

```python
doc_counts = [1000, 5000, 10000, 50000, 100000]

for num_docs in doc_counts:
    # 构建索引
    index = build_index(documents[:num_docs])
    
    # 测量检索延迟
    latency = benchmark_latency(index.search, test_queries)
    
    # 测量索引大小
    index_size = get_index_size(index)
    
    # 测量检索质量（是否因规模增大而下降）
    quality = evaluate_retrieval(index.search, eval_dataset)
```

### 并发测试

```python
concurrency_levels = [1, 2, 4, 8, 16]

for concurrency in concurrency_levels:
    qps, latency = benchmark_concurrent(
        retrieval_fn, test_queries, concurrency
    )
    # 记录: concurrency, qps, latency_p50, latency_p95
```

---

## 7. 结果报告格式

### 标准输出

```json
{
  "experiment": "retrieval_strategy_comparison",
  "timestamp": "2026-06-01T05:00:00Z",
  "dataset": {
    "num_docs": 10000,
    "num_chunks": 50000,
    "num_eval_queries": 500,
    "embedding_model": "BGE-large-zh"
  },
  "results": {
    "vector_only": {
      "precision@5": 0.72,
      "recall@5": 0.76,
      "mrr@5": 0.68,
      "faithfulness": 0.82,
      "answer_relevancy": 0.87,
      "latency_p95_ms": 45
    },
    "hybrid_rerank": {
      "precision@5": 0.83,
      "recall@5": 0.85,
      "mrr@5": 0.79,
      "faithfulness": 0.89,
      "answer_relevancy": 0.92,
      "latency_p95_ms": 175
    }
  }
}
```

---

## 8. 常见陷阱

| 陷阱 | 影响 | 规避 |
|------|------|------|
| 评测集泄露到索引 | 虚高的 recall | 评测集与索引文档分离 |
| 只看检索不看生成 | 检索好但答案差 | 同时评测 retrieval + generation |
| 忽略延迟 | 质量高但不可用 | 同时报告 quality + latency |
| 评测集太小 | 统计不显著 | 至少 200 QA 对 |
| 问题类型单一 | 不能反映真实场景 | 覆盖多种问题类型 |
| 未记录 LLM 版本 | 结果不可复现 | 记录所有依赖版本 |
| 只看平均值 | 忽略 worst case | 按问题类型分解分析 |
