# RAG Quality 事故演练 Playbook

---

## 事故 1：Retrieval Recall Drops（检索召回率下降）

### Symptom（现象）
- RAG 系统回答质量明显下降
- 用户反馈"找不到相关信息"或"回答不准确"
- RAGAS faithfulness/answer_relevancy 分数下降
- 检索返回的文档与 query 相关性降低
- Top-K 结果中相关文档占比从 80% 降到 40%

### Possible Root Causes（可能原因）
1. **Embedding 模型变更**: embedding 模型更新导致新旧向量不兼容
2. **索引损坏**: 向量数据库索引损坏或部分数据丢失
3. **数据质量退化**: 新入库文档质量差（如 OCR 错误、格式混乱）
4. **Chunking 策略问题**: chunk size/overlap 变更导致语义断裂
5. **Query 分布漂移**: 用户 query 模式变化，与索引内容不匹配
6. **向量数据库配置**: HNSW 参数变更（ef_search 降低）导致近似搜索精度下降
7. **Metadata filter 错误**: 过滤条件过严，排除了相关文档

### Metrics to Check
```
# 检索质量指标
- retrieval_recall@k (k=5,10,20)
- retrieval_precision@k
- retrieval_mrr (Mean Reciprocal Rank)
- retrieval_ndcg@k

# 系统指标
- embedding_latency_ms
- vector_db_query_latency_ms
- index_size (document count)
- index_freshness (last update time)

# RAG 端到端指标
- ragas_faithfulness
- ragas_answer_relevancy
- ragas_context_precision
- ragas_context_recall
```

### Logs to Check
```bash
# 检查 embedding 模型版本
grep "model_name\|model_version" /var/log/rag/embedding.log

# 检查索引更新
grep "index_update\|reindex\|delete" /var/log/rag/indexer.log

# 检查 query 和返回结果
grep "query\|retrieved_docs\|score" /var/log/rag/retrieval.log | tail -50

# 检查向量数据库健康
curl http://milvus:9091/healthz
curl http://milvus:9091/api/v1/collection/stats
```

### Profiling Method
```bash
# 1. 对比分析：用已知 query-document pair 测试
python eval_retrieval.py \
  --test_set golden_qa_pairs.json \
  --top_k 10 \
  --output recall_report.json

# 2. Embedding 一致性检查
python check_embedding_consistency.py \
  --old_model "text-embedding-ada-002" \
  --new_model "text-embedding-3-small" \
  --sample_docs 1000

# 3. 索引完整性检查
python check_index_integrity.py \
  --expected_count 100000 \
  --sample_check 1000

# 4. Chunk 质量分析
python analyze_chunks.py \
  --collection documents \
  --check_length --check_overlap --check_encoding
```

### Immediate Mitigation
1. **回滚 embedding 模型**: 如果是模型变更导致，切回旧模型
2. **提高 top_k**: 临时增加检索数量（如 k=5 → k=20）弥补精度下降
3. **降低 score 阈值**: 放宽相似度阈值，召回更多文档
4. **启用 hybrid search**: 同时使用向量搜索 + 关键词搜索
5. **修复 metadata filter**: 检查并放宽过严的过滤条件

### Rollback
- 回滚 embedding 模型版本
- 从备份恢复向量索引
- 回滚 chunking 配置
- 恢复 HNSW 参数（ef_search, M）

### Long-term Fix
1. **Hybrid retrieval**: 向量搜索 + BM25 关键词搜索 + metadata filter 组合
2. **Embedding 版本管理**: embedding 模型更新时自动重建索引
3. **持续评测**: 定期用 golden set 评测 recall，设置告警阈值
4. **Query 理解**: 添加 query expansion/rewriting 提高召回
5. **多路召回**: 使用多个 embedding 模型 + 多种检索策略
6. **索引监控**: 监控索引大小、freshness、query latency

### Postmortem
- **Root Cause**: 明确是模型/数据/配置哪个环节出问题
- **Impact**: 影响了多少 query，持续了多长时间
- **Detection**: 如何更早发现（自动化评测 vs 用户反馈）
- **Prevention**: 如何防止再次发生

### Interview Answer
> "Retrieval recall 下降的排查我会分三步：
> 1. **确认范围**: 用 golden test set（已知 query-document pair）量化 recall 下降幅度，确认是全局性还是特定类型 query
> 2. **定位原因**: 
>    - 检查 embedding 模型是否变更（新旧向量不兼容是最常见原因）
>    - 检查索引完整性（document count 是否正确）
>    - 检查 HNSW 参数（ef_search 是否被降低）
> 3. **验证修复**: 修复后用 golden set 验证 recall 恢复
> 
> 我们建立了持续评测 pipeline：每小时用 200 个 golden pair 测试 recall@10，低于 85% 自动告警。同时实现了 hybrid retrieval（向量 + BM25），单一路径失效时另一路径兜底。"

---

## 事故 2：Reranker Latency Doubles（重排序延迟翻倍）

### Symptom（现象）
- RAG pipeline 端到端延迟从 p50=500ms 增加到 p50=1000ms+
- Reranker 阶段耗时从 100ms 增加到 200ms+
- 用户感知到"搜索变慢了"
- Reranker GPU/CPU 利用率异常

### Possible Root Causes（可能原因）
1. **输入长度增加**: 检索返回的文档变长，reranker 处理时间增加
2. **候选数量增加**: top_k 增大导致 reranker 需要处理更多文档
3. **模型变更**: reranker 模型升级（如从 cross-encoder-small 到 large）
4. **Batch 效率下降**: 输入长度不均匀导致 padding 浪费
5. **GPU 资源争抢**: reranker 与其他模型共享 GPU
6. **内存不足**: 模型被 swap 到 CPU
7. **并发增加**: 请求量增加但 reranker 实例未扩容

### Metrics to Check
```
# Reranker 指标
- reranker_latency_ms (p50, p95, p99)
- reranker_input_length_avg
- reranker_candidate_count
- reranker_batch_size
- reranker_gpu_utilization
- reranker_queue_depth

# 系统指标
- gpu_memory_used (reranker GPU)
- cpu_utilization (if CPU reranker)
- request_rate_per_second
```

### Logs to Check
```bash
# Reranker 延迟分布
grep "rerank_latency\|rerank_time" /var/log/rag/reranker.log | \
  awk '{print $NF}' | sort -n | awk 'BEGIN{c=0}{a[c++]=$1}END{print "p50:",a[int(c*0.5)],"p99:",a[int(c*0.99)]}'

# 输入长度变化
grep "input_length\|num_candidates" /var/log/rag/reranker.log | tail -50

# GPU 内存
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### Profiling Method
```bash
# 1. Reranker 延迟分解
python profile_reranker.py \
  --model cross-encoder/ms-marco-MiniLM-L-12-v2 \
  --input_lengths "128,256,512,1024" \
  --num_candidates "5,10,20,50"

# 2. Batch 效率分析
python analyze_reranker_batch.py \
  --log /var/log/rag/reranker.log \
  --output batch_efficiency.json

# 3. GPU profiling
nsys profile -o reranker_profile python reranker_server.py
```

### Immediate Mitigation
1. **减少候选数量**: 降低传给 reranker 的 top_k（如 20 → 10）
2. **截断文档长度**: 限制每个文档传给 reranker 的最大 token 数
3. **降级模型**: 临时切换到更小的 reranker 模型
4. **扩容**: 增加 reranker 实例
5. **异步 rerank**: 对非实时场景使用异步 reranking

### Rollback
- 回滚 reranker 模型版本
- 恢复之前的 top_k 和 max_length 配置
- 恢复之前的 GPU 分配

### Long-term Fix
1. **动态 top_k**: 根据 retrieval score 分布动态决定传给 reranker 的数量
2. **Early termination**: 如果前几个文档 score 远高于后面的，提前终止
3. **Lightweight reranker**: 使用蒸馏的小模型做初筛，大模型只处理 top 候选
4. **Batch 优化**: 按文档长度分桶，减少 padding 浪费
5. **Cache**: 对相同 query-document pair 缓存 rerank score
6. **GPU 独占**: 给 reranker 独立 GPU，避免资源争抢

### Interview Answer
> "Reranker 延迟翻倍的排查：
> 1. 首先检查输入变化：候选文档数量是否增加？文档长度是否变长？
> 2. 检查 batch 效率：如果文档长度方差大，padding 会浪费大量计算
> 3. 检查资源：GPU 是否被其他模型争抢
> 
> 优化方案：实现动态 top_k——根据 retrieval score 的分布（如 score gap > threshold 就截断），将平均候选数从 20 降到 8，延迟降低 60%。同时按文档长度分桶 batch，减少 padding 浪费 30%。最终 reranker p99 从 400ms 降到 150ms。"

---

## RAG 系统整体监控建议

### 关键 SLI/SLO

| 指标 | SLO | 告警阈值 |
|------|-----|----------|
| Retrieval Recall@10 | ≥ 85% | < 80% |
| Reranker Latency p99 | < 300ms | > 500ms |
| End-to-end Latency p99 | < 3s | > 5s |
| Answer Faithfulness | ≥ 0.85 | < 0.80 |
| Answer Relevancy | ≥ 0.80 | < 0.75 |

### 持续评测 Pipeline

```
每小时:
├── Golden set recall 测试 (200 pairs)
├── Reranker latency benchmark
└── End-to-end latency check

每天:
├── RAGAS 全量评测 (1000 pairs)
├── Embedding drift 检测
└── Index integrity check

每周:
├── 新数据质量审计
├── Query 分布分析
└── A/B 测试结果汇总
```
