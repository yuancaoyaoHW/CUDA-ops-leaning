# LLM Serving 事故演练 Playbook

---

## 事故 1：TTFT Suddenly Increases（首 Token 延迟突增）

### Symptom（现象）
- TTFT (Time To First Token) 从 p50=200ms 突增到 p50=2000ms+
- 用户感知到"等待时间变长"
- Prefill 阶段耗时异常
- 新请求排队时间增加

### Possible Root Causes（可能原因）
1. **Prefill 计算瓶颈**: 输入 prompt 长度突增（如用户发送长文档），prefill 阶段 compute-bound
2. **KV cache 内存压力**: KV cache 接近 OOM，触发 preemption/swap，新请求需要等待内存释放
3. **Batch 调度问题**: 大量 decode 请求占满 batch，新 prefill 请求被排队
4. **GPU 资源争抢**: 其他进程（如 model loading, checkpoint）占用 GPU
5. **Prefix cache 失效**: prefix cache hit rate 下降，原本可复用的 KV 需要重新计算
6. **网络延迟**: 分布式推理中 TP all-reduce 通信延迟增加

### Metrics to Check（需要查看的指标）
```
# 推理指标
- vllm:time_to_first_token_seconds (histogram)
- vllm:num_requests_waiting (gauge)
- vllm:num_requests_running (gauge)
- vllm:gpu_cache_usage_perc (gauge)
- vllm:cpu_cache_usage_perc (gauge)
- vllm:avg_prompt_throughput_toks_per_s (gauge)

# GPU 指标
- gpu_utilization_percent
- gpu_memory_used_bytes
- sm_occupancy

# 系统指标
- request_queue_depth
- prefill_batch_size
- avg_prompt_length
```

### Logs to Check（需要查看的日志）
```bash
# vLLM 日志
grep -i "preempt\|swap\|oom\|waiting" /var/log/vllm/serving.log

# 检查 prompt 长度分布变化
grep "prompt_len" /var/log/vllm/request.log | awk '{print $NF}' | sort -n | tail -20

# 检查 scheduler 决策
grep "schedule\|batch" /var/log/vllm/scheduler.log
```

### Profiling Method（如何 Profiling）
```bash
# 1. Nsight Systems 抓取 prefill timeline
nsys profile --trace=cuda,nvtx -o ttft_debug python -m vllm.entrypoints.openai.api_server ...

# 2. 检查 prefill kernel 耗时
ncu --target-processes all --set full -o prefill_kernel <pid>

# 3. vLLM 内置 profiling
curl http://localhost:8000/debug/pprof/profile?seconds=30 > profile.pb

# 4. 检查 batch 组成
curl http://localhost:8000/debug/scheduler_state
```

### Immediate Mitigation（立即缓解）
1. **限制最大 prompt 长度**: 设置 `--max-model-len` 降低上限
2. **增加 prefill 优先级**: 调整 scheduler policy 优先处理 prefill
3. **扩容**: 增加 GPU 实例分担负载
4. **限流**: 对超长 prompt 请求限流
5. **清理 KV cache**: 如果是内存压力，手动触发 cache eviction

### Rollback（回滚方案）
- 回滚到上一个稳定版本的 serving 配置
- 恢复之前的 `max-model-len` 和 `max-num-seqs` 参数
- 如果是代码变更导致，git revert 并重新部署

### Long-term Fix（长期修复）
1. **Chunked Prefill**: 将长 prompt 分块处理，避免单次 prefill 阻塞
2. **Prefill/Decode 分离**: 使用 disaggregated architecture (DistServe/Splitwise)
3. **动态 batch 策略**: 根据 prompt 长度动态调整 batch 组成
4. **Prefix caching 优化**: 提高 cache hit rate
5. **监控告警**: 设置 TTFT p99 > 阈值的告警

### Postmortem（事后复盘模板）
- **Timeline**: 何时发现 → 何时定位 → 何时修复
- **Root Cause**: 具体原因和触发条件
- **Impact**: 影响了多少用户/请求
- **Detection Gap**: 为什么没有更早发现
- **Prevention**: 如何防止再次发生

### Interview Answer（面试回答）
> "我遇到过 TTFT 突增的问题。首先通过 Prometheus 监控发现 TTFT p99 从 300ms 涨到 3s。排查步骤：
> 1. 检查 `gpu_cache_usage_perc` 发现 KV cache 使用率达到 95%，触发了 preemption
> 2. 检查 `avg_prompt_length` 发现某个客户开始发送 32K token 的长文档
> 3. 用 Nsight Systems 确认 prefill kernel 本身没有回归，是排队导致的延迟
> 
> 短期修复：对超长 prompt 限流 + 增加实例。长期方案：实现 chunked prefill，将长 prompt 分块与 decode 交错执行，TTFT p99 降回 500ms 以内。"

---

## 事故 2：TPOT Regression（Token 生成延迟回归）

### Symptom（现象）
- TPOT (Time Per Output Token) 从 p50=30ms 增加到 p50=60ms+
- 用户感知到"生成速度变慢"
- Decode 阶段每个 token 的生成时间增加
- 整体 throughput 下降

### Possible Root Causes（可能原因）
1. **Batch size 增大**: 更多并发请求导致 decode batch 变大，每个 token 的 attention 计算量增加
2. **KV cache 碎片化**: 内存碎片导致 cache access 效率下降
3. **Attention kernel 回归**: FlashAttention/FlashInfer 版本更新引入性能回归
4. **量化精度问题**: 量化配置变更导致 dequant overhead 增加
5. **CPU-GPU 同步**: 新增的 CPU 逻辑（如 sampling）引入同步等待
6. **NCCL 通信退化**: TP all-reduce 延迟增加

### Metrics to Check
```
- vllm:time_per_output_token_seconds (histogram)
- vllm:num_requests_running (gauge)
- decode_batch_size (gauge)
- attention_kernel_time_ms (histogram)
- sampling_time_ms (histogram)
- nccl_allreduce_time_ms (histogram)
- gpu_memory_bandwidth_utilization
```

### Logs to Check
```bash
# 检查 decode batch size 变化
grep "decode_batch_size\|running_requests" /var/log/vllm/scheduler.log

# 检查 kernel 版本
python -c "import flashinfer; print(flashinfer.__version__)"
python -c "import flash_attn; print(flash_attn.__version__)"

# 检查是否有 CPU-GPU sync
grep "synchronize\|cuda.sync" /var/log/vllm/debug.log
```

### Profiling Method
```bash
# 1. 对比 decode 阶段 kernel timeline
nsys profile --trace=cuda -o decode_debug ...

# 2. 单独 benchmark attention kernel
python benchmark_attention.py --batch=32 --seq_len=2048 --head_dim=128

# 3. 检查 sampling overhead
python -c "import torch; torch.cuda.synchronize(); ..."  # 测量 sampling 时间

# 4. NCCL 通信 profiling
NCCL_DEBUG=INFO python -m vllm.entrypoints.openai.api_server ...
```

### Immediate Mitigation
1. **降低 max_num_seqs**: 减少并发 batch size
2. **回滚 kernel 版本**: 如果是 FlashAttention/FlashInfer 更新导致
3. **关闭新增 feature**: 如果是新功能引入的 overhead
4. **扩容**: 增加实例分担负载

### Rollback
- 回滚 FlashAttention/FlashInfer 版本
- 回滚 vLLM 版本到上一个稳定版
- 恢复之前的 serving 参数

### Long-term Fix
1. **Decode kernel 优化**: 针对大 batch decode 场景优化 attention kernel
2. **动态 batch 上限**: 根据 TPOT SLA 动态调整 max batch size
3. **Sampling 异步化**: 将 sampling 逻辑异步执行，避免 CPU-GPU sync
4. **KV cache defragmentation**: 定期整理 KV cache 内存
5. **性能回归 CI**: 每次 kernel 更新自动跑 benchmark

### Interview Answer
> "TPOT 回归的排查我会按以下步骤：
> 1. 首先区分是 compute-bound 还是 memory-bound：用 Nsight Compute 看 decode attention kernel 的 roofline 位置
> 2. 检查 batch size 变化：如果 running requests 增多，decode batch 变大，每个 token 的 KV cache read 量增加
> 3. 检查 kernel 版本：对比更新前后的 kernel benchmark
> 
> 一个具体案例：FlashInfer 版本更新后 decode attention 在 batch=64 时性能回归 20%。通过 NCU 发现新版本的 memory access pattern 在大 batch 时 L2 cache hit rate 下降。短期回滚版本，长期与 FlashInfer 团队协作修复。"

---

## 事故 3：KV Cache OOM

### Symptom（现象）
- 服务 crash 或请求被大量 reject
- 错误日志出现 "CUDA out of memory" 或 "No available blocks"
- `gpu_cache_usage_perc` 达到 100%
- 新请求全部排队或被拒绝

### Possible Root Causes
1. **并发请求过多**: 同时活跃的请求数超过 KV cache 容量
2. **长序列累积**: 多个长对话同时活跃，KV cache 被占满
3. **内存泄漏**: KV cache block 未正确释放
4. **Preemption 失败**: swap 到 CPU 的机制失效
5. **配置错误**: `gpu_memory_utilization` 设置过低，或 `max_num_seqs` 过高
6. **模型变更**: 新模型的 KV cache per token 更大（如 GQA → MHA）

### Metrics to Check
```
- vllm:gpu_cache_usage_perc (gauge) — 应 < 90%
- vllm:cpu_cache_usage_perc (gauge)
- vllm:num_requests_running (gauge)
- vllm:num_requests_waiting (gauge)
- vllm:num_preemptions_total (counter)
- gpu_memory_used_bytes / gpu_memory_total_bytes
- avg_sequence_length (running requests)
```

### Logs to Check
```bash
# OOM 相关
grep -i "out of memory\|no available blocks\|preempt" /var/log/vllm/serving.log

# Block 分配
grep "allocate\|free\|block" /var/log/vllm/block_manager.log

# 请求长度
grep "seq_len\|prompt_len\|output_len" /var/log/vllm/request.log | tail -50
```

### Profiling Method
```bash
# 1. 监控 block 使用情况
watch -n 1 'curl -s http://localhost:8000/debug/block_manager_state | jq .'

# 2. 分析请求长度分布
python analyze_request_lengths.py --log /var/log/vllm/request.log

# 3. 检查内存泄漏
nvidia-smi --query-gpu=memory.used --format=csv -l 1 > mem_trace.csv
```

### Immediate Mitigation
1. **降低 max_num_seqs**: 立即减少并发请求数
2. **启用 preemption**: 确保 swap/recompute 机制正常工作
3. **限制 max_tokens**: 限制单个请求的最大输出长度
4. **拒绝超长请求**: 对超过阈值的请求返回 413
5. **重启服务**: 如果是内存泄漏，重启清理

### Rollback
- 恢复之前的 `gpu_memory_utilization` 和 `max_num_seqs` 配置
- 回滚模型版本（如果是模型变更导致）

### Long-term Fix
1. **动态 max_num_seqs**: 根据当前 KV cache 使用率动态调整
2. **请求准入控制**: 基于预估 KV cache 需求的准入策略
3. **KV cache 压缩**: 使用 GQA/MQA 减少 KV cache 大小
4. **Offloading**: 将不活跃请求的 KV cache swap 到 CPU/SSD
5. **Prefix caching**: 复用公共 prefix 的 KV cache
6. **监控告警**: cache 使用率 > 80% 时告警

### Interview Answer
> "KV cache OOM 是 LLM serving 中最常见的问题之一。我的排查思路：
> 1. 首先确认是真正的 OOM 还是 block manager 的逻辑 OOM（物理内存还有，但 block 分配逻辑认为满了）
> 2. 检查 `num_requests_running × avg_seq_len × kv_cache_per_token` 是否超过预分配的 cache 容量
> 3. 检查是否有内存泄漏（block 未释放）
> 
> 解决方案分层：短期限流 + preemption；中期实现动态准入控制（根据预估 token 数决定是否接受请求）；长期考虑 disaggregated KV cache（如 Mooncake 的方案）。关键指标是保持 cache 使用率在 85% 以下。"

---

## 事故 4：Prefix Cache Hit Rate Drops（前缀缓存命中率下降）

### Symptom（现象）
- Prefix cache hit rate 从 80% 降到 20%
- TTFT 相应增加（需要重新计算 prefix 的 KV）
- GPU compute utilization 增加
- 相同 prompt 的请求不再享受缓存加速

### Possible Root Causes
1. **Cache eviction 策略问题**: LRU eviction 在流量模式变化时失效
2. **请求模式变化**: 用户 prompt 多样性增加，公共 prefix 减少
3. **Cache 容量不足**: 活跃 prefix 数量超过 cache 容量
4. **Hash 冲突**: prefix hash 计算变更导致 cache miss
5. **部署变更**: 新部署清空了 cache（cold start）
6. **多实例负载均衡**: 请求被分散到不同实例，每个实例 cache 不完整

### Metrics to Check
```
- prefix_cache_hit_rate (gauge)
- prefix_cache_num_entries (gauge)
- prefix_cache_eviction_count (counter)
- unique_prefix_count (gauge)
- ttft_with_cache_hit vs ttft_without_cache_hit
```

### Logs to Check
```bash
# Cache hit/miss
grep "cache_hit\|cache_miss\|evict" /var/log/vllm/cache.log

# Prefix 分布
grep "prefix_hash" /var/log/vllm/request.log | sort | uniq -c | sort -rn | head -20
```

### Profiling Method
```bash
# 1. 分析 prefix 分布
python analyze_prefix_distribution.py --log request.log

# 2. 模拟不同 cache 大小的 hit rate
python simulate_cache_policy.py --trace request_trace.json --cache_sizes 100,500,1000,5000
```

### Immediate Mitigation
1. **增加 cache 容量**: 提高 `gpu_memory_utilization` 给 cache 更多空间
2. **调整 eviction 策略**: 从 LRU 切换到 LFU 或 ARC
3. **Sticky routing**: 将相同 prefix 的请求路由到同一实例
4. **预热 cache**: 用高频 prefix 预热

### Long-term Fix
1. **Shared prefix cache**: 跨实例共享 prefix cache（如 Redis-based）
2. **智能 routing**: 基于 prefix hash 的请求路由
3. **分层 cache**: GPU cache + CPU cache + SSD cache
4. **自适应 eviction**: 根据 prefix 频率动态调整 eviction 策略

### Interview Answer
> "Prefix cache hit rate 下降的排查：首先分析请求 prefix 的分布变化——是 prefix 多样性增加了，还是 cache 被错误 evict 了。我会用 prefix hash 的 frequency 分析来判断。如果是流量模式变化，需要调整 eviction 策略（如从 LRU 到 LFU）。如果是多实例问题，需要实现 prefix-aware routing，将相同 prefix 的请求路由到同一实例。关键指标：cache hit rate 和对应的 TTFT 节省。"

---

## 事故 5：P99 Latency Spikes（P99 延迟尖刺）

### Symptom（现象）
- P99 latency 间歇性飙升（如从 500ms 到 5s）
- P50 正常，但 P99/P999 异常
- 少数请求超时
- 用户投诉"偶尔很慢"

### Possible Root Causes
1. **GC (Garbage Collection)**: Python GC 暂停导致延迟尖刺
2. **Preemption**: 低优先级请求被 preempt 后重新计算
3. **长尾请求**: 少数超长序列请求拖慢整个 batch
4. **GPU 温度降频**: GPU 过热触发 thermal throttling
5. **CUDA context switch**: 多进程共享 GPU 导致 context switch
6. **网络抖动**: 分布式推理中 NCCL 通信偶发延迟
7. **磁盘 I/O**: 日志写入或 checkpoint 导致 I/O 阻塞

### Metrics to Check
```
- request_latency_p50, p95, p99, p999
- gpu_temperature
- gpu_clock_speed (是否降频)
- python_gc_pause_seconds
- preemption_count
- nccl_latency_p99
- disk_io_wait
```

### Logs to Check
```bash
# 找到延迟最高的请求
grep "latency" /var/log/vllm/request.log | sort -t= -k2 -rn | head -10

# GC 日志
grep "gc\|garbage" /var/log/vllm/debug.log

# GPU 降频
nvidia-smi -q -d PERFORMANCE | grep -i "throttle"
```

### Profiling Method
```bash
# 1. 持续监控 GPU 温度和频率
nvidia-smi dmon -s pucvmet -d 1 > gpu_monitor.csv

# 2. Python GC profiling
python -c "import gc; gc.set_debug(gc.DEBUG_STATS); ..."

# 3. 请求级别 tracing
# 在 vLLM 中启用 request tracing
export VLLM_TRACE_FUNCTION=1
```

### Immediate Mitigation
1. **禁用 Python GC**: `gc.disable()` 或调整 GC 阈值
2. **GPU 散热**: 检查风扇/空调，降低 GPU 功耗限制
3. **隔离长尾请求**: 将超长请求路由到专用实例
4. **增加超时**: 适当增加客户端超时避免重试风暴

### Long-term Fix
1. **GC 优化**: 使用 `gc.freeze()` 冻结长期对象，减少 GC 扫描
2. **请求隔离**: 按请求长度分桶，不同桶用不同实例
3. **Preemption 优化**: 使用 recompute 而非 swap 减少恢复时间
4. **GPU 监控**: 温度/频率告警，提前发现降频
5. **Tail latency budget**: 为 P99 设置 latency budget 并自动降级

### Interview Answer
> "P99 尖刺的排查需要区分是系统性的还是偶发的。我的方法：
> 1. 首先看 P99 尖刺的时间模式——是周期性的（可能是 GC）还是随机的（可能是长尾请求）
> 2. 关联 GPU 温度/频率数据，排除 thermal throttling
> 3. 检查被 preempt 的请求比例
> 
> 一个典型案例：Python GC 每 30 秒触发一次 full GC，暂停 200ms。解决方案：`gc.disable()` + 手动在低负载时触发 GC。P99 从 3s 降到 600ms。"

---

## 事故 6：Continuous Batching Fairness Issue（连续批处理公平性问题）

### Symptom（现象）
- 部分请求等待时间远超平均值
- 先到的请求反而后完成（FIFO 违反）
- 短请求被长请求"饿死"
- 用户投诉"我的请求一直在排队"

### Possible Root Causes
1. **长序列霸占 batch**: 长序列请求持续占据 batch slot，新请求无法加入
2. **Preemption 不公平**: 某些请求被反复 preempt
3. **优先级反转**: 低优先级请求占用资源，高优先级请求等待
4. **Batch 组成不合理**: prefill 和 decode 混合比例失衡
5. **Max tokens 设置过大**: 单个请求可以生成过多 token，长期占用 slot

### Metrics to Check
```
- request_waiting_time_distribution
- request_completion_time_distribution
- batch_composition (prefill vs decode ratio)
- preemption_per_request_distribution
- max_waiting_time
- fairness_index (Jain's fairness index)
```

### Logs to Check
```bash
# 等待时间最长的请求
grep "waiting_time" /var/log/vllm/request.log | sort -t= -k2 -rn | head -20

# Preemption 分布
grep "preempt" /var/log/vllm/scheduler.log | awk '{print $NF}' | sort | uniq -c

# Batch 组成
grep "batch_size\|prefill_count\|decode_count" /var/log/vllm/scheduler.log
```

### Immediate Mitigation
1. **限制 max_tokens**: 降低单个请求的最大输出长度
2. **设置 max_waiting_time**: 超时的请求强制调度
3. **增加 preemption 阈值**: 更积极地 preempt 长序列
4. **扩容**: 增加实例减少排队

### Long-term Fix
1. **公平调度器**: 实现 weighted fair queuing
2. **请求分桶**: 按预估长度分桶，短请求优先
3. **时间片轮转**: 对长序列实施时间片，定期让出 slot
4. **SLA-aware scheduling**: 根据 SLA 要求调整优先级
5. **Iteration-level scheduling**: 每个 iteration 重新评估 batch 组成

### Interview Answer
> "Continuous batching 的公平性问题本质是调度问题。我会从 Jain's fairness index 入手量化不公平程度，然后分析 batch 组成——如果 decode 请求长期占满 batch，新 prefill 请求就会饿死。解决方案：实现 iteration-level scheduling，每个 iteration 预留一定比例的 slot 给 prefill；对超过 SLA 的等待请求提升优先级；对超长序列实施 preemption budget。"

---

## 事故 7：Multi-tenant Noisy Neighbor（多租户噪声邻居）

### Symptom（现象）
- 某个租户的请求量突增，导致其他租户延迟增加
- 共享 GPU 上不同租户的 SLA 互相影响
- 某个租户发送大量长序列请求，占满 KV cache
- 其他租户的请求被 preempt 或排队

### Possible Root Causes
1. **无资源隔离**: 多租户共享同一 GPU/实例，无 quota 限制
2. **KV cache 争抢**: 某租户占用过多 KV cache block
3. **Batch 争抢**: 某租户请求占满 batch slot
4. **网络带宽争抢**: 某租户的大量请求占满网络
5. **缺乏限流**: 无 per-tenant rate limiting

### Metrics to Check
```
- per_tenant_request_rate
- per_tenant_latency_p99
- per_tenant_kv_cache_usage
- per_tenant_batch_slot_usage
- per_tenant_token_throughput
- cross_tenant_latency_correlation
```

### Immediate Mitigation
1. **Per-tenant rate limiting**: 立即对突增租户限流
2. **请求优先级**: 降低突增租户的优先级
3. **隔离部署**: 将大租户迁移到独立实例
4. **KV cache quota**: 限制每个租户的 cache 使用量

### Long-term Fix
1. **资源 quota**: 每个租户有明确的 GPU/memory/throughput quota
2. **隔离调度**: 实现 per-tenant fair scheduling
3. **弹性扩容**: 租户流量增加时自动扩容其专属资源
4. **MIG (Multi-Instance GPU)**: 使用 NVIDIA MIG 硬件隔离
5. **优先级队列**: 基于 SLA tier 的多级优先级

### Interview Answer
> "多租户 noisy neighbor 问题的核心是资源隔离。我的方案分三层：
> 1. **准入控制**: per-tenant token bucket rate limiter，防止突发流量
> 2. **调度隔离**: weighted fair queuing，每个租户有保证的 throughput share
> 3. **资源隔离**: 大租户用独立 GPU（或 MIG partition），小租户共享
> 
> 关键指标是 cross-tenant latency correlation——理想情况下一个租户的流量变化不应影响其他租户的 P99。实现时需要在隔离度和资源利用率之间 trade-off。"
