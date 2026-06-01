# LLM Inference Benchmark Protocol

## 目的

定义 LLM 推理框架性能对比的标准流程，确保不同框架、不同配置之间的对比公平、可复现。

---

## 1. 公平对比原则

### 必须控制的变量

| 变量 | 要求 |
|------|------|
| 硬件 | 相同 GPU 型号、相同数量 |
| 模型 | 相同权重文件、相同精度 |
| Workload | 相同 prompt 集合、相同分布 |
| 并发模式 | 相同并发数、相同发送策略 |
| 预热 | 相同预热请求数 |
| 测量时间 | 足够长以达到稳态 |

### 禁止的不公平做法

- ❌ 一个框架用优化配置，另一个用默认配置
- ❌ 不同 workload 分布对比
- ❌ 不同 GPU memory 利用率设置
- ❌ 未预热就开始计时
- ❌ 只跑几个 request 就下结论

---

## 2. 实验流程

### Step 1: 环境准备

```bash
# 记录环境信息
nvidia-smi
python -c "import vllm; print(vllm.__version__)"
python -c "import sglang; print(sglang.__version__)"

# 确认 GPU 独占
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv

# 清理 GPU memory
torch.cuda.empty_cache()
```

### Step 2: 启动 Serving

```bash
# vLLM
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 256 \
    --port 8000

# SGLang
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-7B \
    --mem-fraction-static 0.9 \
    --port 8001
```

### Step 3: 预热

```python
# 发送 10 个预热请求，不计入统计
for i in range(10):
    send_request(warmup_prompt, max_tokens=64)
time.sleep(5)  # 等待系统稳定
```

### Step 4: 执行 Benchmark

```python
# 标准 benchmark 执行
results = run_benchmark(
    endpoint=endpoint,
    workload=workload,
    num_requests=200,       # 至少 200 个请求
    concurrency=16,         # 固定并发
    timeout=300,            # 5 分钟超时
)
```

### Step 5: 收集结果

```python
metrics = {
    "ttft": compute_percentiles(ttft_list),      # p50, p95, p99
    "tpot": compute_percentiles(tpot_list),
    "e2e_latency": compute_percentiles(e2e_list),
    "throughput_tps": total_output_tokens / total_time,
    "qps": num_completed / total_time,
    "gpu_util": avg_gpu_utilization,
    "memory_gb": peak_memory_usage,
    "error_rate": num_errors / num_requests,
}
```

### Step 6: 重复验证

```
每个配置至少跑 3 次，取 median
如果 3 次结果 std > 10%，增加到 5 次
```

---

## 3. Workload 设计

### 合成 Workload（可控）

```python
class SyntheticWorkload:
    """固定长度的合成 workload，用于单变量实验"""
    
    def __init__(self, input_len=512, output_len=256, num_requests=200):
        self.prompts = [generate_random_prompt(input_len) for _ in range(num_requests)]
        self.max_tokens = output_len
```

### 真实 Workload（ShareGPT）

```python
class ShareGPTWorkload:
    """真实对话分布，用于框架对比"""
    
    def __init__(self, dataset_path, num_requests=500):
        # 从 ShareGPT 数据集采样
        # 保留原始 input/output 长度分布
        self.conversations = load_and_sample(dataset_path, num_requests)
```

### Workload 分布记录

每次实验必须记录 workload 的统计信息：

```python
workload_stats = {
    "num_requests": len(prompts),
    "input_len_mean": np.mean(input_lens),
    "input_len_p50": np.median(input_lens),
    "input_len_p95": np.percentile(input_lens, 95),
    "output_len_mean": np.mean(output_lens),
    "output_len_p50": np.median(output_lens),
    "output_len_p95": np.percentile(output_lens, 95),
}
```

---

## 4. 指标计算标准

### TTFT（Time To First Token）

```python
def compute_ttft(request):
    """从发送请求到收到第一个 token 的时间"""
    return request.first_token_time - request.send_time
```

### TPOT（Time Per Output Token）

```python
def compute_tpot(request):
    """每个输出 token 的平均生成时间（不含首 token）"""
    if request.output_token_count <= 1:
        return None
    decode_time = request.end_time - request.first_token_time
    return decode_time / (request.output_token_count - 1)
```

### Throughput

```python
def compute_throughput(results):
    """系统级吞吐量"""
    total_output_tokens = sum(r.output_token_count for r in results)
    total_time = max(r.end_time for r in results) - min(r.send_time for r in results)
    return total_output_tokens / total_time  # tokens/s
```

### Goodput

```python
def compute_goodput(results, sla_ttft_ms=500, sla_tpot_ms=100):
    """满足 SLA 的有效吞吐"""
    good_results = [r for r in results 
                    if r.ttft < sla_ttft_ms and r.tpot < sla_tpot_ms]
    good_tokens = sum(r.output_token_count for r in good_results)
    total_time = max(r.end_time for r in results) - min(r.send_time for r in results)
    return good_tokens / total_time
```

---

## 5. 并发模式

### 固定并发（Closed-loop）

```python
async def closed_loop_benchmark(endpoint, workload, concurrency):
    """维持固定并发数：一个请求完成后立即发下一个"""
    semaphore = asyncio.Semaphore(concurrency)
    
    async def send_one(prompt):
        async with semaphore:
            return await send_request(endpoint, prompt)
    
    tasks = [send_one(p) for p in workload.prompts]
    results = await asyncio.gather(*tasks)
    return results
```

### 固定速率（Open-loop）

```python
async def open_loop_benchmark(endpoint, workload, qps):
    """以固定 QPS 发送请求，不等待响应"""
    interval = 1.0 / qps
    tasks = []
    for prompt in workload.prompts:
        tasks.append(asyncio.create_task(send_request(endpoint, prompt)))
        await asyncio.sleep(interval)
    results = await asyncio.gather(*tasks)
    return results
```

### 选择建议

| 场景 | 模式 | 原因 |
|------|------|------|
| 最大吞吐测试 | Closed-loop, 高并发 | 持续压满 GPU |
| 延迟测试 | Closed-loop, 低并发 | 减少排队影响 |
| SLA 测试 | Open-loop, 固定 QPS | 模拟真实流量 |
| 框架对比 | Closed-loop, 中等并发 | 公平且稳定 |

---

## 6. 参数调优实验设计

### 单参数 Sweep

```python
# 每次只变一个参数
param_sweeps = {
    "max_num_seqs": [64, 128, 256, 512],
    "gpu_memory_utilization": [0.80, 0.85, 0.90, 0.95],
    "enable_chunked_prefill": [True, False],
}

for param_name, values in param_sweeps.items():
    for value in values:
        config = default_config.copy()
        config[param_name] = value
        result = run_benchmark(config)
        # 记录: param_name, value, throughput, latency
```

### 最优配置搜索

```python
# 在单参数 sweep 基础上，组合 top-3 配置做 grid search
best_configs = grid_search(
    max_num_seqs=[128, 256],
    gpu_memory_utilization=[0.9, 0.95],
    enable_chunked_prefill=[True],
)
```

---

## 7. 结果报告格式

### 标准 JSON 输出

```json
{
  "experiment": "framework_compare",
  "timestamp": "2026-06-01T05:00:00Z",
  "environment": {
    "gpu": "NVIDIA A30",
    "gpu_memory_gb": 24,
    "cuda_version": "12.4",
    "model": "Qwen/Qwen2.5-7B",
    "dtype": "float16"
  },
  "workload": {
    "type": "synthetic",
    "num_requests": 200,
    "input_len": 512,
    "output_len": 256,
    "concurrency": 16
  },
  "results": {
    "vllm_default": {
      "throughput_tps": 150.3,
      "ttft_p50_ms": 45.2,
      "ttft_p99_ms": 120.5,
      "tpot_p50_ms": 22.1,
      "tpot_p99_ms": 35.8,
      "gpu_util_pct": 82.5,
      "memory_gb": 18.2
    },
    "sglang_default": { "..." : "..." }
  }
}
```

---

## 8. 常见陷阱

| 陷阱 | 影响 | 规避 |
|------|------|------|
| 未等 serving 完全启动 | 前几个请求超慢 | 预热 + sleep |
| Workload 太短 | 统计不显著 | 至少 200 requests |
| 只看 throughput 不看 latency | 忽略用户体验 | 同时报告两者 |
| 不同 GPU memory 设置 | 不公平对比 | 统一 gpu_memory_utilization |
| 忽略 error rate | 高吞吐但丢请求 | 记录并报告 error_rate |
| 未记录 workload 分布 | 结果不可复现 | 保存完整 workload 文件 |
| 只跑一次 | 结果可能是 outlier | 至少 3 次取 median |
