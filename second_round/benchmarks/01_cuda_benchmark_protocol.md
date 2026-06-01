# CUDA Kernel Benchmark Protocol

## 目的

定义 CUDA kernel 性能测量的标准流程，确保结果可复现、可对比、有统计意义。

---

## 1. 环境标准化

### 硬件信息记录

每次 benchmark 必须记录：

```bash
# GPU 信息
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,clocks.max.memory --format=csv

# CUDA 版本
nvcc --version

# 系统信息
uname -a
cat /proc/cpuinfo | grep "model name" | head -1
```

### 环境隔离

```bash
# 锁定 GPU 频率（避免 thermal throttling 影响）
sudo nvidia-smi -lgc 1410,1410  # 锁定 SM clock
sudo nvidia-smi -lmc 1215,1215  # 锁定 memory clock

# 确认无其他 GPU 进程
nvidia-smi

# 设置 persistence mode
sudo nvidia-smi -pm 1
```

### 理论峰值计算

| GPU | FP32 Peak | FP16 Peak (Tensor Core) | Memory BW Peak |
|-----|-----------|------------------------|----------------|
| A30 | 10.3 TFLOPS | 165 TFLOPS | 933 GB/s |
| V100 | 15.7 TFLOPS | 125 TFLOPS | 900 GB/s |
| A100 | 19.5 TFLOPS | 312 TFLOPS | 2039 GB/s |

---

## 2. 计时方法

### 标准计时模板

```python
import torch
import numpy as np

def benchmark_kernel(kernel_fn, *args, warmup=100, repeat=1000):
    """
    标准 kernel benchmark 函数
    - 使用 CUDA events 计时（精度 ~0.5μs）
    - 预热消除 JIT/cache 效应
    - 多次重复取统计值
    """
    # 预热
    for _ in range(warmup):
        kernel_fn(*args)
    torch.cuda.synchronize()

    # 计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    times = []

    for _ in range(repeat):
        start_event.record()
        kernel_fn(*args)
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))  # ms

    return {
        "median_ms": float(np.median(times)),
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
        "num_runs": repeat,
    }
```

### C++ 计时（纯 CUDA）

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

// Warmup
for (int i = 0; i < 100; i++) {
    kernel<<<grid, block>>>(args...);
}
cudaDeviceSynchronize();

// Benchmark
float total_ms = 0;
for (int i = 0; i < 1000; i++) {
    cudaEventRecord(start);
    kernel<<<grid, block>>>(args...);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    total_ms += ms;
}
float avg_ms = total_ms / 1000;
```

---

## 3. 性能指标计算

### Memory-bound Kernel（vector_add, reduction, softmax, RMSNorm）

```python
def compute_bandwidth(bytes_accessed, time_ms):
    """计算 effective bandwidth (GB/s)"""
    return bytes_accessed / (time_ms * 1e-3) / 1e9

def compute_bandwidth_utilization(effective_bw, peak_bw):
    """计算 bandwidth 利用率"""
    return effective_bw / peak_bw * 100

# 示例：vector_add
# 读 2N 个 float + 写 N 个 float = 3N * 4 bytes
bytes_accessed = 3 * N * 4  # float32
effective_bw = compute_bandwidth(bytes_accessed, median_ms)
util_pct = compute_bandwidth_utilization(effective_bw, 933)  # A30 peak
```

### Compute-bound Kernel（GEMM, FlashAttention）

```python
def compute_tflops(flops, time_ms):
    """计算 TFLOPS"""
    return flops / (time_ms * 1e-3) / 1e12

# 示例：GEMM (M×N×K)
flops = 2 * M * N * K  # multiply + add
tflops = compute_tflops(flops, median_ms)
cublas_pct = tflops / cublas_tflops * 100
```

### Arithmetic Intensity

```python
def compute_arithmetic_intensity(flops, bytes_accessed):
    """FLOP/Byte，用于 roofline 定位"""
    return flops / bytes_accessed

# GEMM: AI = 2MNK / (2*(MK+KN+MN)*dtype_size)
# 大矩阵 AI >> ridge point → compute-bound
# 小 M (decode): AI ≈ 1 → memory-bound
```

---

## 4. 对比基准

### 必须对比的基准

| Kernel 类型 | 对比基准 | 获取方式 |
|------------|---------|---------|
| Vector ops | PyTorch (a + b) | `torch.add(a, b)` |
| Reduction | PyTorch (tensor.sum()) | `tensor.sum()` |
| GEMM | cuBLAS | `torch.mm(A, B)` |
| Softmax | PyTorch F.softmax | `F.softmax(x, dim=-1)` |
| LayerNorm | PyTorch F.layer_norm | `F.layer_norm(x, [H])` |
| Attention | flash-attn 库 | `flash_attn_func(q, k, v)` |

### 对比方法

```python
def compare_with_baseline(custom_fn, baseline_fn, *args, sizes):
    results = []
    for size in sizes:
        test_args = generate_args(size)
        custom = benchmark_kernel(custom_fn, *test_args)
        baseline = benchmark_kernel(baseline_fn, *test_args)
        results.append({
            "size": size,
            "custom_ms": custom["median_ms"],
            "baseline_ms": baseline["median_ms"],
            "speedup": baseline["median_ms"] / custom["median_ms"],
            "pct_of_baseline": custom["median_ms"] / baseline["median_ms"] * 100,
        })
    return results
```

---

## 5. 实验矩阵设计

### 单变量 Sweep

每个 kernel 至少做以下 sweep：

1. **Problem Size Sweep**: 从小到大，找到性能稳定区间
2. **Block Size Sweep**: 32, 64, 128, 256, 512, 1024
3. **数据类型**: float32, float16（如适用）

### 多变量实验（GEMM 专用）

```python
# LLM 常见形状
llm_shapes = [
    (1, 4096, 4096),      # Decode single token
    (8, 4096, 4096),      # Decode batch=8
    (128, 4096, 4096),    # Prefill short
    (1024, 4096, 4096),   # Prefill medium
    (4096, 4096, 4096),   # Prefill long
    (1024, 11008, 4096),  # FFN up projection
    (1024, 4096, 11008),  # FFN down projection
]
```

---

## 6. 结果报告格式

### 标准输出格式

```json
{
  "kernel": "gemm_tiled",
  "gpu": "NVIDIA A30",
  "cuda_version": "12.4",
  "timestamp": "2026-06-01T05:00:00Z",
  "config": {
    "M": 2048, "N": 2048, "K": 2048,
    "block_size": [128, 128, 32],
    "dtype": "float32"
  },
  "results": {
    "median_ms": 1.23,
    "std_ms": 0.05,
    "tflops": 14.0,
    "pct_of_cublas": 65.2,
    "pct_of_peak": 45.3
  },
  "baseline": {
    "cublas_ms": 0.80,
    "cublas_tflops": 21.5
  }
}
```

### Markdown 报告模板

```markdown
## GEMM Benchmark Results

### Environment
- GPU: NVIDIA A30 (24GB)
- CUDA: 12.4
- Driver: 535.xx

### Results (M=N=K)

| Size | Naive | Tiled | Vectorized | Register | cuBLAS | Best % |
|------|-------|-------|-----------|----------|--------|--------|
| 512  | X     | X     | X         | X        | X      | X%     |
| 1024 | X     | X     | X         | X        | X      | X%     |
| 2048 | X     | X     | X         | X        | X      | X%     |

### Key Findings
1. ...
2. ...
```

---

## 7. 常见陷阱与规避

| 陷阱 | 影响 | 规避方法 |
|------|------|---------|
| 未预热 | 首次运行慢（JIT, cache cold） | 100 次预热 |
| GPU 降频 | 结果不稳定 | 锁定频率 |
| 其他进程干扰 | 结果偏高 | 确认 GPU 独占 |
| 数据太小 | launch overhead 占主导 | 测量足够大的 problem size |
| 未 synchronize | 计时不准 | 每次 `cudaDeviceSynchronize()` |
| 编译器优化掉 | kernel 被跳过 | 使用结果（如 checksum） |
| 只报 mean | 被 outlier 影响 | 报 median + p95 + std |
