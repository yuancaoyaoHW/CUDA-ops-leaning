# Roofline Analysis

## 1. 学习目标

- 理解 Roofline 模型的数学定义与图形解读
- 掌握 arithmetic intensity 的计算方法
- 能够判断 kernel 是 compute-bound 还是 memory-bound
- 理解不同 GPU 架构的 roofline 参数
- 能够用 roofline 指导 kernel 优化方向

## 2. 性能问题动机

### 2.1 为什么需要 Roofline？

优化 kernel 前必须回答：**瓶颈是计算还是访存？**

- 如果 compute-bound → 优化计算效率（Tensor Core、ILP）
- 如果 memory-bound → 优化访存（fusion、vectorization、cache）
- 错误判断 → 优化方向错误 → 浪费时间

### 2.2 Roofline 模型

```
Achievable Performance = min(Peak_Compute, Peak_Bandwidth × AI)

其中 AI = Arithmetic Intensity = FLOPs / Bytes_accessed
```

图形上：
- X 轴：Arithmetic Intensity (FLOPs/Byte)
- Y 轴：Performance (FLOPS)
- 两条线的交点 = Ridge Point（拐点）
- 拐点左边 = memory-bound region
- 拐点右边 = compute-bound region

## 3. 核心术语表

| 术语 | 英文全称 | 定义 |
|------|----------|------|
| Roofline | Roofline Model | 性能上限模型 |
| AI | Arithmetic Intensity | 计算密度 = FLOPs / Bytes |
| Ridge Point | Ridge Point | compute 和 memory 线的交点 |
| Peak Compute | Peak Compute | GPU 理论最大计算能力 |
| Peak Bandwidth | Peak Memory Bandwidth | GPU 理论最大内存带宽 |
| Operational Intensity | Operational Intensity | 同 AI |
| Attainable Performance | Attainable Performance | 给定 AI 下的理论最大性能 |

## 4. 指标定义

### 4.1 GPU 参数（A100 SXM 80GB）

| 参数 | 值 |
|------|-----|
| FP32 Peak | 19.5 TFLOPS |
| FP16 Tensor Core Peak | 312 TFLOPS |
| BF16 Tensor Core Peak | 312 TFLOPS |
| FP8 Tensor Core Peak | 624 TFLOPS |
| HBM Bandwidth | 2039 GB/s |
| L2 Bandwidth | ~5 TB/s |
| L1/Shared Bandwidth | ~19 TB/s per SM |

### 4.2 Ridge Points

```
FP32 Ridge Point = 19.5T / 2039G = 9.6 FLOPs/Byte
FP16 TC Ridge Point = 312T / 2039G = 153 FLOPs/Byte
```

### 4.3 常见算子的 AI

| 算子 | AI (FLOPs/Byte) | Bound (FP16 TC) |
|------|-----------------|-----------------|
| Vector Add | 0.17 | Memory |
| Reduction | 0.25 | Memory |
| Softmax | 0.5 | Memory |
| RMSNorm | 0.5 | Memory |
| RoPE | 0.375 | Memory |
| GEMM (M=N=K=4096, FP16) | 1365 | Compute |
| GEMM (M=1, N=K=4096, FP16) | 1 | Memory |
| GEMM (M=32, N=K=4096, FP16) | 32 | Memory |
| FlashAttention (S=4096) | ~1024 | Compute |
| Decode Attention (S_kv=4096) | ~2 | Memory |

## 5. 指标来源

```bash
# 使用 Nsight Compute 获取实际 AI
ncu --set roofline ./my_kernel

# 手动计算
# 1. 确定 FLOPs（从算法分析）
# 2. 确定 Bytes（从 Nsight Compute 的 DRAM throughput）
# 3. AI = FLOPs / Bytes
```

## 6. 正常现象

- Memory-bound kernel 的 achieved bandwidth 接近 peak（>80%）
- Compute-bound kernel 的 achieved TFLOPS 接近 peak（>60% for TC）
- AI 与理论计算一致

## 7. 异常现象

| 异常 | 含义 | 可能原因 |
|------|------|---------|
| Memory-bound 但 BW 低 | 访存效率差 | Non-coalesced access |
| Compute-bound 但 TFLOPS 低 | 计算效率差 | Low occupancy, divergence |
| AI 比理论低 | 额外数据移动 | Cache miss, spilling |
| AI 比理论高 | 数据被缓存 | L2 hit, data reuse |

## 8-20. 关键内容

### 如何使用 Roofline 指导优化

```
Step 1: 计算理论 AI
Step 2: 确定 bound 类型
Step 3: 选择优化方向

If memory-bound:
  → Kernel fusion（减少 global memory 读写）
  → Vectorized load（提高带宽利用）
  → 减少数据精度（FP16/INT8）
  → 利用 cache（tiling）

If compute-bound:
  → 使用 Tensor Core
  → 提高 occupancy
  → 增加 ILP
  → 减少 warp divergence
```

### Benchmark 设计

```python
import torch
import time

def measure_roofline_point(kernel_fn, flops, bytes_accessed, warmup=10, repeat=100):
    """测量一个 kernel 在 roofline 上的位置"""
    # Warmup
    for _ in range(warmup):
        kernel_fn()
    torch.cuda.synchronize()
    
    # Measure
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        kernel_fn()
    end.record()
    torch.cuda.synchronize()
    
    time_ms = start.elapsed_time(end) / repeat
    time_s = time_ms / 1000
    
    achieved_tflops = flops / time_s / 1e12
    achieved_bw = bytes_accessed / time_s / 1e9  # GB/s
    ai = flops / bytes_accessed
    
    return {
        "ai": ai,
        "achieved_tflops": achieved_tflops,
        "achieved_bw_gbs": achieved_bw,
        "time_ms": time_ms,
        "bound": "compute" if ai > 153 else "memory",  # FP16 TC ridge
    }
```

### 习题（选 5 道）

1. A100 的 FP16 Tensor Core ridge point 是多少？如何计算？
2. 为什么 decode attention 是 memory-bound 而 prefill attention 是 compute-bound？
3. 如果一个 kernel 的 AI=5，在 A100 上理论最大性能是多少 TFLOPS？
4. Kernel fusion 如何改变算子的 arithmetic intensity？
5. 如何用 Nsight Compute 验证一个 kernel 的 roofline 位置？

### 调优 checklist

- [ ] 计算 kernel 的理论 AI
- [ ] 确定 bound 类型（compute vs memory）
- [ ] 测量实际 achieved performance
- [ ] 计算 efficiency = achieved / theoretical_max
- [ ] 如果 memory-bound：检查带宽利用率
- [ ] 如果 compute-bound：检查 Tensor Core 利用率
- [ ] 识别优化方向（fusion? vectorization? TC?）
- [ ] 优化后重新测量，验证改善
