# Nsight Profiling Checklist

## 目的

定义使用 Nsight Systems 和 Nsight Compute 进行 GPU profiling 的标准流程，确保能系统化地定位性能瓶颈并产出可展示的分析报告。

---

## 1. 工具选择

| 工具 | 用途 | 粒度 |
|------|------|------|
| Nsight Systems (nsys) | 端到端 timeline 分析 | 系统级：CPU/GPU 交互、kernel 序列 |
| Nsight Compute (ncu) | 单 kernel 深度分析 | Kernel 级：throughput、occupancy、stall |

### 使用顺序

```
Step 1: nsys → 找到热点 kernel（哪个 kernel 最耗时）
Step 2: ncu → 深入分析热点 kernel（为什么慢）
Step 3: 优化 → 重新 nsys 验证整体改善
```

---

## 2. Nsight Systems Checklist

### 2.1 Profile 命令

```bash
# 基础 profile
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=report_name \
    --force-overwrite=true \
    ./your_binary

# Python 程序
nsys profile \
    --trace=cuda,nvtx \
    --python-sampling=true \
    --output=report_name \
    python your_script.py

# vLLM serving profile
nsys profile \
    --trace=cuda,nvtx \
    --duration=30 \
    --delay=10 \
    python -m vllm.entrypoints.openai.api_server --model ...
```

### 2.2 分析 Checklist

- [ ] **总体 GPU 利用率**: GPU 是否大部分时间在执行 kernel？
- [ ] **CPU-GPU 同步**: 是否有不必要的 `cudaDeviceSynchronize`？
- [ ] **Kernel 序列**: kernel 之间是否有 gap（idle time）？
- [ ] **Memory 传输**: 是否有不必要的 H2D/D2H 传输？
- [ ] **热点 Kernel**: 哪个 kernel 占总时间最多？
- [ ] **Kernel Launch Overhead**: 小 kernel 是否 launch overhead 占主导？
- [ ] **NVTX 标注**: 是否能区分 prefill vs decode 阶段？

### 2.3 关键观察点

```
Timeline 中寻找：
1. GPU idle gaps → CPU bottleneck 或同步问题
2. 长时间单 kernel → 优化目标
3. 大量短 kernel → 考虑 fusion 或 CUDA Graph
4. H2D/D2H 传输 → 考虑 pinned memory 或减少传输
```

### 2.4 统计命令

```bash
# 导出 kernel 统计
nsys stats report_name.nsys-rep --report cuda_gpu_kern_sum

# 导出 memory 操作统计
nsys stats report_name.nsys-rep --report cuda_mem_size_sum
```

---

## 3. Nsight Compute Checklist

### 3.1 Profile 命令

```bash
# 完整分析（慢但全面）
ncu --set full \
    --target-processes all \
    --launch-skip 100 \
    --launch-count 10 \
    --export report_%k \
    ./your_binary

# 快速分析（关键指标）
ncu --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    sm__warps_active.avg.pct_of_peak_sustained_elapsed \
    --launch-skip 100 \
    --launch-count 5 \
    ./your_binary

# 指定 kernel 分析
ncu --kernel-name "gemm_tiled" \
    --set full \
    ./your_binary
```

### 3.2 核心指标 Checklist

#### Memory 指标

- [ ] `dram__throughput.avg.pct_of_peak_sustained` — HBM 带宽利用率
  - Memory-bound kernel 目标：> 80%
  - 如果低：检查 coalescing、cache 效率
- [ ] `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum` — Global load bytes
  - 对比理论最小值，判断是否有冗余读取
- [ ] `lts__t_bytes.sum` — L2 cache 流量
  - 高于预期说明 L1 miss 多

#### Compute 指标

- [ ] `sm__throughput.avg.pct_of_peak_sustained` — SM 利用率
  - Compute-bound kernel 目标：> 80%
  - 如果低：检查 occupancy、instruction mix
- [ ] `sm__inst_executed.avg.pct_of_peak_sustained` — 指令吞吐
- [ ] `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained` — FMA 利用率
  - GEMM 核心指标

#### Occupancy 指标

- [ ] `sm__warps_active.avg.pct_of_peak_sustained` — 实际 occupancy
  - 目标：> 50%（不一定越高越好）
- [ ] `launch__occupancy_limit_registers` — register 限制
- [ ] `launch__occupancy_limit_shared_mem` — shared memory 限制
- [ ] `launch__occupancy_limit_blocks` — block 数量限制

#### Stall 指标

- [ ] `smsp__warps_issue_stalled_long_scoreboard_per_warp_active.pct` — Memory stall
  - 高 → memory-bound，需要更多 memory-level parallelism
- [ ] `smsp__warps_issue_stalled_short_scoreboard_per_warp_active.pct` — Compute stall
- [ ] `smsp__warps_issue_stalled_wait_per_warp_active.pct` — Sync stall
  - 高 → `__syncthreads()` 过多或 load imbalance

### 3.3 判断 Bound 类型

```
IF dram_throughput > 80% peak AND sm_throughput < 50%:
    → Memory-bound
    → 优化方向：减少 memory 访问、提高 cache 命中

IF sm_throughput > 80% peak AND dram_throughput < 50%:
    → Compute-bound
    → 优化方向：减少指令数、使用 Tensor Core

IF both < 50%:
    → Latency-bound (stall)
    → 优化方向：提高 occupancy、减少 sync、增加 ILP
```

---

## 4. Roofline Analysis

### 4.1 计算 Arithmetic Intensity

```python
def compute_ai(kernel_name, M=None, N=None, K=None, seq_len=None):
    """计算各 kernel 的 arithmetic intensity"""
    if kernel_name == "vector_add":
        # 1 FLOP per 3 bytes (read 2, write 1, float32)
        return 1 / (3 * 4)  # 0.083 FLOP/Byte
    elif kernel_name == "gemm":
        # 2MNK FLOPS / (2*(MK+KN+MN)*4) bytes
        flops = 2 * M * N * K
        bytes_accessed = 2 * (M*K + K*N + M*N) * 4
        return flops / bytes_accessed
    elif kernel_name == "softmax":
        # ~5N FLOPS / (2N*4) bytes (read + write)
        return 5 / 8  # ~0.625 FLOP/Byte
    elif kernel_name == "flash_attention":
        # ~4*N*d FLOPS per element / ~4*d bytes
        return seq_len  # 随 seq_len 增长
```

### 4.2 绘制 Roofline

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_roofline(gpu_peak_flops, gpu_peak_bw, kernels):
    """绘制 roofline 图"""
    ridge_point = gpu_peak_flops / gpu_peak_bw
    
    ai_range = np.logspace(-2, 3, 100)
    roof = np.minimum(gpu_peak_flops, ai_range * gpu_peak_bw)
    
    plt.figure(figsize=(10, 6))
    plt.loglog(ai_range, roof / 1e12, 'b-', linewidth=2)
    
    for kernel in kernels:
        plt.plot(kernel["ai"], kernel["achieved_flops"] / 1e12,
                 'ro', markersize=10)
        plt.annotate(kernel["name"], (kernel["ai"], kernel["achieved_flops"] / 1e12))
    
    plt.xlabel("Arithmetic Intensity (FLOP/Byte)")
    plt.ylabel("Performance (TFLOPS)")
    plt.title("Roofline Analysis")
    plt.grid(True)
    plt.savefig("roofline.png", dpi=150)
```

---

## 5. Profiling 报告模板

### 单 Kernel 分析报告

```markdown
## Kernel: gemm_tiled (M=2048, N=2048, K=2048)

### Summary
- **Bound type**: Compute-bound
- **SM throughput**: 72% of peak
- **Memory throughput**: 35% of peak
- **Occupancy**: 62%

### Key Metrics
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| SM throughput | 72% | >80% | ⚠️ |
| DRAM throughput | 35% | N/A | ✅ |
| Occupancy | 62% | >50% | ✅ |
| FMA utilization | 68% | >75% | ⚠️ |

### Bottleneck Analysis
- FMA 利用率 68%，低于目标
- 原因：instruction mix 中 non-FMA 指令占比 32%（地址计算、条件判断）
- 优化方向：减少循环开销、使用 register blocking 增加 FMA 比例

### Optimization Suggestions
1. 增大 register tile → 提高 FMA/non-FMA 比例
2. 使用 double buffering → 隐藏 shared memory load latency
3. 考虑 Tensor Core → 大幅提升 compute throughput
```

---

## 6. 常见问题诊断

| 症状 | 可能原因 | 诊断方法 | 解决方案 |
|------|---------|---------|---------|
| Bandwidth 低 | Non-coalesced access | 检查 global load/store pattern | 重排数据布局 |
| Bandwidth 低 | Bank conflict | 检查 shared memory access | 添加 padding |
| SM throughput 低 | Low occupancy | 检查 register/shared mem 使用 | 减少 per-thread 资源 |
| SM throughput 低 | Warp divergence | 检查 branch efficiency | 重构条件逻辑 |
| 高 stall | Memory latency | 检查 long_scoreboard stall | 增加 prefetch / occupancy |
| 高 stall | Sync barrier | 检查 wait stall | 减少 `__syncthreads()` |

---

## 7. 自动化脚本

```bash
#!/bin/bash
# profiling/ncu_profile.sh — 自动化 profiling 脚本

KERNEL_BINARY=$1
OUTPUT_DIR="profiling/reports/$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

echo "=== Profiling $KERNEL_BINARY ==="

# Step 1: Quick overview
ncu --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    sm__warps_active.avg.pct_of_peak_sustained_elapsed \
    --launch-skip 100 --launch-count 5 \
    --csv --log-file $OUTPUT_DIR/overview.csv \
    $KERNEL_BINARY

# Step 2: Full analysis on hottest kernel
ncu --set full \
    --launch-skip 100 --launch-count 3 \
    --export $OUTPUT_DIR/full_report \
    $KERNEL_BINARY

echo "=== Reports saved to $OUTPUT_DIR ==="
```
