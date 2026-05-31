# 02 - Nsight Compute：Kernel 级性能分析

## 1. 学习目标

- 掌握 Nsight Compute（简称 NCU）的核心工作流：采集、分析、对比
- 理解 Speed of Light (SOL) chart 的含义与解读方法
- 能够使用 Memory Workload Analysis 定位显存瓶颈
- 能够使用 Compute Workload Analysis 定位计算瓶颈
- 掌握 Scheduler Statistics 与 Warp State 的关联分析
- 能够针对单个 kernel 给出量化优化建议

## 2. 性能问题动机

在 GPU 程序优化中，Nsight Systems 提供的是时间线级别的宏观视图，而真正的性能瓶颈往往隐藏在单个 kernel 内部。例如：

- 一个 GEMM kernel 只达到了理论峰值的 30%，但从时间线上看不出原因
- 某个 kernel 的 L2 cache hit rate 极低，导致 DRAM 带宽成为瓶颈
- Warp scheduler 大量时间处于 stall 状态，但不知道 stall 原因

Nsight Compute 通过硬件性能计数器（Hardware Performance Counters）提供 kernel 内部的微架构级分析，是定位这类问题的核心工具。

## 3. 核心术语表

| 术语 | 英文全称 | 定义 |
|------|----------|------|
| SOL | Speed of Light | 当前 kernel 相对于硬件理论峰值的利用率百分比 |
| SM | Streaming Multiprocessor | GPU 的基本计算单元 |
| Warp | Warp | 32 个线程组成的调度基本单位 |
| Occupancy | Occupancy | SM 上活跃 warp 数与最大可驻留 warp 数的比值 |
| IPC | Instructions Per Cycle | 每周期执行的指令数 |
| Sector | Sector | L1/L2 cache 的最小传输单位（32 bytes） |
| LSU | Load/Store Unit | 负责内存访问的功能单元 |
| ALU | Arithmetic Logic Unit | 负责算术运算的功能单元 |
| FMA | Fused Multiply-Add | 融合乘加指令 |
| MIO | Memory Input/Output | 内存 I/O 管线 |

## 4. 指标定义

### 4.1 Speed of Light 指标

| 指标名 | 计算公式 | 含义 |
|--------|----------|------|
| SM SOL % | (achieved_compute / peak_compute) × 100 | 计算单元利用率 |
| Memory SOL % | (achieved_bandwidth / peak_bandwidth) × 100 | 显存带宽利用率 |
| Roofline Position | 基于 arithmetic intensity 的位置 | 判断 compute-bound 还是 memory-bound |

### 4.2 Memory Workload 指标

| 指标名 | 含义 |
|--------|------|
| Global Load Throughput | 全局内存读取吞吐量 (GB/s) |
| Global Store Throughput | 全局内存写入吞吐量 (GB/s) |
| L1 Hit Rate | L1 cache 命中率 |
| L2 Hit Rate | L2 cache 命中率 |
| DRAM Throughput | HBM 实际吞吐量 |
| Shared Memory Bank Conflict | 共享内存 bank 冲突次数 |

### 4.3 Compute Workload 指标

| 指标名 | 含义 |
|--------|------|
| Executed IPC | 实际每周期执行指令数 |
| Issued IPC | 每周期发射指令数 |
| Pipe Utilization (FMA) | FMA 管线利用率 |
| Pipe Utilization (ALU) | ALU 管线利用率 |
| Pipe Utilization (Tensor) | Tensor Core 管线利用率 |
| Eligible Warps Per Cycle | 每周期可调度的 warp 数 |

### 4.4 Scheduler Statistics 指标

| 指标名 | 含义 |
|--------|------|
| Active Warps Per Scheduler | 每个 scheduler 的活跃 warp 数 |
| Eligible Warps Per Scheduler | 每个 scheduler 可发射的 warp 数 |
| Issued Warp Per Scheduler | 每个 scheduler 实际发射的 warp 数 |
| No Eligible Reason | warp 不可调度的原因分布 |

## 5. 指标来源

### 5.1 硬件性能计数器

Nsight Compute 通过 GPU 硬件内置的 Performance Monitor (PerfMon) 单元采集数据：

- **SM 级计数器**：每个 SM 独立计数，最终汇总
- **L2 级计数器**：L2 cache 分区级别的计数
- **DRAM 级计数器**：内存控制器级别的计数
- **Warp Scheduler 计数器**：记录每个 cycle 的调度决策

### 5.2 采集模式

```
# 完整采集（所有 section）
ncu --set full -o report ./my_kernel

# 仅采集 SOL 和 Memory
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis -o report ./my_kernel

# 采集指定 kernel
ncu --kernel-name "gemm" --launch-skip 5 --launch-count 3 -o report ./my_kernel
```

### 5.3 注意事项

- NCU 采集会显著降低 kernel 执行速度（10x-100x slowdown）
- 每个 section 需要多次 kernel replay 来采集不同计数器组
- 计数器之间存在互斥关系，不能在同一次 pass 中同时采集

## 6. 正常现象

| 现象 | 说明 |
|------|------|
| SM SOL 60-80% 且 Memory SOL < 30% | 典型的 compute-bound kernel |
| Memory SOL 60-80% 且 SM SOL < 30% | 典型的 memory-bound kernel |
