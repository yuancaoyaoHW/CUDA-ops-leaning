# 06 - Occupancy Analysis（占用率分析）

## 1. 学习目标

- 理解 Occupancy（占用率）的定义及其与 SM（Streaming Multiprocessor）资源的关系
- 掌握影响 occupancy 的三大因素：寄存器、共享内存、线程块大小
- 学会使用 CUDA Occupancy Calculator 和 Nsight Compute 分析 occupancy
- 理解 occupancy 与实际性能之间的非线性关系
- 能够通过调整 kernel 参数提升 occupancy
- 掌握 occupancy 不足时的诊断与优化策略

## 2. 性能问题动机

Occupancy 是 GPU 并行度的核心指标，直接影响延迟隐藏能力：

- 低 occupancy 导致 warp scheduler 无法有效隐藏内存访问延迟
- 寄存器使用过多限制了每个 SM 上可驻留的线程块数量
- 共享内存分配过大挤占了其他线程块的空间
- 线程块大小选择不当导致 SM 资源浪费
- 盲目追求 100% occupancy 可能牺牲每线程可用寄存器，反而降低性能

## 3. 核心术语表

| 术语 | 英文全称 | 定义 |
|------|----------|------|
| Occupancy | Occupancy | 活跃 warp 数与 SM 最大支持 warp 数的比值 |
| SM | Streaming Multiprocessor | GPU 的基本计算单元 |
| Warp | Warp | 32 个线程组成的执行单元 |
| 寄存器文件 | Register File | SM 上的高速存储，每线程私有 |
| 共享内存 | Shared Memory | 线程块内共享的片上存储 |
| 线程块 | Thread Block (CTA) | 协作线程阵列，调度的基本单位 |
| Theoretical Occupancy | Theoretical Occupancy | 基于资源限制计算的最大可能占用率 |
| Achieved Occupancy | Achieved Occupancy | 运行时实际测量的平均占用率 |
| Launch Bounds | Launch Bounds | 编译器提示，限定 kernel 的最大线程数和最小块数 |
| Register Spilling | Register Spilling | 寄存器不足时溢出到 local memory |

## 4. 指标定义

| 指标 | 公式 | 单位 | 含义 |
|------|------|------|------|
| Theoretical Occupancy | active_warps_max / SM_max_warps × 100 | % | 资源允许的最大占用率 |
| Achieved Occupancy | avg(active_warps_runtime) / SM_max_warps × 100 | % | 实际运行时平均占用率 |
| Registers per Thread | 编译器分配 | 个 | 每线程使用的寄存器数 |
| Shared Memory per Block | 静态 + 动态分配 | bytes | 每线程块使用的共享内存 |
| Blocks per SM | min(资源限制) | 个 | 每 SM 可驻留的线程块数 |
| Warps per SM | blocks_per_SM × warps_per_block | 个 | 每 SM 活跃 warp 数 |
| Occupancy Limiter | 限制 occupancy 的主要资源 | - | registers/smem/block_size |

## 5. 指标来源

| 指标 | 数据源 | 获取方式 |
|------|--------|----------|
| Theoretical Occupancy | CUDA Occupancy API | `cudaOccupancyMaxActiveBlocksPerMultiprocessor` |
| Achieved Occupancy | Nsight Compute | `sm__warps_active.avg.pct_of_peak_sustained_active` |
| Registers per Thread | 编译器输出 | `nvcc --ptxas-options=-v` 或 `cuobjdump` |
| Shared Memory | Nsight Compute | `launch__shared_mem_per_block_allocated` |
| SM 硬件限制 | Device Properties | `cudaGetDeviceProperties` |
| Block Size | Launch 配置 | kernel<<<blocks, threads>>> |

## 6. 正常现象

- Achieved occupancy 略低于 theoretical occupancy（调度开销、块数不整除 SM 数）
- Compute-bound kernel 在 50% occupancy 下即可达到峰值性能
- 增加 occupancy 超过某阈值后性能不再提升（已充分隐藏延迟）
- 不同 GPU 架构的 SM 资源限制不同（A100: 64 warps/SM, 65536 regs/SM）
- 小线程块（如 64 threads）的 occupancy 受 blocks-per-SM 上限限制

## 7. 异常现象

- Achieved occupancy 远低于 theoretical（差距 >15%）
- Occupancy 很高但性能很差
- 增加线程块大小后 occupancy 反而下降
- 相同 kernel 在不同 GPU 上 occupancy 差异巨大
- Register spilling 导致 occupancy 提升但性能下降

## 8. 可能原因

| 异常 | 可能原因 |
|------|----------|
| Achieved << Theoretical | 线程块数不足以填满所有 SM；负载不均衡；尾效应 |
| 高 occupancy 低性能 | 内存带宽瓶颈；指令级并行度不足；cache thrashing |
| 增大 block 后 occupancy 降 | 寄存器总量超限；共享内存超限；blocks-per-SM 降为 1 |
| 跨 GPU 差异大 | SM 资源配置不同（寄存器数、共享内存大小、max warps） |
| Spilling 后性能降 | Local memory 访问走 L1/L2/DRAM，延迟远高于寄存器 |

## 9. 验证实验

### 实验 1：Occupancy 与寄存器关系

```python
import torch
import triton
import triton.language as tl

@triton.jit
def kernel_low_reg(x_ptr, out_ptr, N: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * 128 + tl.arange(0, 128)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * 2, mask=mask)

@triton.jit
def kernel_high_reg(x_ptr, out_ptr, N: tl.constexpr):
    """故意使用更多中间变量增加寄存器压力"""
    pid = tl.program_id(0)
    offs = pid * 128 + tl.arange(0, 128)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    a = x * x
    b = a + x
    c = b * a
    d = c + b
    e = d * c
    f = e + d
    g = f * e
    h = g + f
    tl.store(out_ptr + offs, h, mask=mask)
```

### 实验 2：Block Size 对 Occupancy 的影响

```python
import torch

def test_block_size_occupancy():
    """使用 CUDA occupancy API 通过 PyTorch 测试"""
    N = 1024 * 1024
    x = torch.randn(N, device='cuda')
    
    # 不同 block size 的 kernel launch
    block_sizes = [32, 64, 128, 256, 512, 1024]
    
    for bs in block_sizes:
        grid = (N + bs - 1) // bs
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        # Warm-up
        for _ in range(10):
            y = x * 2  # 简单操作
        torch.cuda.synchronize()
        
        start.record()
        for _ in range(100):
            y = x * 2
        end.record()
        torch.cuda.synchronize()
        
        print(f"Block size {bs:4d}: {start.elapsed_time(end)/100:.3f} ms")
```

### 实验 3：共享内存对 Occupancy 的影响

```python
import triton
import triton.language as tl

@triton.jit
def smem_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr, SMEM_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    
    # 分配不同大小的共享内存
    x = tl.load(x_ptr + offs)
    # Triton 自动管理 shared memory
    tl.store(out_ptr + offs, x + 1)

# 通过 ncu 观察不同 SMEM_SIZE 下的 occupancy
# ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active python test.py
```

## 10. 优化方法

| 方法 | 适用场景 | 预期效果 |
|------|----------|----------|
| 减少寄存器使用 | 寄存器是 limiter | 每 SM 可驻留更多 block |
| 使用 `__launch_bounds__` | 已知最优 block size | 编译器优化寄存器分配 |
| 调整 block size | block size 不是 warp 倍数 | 消除资源浪费 |
| 减少共享内存 | smem 是 limiter | 允许更多并发 block |
| 动态共享内存 | 不同场景需不同 smem | 灵活调整 occupancy |
| 使用 `maxrregcount` | 全局限制寄存器 | 提升 occupancy（可能 spill） |
| Kernel fusion | 多个低 occupancy kernel | 合并后提升利用率 |
| 持久化 kernel | grid 太小 | 用循环替代多次 launch |

## 11. 副作用

- 限制寄存器数量可能导致 register spilling，增加 local memory 访问
- 过高 occupancy 可能导致每线程可用 cache 减少，增加 cache miss
- 增大 block size 可能降低调度灵活性（尾效应更严重）
- `__launch_bounds__` 限制了 kernel 的通用性
- 减少共享内存可能需要重新设计算法（如减小 tile size）
- 追求 occupancy 可能牺牲算法效率（ILP、数据复用）

## 12. Profiling 命令模板

```bash
# Nsight Compute 查看 occupancy 详情
ncu --metrics \
  sm__warps_active.avg.pct_of_peak_sustained_active,\
  launch__occupancy_limit_registers,\
  launch__occupancy_limit_shared_mem,\
  launch__occupancy_limit_blocks,\
  launch__registers_per_thread,\
  launch__shared_mem_per_block_allocated,\
  launch__thread_count \
  python my_kernel.py

# 查看 theoretical vs achieved occupancy
ncu --section Occupancy python my_kernel.py

# 编译时查看寄存器使用
nvcc --ptxas-options=-v -o kernel kernel.cu

# 使用 cuobjdump 查看已编译 kernel 的寄存器信息
cuobjdump -res-usage my_binary

# CUDA Occupancy Calculator (命令行)
# 通过 Python API
python -c "
import torch
props = torch.cuda.get_device_properties(0)
print(f'Max threads/SM: {props.max_threads_per_multi_processor}')
print(f'Max threads/block: {props.max_threads_per_block}')
print(f'Regs/SM: {props.regs_per_multiprocessor}')
print(f'Shared mem/SM: {props.max_shared_memory_per_multiprocessor}')
print(f'Warp size: {props.warp_size}')
"

# 完整 occupancy section 报告
ncu --section Occupancy --section LaunchStats -o occupancy_report python my_kernel.py
```

## 13. Benchmark 设计

### 设计原则

1. **控制变量**：每次只改变一个影响 occupancy 的因素
2. **覆盖范围**：测试从低到高的 occupancy 区间
3. **关联性能**：同时记录 occupancy 和实际吞吐量
4. **架构感知**：记录 SM 资源限制作为分析基准

### Occupancy-Performance 关系实验

```python
import torch
import numpy as np

def occupancy_performance_sweep():
    """通过不同 block size 观察 occupancy-performance 关系"""
    N = 16 * 1024 * 1024  # 16M elements
    x = torch.randn(N, device='cuda')
    
    results = []
    block_sizes = [32, 64, 96, 128, 160, 192, 224, 256, 384, 512, 768, 1024]
    
    for bs in block_sizes:
        # Warm-up
        for _ in range(20):
            y = x * 2.0 + 1.0
        torch.cuda.synchronize()
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        times = []
        for _ in range(100):
            start.record()
            y = x * 2.0 + 1.0
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        
        median_ms = np.median(times)
        throughput_gb = N * 4 * 3 / median_ms / 1e6  # read + write
        results.append((bs, median_ms, throughput_gb))
        print(f"Block {bs:4d}: {median_ms:.3f} ms, {throughput_gb:.1f} GB/s")
    
    return results
```

## 14. 实验记录表

| 字段 | 示例值 | 说明 |
|------|--------|------|
| 日期 | 2026-05-31 | 实验日期 |
| GPU 型号 | A100-80GB SXM | 硬件型号 |
| Compute Capability | 8.0 | 架构版本 |
| Max Warps/SM | 64 | SM 最大 warp 数 |
| Max Regs/SM | 65536 | SM 寄存器总数 |
| Max Smem/SM | 164 KB | SM 共享内存上限 |
| Kernel 名称 | matmul_tiled | 被测 kernel |
| Block Size | 256 (16×16) | 线程块配置 |
| Regs/Thread | 32 | 每线程寄存器 |
| Smem/Block | 8192 bytes | 每块共享内存 |
| Theoretical Occ. | 100% | 理论占用率 |
| Achieved Occ. | 87.3% | 实际占用率 |
| Occupancy Limiter | registers | 限制因素 |
| Throughput | 245.6 GFLOPS | 实际吞吐 |
| Peak % | 78.5% | 峰值利用率 |

## 15. 故障树

```
Occupancy 不足
├── 寄存器限制
│   ├── 算法复杂度高 → 简化计算或分阶段
│   ├── 编译器未优化 → 使用 __launch_bounds__
│   ├── 循环展开过度 → 减少 unroll factor
│   └── 中间变量过多 → 重构代码复用变量
├── 共享内存限制
│   ├── Tile size 过大 → 减小 tile，多次迭代
│   ├── 数据类型过宽 → 使用 fp16/int8
│   ├── 多缓冲区 → 减少 buffer 数量
│   └── 未使用动态分配 → 改用动态 shared memory
├── Block size 限制
│   ├── Block 太大 (>1024) → 减小 block size
│   ├── Block 太小 → 增大 block size
│   └── 非 warp 对齐 → 调整为 32 的倍数
└── Grid size 不足
    ├── 问题规模太小 → 增加每线程工作量
    ├── Block 数 < SM 数 → 减小 block size 增加 block 数
    └── 尾效应 → 使用持久化 kernel
```

## 16. 复盘模板

```markdown
## Occupancy 分析复盘

### 实验目标
- Kernel 名称：
- 目标 occupancy：
- 当前 occupancy：

### 资源分析
- Regs/thread: __ → Blocks/SM (reg限制): __
- Smem/block: __ → Blocks/SM (smem限制): __
- Threads/block: __ → Blocks/SM (block限制): __
- 最终 limiter：

### 优化尝试
| 方法 | Occupancy 变化 | 性能变化 | 是否采用 |
|------|---------------|----------|----------|

### 关键发现
- Occupancy 与性能的关系：
- 最优 occupancy 点：
- 性能瓶颈转移：

### 结论
- 最终配置：
- 性能提升：
```

## 17. 常见错误

| # | 错误 | 后果 | 正确做法 |
|---|------|------|----------|
| 1 | 盲目追求 100% occupancy | 寄存器 spill 导致性能下降 | 找到性能最优的 occupancy 点 |
| 2 | 忽略 achieved vs theoretical 差距 | 未发现调度问题 | 两者都要检查 |
| 3 | Block size 非 32 倍数 | 浪费 warp 内线程 | 始终使用 32 的倍数 |
| 4 | 未考虑架构差异 | 代码在新 GPU 上 occupancy 骤降 | 查询 device properties |
| 5 | 只看 occupancy 不看性能 | 优化方向错误 | occupancy 是手段不是目的 |
| 6 | 忽略 register spilling | 以为 occupancy 提升就是优化 | 检查 local memory 流量 |
| 7 | 共享内存分配未对齐 | 实际分配大于请求 | 按 256 bytes 对齐 |
| 8 | 未使用 __launch_bounds__ | 编译器保守分配寄存器 | 明确告知编译器约束 |
| 9 | Grid size 太小 | SM 未被充分利用 | Grid 至少为 SM 数的 4 倍 |
| 10 | 混淆 block 级和 warp 级 occupancy | 分析错误 | 明确以 warp 为单位计算 |

## 18. 习题 20 道

1. 定义 occupancy，并解释为什么它对 GPU 性能重要。
2. A100 GPU 每个 SM 最多支持多少个 warp？多少个线程？
3. 如果一个 kernel 使用 64 个寄存器/线程，block size 为 256，在 A100 上 theoretical occupancy 是多少？
4. 解释 register spilling 的概念及其对性能的影响。
5. `__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)` 的两个参数分别如何影响编译器行为？
6. 为什么 block size 应该是 32 的倍数？如果设为 100 会怎样？
7. 共享内存如何影响 occupancy？给出一个计算示例。
8. 什么是 achieved occupancy？它为什么可能低于 theoretical occupancy？
9. 设计实验：验证 occupancy 与性能之间的非线性关系。
10. 在什么情况下，降低 occupancy 反而能提升性能？
11. 如何使用 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` API？
12. 解释 occupancy limiter 的概念，列举三种可能的 limiter。
13. 如果 grid size 只有 4 个 block，而 GPU 有 108 个 SM，会发生什么？
14. 动态共享内存与静态共享内存在 occupancy 计算中有何区别？
15. 如何通过 Nsight Compute 的 Occupancy section 诊断 occupancy 问题？
16. 解释"尾效应"（tail effect）对 achieved occupancy 的影响。
17. 对比 Volta、Ampere、Hopper 架构的 SM 资源限制差异。
18. 编写代码：自动搜索给定 kernel 的最优 block size。
19. 什么是持久化 kernel（persistent kernel）？它如何解决 occupancy 问题？
20. 如果 occupancy 已经 100% 但性能仍然不佳，应该从哪些方向继续优化？

## 19. 标准答案

1. Occupancy = 活跃 warp 数 / SM 最大 warp 数。重要性：高 occupancy 使 warp scheduler 有更多 warp 可调度，从而隐藏内存访问和指令延迟。

2. A100 (Compute Capability 8.0)：每 SM 最多 64 个 warp = 2048 个线程。最多 32 个 block/SM。

3. 计算：每 block 256 线程 = 8 warps。每 block 寄存器 = 256 × 64 = 16384。SM 总寄存器 65536 / 16384 = 4 blocks。4 blocks × 8 warps = 32 warps。Occupancy = 32/64 = 50%。

4. Register spilling：当 kernel 需要的寄存器超过硬件限制时，编译器将部分变量存储到 local memory（实际在 DRAM 中，经过 L1/L2 cache）。影响：每次 spill 访问延迟从 1 cycle 增加到数百 cycle。

5. `maxThreadsPerBlock`：告知编译器每 block 最大线程数，编译器据此分配寄存器。`minBlocksPerMultiprocessor`：要求编译器确保至少能驻留这么多 block，会限制寄存器使用。

6. 一个 warp = 32 线程。Block size 100 = 3 warps + 4 线程，第 4 个 warp 只有 4 个活跃线程，浪费 28 个线程的执行槽位。应设为 96 或 128。

7. 示例：A100 共享内存 164KB/SM。若 kernel 每 block 用 48KB smem，则 164/48 ≈ 3 blocks/SM。3 blocks × (256/32) warps = 24 warps。Occupancy = 24/64 = 37.5%。

8. Achieved occupancy 是运行时实际平均活跃 warp 比例。低于 theoretical 的原因：grid 不够大、负载不均衡、block 执行时间差异大、尾效应。

9. 实验设计：使用 `maxrregcount` 编译同一 kernel 的多个版本（16/32/48/64/128 regs），分别测量 occupancy 和吞吐量，绘制两者关系曲线。

10. 当 kernel 是 compute-bound 且需要大量寄存器实现数据复用时。例如 GEMM 中增大 tile size 降低 occupancy 但减少全局内存访问，净效果为正。

11. ```cpp
int numBlocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, myKernel, blockSize, dynamicSmemSize);
float occupancy = (float)(numBlocks * blockSize / warpSize) / maxWarpsPerSM;
```

12. Occupancy limiter 是限制 occupancy 的瓶颈资源。三种：(1) 寄存器数量 (2) 共享内存大小 (3) 每 SM 最大 block 数。Nsight Compute 会标注哪个是 limiter。

13. 只有 4 个 SM 有工作，其余 104 个 SM 空闲。GPU 利用率极低（<4%）。应增加 grid size 或使用持久化 kernel。

14. 静态共享内存在编译时确定大小，动态共享内存在 launch 时指定。Occupancy 计算中两者效果相同（都占用 SM 的 smem 资源），但动态分配允许运行时调整。

15. Nsight Compute Occupancy section 显示：theoretical/achieved occupancy、limiter 类型、每种资源允许的 blocks/SM、以及 occupancy 随 block size 变化的曲线图。

16. 尾效应：当 grid 中的 block 数不能被 SM 数整除时，最后一波 block 只占用部分 SM。例如 110 blocks / 108 SMs = 第一波 108 blocks 满载，第二波仅 2 blocks，大部分 SM 空闲。

17. Volta (CC 7.0): 64 warps/SM, 65536 regs, 96KB smem。Ampere (CC 8.0): 64 warps/SM, 65536 regs, 164KB smem。Hopper (CC 9.0): 64 warps/SM, 65536 regs, 228KB smem。主要差异在共享内存容量。

18. ```python
def auto_tune_block_size(kernel_fn, *args, min_bs=32, max_bs=1024):
    best_time, best_bs = float('inf'), min_bs
    for bs in range(min_bs, max_bs+1, 32):
        time_ms = benchmark_kernel(kernel_fn, block_size=bs, *args)
        if time_ms < best_time:
            best_time, best_bs = time_ms, bs
    return best_bs
```

19. 持久化 kernel：launch 的 block 数等于 SM 数 × 每 SM block 数，kernel 内部用循环处理所有数据。解决了 grid 太小的问题，保证所有 SM 始终有工作。适用于 reduction、scan 等操作。

20. 方向：(1) 检查是否 memory-bound，优化访存模式 (2) 提升 ILP（指令级并行）(3) 减少 warp divergence (4) 优化 L1/L2 cache 利用 (5) 使用 Tensor Core (6) 减少同步开销。

## 20. 调优 Checklist

- [ ] 使用 `cudaGetDeviceProperties` 获取 SM 资源限制
- [ ] 使用 Nsight Compute 获取 achieved occupancy
- [ ] 确认 occupancy limiter 是哪种资源
- [ ] Block size 是 32 的倍数
- [ ] 检查 registers/thread（`nvcc -Xptxas -v`）
- [ ] 检查 shared memory/block 分配量
- [ ] 尝试 `__launch_bounds__` 优化寄存器分配
- [ ] 验证 register spilling 情况（local memory 流量）
- [ ] Grid size 至少为 SM 数 × 4
- [ ] 测试多个 block size 找最优点
- [ ] 确认 occupancy 提升确实带来性能提升
- [ ] 记录 occupancy-performance 曲线拐点
- [ ] 检查 achieved vs theoretical 差距原因
- [ ] 考虑是否需要持久化 kernel 策略
