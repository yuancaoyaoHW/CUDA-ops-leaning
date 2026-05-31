# 07 - Memory Bandwidth（显存带宽分析）

## 1. 学习目标

- 理解 GPU 内存层次结构（Register → Shared Memory → L1 → L2 → HBM）的带宽特性
- 掌握 Memory Bandwidth（内存带宽）的理论峰值计算方法
- 学会测量实际带宽利用率并与理论峰值对比
- 理解 memory-bound kernel 的特征与优化方向
- 掌握 coalesced access（合并访问）与 bank conflict 的诊断方法
- 能够通过访存模式优化提升有效带宽

## 2. 性能问题动机

大多数深度学习 kernel 是 memory-bound 的，带宽利用率直接决定性能：

- HBM 带宽是 A100 的硬上限（2039 GB/s），实际利用率常不足 60%
- 非合并访问（uncoalesced access）导致有效带宽骤降至理论值的 1/32
- Shared memory bank conflict 使片上带宽降低为 1/N（N-way conflict）
- 数据类型选择不当（fp32 vs fp16）直接影响带宽需求
- 缺乏对 memory-bound vs compute-bound 的判断导致优化方向错误

## 3. 核心术语表

| 术语 | 英文全称 | 定义 |
|------|----------|------|
| HBM | High Bandwidth Memory | GPU 主存，高带宽堆叠内存 |
| 带宽 | Bandwidth | 单位时间内传输的数据量 |
| 合并访问 | Coalesced Access | 一个 warp 的内存请求合并为最少事务 |
| Bank Conflict | Shared Memory Bank Conflict | 多线程同时访问同一 bank 导致串行化 |
| 事务 | Memory Transaction | 内存控制器的最小传输单位（32B/128B） |
| 有效带宽 | Effective Bandwidth | 应用实际使用的数据量 / 时间 |
| 请求带宽 | Requested Bandwidth | 应用请求的字节数 / 时间 |
| 内存吞吐 | Memory Throughput | 实际传输的总字节数 / 时间（含冗余） |
| Sector | Memory Sector | L2 cache line 的最小访问单位（32 bytes） |
| Cache Line | Cache Line | L1/L2 缓存的基本单位（128 bytes for L1） |

## 4. 指标定义

| 指标 | 公式 | 单位 | 含义 |
|------|------|------|------|
| 理论峰值带宽 | memory_clock × bus_width × 2 (DDR) | GB/s | 硬件带宽上限 |
| 有效带宽 | (bytes_read + bytes_written) / time | GB/s | 应用层实际吞吐 |
| 带宽利用率 | effective_bw / peak_bw × 100 | % | 带宽使用效率 |
| L2 命中率 | l2_hits / (l2_hits + l2_misses) | % | L2 缓存效率 |
| Sector 效率 | requested_sectors / total_sectors | % | 每次事务的有效数据比例 |
| Global Load Efficiency | requested_bytes / transferred_bytes | % | 全局加载效率 |
| Shared Memory Efficiency | 无 conflict 事务 / 总事务 | % | 共享内存访问效率 |

## 5. 指标来源

| 指标 | 数据源 | 获取方式 |
|------|--------|----------|
| DRAM Throughput | Nsight Compute | `dram__bytes.sum.per_second` |
| L2 Throughput | Nsight Compute | `lts__t_bytes.sum.per_second` |
| L1 Throughput | Nsight Compute | `l1tex__t_bytes.sum.per_second` |
| Global Load Efficiency | Nsight Compute | Memory Workload Analysis section |
| Sector 利用率 | Nsight Compute | `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` |
| Bank Conflict | Nsight Compute | `l1tex__data_bank_conflicts_pipe_lsu.sum` |
| 理论峰值 | Device Properties | `memoryClockRate × memoryBusWidth` |

## 6. 正常现象

- 简单 element-wise kernel 带宽利用率可达 80-90%（接近硬件极限）
- L2 cache 对重复访问数据有显著加速效果
- 小数据量（<L2 size）时测量带宽可能超过 HBM 理论值（数据在 cache 中）
- Shared memory 带宽远高于 HBM（A100: ~19 TB/s vs 2 TB/s）
- 矩阵转置等操作的有效带宽低于 copy 操作（访存模式复杂）

## 7. 异常现象

- 有效带宽远低于理论峰值（<50%）
- Global load efficiency 远低于 100%
- 大量 bank conflict 报告
- L2 命中率异常低（数据应该在 cache 中）
- 带宽随数据量增加非线性下降

## 8. 可能原因

| 异常 | 可能原因 |
|------|----------|
| 有效带宽低 | 非合并访问；stride 访问；随机访问模式 |
| Load efficiency 低 | Warp 内线程访问不连续地址；数据未对齐 |
| Bank conflict 多 | Shared memory 访问 stride 为 bank 数的倍数 |
| L2 命中率低 | 工作集超过 L2 容量；访问模式无局部性 |
| 带宽非线性下降 | TLB miss 增加；page fault；NUMA 效应 |

## 9. 验证实验

### 实验 1：带宽测量基准

```python
import torch
import numpy as np

def measure_bandwidth(size_mb, dtype=torch.float32):
    """测量不同数据量下的实际带宽"""
    elem_size = torch.tensor([], dtype=dtype).element_size()
    n_elements = int(size_mb * 1024 * 1024 / elem_size)
    
    x = torch.randn(n_elements, device='cuda', dtype=dtype)
    y = torch.empty_like(x)
    
    # Warm-up
    for _ in range(10):
        y.copy_(x)
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    iterations = 100
    start.record()
    for _ in range(iterations):
        y.copy_(x)
    end.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start.elapsed_time(end) / iterations
    bytes_transferred = n_elements * elem_size * 2  # read + write
    bandwidth_gb = bytes_transferred / elapsed_ms / 1e6
    
    return bandwidth_gb

# 测试不同数据量
for size in [1, 4, 16, 64, 256, 1024, 4096]:
    bw = measure_bandwidth(size)
    print(f"{size:5d} MB: {bw:.1f} GB/s")
```

### 实验 2：合并访问 vs 非合并访问

```python
import triton
import triton.language as tl

@triton.jit
def coalesced_load(x_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    """合并访问：连续线程访问连续地址"""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x, mask=mask)

@triton.jit
def strided_load(x_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr, STRIDE: tl.constexpr):
    """Stride 访问：线程间地址有间隔"""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    strided_offs = offs * STRIDE
    mask = strided_offs < N
    x = tl.load(x_ptr + strided_offs, mask=mask)
    tl.store(out_ptr + offs, x, mask=mask)
```

### 实验 3：数据类型对带宽的影响

```python
def dtype_bandwidth_comparison():
    """比较不同数据类型的带宽利用"""
    size_mb = 256
    dtypes = [torch.float32, torch.float16, torch.bfloat16, torch.int8]
    
    for dtype in dtypes:
        bw = measure_bandwidth(size_mb, dtype)
        print(f"{str(dtype):20s}: {bw:.1f} GB/s")
```

## 10. 优化方法

| 方法 | 适用场景 | 预期效果 |
|------|----------|----------|
| 确保合并访问 | stride/scatter 访问 | 带宽提升 2-32x |
| 使用向量化加载 | 元素粒度访问 | float4 加载提升 4x 效率 |
| Shared memory 中转 | 矩阵转置等 | 将非合并转为合并 |
| 数据对齐 | 地址未对齐 | 减少额外事务 |
| 使用低精度类型 | fp32 可降为 fp16 | 带宽需求减半 |
| Padding 消除 bank conflict | shared memory conflict | 消除串行化 |
| 预取（prefetch） | 可预测的访问模式 | 隐藏内存延迟 |
| 数据布局变换 | AoS → SoA | 提升合并访问率 |

## 11. 副作用

- 向量化加载要求数据对齐，增加内存分配约束
- Shared memory 中转增加 kernel 复杂度和 smem 使用
- Padding 增加内存占用（通常 <5%）
- 低精度类型可能影响数值精度
- SoA 布局可能降低 CPU 端代码可读性
- 预取增加寄存器压力

## 12. Profiling 命令模板

```bash
# 完整内存分析
ncu --section MemoryWorkloadAnalysis python my_kernel.py

# 关键带宽指标
ncu --metrics \
  dram__bytes.sum.per_second,\
  dram__bytes_read.sum.per_second,\
  dram__bytes_write.sum.per_second,\
  lts__t_bytes.sum.per_second,\
  l1tex__t_bytes.sum.per_second,\
  l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
  l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum \
  python my_kernel.py

# 合并访问效率
ncu --metrics \
  smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,\
  smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct \
  python my_kernel.py

# Bank conflict 检测
ncu --metrics \
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
  python my_kernel.py

# 使用 bandwidth benchmark 工具
# CUDA samples 中的 bandwidthTest
/usr/local/cuda/samples/1_Utilities/bandwidthTest/bandwidthTest

# PyTorch 内置带宽测试
python -c "
import torch
from torch.utils.benchmark import Timer
x = torch.randn(256*1024*1024//4, device='cuda')
y = torch.empty_like(x)
t = Timer('y.copy_(x)', globals={'x':x,'y':y})
print(t.blocked_autorange())
"
```

## 13. Benchmark 设计

### 设计原则

1. **数据量覆盖**：从 L2 cache 内到远超 L2 的范围
2. **访问模式分类**：sequential、strided、random
3. **读写分离**：分别测量读带宽、写带宽、读写混合
4. **排除计算干扰**：使用纯 copy 或极简计算

### 标准带宽 Benchmark

```python
import torch
import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class BandwidthResult:
    size_mb: float
    bandwidth_gbs: float
    efficiency_pct: float
    access_pattern: str

def bandwidth_benchmark(peak_bw_gbs: float = 2039.0) -> List[BandwidthResult]:
    """A100 带宽基准测试"""
    results = []
    sizes_mb = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    
    for size_mb in sizes_mb:
        n = size_mb * 1024 * 1024 // 4
        x = torch.randn(n, device='cuda')
        y = torch.empty_like(x)
        
        for _ in range(20):
            y.copy_(x)
        torch.cuda.synchronize()
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        iters = max(10, 1000 // size_mb)
        start.record()
        for _ in range(iters):
            y.copy_(x)
        end.record()
        torch.cuda.synchronize()
        
        ms = start.elapsed_time(end) / iters
        bw = n * 4 * 2 / ms / 1e6
        eff = bw / peak_bw_gbs * 100
        
        results.append(BandwidthResult(size_mb, bw, eff, "sequential"))
    
    return results
```

## 14. 实验记录表

| 字段 | 示例值 | 说明 |
|------|--------|------|
| 日期 | 2026-05-31 | 实验日期 |
| GPU 型号 | A100-80GB | 硬件型号 |
| HBM 理论峰值 | 2039 GB/s | 硬件规格 |
| L2 Cache 大小 | 40 MB | 硬件规格 |
| 数据量 | 256 MB | 测试数据大小 |
| 数据类型 | float32 | 元素类型 |
| 访问模式 | sequential | coalesced/strided/random |
| 有效带宽 | 1756 GB/s | 测量值 |
| 带宽利用率 | 86.1% | 有效/峰值 |
| Global Load Eff. | 98.5% | Nsight Compute |
| L2 Hit Rate | 23.4% | 数据超过 L2 |
| Bank Conflicts | 0 | 无冲突 |
| Kernel Duration | 0.284 ms | Event timing |

## 15. 故障树

```
带宽利用率低
├── 非合并访问
│   ├── Stride 访问 → 重组数据布局为 SoA
│   ├── 随机访问 → 使用 shared memory 或排序
│   ├── 地址未对齐 → 确保起始地址 128B 对齐
│   └── 结构体数组 (AoS) → 转换为 SoA
├── 数据传输冗余
│   ├── 部分 cache line 有效 → 向量化加载
│   ├── 重复加载相同数据 → 使用 shared memory 缓存
│   └── 写后读依赖 → 重排计算顺序
├── Cache 效率低
│   ├── 工作集超 L2 → 分块处理（tiling）
│   ├── Cache thrashing → 调整 block 调度顺序
│   └── L1 bypass 不当 → 调整 cache 策略
├── Shared Memory 瓶颈
│   ├── Bank conflict → padding 或 swizzle
│   └── 带宽饱和 → 减少 smem 访问次数
└── 系统级问题
    ├── PCIe 瓶颈（H2D/D2H） → 使用 pinned memory
    ├── NUMA 效应 → 绑定 CPU 到 GPU 对应 NUMA node
    └── TLB miss → 使用大页（huge pages）
```

## 16. 复盘模板

```markdown
## 带宽分析复盘

### 实验目标
- Kernel 名称：
- 数据量/类型：
- 预期带宽利用率：

### 测量结果
- 有效带宽：__ GB/s
- 带宽利用率：__%
- Global Load Efficiency：__%
- L2 Hit Rate：__%

### 瓶颈分析
- 是否 memory-bound：
- 主要带宽损失来源：
- 访存模式分类：

### 优化措施
| 措施 | 带宽变化 | 性能变化 |
|------|----------|----------|

### 经验总结
- 关键发现：
- 适用场景：
```

## 17. 常见错误

| # | 错误 | 后果 | 正确做法 |
|---|------|------|----------|
| 1 | 用小数据量测带宽 | 数据在 L2 中，结果偏高 | 数据量 > 2× L2 size |
| 2 | 忽略读+写的总字节数 | 带宽计算偏低 | copy = read + write |
| 3 | AoS 布局做向量运算 | 非合并访问 | 转为 SoA |
| 4 | Shared memory stride = 32 | 32-way bank conflict | padding +1 |
| 5 | 未区分 memory-bound 和 compute-bound | 优化方向错误 | 先做 roofline 分析 |
| 6 | 认为带宽利用率应达 100% | 不切实际的目标 | 80-90% 已是优秀 |
| 7 | 忽略 ECC 开销 | 实际可用带宽低于标称 | A100 ECC 约降 5-10% |
| 8 | 混淆 requested vs transferred bytes | 效率计算错误 | 明确区分两者 |
| 9 | 未考虑 cache line 粒度 | 以为访问 1 byte 只传 1 byte | 最小传输 32B sector |
| 10 | 跨 page 边界访问 | TLB miss 增加 | 数据对齐到 page 边界 |

## 18. 习题 20 道

1. A100 GPU 的 HBM2e 理论峰值带宽是多少？如何计算？
2. 解释 coalesced access 的条件。一个 warp 的 32 个线程如何才能合并为一次 128B 事务？
3. 什么是 memory sector？为什么 sector 效率对带宽利用率很重要？
4. 编写代码测量 GPU 的实际内存带宽，并与理论峰值对比。
5. 解释 AoS vs SoA 布局对 GPU 带宽的影响，给出具体示例。
6. Shared memory 的 bank conflict 是什么？如何通过 padding 解决？
7. 如何判断一个 kernel 是 memory-bound 还是 compute-bound？
8. L1 cache 和 L2 cache 对带宽测量有什么影响？如何排除 cache 效应？
9. 向量化加载（vectorized load，如 float4）如何提升带宽利用率？
10. 设计实验：测量 stride-1、stride-2、stride-4、stride-32 访问的带宽差异。
11. 什么是 ECC（Error Correcting Code）？它对可用带宽有多大影响？
12. 解释 GPU 内存层次结构中各级的带宽和延迟特征。
13. 如何使用 Nsight Compute 的 Memory Workload Analysis 诊断带宽问题？
14. 什么是 memory coalescing 的对齐要求？未对齐时会发生什么？
15. 设计一个实验验证 L2 cache 大小对带宽的影响。
16. 解释 pinned memory（页锁定内存）对 Host-Device 传输带宽的影响。
17. 在矩阵转置中，如何使用 shared memory 将非合并写转为合并写？
18. 什么是 memory-level parallelism？它如何帮助隐藏内存延迟？
19. 比较 cudaMemcpy 和 kernel 内访存的带宽差异及原因。
20. 如何通过 tiling 策略提升数据局部性从而提升有效带宽？

## 19. 标准答案

1. A100 HBM2e: 内存时钟 1215 MHz × 总线宽度 5120 bit × 2 (DDR) = 2039 GB/s（开启 ECC 后约 1935 GB/s）。

2. 合并条件：warp 内 32 个线程访问连续的 128 字节区域，且起始地址 128B 对齐。每个线程访问 4B (float)，32×4=128B 恰好一次事务。

3. Sector = 32 bytes，是 L2 到 DRAM 的最小传输单位。如果一次请求只用了 sector 中的部分字节，其余字节浪费带宽。Sector 效率 = 有效字节 / 传输字节。

4. 见实验 1 代码。关键：使用足够大的数据（>2×L2），测量 copy 操作的 read+write 总字节除以时间。

5. AoS: `struct{float x,y,z,w} arr[N]`，线程 i 访问 arr[i].x 时地址间隔 16B，需 4 次事务。SoA: `float x[N],y[N],z[N],w[N]`，线程 i 访问 x[i] 时地址连续，1 次事务。带宽差 4x。

6. Shared memory 分为 32 个 bank，每 bank 宽 4B。当多个线程访问同一 bank 的不同地址时产生 conflict，串行化处理。Padding：将 `smem[32][32]` 改为 `smem[32][33]`，错开 bank 映射。

7. 方法：计算 arithmetic intensity (FLOP/Byte)，与 roofline 模型的拐点比较。或用 ncu 看 compute throughput vs memory throughput 哪个更接近峰值。

8. 小数据量（<L2 size）时数据驻留在 cache 中，测量带宽会超过 HBM 理论值。排除方法：使用远大于 L2 的数据量（A100 L2=40MB，用 >100MB 数据）。

9. float4 一次加载 16 bytes（4 个 float），减少指令数量，且编译器生成 128-bit load 指令（LDG.128），一条指令完成 4 个元素的加载，提升指令效率和带宽利用。

10. 实验设计：分配大数组，分别以 stride 1/2/4/32 读取元素，测量有效带宽。预期：stride-1 接近峰值，stride-32 降至 1/32（每 sector 只用 4B/32B=12.5%）。

11. ECC 为每 256 bits 数据添加额外校验位，占用约 6.25% 带宽。A100 开启 ECC 后可用带宽约降 5-10%。ECC 还增加少量延迟。

12. Register: ~0 cycle, ~20 TB/s; Shared Memory: ~20 cycles, ~19 TB/s; L1: ~30 cycles, ~12 TB/s; L2: ~200 cycles, ~5 TB/s; HBM: ~400-600 cycles, ~2 TB/s。

13. Memory Workload Analysis 显示：各级内存的吞吐量、命中率、sector 效率、请求模式。重点看 "Sectors/Request" 指标，理想值为 1（完美合并）。

14. 对齐要求：warp 首线程地址应为 128B 的倍数。未对齐时，一次 warp 访问可能跨越两个 128B segment，需要两次事务，带宽减半。

15. 实验：从 1MB 到 100MB 逐步增加数据量，测量带宽。在数据量超过 L2 size（40MB）时，带宽应从 L2 带宽（~5 TB/s）降至 HBM 带宽（~2 TB/s）。

16. Pinned memory 避免了 OS page fault 和额外的 staging buffer copy。PCIe 传输带宽：pageable ~12 GB/s，pinned ~25 GB/s（PCIe Gen4 x16 理论 32 GB/s）。

17. 方法：每个 block 将一个 tile 从全局内存合并读入 shared memory，在 shared memory 中转置，再从 shared memory 合并写回全局内存。读写都是合并的。

18. Memory-level parallelism (MLP)：同时发出多个未完成的内存请求。GPU 通过大量 warp 实现 MLP——当一个 warp 等待内存时，scheduler 切换到其他 warp，有效隐藏延迟。

19. cudaMemcpy 使用 DMA 引擎，专门优化大块连续传输，接近理论峰值。Kernel 内访存受合并度、cache 效率、计算交织等影响，通常低于 cudaMemcpy。

20. Tiling：将大矩阵分成小块（tile），每次只处理一个 tile。Tile 大小选择使其适配 L2/shared memory，重复访问同一 tile 内数据时命中 cache，减少 HBM 访问次数。

## 20. 调优 Checklist

- [ ] 确认 kernel 是否 memory-bound（roofline 分析）
- [ ] 计算理论峰值带宽（含 ECC 修正）
- [ ] 测量实际有效带宽并计算利用率
- [ ] 检查 Global Load/Store Efficiency
- [ ] 检查 Sector 效率（sectors per request）
- [ ] 确认数据地址 128B 对齐
- [ ] 确认 warp 内访问模式为合并访问
- [ ] 检查 shared memory bank conflict
- [ ] 考虑向量化加载（float4/int4）
- [ ] 评估数据布局（AoS vs SoA）
- [ ] 检查 L2 cache 利用率
- [ ] 考虑降低数据精度（fp32→fp16）
- [ ] 数据量 > 2×L2 时才测 HBM 带宽
- [ ] 记录完整的带宽-数据量曲线
