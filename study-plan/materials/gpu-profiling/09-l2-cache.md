# 09 - L2 Cache 分析

## 1. 学习目标

- 理解 GPU L2 Cache 的架构、容量与工作机制
- 掌握 L2 命中率（Hit Rate）、吞吐量与驱逐策略的分析方法
- 学会诊断 L2 cache thrashing 和低效利用问题
- 理解 L2 cache 分区（partitioning）与持久化（persistence）策略
- 能够通过数据布局和访问模式优化提升 L2 效率
- 掌握 L2 cache 对不同 kernel 类型性能影响的量化方法

## 2. 性能问题动机

L2 Cache 是 GPU 内存层次中连接 SM 与 HBM 的关键缓冲层：

- A100 的 L2 容量为 40MB，H100 为 50MB，命中与否决定延迟差 5-10x
- 多 SM 并发访问导致 cache thrashing，有效容量远小于物理容量
- 不当的数据访问顺序导致频繁驱逐（eviction），命中率骤降
- 深度学习中 activation、weight、gradient 竞争 L2 空间
- CUDA 11.0+ 提供 L2 persistence 控制，但使用不当反而降低性能

## 3. 核心术语表

| 术语 | 英文全称 | 定义 |
|------|----------|------|
| L2 Cache | Level 2 Cache | GPU 全局共享的二级缓存 |
| Hit Rate | Cache Hit Rate | 请求在 cache 中命中的比例 |
| Miss Rate | Cache Miss Rate | 请求未命中需访问 HBM 的比例 |
| Eviction | Cache Eviction | 缓存满时替换旧数据的操作 |
| Cache Line | Cache Line | L2 缓存的最小管理单位（128 bytes） |
| Sector | Cache Sector | Cache line 内的最小访问单位（32 bytes） |
| Thrashing | Cache Thrashing | 频繁驱逐-重载导致命中率极低 |
| Write-back | Write-back Policy | 数据修改后延迟写回 HBM |
| Persistence | L2 Persistence | 指定数据在 L2 中的驻留优先级 |
| Set-associative | Set-associative Cache | 每个地址映射到固定组内的多路 |

## 4. 指标定义

| 指标 | 公式 | 单位 | 含义 |
|------|------|------|------|
| L2 Hit Rate | l2_hits / (l2_hits + l2_misses) × 100 | % | L2 命中率 |
| L2 Throughput | l2_bytes_transferred / kernel_time | GB/s | L2 数据吞吐 |
| L2 → DRAM Traffic | l2_misses × sector_size | bytes | 未命中导致的 HBM 流量 |
| Sector Hit Rate | sectors_hit / sectors_requested | % | Sector 级命中率 |
| Eviction Rate | evictions / time | evictions/s | 驱逐频率 |
| L2 Bandwidth Utilization | actual_throughput / peak_l2_bw | % | L2 带宽利用率 |
| Compression Ratio | uncompressed_size / compressed_size | ratio | L2 压缩效率（Hopper+） |

## 5. 指标来源

| 指标 | 数据源 | 获取方式 |
|------|--------|----------|
| L2 Hit Rate | Nsight Compute | `lts__t_sector_hit_rate.pct` |
| L2 Throughput | Nsight Compute | `lts__t_bytes.sum.per_second` |
| L2 Sectors Read | Nsight Compute | `lts__t_sectors_op_read.sum` |
| L2 Sectors Write | Nsight Compute | `lts__t_sectors_op_write.sum` |
| DRAM Reads from L2 Miss | Nsight Compute | `dram__sectors_read.sum` |
| L2 Cache Size | Device Properties | `l2CacheSize` |
| L2 Persistence | CUDA API | `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize)` |

## 6. 正常现象

- 工作集 < L2 容量时命中率 >90%
- 工作集 >> L2 容量时命中率降至 10-30%（streaming 模式）
- 矩阵乘法中 weight 矩阵重复访问时 L2 命中率较高
- 首次访问数据时 L2 必然 miss（cold miss）
- 多 kernel 交替执行时 L2 内容被刷新

## 7. 异常现象

- 工作集明显小于 L2 但命中率仍低
- 相同数据重复访问但命中率不升
- L2 throughput 远低于峰值且 DRAM throughput 也低
- 设置 L2 persistence 后性能反而下降
- 不同 block 调度顺序导致命中率差异巨大

## 8. 可能原因

| 异常 | 可能原因 |
|------|----------|
| 小工作集低命中率 | Cache thrashing（多 SM 竞争同一 set）；地址映射冲突 |
| 重复访问不命中 | 中间有其他大量数据访问驱逐了目标数据；LRU 策略不利 |
| L2 和 DRAM 都低 | Kernel 是 compute-bound；或 L1 命中率很高 |
| Persistence 反效果 | 持久化数据挤占了其他更频繁访问的数据空间 |
| Block 顺序影响大 | 不同 block 访问模式导致 cache set 冲突程度不同 |

## 9. 验证实验

### 实验 1：工作集大小与 L2 命中率关系

```python
import torch
import numpy as np

def l2_hit_rate_vs_working_set():
    """测量不同工作集大小下的性能（间接反映 L2 效果）"""
    results = []
    # A100 L2 = 40MB, 测试从 1MB 到 256MB
    sizes_mb = [1, 2, 4, 8, 16, 32, 40, 48, 64, 128, 256]
    
    for size_mb in sizes_mb:
        n = size_mb * 1024 * 1024 // 4  # float32
        x = torch.randn(n, device='cuda')
        y = torch.empty_like(x)
        
        # Warm-up: 让数据进入 L2
        for _ in range(5):
            y.copy_(x)
        torch.cuda.synchronize()
        
        # 第二次访问应该命中 L2（如果 < L2 size）
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(50):
            y.copy_(x)
        end.record()
        torch.cuda.synchronize()
        
        ms = start.elapsed_time(end) / 50
        bw = n * 4 * 2 / ms / 1e6  # GB/s
        results.append((size_mb, bw))
        print(f"{size_mb:4d} MB: {bw:.1f} GB/s")
    
    return results
```

### 实验 2：数据复用与 L2 效果

```python
def l2_reuse_experiment():
    """验证数据复用时 L2 的加速效果"""
    # 小矩阵（适配 L2）vs 大矩阵
    for size in [1024, 2048, 4096, 8192]:
        a = torch.randn(size, size, device='cuda')
        b = torch.randn(size, size, device='cuda')
        
        # Warm-up
        for _ in range(5):
            torch.mm(a, b)
        torch.cuda.synchronize()
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(10):
            torch.mm(a, b)
        end.record()
        torch.cuda.synchronize()
        
        ms = start.elapsed_time(end) / 10
        data_mb = size * size * 4 * 2 / 1024 / 1024
        flops = 2 * size**3 / ms / 1e9
        print(f"Size {size}: {ms:.3f} ms, data={data_mb:.1f}MB, {flops:.1f} TFLOPS")
```

### 实验 3：L2 Persistence API 使用

```cpp
// CUDA C++ 示例：L2 persistence 控制
#include <cuda_runtime.h>

void configure_l2_persistence() {
    // 设置持久化 L2 缓存大小（最多为 L2 总量的一部分）
    size_t persistingL2Size = 20 * 1024 * 1024; // 20MB
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, persistingL2Size);
    
    // 为特定数据流设置 L2 访问策略
    cudaStreamAttrValue attr;
    attr.accessPolicyWindow.base_ptr = (void*)device_ptr;
    attr.accessPolicyWindow.num_bytes = data_size;
    attr.accessPolicyWindow.hitRatio = 1.0f;  // 100% 命中优先
    attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
}
```

## 10. 优化方法

| 方法 | 适用场景 | 预期效果 |
|------|----------|----------|
| Tiling 适配 L2 | 工作集略超 L2 | 命中率从 30% 提升到 80%+ |
| L2 Persistence | 频繁复用的小数据 | 防止被驱逐 |
| Block 调度优化 | 相邻 block 访问相似数据 | 提升空间局部性 |
| 数据压缩 | Hopper+ 架构 | 有效 L2 容量翻倍 |
| 分阶段处理 | 多数据集竞争 L2 | 减少 thrashing |
| Prefetch 到 L2 | 可预测的访问模式 | 减少 cold miss |
| 减少数据精度 | fp32→fp16 | 工作集减半，更易适配 L2 |

## 11. 副作用

- L2 Persistence 占用固定 L2 空间，其他数据可用容量减少
- 过度 tiling 增加 kernel 复杂度和 launch 开销
- Block 调度优化依赖硬件行为，可移植性差
- Prefetch 增加内存带宽压力（即使数据最终未使用）
- 数据压缩有解压开销，对已压缩数据无效

## 12. Profiling 命令模板

```bash
# L2 Cache 完整分析
ncu --section MemoryWorkloadAnalysis_Chart python my_kernel.py

# L2 关键指标
ncu --metrics \
  lts__t_sector_hit_rate.pct,\
  lts__t_bytes.sum.per_second,\
  lts__t_sectors_op_read.sum,\
  lts__t_sectors_op_write.sum,\
  lts__t_sectors_op_atom.sum,\
  dram__bytes_read.sum,\
  dram__bytes_write.sum \
  python my_kernel.py

# L2 vs DRAM 流量对比
ncu --metrics \
  lts__t_bytes.sum,\
  dram__bytes.sum \
  python my_kernel.py

# 查看 L2 cache 大小
python -c "
import torch
props = torch.cuda.get_device_properties(0)
print(f'L2 Cache Size: {props.l2_cache_size / 1024 / 1024:.1f} MB')
"

# 多 kernel 场景下的 L2 行为
nsys profile --stats=true --gpu-metrics-device=all python multi_kernel.py

# 详细 memory chart
ncu --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables \
  -o l2_analysis python my_kernel.py
```

## 13. Benchmark 设计

### 设计原则

1. **工作集扫描**：从远小于 L2 到远大于 L2 的范围
2. **复用模式**：测试单次访问 vs 多次复用的差异
3. **并发竞争**：模拟多 kernel/多 stream 竞争 L2 的场景
4. **隔离测量**：排除 L1 cache 和计算的干扰

### L2 效率 Benchmark

```python
import torch
import numpy as np

def l2_benchmark():
    """L2 cache 效率基准测试"""
    results = []
    
    # 测试不同复用次数下的带宽
    size_mb = 20  # 小于 L2 (40MB)
    n = size_mb * 1024 * 1024 // 4
    x = torch.randn(n, device='cuda')
    y = torch.empty_like(x)
    
    for reuse_count in [1, 2, 4, 8, 16, 32]:
        # Warm-up
        for _ in range(5):
            y.copy_(x)
        torch.cuda.synchronize()
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(reuse_count * 10):
            y.copy_(x)
        end.record()
        torch.cuda.synchronize()
        
        ms = start.elapsed_time(end) / (reuse_count * 10)
        bw = n * 4 * 2 / ms / 1e6
        results.append((reuse_count, bw))
        print(f"Reuse {reuse_count:2d}x: {bw:.1f} GB/s")
    
    return results
```

## 14. 实验记录表

| 字段 | 示例值 | 说明 |
|------|--------|------|
| 日期 | 2026-05-31 | 实验日期 |
| GPU 型号 | A100-80GB | 硬件型号 |
| L2 Cache 大小 | 40 MB | 硬件规格 |
| L2 峰值带宽 | ~5 TB/s | 硬件规格 |
| 工作集大小 | 32 MB | 测试数据量 |
| 访问模式 | sequential read | 访问类型 |
| L2 Hit Rate | 87.3% | 命中率 |
| L2 Throughput | 3.2 TB/s | 实际吞吐 |
| DRAM Traffic | 4.1 MB | 未命中流量 |
| Kernel Duration | 0.156 ms | 执行时间 |
| L2 Persistence | 未启用 | 是否使用 |
| 备注 | 数据适配 L2 | 额外说明 |

## 15. 故障树

```
L2 Cache 效率低
├── 命中率低（工作集 < L2）
│   ├── Cache set 冲突 → 调整数据对齐/布局
│   ├── 多 SM 竞争同一 set → 调整 block 调度
│   ├── 中间 kernel 刷新 L2 → 合并 kernel 或使用 persistence
│   └── 写操作驱逐读数据 → 分离读写数据区域
├── 命中率低（工作集 > L2）
│   ├── 未做 tiling → 分块处理适配 L2
│   ├── 数据精度过高 → 降精度减小工作集
│   └── 无法 tiling → 接受 streaming 模式
├── L2 吞吐低
│   ├── 请求数不足 → 增加并发访问
│   ├── Bank 冲突 → 调整访问模式
│   └── Kernel 是 compute-bound → L2 非瓶颈
└── Persistence 无效
    ├── 持久化数据不是热点 → 重新选择持久化目标
    ├── 持久化空间过大 → 减小 persisting size
    └── 其他数据被挤出 → 平衡 persisting 与 streaming
```

## 16. 复盘模板

```markdown
## L2 Cache 分析复盘

### 实验目标
- Kernel 名称：
- 工作集大小：
- 预期 L2 行为：

### 测量结果
- L2 Hit Rate：__%
- L2 Throughput：__ TB/s
- DRAM Traffic：__ MB
- 工作集 vs L2 容量：

### 分析
- 命中率是否符合预期：
- 瓶颈在 L2 还是 DRAM：
- 是否存在 thrashing：

### 优化措施
| 措施 | Hit Rate 变化 | 性能变化 |
|------|--------------|----------|

### 经验总结
- L2 对该 kernel 的重要程度：
- 最有效的优化手段：
```

## 17. 常见错误

| # | 错误 | 后果 | 正确做法 |
|---|------|------|----------|
| 1 | 假设 L2 对所有 kernel 都重要 | 浪费优化时间 | 先确认是否 memory-bound |
| 2 | 用小数据测带宽当作 HBM 带宽 | 结果偏高（实际是 L2 带宽） | 数据 > 2×L2 size |
| 3 | Persistence 设置过大 | 挤占其他数据空间 | 只持久化真正热点数据 |
| 4 | 忽略 L2 sector 粒度 | 以为命中就是全部有效 | 检查 sector 利用率 |
| 5 | 不考虑多 kernel 间 L2 竞争 | 单 kernel 优化后整体变慢 | 全局视角分析 L2 使用 |
| 6 | 混淆 L1 和 L2 的作用 | 优化错误层级 | L1 per-SM，L2 全局共享 |
| 7 | 忽略 write-back 流量 | 低估 DRAM 带宽需求 | 同时检查读写流量 |
| 8 | 假设 L2 是 LRU | 实际策略更复杂 | 通过实验验证行为 |
| 9 | 未考虑 ECC 对 L2 的影响 | 有效容量计算错误 | ECC 不影响 L2 容量 |
| 10 | Tiling 时 tile 大小不考虑 L2 | Tile 过大仍然 thrash | Tile 工作集 < L2/SM数 |

## 18. 习题 20 道

1. A100 和 H100 的 L2 cache 大小分别是多少？
2. L2 cache line 和 sector 的大小分别是多少？它们的关系是什么？
3. 解释 L2 hit rate 对 memory-bound kernel 性能的影响。
4. 设计实验：验证工作集大小与 L2 命中率的关系。
5. 什么是 cache thrashing？在 GPU 上什么情况下容易发生？
6. 解释 L2 persistence API 的工作原理和适用场景。
7. 如何通过 Nsight Compute 获取 L2 hit rate 和 throughput？
8. 在矩阵乘法中，哪些数据适合驻留在 L2 中？为什么？
9. L2 cache 的 set-associative 结构如何影响命中率？
10. 设计实验：测量 L2 cache 的实际带宽（区别于 HBM 带宽）。
11. 多 kernel 并发执行时，L2 cache 如何被共享？有什么问题？
12. 解释 tiling 策略如何帮助数据适配 L2 cache。
13. L2 write-back 策略对写密集型 kernel 有什么影响？
14. 如何判断一个 kernel 的性能瓶颈在 L2 还是 HBM？
15. Hopper 架构的 L2 cache 压缩功能如何工作？
16. 解释 block 调度顺序如何影响 L2 cache 效率。
17. 在 Transformer 推理中，KV cache 与 L2 的关系是什么？
18. 设计一个实验验证 L2 persistence 的效果。
19. L2 cache 对 atomic 操作的性能有什么影响？
20. 如何在不修改 kernel 代码的情况下改善 L2 利用率？

## 19. 标准答案

1. A100: 40MB L2 cache。H100: 50MB L2 cache。L2 容量随架构升级逐步增大。

2. Cache line = 128 bytes，sector = 32 bytes。一个 cache line 包含 4 个 sector。L2 以 sector 为最小传输单位，但以 cache line 为管理（分配/驱逐）单位。

3. 高 hit rate 意味着数据从 L2 返回（~200 cycles），低 hit rate 需访问 HBM（~400-600 cycles）。对 memory-bound kernel，命中率从 50% 提升到 90% 可带来接近 2x 性能提升。

4. 见实验 1 代码。关键：从 1MB 到 256MB 逐步增加数据量，测量带宽。在 ~40MB 处应观察到带宽拐点（从 L2 带宽降至 HBM 带宽）。

5. Cache thrashing：数据被加载到 cache 后很快被驱逐，下次访问又 miss。GPU 上常见原因：多 SM 同时访问映射到同一 cache set 的不同地址，超过 set 的 associativity。

6. L2 Persistence API 允许指定某些数据在 L2 中的驻留优先级。通过 `cudaDeviceSetLimit` 设置持久化空间大小，通过 stream attribute 指定哪些数据使用持久化策略。适用于频繁复用的小数据（如 weight matrix）。

7. 使用 `ncu --metrics lts__t_sector_hit_rate.pct,lts__t_bytes.sum.per_second`。或使用 `ncu --section MemoryWorkloadAnalysis` 获取完整内存分析。

8. 在 GEMM C=A×B 中，如果 A 的行被多个 output tile 复用，A 适合驻留 L2。通常选择较小的矩阵（如 weight）驻留，较大的（如 activation）streaming 访问。

9. Set-associative：每个地址映射到固定 set，set 内有 N 路（N-way）。如果超过 N 个不同地址映射到同一 set，必须驱逐。GPU L2 通常 16-way，但大量 SM 并发访问仍可能超过。

10. 实验：使用小于 L2 的数据（如 20MB），warm-up 后反复 copy。测量带宽应显著高于 HBM 峰值（A100 L2 ~5 TB/s vs HBM ~2 TB/s）。

11. 多 kernel 并发时共享同一 L2。问题：kernel A 的数据可能驱逐 kernel B 的热数据。解决：使用 L2 persistence 保护关键数据，或错开 kernel 执行。

12. Tiling：将大矩阵分成小 tile，每次只处理一个 tile。选择 tile 大小使所有输入 tile 的总大小 < L2 容量。同一 tile 被多次访问时命中 L2，避免反复从 HBM 加载。

13. Write-back：写操作先更新 L2 中的 cache line，标记为 dirty，延迟写回 HBM。优点：减少 HBM 写流量（多次写同一位置只写回一次）。缺点：dirty line 驱逐时需要额外写回带宽。

14. 方法：比较 L2 throughput 和 DRAM throughput。如果 DRAM throughput 接近峰值，瓶颈在 HBM。如果 L2 throughput 高但 DRAM 低，数据主要在 L2 服务。也可看 L2 hit rate。

15. Hopper L2 压缩：硬件自动检测可压缩数据模式（如零值、重复值），在 L2 中以压缩格式存储。有效容量可增加 ~2x（取决于数据可压缩性）。对应用透明。

16. GPU 按 block ID 顺序调度 block 到 SM。如果相邻 block 访问相邻数据，它们可能共享 L2 中的 cache line。如果 block 调度导致远距离数据同时竞争 L2，thrashing 加剧。

17. KV cache 在 Transformer 推理中随序列长度增长。短序列时 KV cache 适配 L2，attention 计算快。长序列时 KV cache 超过 L2，每次 attention 都需从 HBM 加载，成为瓶颈。

18. 实验：选择一个频繁复用小数据的 kernel。分别在启用和未启用 L2 persistence 下运行，对比 L2 hit rate 和执行时间。确保持久化数据确实被多次访问。

19. L2 cache 对 atomic 操作有加速作用：如果多个线程 atomic 到同一地址，L2 可以在 cache 内完成合并（L2 atomic），避免每次都访问 HBM。A100+ 支持 L2 resident atomic。

20. 不修改 kernel 的方法：(1) 使用 L2 persistence API (2) 调整 kernel launch 顺序减少竞争 (3) 降低数据精度减小工作集 (4) 使用 CUDA Graph 优化 kernel 间 L2 复用 (5) 调整 stream 并发度。

## 20. 调优 Checklist

- [ ] 确认 L2 cache 大小（`cudaGetDeviceProperties`）
- [ ] 计算 kernel 工作集大小，与 L2 容量对比
- [ ] 使用 ncu 测量 L2 hit rate
- [ ] 检查 L2 throughput 是否接近峰值
- [ ] 分析 DRAM traffic 是否由 L2 miss 导致
- [ ] 评估是否适合使用 L2 persistence
- [ ] 考虑 tiling 策略适配 L2
- [ ] 检查多 kernel 间的 L2 竞争
- [ ] 评估降低数据精度的可行性
- [ ] 验证 block 调度对 L2 效率的影响
- [ ] 测量优化前后的 L2 hit rate 变化
- [ ] 确认 L2 优化带来了实际性能提升
