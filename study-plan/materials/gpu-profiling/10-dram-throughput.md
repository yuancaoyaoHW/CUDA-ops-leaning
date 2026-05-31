# 10 - DRAM Throughput（显存吞吐分析）

## 1. 学习目标

- 理解 GPU DRAM（HBM, High Bandwidth Memory）的物理架构与带宽特性
- 掌握 DRAM throughput 的测量方法与理论峰值计算
- 学会区分 DRAM read/write throughput 及其对性能的影响
- 理解 DRAM 利用率与 kernel 性能瓶颈的关系
- 能够诊断 DRAM 带宽饱和与未充分利用的问题
- 掌握通过减少 DRAM 访问量来优化性能的策略

## 2. 性能问题动机

DRAM（HBM）是 GPU 内存层次的最底层，带宽有限且延迟最高：

- A100 HBM2e 峰值 2039 GB/s，实际可达约 1800 GB/s（含 ECC 开销）
- 大量深度学习 kernel 受限于 DRAM 带宽而非计算能力
- DRAM 带宽饱和后，增加计算量不会降低性能（免费计算）
- 不当的访存模式导致 DRAM 实际吞吐远低于峰值
- 多 kernel 并发时 DRAM 带宽竞争导致性能不可预测

## 3. 核心术语表

| 术语 | 英文全称 | 定义 |
|------|----------|------|
| HBM | High Bandwidth Memory | 高带宽堆叠内存，GPU 主存技术 |
| DRAM Throughput | DRAM Throughput | 单位时间内 DRAM 传输的数据量 |
| Memory Controller | Memory Controller | 管理 DRAM 访问调度的硬件单元 |
| Channel | Memory Channel | DRAM 的独立访问通道 |
| Bank | DRAM Bank | DRAM 内部的独立存储阵列 |
| Row Buffer | Row Buffer | DRAM bank 中缓存当前活跃行的缓冲区 |
| Row Hit | Row Buffer Hit | 访问地址在当前活跃行中 |
| Row Miss | Row Buffer Miss | 需要激活新行的访问 |
| Burst Length | Burst Length | 单次传输的连续数据量 |
| ECC | Error Correcting Code | 错误校正码，占用部分带宽 |

## 4. 指标定义

| 指标 | 公式 | 单位 | 含义 |
|------|------|------|------|
| DRAM Read Throughput | dram_bytes_read / kernel_time | GB/s | DRAM 读吞吐 |
| DRAM Write Throughput | dram_bytes_write / kernel_time | GB/s | DRAM 写吞吐 |
| DRAM Total Throughput | (read + write) / kernel_time | GB/s | DRAM 总吞吐 |
| DRAM Utilization | actual_throughput / peak_throughput × 100 | % | DRAM 带宽利用率 |
| Bytes per FLOP | dram_bytes / flop_count | B/FLOP | 每次浮点运算的内存需求 |
| Arithmetic Intensity | flop_count / dram_bytes | FLOP/B | 计算密度（roofline 关键指标） |
| Read/Write Ratio | read_bytes / write_bytes | ratio | 读写比例 |

## 5. 指标来源

| 指标 | 数据源 | 获取方式 |
|------|--------|----------|
| DRAM Read Bytes | Nsight Compute | `dram__bytes_read.sum` |
| DRAM Write Bytes | Nsight Compute | `dram__bytes_write.sum` |
| DRAM Throughput | Nsight Compute | `dram__bytes.sum.per_second` |
| DRAM Utilization | Nsight Compute | `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` |
| Peak Bandwidth | Device Specs | memory_clock × bus_width × 2 |
| Sectors Read | Nsight Compute | `dram__sectors_read.sum` |
| Sectors Write | Nsight Compute | `dram__sectors_write.sum` |

## 6. 正常现象

- 简单 copy kernel DRAM 利用率达 80-90%
- Element-wise 操作（如 ReLU、add）接近 DRAM 带宽上限
- GEMM 等计算密集型 kernel DRAM 利用率较低（compute-bound）
- 开启 ECC 后实际可用带宽降低约 5-10%
- 读吞吐通常高于写吞吐（大多数 kernel 读多写少）

## 7. 异常现象

- Memory-bound kernel 的 DRAM 利用率低于 60%
- DRAM throughput 波动大（同一 kernel 多次运行）
- 读写比例与算法预期不符
- DRAM throughput 高但 kernel 性能差
- 小数据量时 DRAM throughput 异常低

## 8. 可能原因

| 异常 | 可能原因 |
|------|----------|
| 利用率低 | 非合并访问；L2 miss pattern 不利于 DRAM 调度；bank conflict |
| 波动大 | GPU 频率动态调整；thermal throttling；后台进程干扰 |
| 读写比异常 | Write-back 策略导致额外写；atomic 操作的 read-modify-write |
| 高吞吐低性能 | 大量冗余数据传输（低 sector 效率）；数据未被有效使用 |
| 小数据低吞吐 | 数据在 L2 中服务，未到达 DRAM；launch overhead 占比大 |

## 9. 验证实验

### 实验 1：DRAM 峰值带宽测量

```python
import torch
import numpy as np

def measure_dram_peak():
    """测量接近 DRAM 峰值的带宽"""
    # 使用远大于 L2 的数据确保访问 DRAM
    size_gb = 2  # 2GB >> L2 (40MB)
    n = size_gb * 1024 * 1024 * 1024 // 4
    x = torch.randn(n, device='cuda')
    y = torch.empty_like(x)
    
    # Warm-up
    for _ in range(5):
        y.copy_(x)
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(20):
        y.copy_(x)
    end.record()
    torch.cuda.synchronize()
    
    ms = start.elapsed_time(end) / 20
    bw = n * 4 * 2 / ms / 1e6  # read + write
    print(f"DRAM Bandwidth: {bw:.1f} GB/s")
    print(f"Utilization: {bw/2039*100:.1f}% of theoretical peak")
    return bw
```

### 实验 2：Arithmetic Intensity 与 DRAM 关系

```python
def arithmetic_intensity_experiment():
    """不同计算密度下的 DRAM 行为"""
    N = 64 * 1024 * 1024  # 256MB
    x = torch.randn(N, device='cuda')
    
    operations = {
        'copy (AI=0)': lambda: x.clone(),
        'add scalar (AI=0.25)': lambda: x + 1.0,
        'mul+add (AI=0.5)': lambda: x * 2.0 + 1.0,
        'polynomial (AI=2)': lambda: x*x*x + x*x + x + 1,
    }
    
    for name, op in operations.items():
        # Warm-up
        for _ in range(10):
            op()
        torch.cuda.synchronize()
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(50):
            y = op()
        end.record()
        torch.cuda.synchronize()
        
        ms = start.elapsed_time(end) / 50
        print(f"{name:25s}: {ms:.3f} ms")
```

### 实验 3：ECC 对带宽的影响

```bash
# 需要 root 权限
# 关闭 ECC（需要重启 GPU）
sudo nvidia-smi -e 0
# 开启 ECC
sudo nvidia-smi -e 1

# 在两种模式下分别运行带宽测试对比
```

## 10. 优化方法

| 方法 | 适用场景 | 预期效果 |
|------|----------|----------|
| 合并访问 | 非合并导致低效 | DRAM 利用率提升至 80%+ |
| 减少数据精度 | fp32→fp16/bf16 | DRAM 流量减半 |
| Kernel fusion | 多个 memory-bound kernel | 减少中间结果的 DRAM 读写 |
| 增加计算复用 | 数据只读一次多次计算 | 提升 arithmetic intensity |
| Tiling + shared memory | 数据可复用 | 减少 DRAM 访问次数 |
| 压缩/稀疏 | 数据有冗余 | 减少实际传输量 |
| 异步预取 | 可预测访问模式 | 隐藏 DRAM 延迟 |

## 11. 副作用

- 降低精度可能影响模型收敛或推理精度
- Kernel fusion 增加 kernel 复杂度，可能降低 occupancy
- Tiling 增加代码复杂度和共享内存使用
- 压缩/解压有计算开销
- 过度优化 DRAM 可能将瓶颈转移到计算单元

## 12. Profiling 命令模板

```bash
# DRAM 吞吐详细分析
ncu --metrics \
  dram__bytes.sum.per_second,\
  dram__bytes_read.sum.per_second,\
  dram__bytes_write.sum.per_second,\
  dram__bytes_read.sum,\
  dram__bytes_write.sum,\
  gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed \
  python my_kernel.py

# DRAM sectors 分析
ncu --metrics \
  dram__sectors_read.sum,\
  dram__sectors_write.sum,\
  dram__sectors.sum.per_second \
  python my_kernel.py

# 完整内存层次分析
ncu --section MemoryWorkloadAnalysis python my_kernel.py

# Speed of Light 快速判断瓶颈
ncu --section SpeedOfLight python my_kernel.py

# 查看 GPU 内存规格
nvidia-smi -q -d MEMORY

# 带宽测试工具
/usr/local/cuda/extras/demo_suite/bandwidthTest --device=0 --mode=shmoo
```

## 13. Benchmark 设计

### 设计原则

1. **数据量足够大**：远超 L2 cache 确保测量 DRAM
2. **访问模式控制**：分别测试顺序、随机、stride 模式
3. **读写分离**：独立测量读带宽和写带宽
4. **排除干扰**：锁频、无后台进程、充分 warm-up

### DRAM Throughput Benchmark

```python
import torch
import numpy as np

def dram_throughput_benchmark():
    """DRAM 吞吐基准测试"""
    # 确保数据远大于 L2
    size_mb = 512  # 512MB >> 40MB L2
    n = size_mb * 1024 * 1024 // 4
    
    x = torch.randn(n, device='cuda')
    y = torch.empty_like(x)
    
    tests = {
        'read-only': lambda: x.sum(),
        'write-only': lambda: y.fill_(1.0),
        'read-write (copy)': lambda: y.copy_(x),
        'read-write (add)': lambda: torch.add(x, 1.0, out=y),
    }
    
    for name, fn in tests.items():
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        times = []
        for _ in range(50):
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        
        median_ms = np.median(times)
        # 估算字节数（简化）
        if 'copy' in name or 'add' in name:
            total_bytes = n * 4 * 2
        elif 'read' in name:
            total_bytes = n * 4
        else:
            total_bytes = n * 4
        
        bw = total_bytes / median_ms / 1e6
        print(f"{name:20s}: {median_ms:.3f} ms, {bw:.1f} GB/s")
```

## 14. 实验记录表

| 字段 | 示例值 | 说明 |
|------|--------|------|
| 日期 | 2026-05-31 | 实验日期 |
| GPU 型号 | A100-80GB SXM | 硬件型号 |
| HBM 类型 | HBM2e | 内存技术 |
| 理论峰值 | 2039 GB/s | 硬件规格 |
| ECC 状态 | Enabled | 是否开启 |
| GPU 频率 | 1410 MHz (locked) | 是否锁频 |
| 数据量 | 512 MB | 测试数据大小 |
| DRAM Read | 1423 GB/s | 读吞吐 |
| DRAM Write | 356 GB/s | 写吞吐 |
| DRAM Total | 1779 GB/s | 总吞吐 |
| DRAM Utilization | 87.2% | 利用率 |
| Arithmetic Intensity | 0.25 FLOP/B | 计算密度 |
| Kernel Duration | 0.547 ms | 执行时间 |

## 15. 故障树

```
DRAM 吞吐不足
├── 访存模式问题
│   ├── 非合并访问 → 优化为连续访问
│   ├── 随机访问 → 排序/分组后访问
│   ├── Stride 过大 → 数据布局变换
│   └── 未对齐 → 确保 128B 对齐
├── L2 miss pattern 不利
│   ├── 大量 conflict miss → 调整数据布局
│   ├── 请求分散到少数 channel → 均衡 channel 使用
│   └── Row buffer miss 多 → 提升访问局部性
├── 硬件限制
│   ├── ECC 开销 → 评估是否可关闭
│   ├── Thermal throttling → 改善散热
│   ├── 频率降低 → 锁定频率
│   └── 硬件故障 → nvidia-smi 检查错误
├── 非 DRAM 瓶颈
│   ├── Compute-bound → DRAM 低利用率正常
│   ├── L2 命中率高 → 数据未到达 DRAM
│   └── Launch overhead → 增大 kernel 工作量
└── 带宽竞争
    ├── 多 kernel 并发 → 串行化或优先级控制
    ├── Host-Device 传输 → 使用 pinned memory + overlap
    └── P2P 传输 → 调度避免冲突
```

## 16. 复盘模板

```markdown
## DRAM Throughput 分析复盘

### 实验目标
- Kernel 名称：
- 预期 DRAM 行为：[memory-bound/compute-bound]
- 预期利用率：

### 测量结果
- DRAM Read：__ GB/s
- DRAM Write：__ GB/s
- DRAM Utilization：__%
- Arithmetic Intensity：__ FLOP/B

### 分析
- 是否达到预期利用率：
- 瓶颈确认：[DRAM/compute/other]
- 冗余传输量：

### 优化方向
| 方向 | 预期效果 | 实际效果 |
|------|----------|----------|

### 经验总结
```

## 17. 常见错误

| # | 错误 | 后果 | 正确做法 |
|---|------|------|----------|
| 1 | 用小数据测 DRAM 带宽 | 数据在 L2，结果不代表 DRAM | 数据 > 2× L2 size |
| 2 | 忽略 ECC 对峰值的影响 | 利用率计算偏低 | 用 ECC-adjusted 峰值 |
| 3 | 对 compute-bound kernel 优化 DRAM | 无效优化 | 先确认瓶颈类型 |
| 4 | 只看 throughput 不看 utilization | 无法判断优化空间 | 始终计算利用率 |
| 5 | 忽略读写比例 | 优化方向错误 | 分别分析读写 |
| 6 | 混淆 requested vs transferred bytes | 效率计算错误 | 用 ncu 获取实际传输量 |
| 7 | 未考虑 write-allocate | 低估写操作的 DRAM 流量 | 写 miss 时先读再写 |
| 8 | 假设所有 DRAM 访问等延迟 | 忽略 row hit/miss 差异 | 优化访问局部性 |
| 9 | 不锁频测带宽 | 结果不可复现 | nvidia-smi -lgc |
| 10 | 忽略 kernel fusion 的带宽收益 | 错过最大优化机会 | 优先考虑 fusion |

## 18. 习题 20 道

1. A100 HBM2e 的理论峰值带宽如何计算？给出具体数值。
2. 什么是 DRAM utilization？80% 和 50% 分别意味着什么？
3. 解释 arithmetic intensity 的概念及其与 roofline 模型的关系。
4. 为什么 element-wise 操作（如 ReLU）几乎总是 memory-bound？
5. 编写代码测量 GPU 的实际 DRAM 峰值带宽。
6. ECC 如何影响 DRAM 可用带宽？影响幅度约为多少？
7. 什么是 write-allocate 策略？它如何增加 DRAM 流量？
8. 解释 HBM 的 channel 和 bank 结构对带宽的影响。
9. 设计实验：验证 kernel fusion 对 DRAM 流量的减少效果。
10. 在什么情况下 DRAM throughput 高但性能仍然差？
11. 如何通过 Nsight Compute 的 Speed of Light 判断 DRAM 是否是瓶颈？
12. 比较 A100、H100、H200 的 HBM 带宽规格。
13. 解释为什么降低数据精度（fp32→fp16）能直接提升 memory-bound kernel 性能。
14. 什么是 DRAM row buffer？row hit 和 row miss 的延迟差异是多少？
15. 设计实验：测量不同 batch size 下 DRAM 利用率的变化。
16. 多 GPU 训练中，DRAM 带宽如何与通信带宽交互？
17. 解释 "免费计算"（free compute）的概念——何时增加计算不影响性能？
18. 如何估算一个 kernel 的理论最小执行时间（基于 DRAM 带宽）？
19. Transformer 中哪些操作是 DRAM-bound？哪些是 compute-bound？
20. 设计一个实验验证 DRAM bandwidth 是否随数据量线性扩展。

## 19. 标准答案

1. A100 HBM2e: 内存时钟 1215 MHz × 总线宽度 5120 bit × 2 (DDR) / 8 = 1,556,480 MB/s ≈ 2039 GB/s（部分计算取决于具体 SKU）。

2. DRAM utilization = 实际吞吐/峰值吞吐。80% 表示接近硬件极限，优化空间有限。50% 表示有显著优化空间，可能存在访存模式问题。

3. Arithmetic Intensity (AI) = FLOPs / Bytes。Roofline 模型中，AI < 拐点时 kernel 是 memory-bound，AI > 拐点时是 compute-bound。拐点 = peak_compute / peak_bandwidth。

4. ReLU: 每元素 1 次比较 + 1 次读 + 1 次写 = 1 FLOP / 8 Bytes (fp32)。AI = 0.125，远低于 roofline 拐点（A100 约 100 FLOP/B for fp32），因此必然 memory-bound。

5. 见实验 1 代码。关键：使用 >1GB 数据，copy 操作，计算 read+write 总字节/时间。

6. ECC 为每 256 bits 添加校验位，占用约 6.25% 存储空间和带宽。A100 开启 ECC 后实际可用带宽约 1900-1935 GB/s（降低 5-7%）。

7. Write-allocate：写 miss 时先将整个 cache line 从 DRAM 读入 cache，再修改。即使只写 4 bytes，也需要先读 128 bytes。增加了读流量。解决：使用 streaming store 绕过 cache。

8. HBM 有多个 channel（A100: 32 channels），每个 channel 独立访问。Bank 是 channel 内的并行单元。地址均匀分布到各 channel/bank 时带宽最大。集中访问少数 channel 会成为瓶颈。

9. 实验：(1) 分别运行 y=relu(x) 和 z=relu(x)+1 两个 kernel (2) 融合为一个 kernel z=relu(x)+1。测量 DRAM 总流量。融合版本应减少一次完整的读写（节省 2×N×4 bytes）。

10. 情况：大量冗余传输（sector 效率低）——DRAM 传输了很多数据但应用只用了一小部分。或者 DRAM 带宽被非关键路径占用。

11. Speed of Light section 显示 compute 和 memory 的利用率百分比。如果 memory% >> compute%，则 DRAM 是瓶颈。如果两者都低，可能有其他瓶颈（如 latency）。

12. A100: HBM2e, 2039 GB/s, 80GB。H100: HBM3, 3350 GB/s, 80GB。H200: HBM3e, 4800 GB/s, 141GB。

13. fp32→fp16：每元素从 4B 降为 2B，相同元素数的 DRAM 流量减半。Memory-bound kernel 性能近似与带宽需求成反比，因此性能接近翻倍。

14. Row buffer 是 DRAM bank 中缓存当前活跃行的 SRAM。Row hit: ~10ns（直接从 buffer 读）。Row miss: ~30-50ns（需要 precharge + activate 新行）。差异约 3-5x。

15. 实验：固定模型，从 batch=1 到 batch=256 逐步增加。小 batch 时 kernel 太短（launch overhead 占比大），DRAM 利用率低。大 batch 时数据量足够，利用率趋近峰值。

16. 训练中 DRAM 带宽用于前向/反向计算，通信带宽用于梯度同步。理想情况下计算与通信重叠。如果 DRAM 带宽不足导致计算变慢，通信等待时间增加，整体效率下降。

17. "免费计算"：当 kernel 是 memory-bound 时，DRAM 传输时间决定总时间。在此期间增加计算（只要不超过计算峰值）不会增加总时间。例如在 element-wise kernel 中加入额外算术操作。

18. 理论最小时间 = total_DRAM_bytes / peak_DRAM_bandwidth。例如：copy 1GB 数据，最小时间 = 2GB (read+write) / 2039 GB/s ≈ 0.98 ms。

19. DRAM-bound：LayerNorm、Softmax、element-wise activation、embedding lookup。Compute-bound：矩阵乘法（GEMM）、卷积（大 batch）。边界：Attention（取决于序列长度）。

20. 实验：从 64MB 到 8GB 逐步增加 copy 数据量，测量带宽。预期：超过 L2 后带宽稳定在 DRAM 峰值附近，与数据量无关（线性扩展 = 带宽恒定）。

## 20. 调优 Checklist

- [ ] 确认 kernel 是否 memory-bound（Speed of Light）
- [ ] 计算理论峰值带宽（含 ECC 修正）
- [ ] 测量实际 DRAM throughput（read + write）
- [ ] 计算 DRAM utilization
- [ ] 计算 arithmetic intensity
- [ ] 检查是否有冗余 DRAM 传输（sector 效率）
- [ ] 评估 kernel fusion 机会
- [ ] 评估降低数据精度的可行性
- [ ] 检查是否存在 write-allocate 额外流量
- [ ] 确认数据量足够大（排除 L2 cache 效应）
- [ ] 锁定 GPU 频率确保测量稳定
- [ ] 与理论最小执行时间对比
- [ ] 记录优化前后的 DRAM 流量变化
