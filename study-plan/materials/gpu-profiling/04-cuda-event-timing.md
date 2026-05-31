# 04 - CUDA Event Timing

## 1. 学习目标

- 掌握 CUDA Event（CUDA 事件）的创建、记录与同步机制
- 理解 Event-based timing 与 CPU wall-clock timing 的本质区别
- 能够在多流（multi-stream）场景下正确测量 kernel 执行时间
- 学会排除首次启动开销（launch overhead）对计时的干扰
- 掌握统计学方法处理多次测量结果的波动
- 能够将 Event timing 集成到自动化 benchmark 框架中

## 2. 性能问题动机

在 GPU 程序优化中，精确测量 kernel 执行时间是最基础的需求。常见痛点包括：

- 使用 `time.time()` 或 `std::chrono` 测量 GPU kernel 时间，结果包含了 launch latency 和同步开销
- 多流并发执行时无法区分各 kernel 的独立耗时
- 首次调用 kernel 因 JIT 编译或 context 初始化导致测量偏高
- 不同硬件上 timer resolution 不同，导致短 kernel 测量不准
- 缺乏系统化的 warm-up 和统计方法，导致性能数据不可复现

## 3. 核心术语表

| 术语 | 英文全称 | 定义 |
|------|----------|------|
| CUDA Event | CUDA Event | GPU 时间线上的标记点，用于记录时间戳 |
| 流 | Stream | GPU 命令的有序队列，同一流内命令顺序执行 |
| 同步 | Synchronization | 等待 GPU 操作完成的阻塞操作 |
| Launch Overhead | Kernel Launch Overhead | 从 CPU 发起 kernel 到 GPU 开始执行的延迟 |
| Elapsed Time | Elapsed Time | 两个 Event 之间的 GPU 时钟差值 |
| Warm-up | Warm-up | 正式测量前的预热运行，消除初始化开销 |
| Timer Resolution | Timer Resolution | 计时器能区分的最小时间间隔 |
| Default Stream | Default Stream (Stream 0) | 隐式同步的默认命令流 |
| Blocking Sync | Blocking Synchronization | CPU 线程阻塞等待 GPU 完成 |
| Spin Wait | Spin Wait | CPU 忙等待 GPU 事件完成 |

## 4. 指标定义

| 指标 | 公式 | 单位 | 含义 |
|------|------|------|------|
| Kernel Duration | `cudaEventElapsedTime(stop - start)` | ms | 单次 kernel GPU 执行时间 |
| Mean Duration | `Σ(duration_i) / N` | ms | N 次测量的平均执行时间 |
| Std Deviation | `sqrt(Σ(d_i - mean)² / (N-1))` | ms | 测量波动程度 |
| CV (变异系数) | `std / mean × 100%` | % | 相对波动，用于跨 kernel 比较 |
| Percentile P99 | 排序后第 99% 位置的值 | ms | 尾延迟指标 |
| Warm-up Overhead | `first_run - mean(subsequent)` | ms | 首次运行额外开销 |
| Timer Granularity | 硬件最小可分辨时间 | μs | 约 0.5μs (现代 GPU) |

## 5. 指标来源

| 指标 | 数据源 | 获取方式 |
|------|--------|----------|
| Kernel Duration | CUDA Runtime API | `cudaEventElapsedTime()` |
| GPU Clock | GPU 硬件计数器 | Event 内部使用 SM clock |
| Stream 关联 | CUDA Driver | Event 绑定到特定 stream |
| Launch Latency | Nsight Systems | timeline 中 API 调用到 kernel 开始的间隔 |
| Resolution | 设备属性 | `cudaDeviceGetAttribute` (clockRate) |

## 6. 正常现象

- 前 1-3 次运行耗时明显高于后续运行（JIT、context 初始化）
- 相同 kernel 多次运行有 ±2-5% 的波动（GPU 频率动态调整 DVFS）
- 极短 kernel（<10μs）测量精度受 timer resolution 限制
- Default stream 上的 Event timing 包含隐式同步开销
- 多次测量的分布呈轻微右偏（偶发调度延迟）

## 7. 异常现象

- 同一 kernel 测量结果波动超过 20%
- Event elapsed time 为 0 或负值
- 测量时间远大于 Nsight Compute 报告的 kernel duration
- 多流场景下 Event 时间出现重叠或不合理的大值
- Warm-up 后仍有周期性的耗时尖峰

## 8. 可能原因

| 异常 | 可能原因 |
|------|----------|
| 波动超 20% | GPU 频率未锁定；后台有其他 GPU 任务；TDP 限制触发降频 |
| Elapsed time = 0 | start/stop 记录在同一时钟周期；kernel 未实际执行 |
| 时间远大于预期 | Event 记录在错误的 stream；中间插入了隐式同步 |
| 多流时间不合理 | Event 跨流使用但未正确同步；流间依赖未处理 |
| 周期性尖峰 | GPU 温度触发 thermal throttling；ECC 错误校正 |

## 9. 验证实验

### 实验 1：Event Timing vs CPU Timing 对比

```python
import torch
import time

def compare_timing(size=4096, iterations=100):
    a = torch.randn(size, size, device='cuda')
    b = torch.randn(size, size, device='cuda')
    
    # Warm-up
    for _ in range(10):
        torch.mm(a, b)
    torch.cuda.synchronize()
    
    # CPU timing
    torch.cuda.synchronize()
    cpu_start = time.perf_counter()
    for _ in range(iterations):
        torch.mm(a, b)
    torch.cuda.synchronize()
    cpu_elapsed = (time.perf_counter() - cpu_start) / iterations * 1000
    
    # CUDA Event timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    times = []
    for _ in range(iterations):
        start_event.record()
        torch.mm(a, b)
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))
    
    print(f"CPU timing: {cpu_elapsed:.3f} ms")
    print(f"Event timing mean: {sum(times)/len(times):.3f} ms")
    print(f"Event timing std: {(sum((t-sum(times)/len(times))**2 for t in times)/(len(times)-1))**0.5:.3f} ms")
```

### 实验 2：Warm-up 效果验证

```python
def warmup_experiment(size=2048):
    a = torch.randn(size, size, device='cuda')
    b = torch.randn(size, size, device='cuda')
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    times = []
    for i in range(50):
        start.record()
        torch.mm(a, b)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    print("First 5 runs:", [f"{t:.3f}" for t in times[:5]])
    print("Last 5 runs:", [f"{t:.3f}" for t in times[-5:]])
    print(f"Warm-up overhead: {times[0] - sum(times[10:])/len(times[10:]):.3f} ms")
```

### 实验 3：多流独立计时

```python
def multi_stream_timing(size=2048):
    a = torch.randn(size, size, device='cuda')
    b = torch.randn(size, size, device='cuda')
    
    streams = [torch.cuda.Stream() for _ in range(4)]
    events = [(torch.cuda.Event(enable_timing=True), 
               torch.cuda.Event(enable_timing=True)) for _ in range(4)]
    
    # Warm-up
    for _ in range(5):
        torch.mm(a, b)
    torch.cuda.synchronize()
    
    for i, stream in enumerate(streams):
        with torch.cuda.stream(stream):
            events[i][0].record()
            torch.mm(a, b)
            events[i][1].record()
    
    torch.cuda.synchronize()
    for i in range(4):
        print(f"Stream {i}: {events[i][0].elapsed_time(events[i][1]):.3f} ms")
```

## 10. 优化方法

| 方法 | 适用场景 | 预期效果 |
|------|----------|----------|
| 锁定 GPU 频率 | 所有 benchmark | 减少波动至 <2% |
| 充分 warm-up (10-50 次) | 首次运行偏高 | 消除 JIT/context 开销 |
| 多次测量取中位数 | 存在离群值 | 比均值更鲁棒 |
| 独立 Event 对 per kernel | 多 kernel 流水线 | 精确定位瓶颈 |
| 使用 non-default stream | 避免隐式同步 | 更准确的独立计时 |
| 批量测量减少同步 | 短 kernel 测量 | 摊薄同步开销 |

## 11. 副作用

- `cudaEventSynchronize()` 会阻塞 CPU 线程，影响 CPU-GPU 重叠
- 频繁创建/销毁 Event 对象有内存分配开销
- 每次 `record()` 在 stream 中插入一条命令，极端情况影响 kernel 调度
- 锁定 GPU 频率会降低实际性能（固定在非 boost 频率）
- 过多的同步点破坏了流水线并行性

## 12. Profiling 命令模板

```bash
# 锁定 GPU 频率（需要 root 权限）
sudo nvidia-smi -lgc 1410,1410  # 锁定到 1410 MHz
sudo nvidia-smi -rgc             # 恢复自动调频

# 查看当前 GPU 频率
nvidia-smi -q -d CLOCK

# 使用 nvprof 验证 Event timing 准确性（旧版）
nvprof --print-gpu-trace python my_benchmark.py

# 使用 Nsight Systems 对比
nsys profile --stats=true -o event_timing_report python my_benchmark.py

# 查看 GPU 温度（排除 thermal throttling）
nvidia-smi -q -d TEMPERATURE

# 检查 ECC 错误
nvidia-smi -q -d ECC

# Python 中锁频后运行 benchmark
nvidia-smi -lgc 1410,1410 && python benchmark.py && nvidia-smi -rgc
```

## 13. Benchmark 设计

### 设计原则

1. **隔离性**：每个 benchmark 独立运行，不受前序操作影响
2. **可复现性**：固定随机种子、GPU 频率、输入数据
3. **统计显著性**：至少 100 次迭代，报告均值/中位数/P95/P99
4. **Warm-up 充分**：至少 10 次预热，丢弃前 N 次结果

### 标准 Benchmark 模板

```python
import torch
import numpy as np
from dataclasses import dataclass

@dataclass
class BenchmarkResult:
    mean_ms: float
    median_ms: float
    std_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    num_iterations: int
    warmup_iterations: int

def benchmark_kernel(fn, warmup=20, iterations=200, **kwargs):
    """标准 kernel benchmark 函数"""
    # Warm-up
    for _ in range(warmup):
        fn(**kwargs)
    torch.cuda.synchronize()
    
    # Measurement
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    
    for _ in range(iterations):
        start.record()
        fn(**kwargs)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    times = np.array(times)
    return BenchmarkResult(
        mean_ms=times.mean(),
        median_ms=np.median(times),
        std_ms=times.std(ddof=1),
        p95_ms=np.percentile(times, 95),
        p99_ms=np.percentile(times, 99),
        min_ms=times.min(),
        max_ms=times.max(),
        num_iterations=iterations,
        warmup_iterations=warmup
    )
```

## 14. 实验记录表

| 字段 | 示例值 | 说明 |
|------|--------|------|
| 日期 | 2026-05-31 | 实验日期 |
| GPU 型号 | A100-80GB | 硬件型号 |
| Driver 版本 | 535.129.03 | nvidia-smi 输出 |
| CUDA 版本 | 12.2 | nvcc --version |
| GPU 频率 | 1410 MHz (locked) | 是否锁频 |
| GPU 温度 | 42°C | 实验时温度 |
| Kernel 名称 | matmul_4096x4096 | 被测 kernel |
| Warm-up 次数 | 20 | 预热迭代数 |
| 测量次数 | 200 | 正式迭代数 |
| Mean (ms) | 2.341 | 平均值 |
| Median (ms) | 2.338 | 中位数 |
| Std (ms) | 0.045 | 标准差 |
| CV (%) | 1.92 | 变异系数 |
| P99 (ms) | 2.456 | 99 分位 |
| 备注 | baseline measurement | 额外说明 |

## 15. 故障树

```
Event Timing 结果异常
├── 测量值波动大 (CV > 5%)
│   ├── GPU 频率未锁定 → nvidia-smi -lgc 锁频
│   ├── Thermal throttling → 检查温度，增加散热
│   ├── 后台 GPU 进程 → nvidia-smi 检查，kill 干扰进程
│   └── 内存带宽竞争 → 减少并发访存操作
├── 测量值为 0
│   ├── Kernel 未实际执行 → 检查 launch 参数
│   ├── Event 未正确 record → 确认 record() 在 kernel 前后
│   └── Timer resolution 不足 → kernel 太短，增加工作量
├── 测量值偏大
│   ├── 包含了隐式同步 → 使用 non-default stream
│   ├── Event 在错误 stream → 确认 stream 参数
│   ├── 中间有内存分配 → 预分配所有 tensor
│   └── Page fault (统一内存) → 预取数据到 GPU
└── 首次运行异常高
    ├── CUDA context 初始化 → 增加 warm-up
    ├── JIT 编译 (cuBLAS) → 首次调用触发 plan 选择
    └── 内存首次分配 → 预分配 + warm-up
```

## 16. 复盘模板

```markdown
## Event Timing 实验复盘

### 实验目标
- 测量目标：[kernel 名称/操作]
- 预期结果：[预期耗时范围]

### 实验配置
- GPU/频率/温度：
- Warm-up/迭代次数：
- 输入规模：

### 实际结果
- Mean/Median/Std/CV：
- 是否符合预期：[是/否]

### 问题与发现
- 遇到的异常：
- 根因分析：
- 解决方法：

### 经验总结
- 本次学到的关键点：
- 下次改进方向：
```

## 17. 常见错误

| # | 错误 | 后果 | 正确做法 |
|---|------|------|----------|
| 1 | 未调用 `synchronize()` 就读取 elapsed time | 返回 0 或错误值 | record 后必须 synchronize |
| 2 | 在 default stream 测量多 kernel | 包含隐式同步开销 | 使用独立 stream |
| 3 | 未 warm-up | 首次运行偏高污染统计 | 至少 10 次 warm-up |
| 4 | Event 对象重复使用未重新 record | 读到旧的时间戳 | 每次测量重新 record |
| 5 | 用 CPU time 代替 Event time | 包含 launch overhead | 使用 CUDA Event |
| 6 | 测量中包含 `torch.cuda.synchronize()` | 同步本身有开销 | 只在最后同步一次 |
| 7 | 未锁定 GPU 频率 | 波动大，不可复现 | `nvidia-smi -lgc` |
| 8 | 样本量太少 (<10) | 统计不显著 | 至少 100 次迭代 |
| 9 | 忽略离群值 | 均值被拉高 | 使用中位数或 trimmed mean |
| 10 | 跨流使用 Event 未处理依赖 | 时间计算错误 | 使用 `cudaStreamWaitEvent` |

## 18. 习题 20 道

1. CUDA Event 的 `enable_timing=True` 参数的作用是什么？不设置会怎样？
2. 为什么 `cudaEventElapsedTime` 比 CPU `clock_gettime` 更适合测量 GPU kernel 时间？
3. 解释 CUDA Event 的 timer resolution 约为多少？这对测量短 kernel 有什么影响？
4. 编写代码：使用 CUDA Event 测量一个 4096×4096 矩阵乘法的执行时间。
5. 什么是 warm-up？为什么第一次 kernel 调用通常比后续调用慢？
6. 在多流场景下，如何正确测量每个流中 kernel 的独立执行时间？
7. `cudaEventRecord` 在 stream 中的语义是什么？它是否会阻塞 GPU 执行？
8. 如何区分 kernel execution time 和 kernel launch latency？
9. 设计实验：验证 GPU 频率锁定对测量稳定性的影响。
10. 为什么建议使用中位数而非均值作为 kernel 性能的代表值？
11. `cudaEventSynchronize` 和 `cudaDeviceSynchronize` 的区别是什么？
12. 如何在 PyTorch 中使用 `torch.cuda.Event` 进行计时？写出完整代码。
13. 解释为什么在 default stream 上测量可能包含隐式同步开销。
14. 设计一个实验来测量 `cudaEventRecord` 本身的开销。
15. 如何处理测量数据中的离群值？列举至少两种方法。
16. 在 A100 上，一个 kernel 执行时间为 5μs，使用 Event timing 测量的相对误差约为多少？
17. 编写一个完整的 benchmark 函数，输出 mean、median、std、P95、P99。
18. 解释 `cudaEventCreateWithFlags(cudaEventBlockingSync)` 的作用和适用场景。
19. 如何验证你的 Event timing 结果与 Nsight Compute 报告的 kernel duration 一致？
20. 设计实验：比较同一 kernel 在不同 GPU 频率下的执行时间变化。

## 19. 标准答案

1. `enable_timing=True` 使 Event 记录 GPU 时间戳。不设置则 `elapsed_time()` 会抛出错误，因为 Event 未携带时间信息。非计时 Event 开销更低，适合纯同步用途。

2. CPU 计时包含 kernel launch overhead、driver 开销和同步等待时间。CUDA Event 直接在 GPU 时间线上打点，测量的是纯 GPU 执行时间，不受 CPU-GPU 交互延迟影响。

3. Timer resolution 约 0.5μs（取决于 GPU 架构）。对于执行时间 <10μs 的短 kernel，相对误差可达 5-10%，建议批量执行后取平均。

4. 见实验 1 代码。关键点：创建 Event 对、record 包围 kernel、synchronize 后调用 elapsed_time。

5. Warm-up 是正式测量前的预热运行。首次调用慢的原因：CUDA context 初始化、cuBLAS handle 创建、JIT 编译最优 kernel、GPU 从低功耗状态唤醒、内存首次分配触发 page table 建立。

6. 每个流创建独立的 Event 对，在各自流中 record。所有流完成后统一 synchronize，再分别计算各流的 elapsed time。见实验 3 代码。

7. `cudaEventRecord` 在指定 stream 的命令队列中插入一个"记录时间戳"的命令。它不阻塞 GPU，只是标记一个时间点。当 GPU 执行到该命令时才实际记录时间戳。

8. Kernel execution time：GPU 实际执行 kernel 的时间（Event timing 测量）。Launch latency：CPU 调用 kernel launch API 到 GPU 开始执行的延迟（需 Nsight Systems timeline 观察）。

9. 实验设计：分别在 `nvidia-smi -lgc 1410,1410`（锁频）和自动调频下运行 200 次相同 kernel，比较两组的 CV（变异系数）。预期锁频后 CV < 2%，自动调频 CV 可能 5-15%。

10. GPU 测量数据通常右偏（偶发的调度延迟、中断等导致少数高值）。中位数不受离群值影响，更能代表"典型"性能。均值会被少数高值拉高。

11. `cudaEventSynchronize`：等待特定 Event 完成，粒度更细。`cudaDeviceSynchronize`：等待设备上所有流的所有操作完成，开销更大。Benchmark 中优先使用前者。

12. 见实验 1 中 PyTorch 版本代码。核心 API：`torch.cuda.Event(enable_timing=True)`、`.record()`、`.elapsed_time(other_event)`。

13. Default stream (stream 0) 与其他流有隐式同步语义：在 default stream 上的操作会等待所有其他流完成。因此 Event 可能记录了等待其他流的时间。

14. 设计：record 两个紧邻的 Event（中间无 kernel），测量 elapsed time。重复 1000 次取均值，该值即为 record 开销的上界。典型值 <1μs。

15. 方法一：使用 trimmed mean（去掉最高/最低 5% 后取均值）。方法二：使用 IQR（四分位距）方法，排除 Q1-1.5×IQR 以下和 Q3+1.5×IQR 以上的值。方法三：直接使用中位数。

16. Timer resolution ≈ 0.5μs，kernel 时间 5μs，相对误差 ≈ 0.5/5 = 10%。建议将多次调用合并测量后除以次数来降低相对误差。

17. 见 Benchmark 设计章节的 `benchmark_kernel` 函数。

18. `cudaEventBlockingSync` 使 `cudaEventSynchronize` 时 CPU 线程进入阻塞等待（让出 CPU），而非默认的 spin wait（忙等）。适用于不需要最低延迟、但希望降低 CPU 占用的场景。

19. 方法：对同一 kernel 分别用 Event timing 和 `ncu --metrics gpu__time_duration.avg` 测量，比较两者差异。正常情况下差异 <5%。若差异大，检查是否有隐式同步或 Event 位置错误。

20. 实验设计：使用 `nvidia-smi -lgc` 分别锁定到 3-5 个不同频率（如 900/1200/1410/1800 MHz），每个频率下运行 100 次相同 kernel，记录 mean 和 std。绘制频率-执行时间曲线，验证是否线性关系。

## 20. 调优 Checklist

- [ ] 确认 GPU 频率已锁定（`nvidia-smi -lgc`）
- [ ] 确认无其他 GPU 进程运行（`nvidia-smi` 检查）
- [ ] 确认 GPU 温度在正常范围（<80°C）
- [ ] Event 创建时设置 `enable_timing=True`
- [ ] Warm-up 至少 10 次迭代
- [ ] 正式测量至少 100 次迭代
- [ ] 使用 non-default stream 避免隐式同步
- [ ] 每次迭代独立 record start/end Event
- [ ] 报告 mean、median、std、P95、P99
- [ ] CV < 5% 才认为测量稳定
- [ ] 与 Nsight Compute 结果交叉验证
- [ ] 记录完整实验环境信息
- [ ] 实验结束后恢复 GPU 频率（`nvidia-smi -rgc`）
- [ ] 数据保存到版本控制的实验记录中
