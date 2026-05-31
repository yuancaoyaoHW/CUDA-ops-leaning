# 03 - PyTorch Profiler 性能分析

## 1. 学习目标

- 掌握 PyTorch Profiler（torch.profiler）的完整使用流程
- 理解 Profiler 输出中各项指标的含义与关联
- 能够通过 Profiler 定位 GPU 训练中的性能瓶颈
- 学会将 Profiler 数据导出至 TensorBoard 和 Chrome Trace 进行可视化
- 掌握 Profiler 与 Nsight Systems 的协同使用方法

## 2. 性能问题动机

在深度学习训练中，常见以下场景需要 PyTorch Profiler：

- GPU 利用率低但不知道时间花在哪里
- 训练吞吐量（throughput）远低于理论峰值
- 某些 operator 耗时异常但缺乏定量数据
- 数据加载与计算之间存在隐性等待
- 内存分配/释放导致的性能抖动
- 需要对比不同实现方案的实际性能差异

PyTorch Profiler 提供了从 Python 层到 CUDA kernel 层的端到端性能视图，是定位训练瓶颈的第一工具。

## 3. 核心术语表

| 术语 | 英文全称 | 定义 |
|------|----------|------|
| Profiler | Profiler | 性能分析器，收集程序执行的时间与资源使用数据 |
| Trace | Execution Trace | 程序执行的时间线记录 |
| Operator | Operator | PyTorch 中的基本计算单元（如 aten::mm） |
| CUDA Activity | CUDA Activity | GPU 上实际执行的 CUDA 操作 |
| CPU Time | CPU Time | 操作在 CPU 侧的耗时（含等待 GPU） |
| CUDA Time | CUDA Time / Device Time | 操作在 GPU 侧的实际执行耗时 |
| Self Time | Self Time | 不含子操作的自身耗时 |
| Memory Event | Memory Allocation Event | 内存分配/释放事件 |
| Schedule Wait | Schedule Wait | kernel 从提交到实际执行的等待时间 |
| Tensor Core | Tensor Core | GPU 中专用于矩阵运算的硬件单元 |

## 4. 指标定义

| 指标 | 公式/定义 | 单位 | 意义 |
|------|-----------|------|------|
| CPU Total Time | 操作从调用到返回的总时间 | μs/ms | 反映 Python 侧开销 |
| CUDA Total Time | GPU kernel 实际执行时间之和 | μs/ms | 反映 GPU 计算负载 |
| Self CPU Time | 不含子调用的 CPU 时间 | μs/ms | 定位 CPU 热点 |
| Self CUDA Time | 不含子调用的 CUDA 时间 | μs/ms | 定位 GPU 热点 |
| GPU Utilization | GPU 活跃时间 / 总时间 × 100% | % | GPU 是否被充分利用 |
| Memory Usage | 当前已分配的 GPU 显存 | MB/GB | 显存压力 |
| FLOPS | 浮点运算次数 / 时间 | TFLOPS | 计算效率 |
| Bandwidth | 数据传输量 / 时间 | GB/s | 访存效率 |

## 5. 指标来源

| 指标 | 采集方式 | 工具/API |
|------|----------|----------|
| CPU/CUDA Time | 事件插桩 | torch.profiler.profile() |
| GPU Utilization | CUPTI 回调 | ProfilerActivity.CUDA |
| Memory Usage | 分配器 hook | profile_memory=True |
| FLOPS | 算子计算量估算 | with_flops=True |
| Stack Trace | Python 栈采样 | with_stack=True |
| Kernel 参数 | CUPTI Activity | record_shapes=True |
| 模块归属 | Module hook | with_modules=True |

## 6. 正常现象

- GPU Utilization 在 85%+ 且 kernel 之间间隙 < 5μs
- CPU Time 略大于 CUDA Time（正常的 launch overhead）
- 内存使用在 warmup 后趋于稳定
- 前几个 iteration 耗时较长（JIT 编译、cudnn benchmark）
- backward 耗时约为 forward 的 2-3 倍
- 小 kernel 的 launch overhead 占比较高但总时间占比低
- DataLoader prefetch 与计算重叠良好

## 7. 异常现象

- GPU Utilization < 50% 且存在大量 CPU 空闲段
- CUDA Time 远小于 CPU Time（CPU 瓶颈）
- 内存使用持续增长（内存泄漏）
- 大量小 kernel 连续执行，间隙时间占比 > 30%
- cudaMemcpy（H2D/D2H）频繁出现在关键路径
- cudaStreamSynchronize 占据大量 CPU 时间
- 某个 operator 的 Self CUDA Time 异常高
- backward 中出现非预期的 CPU 操作

## 8. 可能原因

| 异常现象 | 可能原因 |
|----------|----------|
| GPU 利用率低 | 数据加载慢、CPU 预处理瓶颈、频繁同步 |
| CPU Time >> CUDA Time | Python GIL 竞争、复杂的控制流、动态图开销 |
| 内存持续增长 | 梯度未释放、中间张量被引用、日志保存张量 |
| 大量小 kernel | 逐元素操作未融合、动态 shape 导致无法优化 |
| 频繁 H2D 拷贝 | 数据未 pin_memory、CPU tensor 参与 GPU 计算 |
| 同步开销大 | .item()/.numpy() 调用、print 中间结果、断言 |
| 单 operator 耗时高 | 算法选择不优、未使用 Tensor Core、shape 不对齐 |

## 9. 验证实验

### 实验 1：基础 Profiling 采集

```python
import torch
from torch.profiler import profile, ProfilerActivity, schedule

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    with_flops=True,
    with_modules=True,
) as prof:
    for step, (data, target) in enumerate(train_loader):
        if step >= 5:
            break
        data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        prof.step()
```

### 实验 2：对比 torch.compile 前后性能

```python
model_eager = MyModel().cuda()
model_compiled = torch.compile(model_eager)

# 分别 profile 两个模型，对比 kernel 数量和总耗时
for model, tag in [(model_eager, "eager"), (model_compiled, "compiled")]:
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            output = model(input_tensor)
    print(f"[{tag}] {prof.key_averages().table(sort_by='cuda_time_total', row_limit=10)}")
```

### 实验 3：内存 Profiling

```python
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True,
    record_shapes=True,
) as prof:
    output = model(input_tensor)
    loss = criterion(output, target)
    loss.backward()

# 查看内存分配热点
print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
```
