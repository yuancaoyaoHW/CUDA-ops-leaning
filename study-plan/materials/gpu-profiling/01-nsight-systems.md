# Nsight Systems

## 1. 学习目标

- 理解 Nsight Systems 的定位：系统级 timeline 分析工具
- 掌握 Nsight Systems 的基本使用流程（profile → 分析 → 优化）
- 能够识别 CPU-GPU 交互瓶颈、kernel overlap、launch gap
- 理解 NVTX 标注的使用方法
- 掌握常见性能问题的 timeline 特征

## 2. 性能问题动机

### 2.1 为什么需要 Nsight Systems？

Nsight Compute 分析单个 kernel 的效率，但无法回答：
- Kernel 之间有多少空闲时间？
- CPU 和 GPU 是否有效重叠？
- 数据传输是否与计算重叠？
- 哪个阶段是整体瓶颈？

Nsight Systems 提供**系统级 timeline 视图**，回答"时间花在哪里"的问题。

### 2.2 典型问题场景

1. **GPU 空闲**：kernel 之间有大量 gap → launch overhead 或 CPU 瓶颈
2. **串行执行**：多 stream 的 kernel 没有并发 → stream 配置错误
3. **同步阻塞**：频繁的 cudaDeviceSynchronize → 不必要的同步
4. **数据传输瓶颈**：H2D/D2H 占比高 → 需要 overlap 或减少传输

## 3. 核心术语表

| 术语 | 英文全称 | 定义 |
|------|----------|------|
| Timeline | Timeline View | 时间轴视图，显示所有 GPU/CPU 活动 |
| NVTX | NVIDIA Tools Extension | 用户自定义的代码标注 API |
| API Trace | CUDA API Trace | 记录所有 CUDA API 调用 |
| Kernel Row | Kernel Row | Timeline 中显示 GPU kernel 执行的行 |
| Stream Row | Stream Row | 按 CUDA stream 分组的活动 |
| GPU Idle | GPU Idle Time | GPU 无 kernel 执行的时间 |
| Launch Gap | Launch Gap | 两个 kernel 之间的空闲时间 |
| Occupancy | Theoretical Occupancy | Nsight Systems 中显示的 kernel occupancy |

## 4. 指标定义

| 指标 | 定义 | 计算方式 |
|------|------|---------|
| GPU Utilization | GPU 有 kernel 执行的时间比例 | kernel_time / total_time |
| Kernel Duration | 单个 kernel 的执行时间 | end - start |
| Launch Overhead | kernel launch 到开始执行的延迟 | kernel_start - api_call_time |
| Memory Transfer Time | H2D/D2H 传输时间 | transfer_end - transfer_start |
| API Call Duration | CUDA API 调用的 CPU 端耗时 | api_return - api_call |
| Stream Concurrency | 多 stream 并行执行的时间比例 | overlap_time / total_time |

## 5. 指标来源

```bash
# 基本 profiling
nsys profile --trace=cuda,nvtx,osrt --output=report ./my_program

# 详细 profiling（包含 CUDA API）
nsys profile --trace=cuda,nvtx,cudnn,cublas --cuda-memory-usage=true \
    --output=report ./my_program

# Python 程序
nsys profile --trace=cuda,nvtx,python --python-sampling=true \
    python train.py

# 只 profile 特定时间段
nsys profile --delay=5 --duration=10 ./my_program
```

## 6. 正常现象

- GPU kernel 紧密排列，几乎无 gap
- 多 stream 有明显的并行执行
- H2D/D2H 与 compute 重叠
- NVTX 标注的各阶段时间合理

## 7. 异常现象

| 异常 | Timeline 特征 | 可能原因 |
|------|--------------|---------|
| 大量 GPU idle | Kernel 行有明显空白 | CPU 瓶颈、同步过多 |
| Kernel 串行 | 多 stream 但无重叠 | Default stream 或依赖错误 |
| 长 launch gap | API 调用到 kernel 开始有延迟 | CPU 计算阻塞、Python GIL |
| 频繁小 kernel | 大量短 kernel 密集排列 | 需要 kernel fusion 或 CUDA Graph |
| H2D/D2H 阻塞 | 传输期间 GPU idle | 需要 async transfer + overlap |

## 8-12. Profiling 命令模板

```bash
# 1. 基本 profile
nsys profile -o baseline ./inference_server

# 2. 带 NVTX 标注
nsys profile --trace=cuda,nvtx -o annotated python model.py

# 3. 只看 CUDA kernel
nsys profile --trace=cuda --cuda-memory-usage=true -o kernels ./app

# 4. 多 GPU
nsys profile --trace=cuda,nvtx,nccl -o multi_gpu \
    torchrun --nproc_per_node=4 train.py

# 5. 生成统计报告
nsys stats report.nsys-rep

# 6. 导出为 JSON（程序化分析）
nsys export --type=json report.nsys-rep

# 7. 查看 kernel 统计
nsys stats --report cuda_gpu_kern_sum report.nsys-rep

# 8. 查看 CUDA API 统计
nsys stats --report cuda_api_sum report.nsys-rep
```

### NVTX 标注示例

```python
import torch
import nvtx

# 方式 1: decorator
@nvtx.annotate("forward_pass", color="blue")
def forward(model, input):
    return model(input)

# 方式 2: context manager
with nvtx.annotate("attention", color="green"):
    attn_output = attention(q, k, v)

# 方式 3: range push/pop
nvtx.range_push("data_loading")
batch = next(dataloader)
nvtx.range_pop()
```

## 13. Benchmark 设计

```python
# 使用 NVTX 标注关键阶段
import nvtx
import torch

def benchmark_inference(model, inputs, warmup=10, repeat=100):
    # Warmup
    for _ in range(warmup):
        model(inputs)
    torch.cuda.synchronize()
    
    # Profile region
    nvtx.range_push("benchmark_region")
    for i in range(repeat):
        nvtx.range_push(f"iteration_{i}")
        
        nvtx.range_push("forward")
        output = model(inputs)
        nvtx.range_pop()
        
        nvtx.range_push("postprocess")
        result = output.argmax(dim=-1)
        nvtx.range_pop()
        
        nvtx.range_pop()  # iteration
    
    torch.cuda.synchronize()
    nvtx.range_pop()  # benchmark_region
```

## 14. 实验记录表

| 实验 | 配置 | GPU Util | Kernel Gap | 瓶颈 | 优化方向 |
|------|------|----------|-----------|------|---------|
| Baseline | batch=1, no graph | 45% | 5μs avg | Launch overhead | CUDA Graph |
| + CUDA Graph | batch=1, graph | 78% | <1μs | Memory BW | Batch up |
| + Batch=32 | batch=32, graph | 92% | <1μs | Compute | 已饱和 |

## 15. 故障树

```
GPU Utilization 低
├── Kernel 之间有 gap
│   ├── Launch overhead 大 → CUDA Graph
│   ├── CPU 计算阻塞 → 异步化 / C++ 扩展
│   └── 不必要的同步 → 移除 synchronize
├── Kernel 本身短
│   ├── 小 kernel 多 → Kernel fusion
│   └── 问题规模小 → 增大 batch
└── 数据传输阻塞
    ├── 同步传输 → 改为 async
    └── 传输量大 → 减少传输 / pinned memory
```

## 16. 复盘模板

```markdown
## Nsight Systems 分析报告

### 环境
- GPU: 
- Driver: 
- CUDA: 
- 程序: 

### Timeline 观察
- GPU utilization: 
- 主要 kernel: 
- 最大 gap: 
- 并发情况: 

### 瓶颈识别
- 类型: [CPU-bound / Launch-bound / Memory-bound / Sync-bound]
- 证据: 
- 影响: 

### 优化方案
- 方案: 
- 预期收益: 
- 实施难度: 

### 验证结果
- 优化前: 
- 优化后: 
- 实际收益: 
```

## 17. 常见错误

1. **Profile 时间太长**：生成巨大的 report 文件 → 用 `--duration` 限制
2. **忘记 warmup**：前几次迭代包含 JIT 编译 → 用 `--delay` 跳过
3. **Python overhead 干扰**：Python 解释器开销被计入 → 用 `--python-sampling`
4. **多 GPU 不同步**：各 GPU 的 timeline 不对齐 → 用 `nsys` 统一 profile
5. **NVTX 标注过多**：标注本身有开销 → 只标注关键区域

## 18. 习题 20 道

1. Nsight Systems 和 Nsight Compute 的区别是什么？各自适用什么场景？
2. 如何用 Nsight Systems 识别 CPU-GPU 同步瓶颈？
3. NVTX 标注的三种使用方式是什么？
4. 如何测量 kernel launch overhead？在 timeline 中如何识别？
5. 多 stream 并发在 timeline 中是什么样子？如何验证是否真正并发？
6. `nsys stats` 命令能生成哪些统计报告？
7. 如何 profile PyTorch 训练脚本？需要注意什么？
8. GPU idle time 高的三种常见原因是什么？
9. 如何用 Nsight Systems 分析 NCCL 通信？
10. CUDA Graph 在 timeline 中与普通 kernel launch 有什么区别？
11. 如何导出 Nsight Systems 数据进行程序化分析？
12. `--delay` 和 `--duration` 参数的作用是什么？
13. 如何识别 Python GIL 导致的 GPU 空闲？
14. Nsight Systems 中 memory transfer 行显示什么信息？
15. 如何用 Nsight Systems 验证 compute-transfer overlap？
16. Profile 大型分布式训练时有什么注意事项？
17. 如何从 timeline 中计算 GPU utilization？
18. Nsight Systems 的 Expert Systems 功能是什么？
19. 如何比较两次 profile 的结果？
20. Nsight Systems 对程序性能的影响（overhead）有多大？

## 19. 标准答案

1. Nsight Systems：系统级 timeline，分析 CPU-GPU 交互、kernel 调度、多 stream 并发。Nsight Compute：kernel 级，分析单个 kernel 的 compute/memory 效率。前者回答"时间花在哪"，后者回答"kernel 为什么慢"。

2. 在 timeline 中查找：(a) cudaDeviceSynchronize 调用导致 CPU 等待；(b) GPU idle 期间 CPU 在做计算；(c) API trace 中同步调用的频率和耗时。

3. (a) `@nvtx.annotate()` decorator；(b) `with nvtx.annotate():` context manager；(c) `nvtx.range_push()/range_pop()` 手动标注。

4. 在 timeline 中测量 CUDA API 调用时间点到对应 kernel 开始执行的时间差。典型值 5-10μs。如果 > 20μs 说明有 CPU 端阻塞。

5. 多 stream 并发时，不同 stream 行的 kernel 在时间上重叠。验证：(a) 查看 timeline 是否有重叠；(b) 检查总时间是否小于各 stream 时间之和。

(后续答案略)

## 20. 调优 checklist

- [ ] 确认 profile 包含 warmup 后的稳定阶段
- [ ] 检查 GPU utilization（目标 > 80%）
- [ ] 识别最大的 kernel gap 并分析原因
- [ ] 检查是否有不必要的 cudaDeviceSynchronize
- [ ] 验证多 stream 是否真正并发
- [ ] 检查 H2D/D2H 是否与 compute 重叠
- [ ] 识别最耗时的 kernel（是否可以优化或 fuse）
- [ ] 检查 Python/CPU 端是否有阻塞
- [ ] 验证 CUDA Graph 是否正确应用
- [ ] 对比优化前后的 timeline
