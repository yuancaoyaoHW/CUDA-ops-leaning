# CUDA Stream 与 CUDA Graph

## 1. 学习目标

- 理解 CUDA Stream（流）的概念与并发执行模型
- 掌握多 stream 并发的使用场景与同步机制
- 理解 CUDA Graph（图）的动机、构建方式与性能优势
- 能够将 stream-based 代码迁移到 graph-based 执行
- 掌握 graph 在 LLM 推理中的应用（decode phase）

## 2. 前置知识

- CUDA 执行模型
- Kernel launch 开销概念
- LLM 推理中 prefill/decode 的区别

## 3. 核心术语表

| 术语 | 英文全称 | 定义 |
|------|----------|------|
| Stream | CUDA Stream | 一个有序的 GPU 操作队列，同一 stream 内操作顺序执行 |
| Default Stream | Default Stream | 未指定 stream 时使用的默认流（与其他 stream 有隐式同步） |
| Event | CUDA Event | stream 中的时间标记点，用于同步和计时 |
| Graph | CUDA Graph | 预录制的 GPU 操作 DAG，一次 launch 执行整个图 |
| Graph Node | Graph Node | graph 中的一个操作（kernel、memcpy、event 等） |
| Graph Instance | Executable Graph | 编译后可执行的 graph 实例 |
| Stream Capture | Stream Capture | 将 stream 上的操作录制为 graph 的机制 |
| Kernel Launch Overhead | Kernel Launch Overhead | 每次 kernel launch 的 CPU 端开销（~5-10μs） |
| Concurrency | Concurrency | 多个 stream 上的操作可以并行执行 |

## 4. 动机

### 4.1 CUDA Stream 的动机

默认情况下，所有 GPU 操作在 default stream 中顺序执行。但很多操作可以并行：
- 计算与数据传输重叠（compute + H2D/D2H）
- 独立 kernel 并行执行
- 多 GPU 间通信与计算重叠

### 4.2 CUDA Graph 的动机

LLM decode phase 的特点：
- 每个 token 生成需要执行相同的 kernel 序列
- 每个 kernel 计算量小（batch=1 时）
- Kernel launch overhead 占比高（可能 > 50% 时间）

CUDA Graph 解决方案：
- 预录制整个 kernel 序列为一个 graph
- 一次 launch 执行所有 kernel → 消除逐个 launch 的 CPU 开销
- 典型加速：decode latency 降低 20-40%

### 4.3 性能对比

```
Without Graph (decode, batch=1):
  kernel_1 launch (5μs) + exec (2μs) + kernel_2 launch (5μs) + exec (3μs) + ...
  Total: 50 kernels × (5μs launch + 3μs exec) = 400μs

With Graph:
  graph launch (10μs) + all kernels exec (150μs)
  Total: 160μs → 2.5x speedup
```

## 5. 数学定义

### 5.1 Stream 并发条件

两个操作可以并发执行当且仅当：
1. 它们在不同的 stream 中
2. 它们之间没有显式或隐式的依赖关系
3. GPU 有足够的资源同时执行

### 5.2 Graph 执行时间

```
T_graph = T_launch_once + T_execute_all_nodes
T_stream = Σ(T_launch_i + T_execute_i)  // for sequential nodes

Speedup = T_stream / T_graph
        ≈ (N × T_launch + T_compute) / (T_launch_once + T_compute)
        ≈ N × T_launch / T_launch_once  // when launch-dominated
```

## 6. 推导逻辑

### 6.1 Stream 执行模型

```
Stream A: [kernel_1] → [kernel_2] → [kernel_3]
Stream B: [kernel_4] → [kernel_5]
                ↑
                Event dependency from Stream A

Timeline:
Stream A: |--k1--|--k2--|--k3--|
Stream B:       |--k4--|      |--k5--|
                              ↑ waits for event after k2
```

### 6.2 Graph 构建方式

方式一：显式 API
```
cudaGraphCreate → cudaGraphAddKernelNode → ... → cudaGraphInstantiate → cudaGraphLaunch
```

方式二：Stream Capture（推荐）
```
cudaStreamBeginCapture → 正常执行 kernel → cudaStreamEndCapture → cudaGraphInstantiate
```

### 6.3 Graph 在 vLLM 中的应用

vLLM 对 decode phase 使用 CUDA Graph：
1. 预先为不同 batch size 录制 graph（batch=1,2,4,8,...）
2. Decode 时选择匹配的 graph 执行
3. 通过 `cudaGraphExecUpdateNode` 更新输入指针（避免重建 graph）
4. 限制：graph 内不能有动态 shape 或条件分支

## 7. 算子流程

### 7.1 Multi-Stream 并发

```cuda
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// 并发执行两个独立 kernel
kernel_A<<<grid, block, 0, stream1>>>(data_A);
kernel_B<<<grid, block, 0, stream2>>>(data_B);

// 同步
cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);
```

### 7.2 Compute-Transfer Overlap

```cuda
cudaStream_t compute_stream, transfer_stream;
cudaStreamCreate(&compute_stream);
cudaStreamCreate(&transfer_stream);

for (int i = 0; i < num_chunks; i++) {
    // 传输第 i+1 块数据（与第 i 块计算重叠）
    if (i + 1 < num_chunks) {
        cudaMemcpyAsync(d_input[next], h_input[next], size, 
                       cudaMemcpyHostToDevice, transfer_stream);
    }
    // 计算第 i 块
    kernel<<<grid, block, 0, compute_stream>>>(d_input[curr], d_output[curr]);
    
    // 传输第 i 块结果回 host
    cudaMemcpyAsync(h_output[curr], d_output[curr], size,
                   cudaMemcpyDeviceToHost, transfer_stream);
}
```

### 7.3 CUDA Graph - Stream Capture

```cuda
// Step 1: 录制
cudaGraph_t graph;
cudaStream_t stream;
cudaStreamCreate(&stream);

cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

// 正常写 kernel launch（不会真正执行）
layer_norm<<<grid, block, 0, stream>>>(input, output, gamma, beta, N);
attention<<<grid2, block2, 0, stream>>>(q, k, v, output, ...);
ffn<<<grid3, block3, 0, stream>>>(input, output, weights, ...);

cudaStreamEndCapture(stream, &graph);

// Step 2: 实例化
cudaGraphExec_t graph_exec;
cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);

// Step 3: 执行（可重复调用）
for (int token = 0; token < max_tokens; token++) {
    // 更新输入指针（如果需要）
    cudaGraphLaunch(graph_exec, stream);
    cudaStreamSynchronize(stream);
}

// Cleanup
cudaGraphExecDestroy(graph_exec);
cudaGraphDestroy(graph);
```

## 8. PyTorch baseline

```python
import torch

# PyTorch CUDA Graph API
def decode_step(model, input_ids, kv_cache):
    """单步 decode"""
    with torch.no_grad():
        logits = model(input_ids, kv_cache=kv_cache)
    return logits[:, -1, :]

# 使用 torch.cuda.graph
static_input = torch.zeros(1, 1, dtype=torch.long, device='cuda')
static_kv = ...  # pre-allocated

# Warmup
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        decode_step(model, static_input, static_kv)
torch.cuda.current_stream().wait_stream(s)

# Capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output = decode_step(model, static_input, static_kv)

# Replay
for token_id in token_ids:
    static_input.fill_(token_id)
    g.replay()
    next_token = static_output.argmax(dim=-1)
```

## 9. CUDA 实现思路

### 9.1 Event-based 同步

```cuda
cudaEvent_t event;
cudaEventCreate(&event);

// Stream A 完成后通知 Stream B
kernel_A<<<grid, block, 0, streamA>>>(...);
cudaEventRecord(event, streamA);

cudaStreamWaitEvent(streamB, event, 0);  // B 等待 A
kernel_B<<<grid, block, 0, streamB>>>(...);
```

### 9.2 Graph 更新（避免重建）

```cuda
// 更新 graph 中 kernel 的参数
cudaGraphExecKernelNodeSetParams(graph_exec, kernel_node, &new_params);

// 或者整体更新
cudaGraphExecUpdate(graph_exec, new_graph, &update_result);
```

## 10. Triton 实现思路

Triton 本身不直接支持 CUDA Graph，但可以通过 PyTorch 集成使用：

```python
import triton
import triton.language as tl
import torch

@triton.jit
def my_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, x * 2, mask=mask)

# 包装为 PyTorch autograd function 后可被 CUDA Graph capture
def triton_op(x):
    y = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, 1024),)
    my_kernel[grid](x, y, n, BLOCK=1024)
    return y

# Capture with CUDA Graph
static_x = torch.randn(1024, device='cuda')
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_y = triton_op(static_x)

# Replay
static_x.copy_(new_data)
g.replay()
```

## 11. Memory Access 分析

### Stream 并发
- 多 stream 并发时，多个 kernel 共享 L2 cache → 可能互相 evict
- Memory-bound kernel 并发可能导致带宽争抢

### CUDA Graph
- Graph 内 kernel 的 memory 布局在 capture 时确定
- 使用 graph 时需要 pre-allocate 所有 buffer（static shape）
- Graph replay 不会重新分配内存

## 12. Parallelism 分析

### Stream 级并发
- 不同 stream 的 kernel 可以在不同 SM 上并行
- 前提：单个 kernel 不占满所有 SM
- 小 kernel 并发效果好，大 kernel 并发收益小

### Graph 级优化
- Graph 内的独立节点可以并发执行
- Graph 消除了 CPU-GPU 之间的 launch 往返

## 13. Compute-bound / Memory-bound 判断

| 场景 | 瓶颈 | Graph 收益 |
|------|------|-----------|
| Decode (batch=1) | Launch overhead | 高（20-40%） |
| Decode (batch=64) | Compute | 低（<5%） |
| Prefill (long seq) | Compute/Memory | 无（shape 动态） |
| 小 kernel 序列 | Launch overhead | 高 |
| 大 kernel 序列 | Compute | 低 |

## 14. Profiling 指标

| 指标 | 工具 | 含义 |
|------|------|------|
| Kernel Launch Overhead | Nsight Systems | CPU 端 launch 到 GPU 开始执行的时间 |
| Stream Concurrency | Nsight Systems | 多 stream 并行执行的时间比例 |
| GPU Idle Time | Nsight Systems | GPU 空闲时间（launch gap） |
| Graph Launch Time | Nsight Systems | graph launch 的 CPU 开销 |
| Occupancy per Kernel | Nsight Compute | 每个 kernel 的 SM 利用率 |

## 15. Benchmark 设计

```python
import torch
import time

def benchmark_with_without_graph(model, input_ids, kv_cache, n_tokens=100):
    """对比有无 CUDA Graph 的 decode 性能"""
    
    # Without graph
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_tokens):
        output = model.decode_step(input_ids, kv_cache)
    torch.cuda.synchronize()
    time_no_graph = time.perf_counter() - start
    
    # With graph (after capture)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_tokens):
        graph.replay()
    torch.cuda.synchronize()
    time_with_graph = time.perf_counter() - start
    
    print(f"Without graph: {time_no_graph/n_tokens*1000:.3f} ms/token")
    print(f"With graph: {time_with_graph/n_tokens*1000:.3f} ms/token")
    print(f"Speedup: {time_no_graph/time_with_graph:.2f}x")
```

## 16. 常见错误

| 错误 | 原因 | 解决 |
|------|------|------|
| Graph capture 失败 | Capture 期间有 CPU-GPU 同步 | 移除 synchronize 调用 |
| Graph replay 结果错误 | 输入 shape 变化 | 使用 static buffer + copy |
| 多 stream 无并发 | 使用了 default stream | 创建独立 stream |
| Graph 内存泄漏 | 未 destroy graph/exec | 正确管理生命周期 |
| Capture 中有条件分支 | if 语句依赖 GPU 数据 | 重构为无分支或多个 graph |
| Stream 同步死锁 | 循环依赖 | 检查 event 依赖关系 |

## 17. 实验任务

### 实验 1：Stream 并发
- 创建 4 个 stream，每个执行一个小 kernel
- 用 Nsight Systems 观察并发执行
- 对比单 stream 顺序执行的时间

### 实验 2：Compute-Transfer Overlap
- 实现 pipeline：H2D → compute → D2H
- 用 2 个 stream 实现 double buffering
- 测量 overlap 带来的加速

### 实验 3：CUDA Graph Decode
- 模拟 LLM decode：10 个小 kernel 顺序执行
- 用 stream capture 录制为 graph
- 对比 replay vs 逐个 launch 的延迟

### 实验 4：Graph 限制探索
- 尝试在 graph 中使用动态 shape → 观察失败
- 尝试在 graph 中使用 cudaMalloc → 观察失败
- 理解 graph 的适用边界

## 18. 习题 20 道

1. CUDA Stream 的默认行为是什么？为什么需要多 stream？
2. 两个 kernel 在不同 stream 中一定会并发执行吗？列举不会的情况。
3. cudaEvent 的两个主要用途是什么？
4. 解释 cudaStreamWaitEvent 的语义。
5. CUDA Graph 解决了什么性能问题？
6. Stream Capture 和显式 Graph API 的优缺点对比。
7. 为什么 CUDA Graph 特别适合 LLM decode phase？
8. Graph 内可以有条件分支吗？如何处理？
9. vLLM 如何为不同 batch size 管理 CUDA Graph？
10. Graph replay 时如何更新输入数据？
11. 什么情况下 CUDA Graph 收益很小？
12. 多 stream 并发时 L2 cache 会有什么问题？
13. cudaStreamCaptureModeGlobal vs Relaxed 的区别？
14. 如何测量 kernel launch overhead？
15. Graph instantiation 的开销大吗？应该何时做？
16. PyTorch 的 torch.cuda.graph API 有什么限制？
17. 如何在 graph 中处理 dynamic shape（如不同 sequence length）？
18. CUDA Graph 与 TensorRT 的 graph optimization 有什么区别？
19. 多 GPU 场景下 CUDA Graph 如何使用？
20. Graph 中的 kernel 可以使用不同的 stream 吗？

## 19. 标准答案

1. 默认所有操作在 default stream 中顺序执行。多 stream 允许并发执行独立操作，提高 GPU 利用率。

2. 不一定。不会并发的情况：(a) GPU 资源不足（一个 kernel 占满所有 SM）；(b) 使用了 legacy default stream（隐式同步）；(c) 有 event 依赖。

3. (a) 跨 stream 同步（cudaStreamWaitEvent）；(b) 精确计时（cudaEventElapsedTime）。

4. 语义：指定 stream 等待 event 被 record 后才继续执行后续操作。不阻塞 CPU。

5. 解决 kernel launch overhead 问题。当 kernel 执行时间很短时，CPU 端逐个 launch 的开销占比高。Graph 将整个序列预编译，一次 launch 执行全部。

6. Stream Capture：代码改动小，直接录制现有代码；但灵活性低。显式 API：完全控制 graph 结构，可以表达复杂依赖；但代码量大。

7. Decode phase 特点：(a) 每 token 执行相同 kernel 序列；(b) batch 小时每个 kernel 计算量小；(c) launch overhead 占比高。Graph 消除重复 launch 开销。

8. 不能有运行时条件分支。处理方式：(a) 为每个分支预建不同 graph；(b) 使用 conditional node（CUDA 12.4+）；(c) 将条件逻辑移到 graph 外。

9. vLLM 预先为 batch_size = 1,2,4,8,16,32,... 各录制一个 graph。运行时根据当前 batch size 选择最接近的 graph（padding 到对应 size）。

10. 使用 static buffer：graph capture 时使用固定地址的 buffer，replay 前通过 cudaMemcpy 更新 buffer 内容。或使用 cudaGraphExecKernelNodeSetParams 更新参数。

11. (a) Kernel 本身计算量大（launch overhead 占比小）；(b) Shape 动态变化频繁；(c) 需要 CPU-GPU 交互的逻辑。

12. 多 kernel 并发共享 L2 cache，可能互相 evict 对方的数据，导致 cache hit rate 下降。对 memory-bound kernel 影响更大。

13. Global：capture 期间所有 stream 的操作都被录制。Relaxed：只录制显式加入 capture 的 stream。Global 更安全但限制更多。

14. 使用 Nsight Systems 查看 kernel launch 到 kernel start 的时间差。或用 CUDA event 测量空 kernel 的 launch-to-completion 时间。

15. Instantiation 开销较大（ms 级），应在初始化阶段做一次，之后重复 replay。不要在热路径中 instantiate。

16. 限制：(a) capture 期间不能有 CPU-GPU 同步；(b) 不能有动态内存分配；(c) 不能有 Python 控制流依赖 GPU 结果；(d) 所有 tensor 必须 pre-allocated。

17. 方案：(a) 为常见 length 预建多个 graph；(b) padding 到固定 length；(c) 只对 decode（固定 seq_len=1）使用 graph，prefill 不用。

18. CUDA Graph 是执行级优化（减少 launch overhead），不改变计算。TensorRT 是编译级优化（kernel fusion、precision 转换、layout 优化），改变计算图结构。

19. 多 GPU 场景：每个 GPU 有自己的 graph。跨 GPU 通信（NCCL）可以包含在 graph 中（CUDA 12+支持）。需要确保所有 GPU 同步 capture。

20. 可以。Graph 中的节点可以指定不同 stream，graph 会保持依赖关系。这允许 graph 内部的并发执行。

## 20. 复习卡片 30 张

1. Q: CUDA Stream 是什么？ A: 有序的 GPU 操作队列，同一 stream 内顺序执行，不同 stream 可并发。
2. Q: Default stream 的特殊性？ A: Legacy default stream 与其他 stream 有隐式同步；per-thread default stream 无此限制。
3. Q: 如何创建 stream？ A: `cudaStreamCreate(&stream)` 或 `cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)`。
4. Q: cudaEvent 如何计时？ A: `cudaEventRecord(start, stream)` → 操作 → `cudaEventRecord(stop, stream)` → `cudaEventElapsedTime(&ms, start, stop)`。
5. Q: Stream 并发的必要条件？ A: 不同 stream + 无依赖 + GPU 资源充足。
6. Q: CUDA Graph 的核心优势？ A: 消除逐个 kernel launch 的 CPU 开销，一次 launch 执行整个操作序列。
7. Q: Graph 适用场景？ A: 固定 shape、重复执行的 kernel 序列，如 LLM decode。
8. Q: Graph 不适用场景？ A: 动态 shape、需要 CPU 决策、单次执行的操作。
9. Q: Stream Capture 的基本流程？ A: BeginCapture → 正常 launch kernel → EndCapture → Instantiate → Launch。
10. Q: Graph 中能否 cudaMalloc？ A: 不能。所有内存必须预分配。
11. Q: Kernel launch overhead 典型值？ A: 5-10μs（CPU 端）。
12. Q: Graph launch overhead 典型值？ A: 10-20μs（一次性，不随 kernel 数增加）。
13. Q: vLLM 如何使用 CUDA Graph？ A: 为不同 batch size 预录制 graph，decode 时选择匹配的 graph replay。
14. Q: Graph replay 时如何传入新数据？ A: 使用 static buffer，replay 前 memcpy 新数据到 buffer。
15. Q: cudaStreamSynchronize vs cudaDeviceSynchronize？ A: 前者只等待指定 stream，后者等待所有 stream。
16. Q: 多 stream 的 L2 cache 问题？ A: 并发 kernel 共享 L2，可能互相 evict。
17. Q: Graph instantiation 做了什么？ A: 验证 graph 结构、分配资源、编译执行计划。开销大，应只做一次。
18. Q: PyTorch CUDA Graph API？ A: `torch.cuda.CUDAGraph()` + `torch.cuda.graph(g)` context manager。
19. Q: Graph 中的 conditional node？ A: CUDA 12.4+ 支持，允许 graph 内有条件分支。
20. Q: 如何测量 stream 并发效果？ A: Nsight Systems timeline 查看多 stream kernel 的重叠程度。
21. Q: cudaStreamCaptureModeGlobal 含义？ A: Capture 期间所有 stream 的操作都被录制到 graph。
22. Q: Graph update vs rebuild？ A: 小改动用 update（更新参数），结构变化需 rebuild。
23. Q: Decode batch=1 时 graph 加速比？ A: 典型 20-40%，取决于 kernel 数量和大小。
24. Q: 为什么 prefill 不用 graph？ A: Prefill 的 sequence length 动态变化，无法预录制。
25. Q: cudaStreamWaitEvent 是否阻塞 CPU？ A: 不阻塞 CPU，只让 GPU stream 等待。
26. Q: 如何在 Nsight Systems 中识别 launch overhead？ A: 查看 CUDA API 行和 kernel 行之间的时间差。
27. Q: Graph 中多个独立节点会并发吗？ A: 会，如果它们没有依赖关系且 GPU 资源充足。
28. Q: cudaEventDisableTiming flag 的作用？ A: 创建不记录时间的 event，减少开销，仅用于同步。
29. Q: Stream priority 的作用？ A: 高优先级 stream 的 kernel 优先被调度执行。
30. Q: CUDA Graph 与 TorchScript/torch.compile 的关系？ A: torch.compile 可以自动使用 CUDA Graph（通过 cudagraphs backend）。