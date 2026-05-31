# 08 - Warp Stall（Warp 停顿分析）

## 1. 学习目标

- 理解 Warp Stall（Warp 停顿）的概念及其对 GPU 吞吐量的影响
- 掌握各类 stall reason 的含义与诊断方法
- 学会使用 Nsight Compute 的 Warp State Statistics 分析停顿分布
- 理解 warp scheduler 如何通过切换 warp 隐藏停顿延迟
- 能够根据 stall 类型选择对应的优化策略
- 掌握 warp divergence 的检测与消除方法

## 2. 性能问题动机

Warp 是 GPU 执行的基本调度单位，停顿直接导致计算资源闲置：

- Memory stall 占比过高说明访存延迟未被充分隐藏
- Instruction stall 暗示指令级并行度（ILP, Instruction-Level Parallelism）不足
- Synchronization stall 表明 barrier 过于频繁或负载不均
- Warp divergence 导致同一 warp 内线程串行执行不同分支
- 不同 stall 类型需要完全不同的优化策略，误判会浪费优化时间

## 3. 核心术语表

| 术语 | 英文全称 | 定义 |
|------|----------|------|
| Warp Stall | Warp Stall | Warp 无法发射下一条指令的状态 |
| Stall Reason | Stall Reason | 导致 warp 停顿的具体原因分类 |
| Warp Scheduler | Warp Scheduler | 每周期从就绪 warp 中选择一个发射指令 |
| Eligible Warp | Eligible Warp | 当前可以发射指令的 warp |
| Active Warp | Active Warp | 驻留在 SM 上的 warp（含 stalled） |
| Warp Divergence | Warp Divergence | 同一 warp 内线程执行不同分支路径 |
| Predication | Predication | 用谓词寄存器控制指令执行，避免分支 |
| ILP | Instruction-Level Parallelism | 指令级并行度 |
| Latency Hiding | Latency Hiding | 通过切换 warp 隐藏操作延迟 |
| Issue Slot | Issue Slot | Warp scheduler 每周期的指令发射槽 |

## 4. 指标定义

| 指标 | 公式 | 单位 | 含义 |
|------|------|------|------|
| Warp Cycles Per Issued Instruction | total_cycles / issued_instructions | cycles | 平均每条指令的 warp 等待周期 |
| Stall Percentage | stall_cycles / total_cycles × 100 | % | 各类 stall 占总周期比例 |
| Eligible Warps Per Cycle | avg(eligible_warps) | 个 | 每周期可调度 warp 数 |
| Issue Efficiency | issued_cycles / active_cycles × 100 | % | 发射槽利用率 |
| Branch Efficiency | uniform_branches / total_branches × 100 | % | 非 divergent 分支比例 |
| Warp Execution Efficiency | active_threads / 32 × 100 | % | 平均每 warp 活跃线程比例 |

## 5. 指标来源

| 指标 | 数据源 | 获取方式 |
|------|--------|----------|
| Stall Reasons | Nsight Compute | Warp State Statistics section |
| Eligible Warps | Nsight Compute | `smsp__warps_eligible.avg` |
| Issue Efficiency | Nsight Compute | `smsp__issue_active.avg.pct_of_peak_sustained_active` |
| Branch Efficiency | Nsight Compute | `smsp__sass_branch_targets_threads_uniform.pct` |
| Divergent Branches | Nsight Compute | `smsp__sass_branch_targets_threads_divergent.sum` |
| Cycles Per Instruction | Nsight Compute | Scheduler Statistics section |

## 6. 正常现象

- Memory-bound kernel 中 "Stall Long Scoreboard"（等待全局内存）占主导（40-70%）
- Compute-bound kernel 中 "Stall Math Pipe Throttle" 占主导
- 少量 "Stall Barrier"（<10%）在使用 shared memory 的 kernel 中正常
- Eligible warps > 1 表示 scheduler 有选择余地，延迟隐藏良好
- 简单 if-else 导致的轻微 divergence（<5% 效率损失）

## 7. 异常现象

- "Stall Long Scoreboard" 超过 80% 且 occupancy 已经很高
- "Stall Not Selected" 占比高但 eligible warps 也高
- "Stall Barrier" 超过 30%
- Branch efficiency 低于 80%
- Eligible warps 接近 0（scheduler 饥饿）

## 8. 可能原因

| 异常 | 可能原因 |
|------|----------|
| Long Scoreboard 过高 | 全局内存延迟未隐藏；occupancy 不足；访存模式差 |
| Not Selected 高 | Warp 就绪但 scheduler 带宽不足；指令混合不均 |
| Barrier 过高 | __syncthreads 过频繁；block 内负载不均 |
| Branch efficiency 低 | 数据依赖的条件分支；warp 内线程走不同路径 |
| Scheduler 饥饿 | Occupancy 极低；所有 warp 同时等待同一资源 |
| Math Pipe Throttle | 计算密集且 pipeline 满载（正常，说明 compute-bound） |

## 9. 验证实验

### 实验 1：不同 Stall 类型的 Kernel

```python
import torch
import triton
import triton.language as tl

@triton.jit
def memory_bound_kernel(x_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    """Memory-bound: 大量全局内存访问，少量计算"""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + 1.0, mask=mask)

@triton.jit
def compute_bound_kernel(x_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    """Compute-bound: 少量访存，大量计算"""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    # 大量计算
    for _ in range(100):
        x = x * x + x
        x = tl.math.sqrt(tl.abs(x) + 1e-6)
    tl.store(out_ptr + offs, x, mask=mask)
```

### 实验 2：Warp Divergence 影响

```python
@triton.jit
def divergent_kernel(x_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    """故意制造 warp divergence"""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    
    # 线程 ID 奇偶决定不同计算路径 → divergence
    thread_id = offs % 2
    result = tl.where(thread_id == 0, x * x + x, tl.math.sqrt(tl.abs(x) + 1e-6))
    tl.store(out_ptr + offs, result, mask=mask)

@triton.jit
def uniform_kernel(x_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    """无 divergence 的等价计算"""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    
    # 所有线程执行相同路径
    result = x * x + x
    tl.store(out_ptr + offs, result, mask=mask)
```

### 实验 3：Barrier Stall 分析

```python
@triton.jit
def barrier_heavy_kernel(x_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    """频繁同步导致 barrier stall"""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    
    # 模拟频繁的 reduction 操作（隐含 barrier）
    for _ in range(10):
        total = tl.sum(x, axis=0)
        x = x + total / BLOCK
    
    tl.store(out_ptr + offs, x, mask=mask)
```

## 10. 优化方法

| 方法 | 适用 Stall 类型 | 预期效果 |
|------|----------------|----------|
| 提升 occupancy | Long Scoreboard | 更多 warp 隐藏内存延迟 |
| 优化访存模式 | Long Scoreboard | 减少内存延迟本身 |
| 增加 ILP | Short Scoreboard | 单 warp 内更多独立指令 |
| 减少 barrier | Barrier | 降低同步频率 |
| 消除 divergence | Divergent Branch | 所有线程走同一路径 |
| 使用 predication | Divergent Branch | 短分支用谓词替代 |
| 数据重排 | Divergent Branch | 相似数据聚集到同一 warp |
| 循环展开 | Short Scoreboard | 暴露更多 ILP |
| 软件流水线 | Long Scoreboard | 计算与访存重叠 |

## 11. 副作用

- 提升 occupancy 可能减少每线程寄存器，增加 spilling
- 循环展开增加寄存器压力和代码体积
- 消除 divergence 可能增加冗余计算
- 减少 barrier 可能引入数据竞争（需仔细验证正确性）
- 软件流水线增加代码复杂度和寄存器使用
- Predication 对长分支体无效（两条路径都执行）

## 12. Profiling 命令模板

```bash
# 完整 Warp State 分析
ncu --section WarpStateStatistics python my_kernel.py

# Scheduler 统计
ncu --section SchedulerStatistics python my_kernel.py

# 关键 stall 指标
ncu --metrics \
  smsp__warps_eligible.avg.per_cycle_active,\
  smsp__issue_active.avg.pct_of_peak_sustained_active,\
  smsp__cycles_active.avg.pct_of_peak_sustained_elapsed,\
  smsp__sass_branch_targets_threads_uniform.pct \
  python my_kernel.py

# Stall reason 详细分解
ncu --metrics \
  smsp__average_warp_latency_per_inst_issued.ratio,\
  smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
  smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,\
  smsp__warp_issue_stalled_wait_per_warp_active.pct,\
  smsp__warp_issue_stalled_barrier_per_warp_active.pct,\
  smsp__warp_issue_stalled_not_selected_per_warp_active.pct \
  python my_kernel.py

# Source-level stall 分析
ncu --section SourceCounters --source-level all python my_kernel.py

# 结合 Nsight Systems 看整体 timeline
nsys profile --stats=true -o warp_stall_analysis python my_kernel.py
```

## 13. Benchmark 设计

### 设计原则

1. **隔离 stall 类型**：设计专门触发特定 stall 的 kernel
2. **对比实验**：同一算法的 stall-heavy vs stall-free 版本
3. **量化影响**：测量 stall 减少后的性能提升比例
4. **关联 occupancy**：在不同 occupancy 下观察 stall 变化

### Stall 类型对比 Benchmark

```python
import torch
import numpy as np

def stall_benchmark():
    """对比不同 stall 特征的 kernel 性能"""
    N = 16 * 1024 * 1024
    x = torch.randn(N, device='cuda')
    out = torch.empty_like(x)
    
    kernels = {
        'memory_bound': lambda: memory_bound_kernel[(N//256,)](x, out, N, 256),
        'compute_bound': lambda: compute_bound_kernel[(N//256,)](x, out, N, 256),
        'divergent': lambda: divergent_kernel[(N//256,)](x, out, N, 256),
    }
    
    for name, fn in kernels.items():
        # Warm-up
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
        
        print(f"{name:15s}: {np.median(times):.3f} ms (use ncu for stall breakdown)")
```

## 14. 实验记录表

| 字段 | 示例值 | 说明 |
|------|--------|------|
| 日期 | 2026-05-31 | 实验日期 |
| GPU 型号 | A100-80GB | 硬件型号 |
| Kernel 名称 | attention_fwd | 被测 kernel |
| Achieved Occupancy | 62.5% | 实际占用率 |
| Eligible Warps/Cycle | 3.2 | 可调度 warp 数 |
| Issue Efficiency | 45.3% | 发射效率 |
| Top Stall #1 | Long Scoreboard: 42% | 主要停顿 |
| Top Stall #2 | Barrier: 18% | 次要停顿 |
| Top Stall #3 | Not Selected: 15% | 第三停顿 |
| Branch Efficiency | 97.2% | 分支效率 |
| Warp Exec Efficiency | 99.1% | 执行效率 |
| Kernel Duration | 1.234 ms | 执行时间 |
| 优化后 Duration | 0.891 ms | 优化后时间 |
| 提升比例 | 27.8% | 性能提升 |

## 15. 故障树

```
Warp Stall 导致性能不佳
├── Long Scoreboard (全局内存等待)
│   ├── Occupancy 不足 → 增加活跃 warp 数
│   ├── 非合并访问 → 优化访存模式
│   ├── 缺乏 ILP → 循环展开/软件流水线
│   └── 数据不在 cache → 改善局部性
├── Short Scoreboard (寄存器依赖)
│   ├── 指令链依赖 → 重排指令增加 ILP
│   ├── 特殊函数单元等待 → 减少 SFU 使用
│   └── Tensor Core 等待 → 增加 MMA 间独立指令
├── Barrier (同步等待)
│   ├── __syncthreads 过频繁 → 合并同步点
│   ├── Block 内负载不均 → 均衡工作分配
│   └── Reduction 操作 → 使用 warp-level primitives
├── Not Selected (就绪但未被选中)
│   ├── Scheduler 带宽限制 → 正常现象
│   └── 指令类型冲突 → 混合不同类型指令
├── Divergent Branch
│   ├── 数据依赖条件 → 数据重排/predication
│   ├── 边界检查 → 分离边界 block
│   └── 不规则数据结构 → 重新设计算法
└── Math Pipe Throttle
    └── 计算单元满载 → 正常（compute-bound）
```

## 16. 复盘模板

```markdown
## Warp Stall 分析复盘

### 实验目标
- Kernel 名称：
- 当前性能瓶颈假设：

### Stall 分布
| Stall Reason | 占比 | 是否异常 |
|-------------|------|----------|
| Long Scoreboard | | |
| Short Scoreboard | | |
| Barrier | | |
| Not Selected | | |
| Other | | |

### 分析结论
- 主要瓶颈类型：[memory/compute/sync/divergence]
- 根因：
- Eligible warps/cycle：

### 优化措施与效果
| 措施 | Stall 变化 | 性能变化 |
|------|-----------|----------|

### 经验总结
- 关键发现：
- 适用的优化模式：
```

## 17. 常见错误

| # | 错误 | 后果 | 正确做法 |
|---|------|------|----------|
| 1 | 只看 stall 占比不看绝对值 | 误判优化优先级 | 结合 kernel duration 分析 |
| 2 | 对 compute-bound kernel 优化 memory stall | 无效优化 | 先确认瓶颈类型 |
| 3 | 用 if-else 处理 warp 边界 | 引入 divergence | 使用 mask/predication |
| 4 | 过度同步 | barrier stall 增加 | 最小化 __syncthreads |
| 5 | 忽略 "Not Selected" | 以为是问题 | 通常是正常调度行为 |
| 6 | 盲目提升 occupancy 解决所有 stall | 可能适得其反 | 针对具体 stall 类型优化 |
| 7 | 混淆 active warp 和 eligible warp | 分析错误 | active 含 stalled，eligible 不含 |
| 8 | 忽略 warp divergence 的累积效应 | 低估性能损失 | 嵌套分支损失指数增长 |
| 9 | 在 Triton 中手动管理 warp | 不适用 | Triton 自动处理，关注算法层面 |
| 10 | 未区分 stall 是否可隐藏 | 过度优化 | 有足够 eligible warp 时 stall 可隐藏 |

## 18. 习题 20 道

1. 什么是 warp stall？为什么它是 GPU 性能分析的核心指标？
2. 列举并解释 5 种主要的 stall reason。
3. "Long Scoreboard" stall 的含义是什么？它通常由什么操作触发？
4. 解释 warp scheduler 如何通过切换 warp 来隐藏延迟。
5. 什么是 eligible warp？它与 active warp 的区别是什么？
6. 如果 eligible warps per cycle 接近 0，说明什么问题？如何解决？
7. 什么是 warp divergence？给出一个会导致 divergence 的代码示例。
8. 解释 predication 如何避免短分支的 divergence 开销。
9. "Stall Barrier" 高说明什么？列举三种降低 barrier stall 的方法。
10. 设计实验：对比 memory-bound 和 compute-bound kernel 的 stall 分布差异。
11. 什么是 ILP（指令级并行）？它如何帮助减少 stall？
12. 解释 "Stall Math Pipe Throttle" 的含义。这是好事还是坏事？
13. 如何使用 Nsight Compute 的 Source Counters 定位具体哪行代码导致 stall？
14. 软件流水线（software pipelining）如何减少 memory stall？
15. 在什么情况下，高 occupancy 仍然无法解决 long scoreboard stall？
16. 解释 warp execution efficiency 指标及其与 divergence 的关系。
17. 如何通过数据重排将 divergent 访问转为 uniform 访问？
18. "Stall Short Scoreboard" 与 "Long Scoreboard" 的区别是什么？
19. 设计一个实验验证：减少 barrier 数量对性能的影响。
20. 如果一个 kernel 的 stall 分布均匀（无明显主导），应该如何优化？

## 19. 标准答案

1. Warp stall 是 warp 无法在当前周期发射指令的状态。核心指标因为：GPU 性能 = 有效指令吞吐，stall 直接降低吞吐，且 stall 类型指明优化方向。

2. (1) Long Scoreboard：等待全局内存/L2 返回数据 (2) Short Scoreboard：等待寄存器依赖（如 SFU、Tensor Core） (3) Barrier：等待 __syncthreads (4) Not Selected：就绪但未被 scheduler 选中 (5) Math Pipe Throttle：计算管线满载。

3. Long Scoreboard：warp 发射了一条长延迟指令（通常是全局内存加载），正在等待结果写回寄存器。触发操作：global load/store、L2 cache miss、atomic 操作。

4. Warp scheduler 每周期检查所有 active warp，从 eligible（就绪）warp 中选一个发射指令。当 warp A 等待内存时，scheduler 切换到 warp B 执行，A 的延迟被 B 的执行时间"隐藏"。

5. Active warp：驻留在 SM 上的所有 warp（含正在 stall 的）。Eligible warp：当前周期可以发射指令的 warp（无 stall、无依赖）。Eligible ⊆ Active。

6. Eligible ≈ 0 说明所有 active warp 都在 stall（scheduler 饥饿）。解决：(1) 提升 occupancy 增加 active warp (2) 优化访存减少 stall 时间 (3) 增加 ILP 使单 warp 有更多可发射指令。

7. Warp divergence：同一 warp 的 32 个线程遇到条件分支时走不同路径，必须串行执行两条路径。示例：`if (threadIdx.x % 2 == 0) { a(); } else { b(); }` — 奇偶线程走不同分支。

8. Predication：编译器将短 if-else 转为条件执行——两条路径的指令都执行，但用谓词寄存器控制哪些线程的结果有效。避免了分支跳转和串行化，但两条路径的指令都消耗周期。

9. Barrier stall 高说明线程块内同步过于频繁或负载不均。降低方法：(1) 合并多次 __syncthreads 为一次 (2) 使用 warp-level primitives（__shfl）替代 block-level sync (3) 均衡 block 内工作负载。

10. 实验：编写两个 kernel——一个只做 load/store（memory-bound），一个做大量 FMA（compute-bound）。用 ncu --section WarpStateStatistics 分别 profile，对比 stall 分布。

11. ILP：单线程内多条独立指令可同时在不同功能单元执行。帮助减少 stall：当指令 A 等待结果时，独立的指令 B 可以发射，减少 warp 的 stall 周期。

12. Math Pipe Throttle：计算管线（如 FMA 单元）已满载，新指令必须等待。这通常是好事——说明 kernel 是 compute-bound 且充分利用了计算资源。

13. 使用 `ncu --section SourceCounters --source-level all`，报告会显示每行源代码对应的 stall 采样数。高 stall 行即为热点，可针对性优化。

14. 软件流水线：将循环迭代 i 的计算与迭代 i+1 的数据加载重叠。当前迭代计算时，下一迭代的数据已在加载中，减少了等待数据的 stall 时间。

15. 当所有 warp 同时访问同一内存区域（如 atomic 操作）或访存模式极差（完全随机访问）时，即使 occupancy 100%，所有 warp 可能同时 stall，无法互相隐藏。

16. Warp execution efficiency = 平均每条指令的活跃线程数 / 32 × 100%。Divergence 导致部分线程被 mask 掉，降低该指标。100% 表示无 divergence。

17. 方法：对数据按条件分支的判断值排序/分组，使同一 warp 内的线程具有相同的分支条件。例如将正数和负数分别聚集，避免 warp 内混合。

18. Short Scoreboard：等待短延迟操作（如 shared memory load、SFU 计算，~20 cycles）。Long Scoreboard：等待长延迟操作（如 global memory load，~200-600 cycles）。前者更容易通过 ILP 隐藏。

19. 实验：实现同一 reduction 算法的两个版本——(A) 每步都 __syncthreads (B) 使用 warp shuffle 减少 sync。对比 barrier stall 占比和总执行时间。

20. Stall 分布均匀说明没有单一瓶颈，kernel 可能已经比较均衡。优化方向：(1) 整体减少指令数（算法优化） (2) 同时改善多个维度 (3) 考虑是否已接近硬件极限（检查 roofline 位置）。

## 20. 调优 Checklist

- [ ] 使用 ncu --section WarpStateStatistics 获取 stall 分布
- [ ] 确认主要 stall 类型（Long/Short Scoreboard、Barrier、Divergence）
- [ ] 检查 eligible warps per cycle（>1 为健康）
- [ ] 检查 issue efficiency
- [ ] 确认 kernel 是 memory-bound 还是 compute-bound
- [ ] 若 Long Scoreboard 主导：检查访存模式和 occupancy
- [ ] 若 Barrier 主导：减少同步点或使用 warp primitives
- [ ] 若 Divergence 明显：检查 branch efficiency
- [ ] 考虑增加 ILP（循环展开、指令重排）
- [ ] 使用 Source Counters 定位具体热点行
- [ ] 优化后重新 profile 确认 stall 分布变化
- [ ] 验证性能提升与 stall 减少的相关性
