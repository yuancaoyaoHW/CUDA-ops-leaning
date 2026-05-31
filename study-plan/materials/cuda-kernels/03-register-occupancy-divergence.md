# Register Pressure、Occupancy 与 Warp Divergence

## 1. 学习目标

- 理解 register pressure（寄存器压力）对 occupancy 的影响
- 掌握 occupancy 的计算方法和优化策略
- 理解 warp divergence（warp 分支分歧）的成因和性能影响
- 能够使用 `--ptxas-options=-v` 和 occupancy calculator 分析 kernel
- 掌握 `__launch_bounds__` 的使用方法

## 2. 前置知识

- CUDA 执行模型（thread/block/warp/SM）
- GPU 内存层次结构
- 基本的编译器优化概念

## 3. 核心术语表

| 术语 | 英文全称 | 定义 |
|------|----------|------|
| Register | Register | 每个 thread 私有的最快存储单元 |
| Register File | Register File | SM 上所有 register 的集合（如 A100: 65536 × 32-bit） |
| Register Pressure | Register Pressure | kernel 使用过多 register 导致 occupancy 下降 |
| Register Spilling | Register Spilling | register 不够用时溢出到 local memory（实际在 global memory） |
| Occupancy | Occupancy | SM 上活跃 warp 数 / 最大可驻留 warp 数 |
| Theoretical Occupancy | Theoretical Occupancy | 基于资源限制计算的最大可能 occupancy |
| Achieved Occupancy | Achieved Occupancy | 实际运行时测量的平均 occupancy |
| Warp Divergence | Warp Divergence | 同一 warp 内 thread 执行不同分支路径 |
| Predication | Predication | 短分支的硬件优化，用 predicate 寄存器控制执行 |
| Launch Bounds | Launch Bounds | 编译器提示，指定 kernel 的最大 thread 数和最小 block 数 |

## 4. 动机

### 4.1 Register Pressure 的影响

每个 SM 的 register file 是固定的（如 A100: 65536 个 32-bit register）。如果一个 thread 使用 64 个 register：
- 一个 warp (32 threads) 需要 64 × 32 = 2048 个 register
- 一个 SM 最多驻留 65536 / 2048 = 32 个 warp
- 如果 block size = 256 (8 warps)，最多 4 个 block

但如果 thread 使用 128 个 register：
- 一个 warp 需要 128 × 32 = 4096 个 register
- 一个 SM 最多 16 个 warp → occupancy 大幅下降

### 4.2 Occupancy 不是越高越好

- 高 occupancy 有助于隐藏 memory latency（更多 warp 可切换）
- 但过高 occupancy 可能意味着每个 thread 的 register 不够 → spilling → 性能下降
- 最优点通常在 50%-75% occupancy

### 4.3 Warp Divergence 的代价

```cuda
if (threadIdx.x < 16) {
    // Path A: 只有 lane 0-15 执行
} else {
    // Path B: 只有 lane 16-31 执行
}
// 两条路径串行执行，warp 效率降为 50%
```

## 5. 数学定义

### 5.1 Occupancy 计算

```
occupancy = active_warps / max_warps_per_SM

active_warps = active_blocks × warps_per_block
warps_per_block = ceil(block_size / 32)
```

限制因素（取最小值）：
```
max_blocks_by_registers = floor(registers_per_SM / (regs_per_thread × block_size))
max_blocks_by_smem = floor(shared_mem_per_SM / smem_per_block)
max_blocks_by_threads = floor(max_threads_per_SM / block_size)
max_blocks_by_limit = max_blocks_per_SM  // 硬件限制，如 A100: 32

active_blocks = min(max_blocks_by_registers, max_blocks_by_smem, 
                    max_blocks_by_threads, max_blocks_by_limit)
```

### 5.2 Register Spilling 代价

```
spill_penalty ≈ spilled_bytes × memory_latency / compute_time
```

Local memory spill 走 L1 → L2 → HBM 路径，延迟约 200-400 cycles。

### 5.3 Divergence 效率

```
divergence_efficiency = active_threads / 32

// 对于 if-else with 50% branch:
efficiency = 16/32 = 50% (每条路径)
overall_time = time_path_A + time_path_B (串行)
```

## 6. 推导逻辑

### 6.1 Register 分配策略

编译器（ptxas）为每个 thread 分配 register：
1. 分析 kernel 中的活跃变量（live variables）
2. 进行 register allocation（图着色算法）
3. 如果需要的 register > 可用量 → spill 到 local memory
4. `--maxrregcount=N` 可限制最大 register 数

### 6.2 Occupancy 与 Latency Hiding

GPU 通过 warp 切换隐藏 memory latency：
```
required_warps = memory_latency / instruction_throughput

// 例：memory latency = 400 cycles, 每 cycle 发射 1 条指令
// 需要 400 个 warp-instructions 来隐藏 → 约 12-13 个 warp
// 对应 occupancy ≈ 13/64 ≈ 20% (A100 max 64 warps/SM)
```

实际中，compute-bound kernel 不需要高 occupancy（计算本身就能隐藏 latency）。

### 6.3 Warp Divergence 的硬件处理

Volta+ 架构（Independent Thread Scheduling）：
- 每个 thread 有独立的 program counter 和 call stack
- Divergent thread 可以独立推进
- 但仍然需要 reconvergence 才能恢复 SIMT 效率
- `__syncwarp()` 显式同步 warp 内 thread

## 7. 算子流程

### 7.1 分析 Register Usage

```bash
# 编译时查看 register 使用
nvcc -Xptxas -v kernel.cu -o kernel
# 输出示例：
# ptxas info: Used 32 registers, 4096 bytes smem, 0 bytes cmem

# 限制 register 数量
nvcc --maxrregcount=32 kernel.cu -o kernel
```

### 7.2 使用 Launch Bounds

```cuda
// 告诉编译器：最多 256 threads/block，至少 2 blocks/SM
__global__ void __launch_bounds__(256, 2)
my_kernel(...) {
    // 编译器会优化 register 分配以满足约束
}
```

### 7.3 Occupancy Calculator API

```cuda
#include <cuda_runtime.h>

int main() {
    int block_size = 256;
    int min_grid_size, grid_size;
    
    // 自动计算最优 block size
    cudaOccupancyMaxPotentialBlockSize(
        &min_grid_size, &block_size, my_kernel, 0, 0);
    
    // 计算给定 block size 的 occupancy
    int max_active_blocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks, my_kernel, block_size, 0);
    
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    float occupancy = (float)(max_active_blocks * block_size / 32) 
                    / prop.maxThreadsPerMultiProcessor * 32;
    printf("Occupancy: %.2f%%\n", occupancy * 100);
}
```

## 8. PyTorch baseline

```python
import torch
from torch.profiler import profile, ProfilerActivity

# 一个会产生 divergence 的操作
def conditional_op(x: torch.Tensor, threshold: float) -> torch.Tensor:
    """PyTorch 自动处理，无需手动管理 divergence"""
    return torch.where(x > threshold, x * 2, x * 0.5)

x = torch.randn(1 << 20, device='cuda')
result = conditional_op(x, 0.0)
```

## 9. CUDA 实现思路

### 9.1 Divergence 示例与优化

```cuda
// Bad: warp divergence
__global__ void divergent_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (idx % 2 == 0) {  // 50% divergence within each warp
            data[idx] = data[idx] * 2.0f;
        } else {
            data[idx] = data[idx] * 0.5f;
        }
    }
}

// Better: reorganize data so that divergence aligns with warp boundaries
__global__ void non_divergent_kernel(float* even_data, float* odd_data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n/2) {
        even_data[idx] = even_data[idx] * 2.0f;  // 所有 thread 走同一路径
    }
}

// Alternative: use predication for short branches (compiler does this automatically)
__global__ void predicated_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        float factor = (idx % 2 == 0) ? 2.0f : 0.5f;  // 短分支，编译器用 predication
        data[idx] = val * factor;
    }
}
```

### 9.2 Register Pressure 优化

```cuda
// High register pressure: many live variables
__global__ void high_reg_kernel(float* a, float* b, float* c, 
                                 float* d, float* e, float* f, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v1 = a[idx], v2 = b[idx], v3 = c[idx];
        float v4 = d[idx], v5 = e[idx], v6 = f[idx];
        // 所有变量同时活跃 → 高 register pressure
        float result = v1*v2 + v3*v4 + v5*v6;
        a[idx] = result;
    }
}

// Optimized: reduce live variables by recomputing or staging
__global__ void low_reg_kernel(float* a, float* b, float* c,
                                float* d, float* e, float* f, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float result = a[idx] * b[idx];  // v1, v2 dead after this
        result += c[idx] * d[idx];        // v3, v4 dead after this
        result += e[idx] * f[idx];        // v5, v6 dead after this
        a[idx] = result;
    }
}
```

## 10. Triton 实现思路

```python
import triton
import triton.language as tl

@triton.jit
def conditional_kernel(
    x_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Triton 处理 divergence: 编译器自动优化
    # 使用 tl.where 避免显式分支
    result = tl.where(x > 0, x * 2.0, x * 0.5)
    
    tl.store(out_ptr + offsets, result, mask=mask)
```

**Triton 的优势**：
- 程序员不需要手动管理 register allocation
- 编译器自动选择最优的 register 使用策略
- `tl.where` 自动处理 divergence
- `num_warps` 参数控制 block 内 warp 数量

## 11. Memory Access 分析

### Register Spilling 的访存代价

```
Spill store: register → L1 cache → (可能) L2 → HBM
Spill load:  HBM → L2 → L1 → register

每次 spill load/store 增加约 8-400 cycles 延迟
（取决于是否命中 L1/L2）
```

### 如何检测 Spilling

```bash
# 编译时查看
nvcc -Xptxas -v kernel.cu
# 输出: "Used 64 registers, 16 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads"

# Nsight Compute 中查看
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum ./kernel
```

## 12. Parallelism 分析

### Occupancy vs Performance 关系

```
Case 1: Memory-bound kernel
- 低 occupancy → 无法隐藏 memory latency → 性能差
- 提高 occupancy → 更多 warp 可切换 → 性能提升
- 但超过某个点后收益递减

Case 2: Compute-bound kernel  
- 计算本身就能填满 pipeline
- Occupancy 50% 可能就够了
- 过高 occupancy 反而因 register spilling 降低性能
```

### 实际案例：GEMM

- cuBLAS GEMM 通常 occupancy 只有 25-50%
- 但每个 thread 使用大量 register 做 tile 计算
- 高 ILP（Instruction Level Parallelism）补偿低 occupancy

## 13. Compute-bound / Memory-bound 判断

```
如果 kernel 的 achieved bandwidth 接近 peak bandwidth → memory-bound
如果 kernel 的 achieved FLOPS 接近 peak FLOPS → compute-bound

对于 register pressure 相关的 kernel：
- Spilling 导致额外 memory traffic → 可能从 compute-bound 变为 memory-bound
- 需要在 register 使用和 occupancy 之间找平衡
```

## 14. Profiling 指标

| 指标 | 工具 | 含义 |
|------|------|------|
| Registers Per Thread | nvcc -Xptxas -v | 每个 thread 使用的 register 数 |
| Theoretical Occupancy | Occupancy Calculator | 资源限制下的最大 occupancy |
| Achieved Occupancy | Nsight Compute | 实际运行时的平均 occupancy |
| Warp Execution Efficiency | Nsight Compute | 活跃 thread 占 warp 的比例 |
| Branch Efficiency | Nsight Compute | 非 divergent 分支占比 |
| Local Memory Overhead | Nsight Compute | spill 导致的额外访存 |
| Eligible Warps | Nsight Compute | 每 cycle 可调度的 warp 数 |
| Stall Reasons | Nsight Compute | warp stall 的原因分布 |

## 15. Benchmark 设计

### 实验 1：Register Pressure vs Occupancy

```python
# 设计不同 register 使用量的 kernel，测量性能
# 变量：register count = 16, 32, 48, 64, 80, 96, 128
# 指标：kernel time, achieved occupancy, effective bandwidth
```

### 实验 2：Divergence 影响

```python
# 设计不同 divergence 比例的 kernel
# 变量：divergence ratio = 0%, 25%, 50%, 75%, 100%
# 指标：kernel time, warp execution efficiency
```

### 实验 3：Launch Bounds 效果

```python
# 对比有无 __launch_bounds__ 的性能差异
# 变量：maxThreadsPerBlock = 128, 256, 512
# 指标：register count, occupancy, kernel time
```

## 16. 常见错误

1. **盲目追求 100% occupancy**：牺牲 register 导致 spilling，反而更慢
2. **忽略 register spilling**：编译时不检查 `-v` 输出
3. **在 warp 内做数据依赖的分支**：导致严重 divergence
4. **使用过大的 block size**：register 和 shared memory 不够分
5. **不使用 `__launch_bounds__`**：编译器无法优化 register 分配
6. **混淆 theoretical 和 achieved occupancy**：前者是上限，后者受 workload 影响
7. **忽略 ILP**：低 occupancy 可以通过高 ILP 补偿
8. **在循环内声明大数组**：编译器可能放入 local memory

## 17. 实验任务

### 任务 1：Occupancy 实验
编写一个 kernel，通过调整局部变量数量控制 register 使用量。测量不同 register 数量下的 occupancy 和性能。

### 任务 2：Divergence 消除
给定一个有 50% divergence 的 kernel，重构数据布局消除 divergence，对比性能。

### 任务 3：Launch Bounds 调优
对一个 compute-intensive kernel 尝试不同的 `__launch_bounds__` 配置，找到最优点。

### 任务 4：Spilling 检测与修复
编写一个使用过多 register 的 kernel，检测 spilling，然后通过算法重构减少 live variables。

## 18. 习题 20 道

1. A100 每个 SM 有多少个 32-bit register？每个 thread 最多能用多少？
2. 如果 kernel 使用 64 registers/thread，block size = 256，A100 上每个 SM 最多驻留几个 block？
3. 什么是 register spilling？spill 的数据存储在哪里？
4. `__launch_bounds__(256, 4)` 的两个参数分别是什么含义？
5. 为什么 cuBLAS GEMM 的 occupancy 通常只有 25-50% 但性能很高？
6. Warp divergence 在什么情况下不会造成性能损失？
7. Volta 架构引入的 Independent Thread Scheduling 解决了什么问题？
8. 如何用 nvcc 编译选项查看 kernel 的 register 使用量？
9. `cudaOccupancyMaxPotentialBlockSize` 的作用是什么？
10. 一个 warp 内 32 个 thread 中有 8 个走 if 分支、24 个走 else 分支，执行效率是多少？
11. 为什么 `--maxrregcount` 可能导致性能下降？
12. Achieved occupancy 低于 theoretical occupancy 的可能原因？
13. 如何在 Nsight Compute 中查看 warp stall 原因？
14. 什么是 ILP？它如何补偿低 occupancy？
15. 对于 memory-bound kernel，occupancy 从 25% 提升到 50% 预期性能提升多少？
16. Register pressure 和 shared memory 使用量如何共同影响 occupancy？
17. 什么情况下应该选择更小的 block size？
18. Predication 和 branch 的区别是什么？编译器何时选择 predication？
19. 如何判断一个 kernel 的性能瓶颈是 register spilling 而非其他原因？
20. 设计一个实验验证 occupancy 对 memory-bound kernel 性能的影响。

## 19. 标准答案

1. A100: 65536 个 32-bit register/SM。每个 thread 最多 255 个 register。
2. 64 × 256 = 16384 registers/block。65536 / 16384 = 4 blocks/SM。4 × 8 warps = 32 warps。Occupancy = 32/64 = 50%。
3. Register spilling 是编译器将放不下的变量存到 local memory（物理上在 global memory，经过 L1/L2 cache）。
4. 第一个参数：每个 block 最大 thread 数。第二个参数：每个 SM 最少 block 数。编译器据此优化 register 分配。
5. 因为 GEMM 是 compute-bound，每个 thread 做大量 FMA 运算，高 ILP 填满了执行 pipeline，不需要靠 warp 切换隐藏 latency。
6. 当分支条件按 warp 边界对齐时（如 `warpId < threshold`），整个 warp 走同一路径，无 divergence。
7. 解决了 Volta 之前 warp 内 thread 必须在同一 PC 的限制，允许 thread 独立推进，改善了 fine-grained synchronization。
8. `nvcc -Xptxas -v kernel.cu` 或 `nvcc --ptxas-options=-v kernel.cu`。
9. 自动计算能达到最高 occupancy 的最小 grid size 和最优 block size。
10. 两条路径串行执行：if 路径效率 8/32 = 25%，else 路径效率 24/32 = 75%。总时间 = time_if + time_else。
11. 限制 register 数量可能导致 spilling，spill load/store 的延迟可能超过 occupancy 提升带来的收益。
12. 可能原因：grid size 不够大（block 数 < SM 数 × blocks/SM）、workload 不均匀、tail effect。
13. Nsight Compute → Warp State Statistics → 查看 stall_not_selected, stall_memory_dependency 等。
14. ILP = Instruction Level Parallelism。单个 thread 内多条独立指令可以流水线执行，减少对 warp 切换的依赖。
15. 理论上接近 2x（如果完全 memory latency bound），实际通常 1.3-1.7x（因为还有其他因素）。
16. 两者独立限制 active blocks。最终 occupancy 取两者限制的较小值。需要同时优化。
17. 当 kernel 使用大量 register 或 shared memory 时，小 block size 允许更多 block 驻留。
18. Predication：两条路径都执行但用 predicate mask 控制写回。Branch：跳转指令。编译器对短分支（几条指令）用 predication。
19. 检查 Nsight Compute 的 local memory traffic 指标。如果 `l1tex__t_sectors_pipe_lsu_mem_local` 非零且占比高，说明 spilling 是瓶颈。
20. 设计一个纯 memory copy kernel，通过 `__launch_bounds__` 控制 occupancy（如 25%, 50%, 75%, 100%），测量 effective bandwidth。

## 20. 复习卡片 30 张

1. Q: A100 每个 SM 的 register file 大小？ A: 65536 × 32-bit = 256KB
2. Q: 每个 thread 最多使用多少 register？ A: 255
3. Q: Occupancy 的定义？ A: active warps / max warps per SM
4. Q: A100 每个 SM 最多驻留多少 warp？ A: 64
5. Q: A100 每个 SM 最多驻留多少 block？ A: 32
6. Q: Register spilling 的目标存储？ A: Local memory（物理在 HBM，经 L1/L2 cache）
7. Q: 如何编译时查看 register 使用？ A: nvcc -Xptxas -v
8. Q: `__launch_bounds__(M, N)` 含义？ A: M=max threads/block, N=min blocks/SM
9. Q: Warp divergence 何时发生？ A: 同一 warp 内 thread 走不同分支
10. Q: Divergence 的硬件处理方式？ A: 串行执行各分支路径
11. Q: Predication 适用条件？ A: 短分支（几条指令），编译器自动选择
12. Q: Volta+ 的 Independent Thread Scheduling？ A: 每个 thread 有独立 PC，可独立推进
13. Q: `__syncwarp()` 的作用？ A: 显式同步 warp 内 thread 到同一执行点
14. Q: Occupancy 不是越高越好的原因？ A: 高 occupancy 可能导致 register spilling
15. Q: ILP 如何补偿低 occupancy？ A: 单 thread 内多条独立指令填满 pipeline
16. Q: `cudaOccupancyMaxPotentialBlockSize` 的作用？ A: 自动计算最优 block size
17. Q: `--maxrregcount=N` 的风险？ A: 可能导致 spilling，反而更慢
18. Q: Achieved vs Theoretical occupancy？ A: Achieved 是实测值，可能因 grid 太小等原因低于 theoretical
19. Q: Memory-bound kernel 需要高 occupancy 的原因？ A: 需要更多 warp 切换来隐藏 memory latency
20. Q: Compute-bound kernel 对 occupancy 的需求？ A: 较低即可，计算本身隐藏 latency
21. Q: 如何检测 register spilling？ A: 编译输出看 spill loads/stores，或 Nsight Compute 看 local memory traffic
22. Q: Block size 选择的经验法则？ A: 128 或 256，是 warp size 的倍数
23. Q: Warp execution efficiency 的定义？ A: 平均活跃 thread 数 / 32
24. Q: Branch efficiency 的定义？ A: 非 divergent 分支数 / 总分支数
25. Q: Eligible warps 指标的含义？ A: 每 cycle 可被调度的 warp 数
26. Q: Stall reason: memory dependency？ A: Warp 等待 memory 操作完成
27. Q: Stall reason: not selected？ A: Warp ready 但 scheduler 选了其他 warp
28. Q: 如何消除 warp divergence？ A: 重组数据使分支按 warp 边界对齐
29. Q: Register 数量与 occupancy 的关系？ A: 反比关系，register 越多 occupancy 越低
30. Q: Shared memory 与 register 对 occupancy 的影响？ A: 独立限制，取较严格的限制
