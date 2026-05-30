# Day 1：Parallel Reduction

## 学习目标
- 实现 5 个版本的 parallel reduction kernel
- 理解 bank conflict、warp divergence、warp shuffle
- 能闭卷手写 warp shuffle reduction
- 理解 Ring AllReduce 原理

---

## 上午（3h）- 实现 5 个版本

### V1: Interleaved Addressing

```cuda
__global__ void reduce_v1(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {  // 问题：warp divergence
            sdata[tid] += sdata[tid + s];  // 问题：bank conflict（stride 为偶数）
        }
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = sdata[0];
}
```

问题分析：
- `tid % (2*s) == 0`：导致 warp 内只有部分 thread 活跃 → warp divergence
- `sdata[tid + s]`：当 s 为 2 的幂时，访问 stride 为偶数 → bank conflict

### V2: Sequential Addressing

```cuda
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}
```

改进：连续的 thread 做加法，消除 bank conflict。但前半 warp 活跃、后半 idle → 仍有 divergence。

### V3: First Add During Load

```cuda
int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
sdata[tid] = input[i] + input[i + blockDim.x];  // 加载时就做第一次加法
```

改进：每个 thread 在加载阶段就处理 2 个元素，减少 idle thread。

### V4: Warp Unrolling

```cuda
// 当 s <= 32（一个 warp）时，不需要 __syncthreads()
if (tid < 32) {
    volatile float* smem = sdata;  // volatile 防止编译器优化
    smem[tid] += smem[tid + 32];
    smem[tid] += smem[tid + 16];
    smem[tid] += smem[tid + 8];
    smem[tid] += smem[tid + 4];
    smem[tid] += smem[tid + 2];
    smem[tid] += smem[tid + 1];
}
```

改进：warp 内 thread 天然同步（SIMT），省去 `__syncthreads()` 开销。

### V5: Warp Shuffle（最终版，面试手写目标）

```cuda
__global__ void reduce_v5(float* input, float* output, int n) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (i < n) ? input[i] : 0.0f;

    // Warp-level reduction using shuffle
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    // 每个 warp 的结果写入 shared memory
    __shared__ float warp_sums[32];  // 最多 32 个 warp
    int lane = tid % 32;
    int warp_id = tid / 32;
    if (lane == 0) warp_sums[warp_id] = val;
    __syncthreads();

    // 第一个 warp 做最终 reduction
    if (warp_id == 0) {
        val = (tid < blockDim.x / 32) ? warp_sums[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (tid == 0) output[blockIdx.x] = val;
    }
}
```

优势：完全消除 shared memory 的 bank conflict 和 latency（register 间直接交换）。

### 每个版本记录的指标

用 Nsight Compute 跑每个版本：
```bash
ncu --set full -o reports/reduce_v1 python bench_reduction.py --version 1
```

记录：
- `dram__bytes_read.sum` / time → achieved bandwidth (GB/s)
- `sm__warps_active.avg.pct_of_peak_sustained_active` → occupancy
- 相对 V1 的加速比

---

## 下午（2h）- 理论深入

### 必读材料
- Mark Harris "Optimizing Parallel Reduction in CUDA"（NVIDIA slides）

### 核心概念

**Bank Conflict**：
- Shared memory 有 32 个 bank，每个 bank 4 bytes 宽
- 同一个 warp 的 32 个 thread 同时访问 shared memory
- 如果多个 thread 访问同一个 bank 的不同地址 → 串行化（N-way conflict）
- 如果访问同一个 bank 的同一个地址 → broadcast（无冲突）
- 避免方法：让连续 thread 访问连续地址（stride=1），或加 padding

**Warp Divergence**：
- 一个 warp 32 个 thread 执行相同指令（SIMT）
- if-else 导致部分 thread 走 if、部分走 else → 两个分支串行执行
- 代价：执行时间 = if 时间 + else 时间（而非 max）
- 避免方法：让同一个 warp 的 thread 走相同分支

**`__shfl_down_sync(mask, val, offset)`**：
- 语义：warp 内 lane i 的 thread 获取 lane i+offset 的 val
- 零延迟（register 间直接交换，不经过 shared memory）
- mask = 0xffffffff 表示所有 32 个 lane 参与

---

## 晚上（1.5h）- 分布式：Ring AllReduce

### 原理

N 个 GPU，每个 GPU 有大小为 D 的数据，目标：每个 GPU 得到所有数据的 sum。

**ReduceScatter 阶段**（N-1 步）：
- 数据切成 N 份，每份大小 D/N
- 每步：每个 GPU 发送一份给下一个 GPU，接收一份并累加
- N-1 步后：每个 GPU 持有一份完整的 partial sum

**AllGather 阶段**（N-1 步）：
- 每步：每个 GPU 发送自己的完整份给下一个 GPU，接收一份
- N-1 步后：每个 GPU 持有所有份的完整 sum

**通信量分析**：
```
每步每个 GPU 发送 D/N 数据
总步数：2*(N-1)
总发送量：2*(N-1) * D/N
当 N 很大时 → 约 2D（和 GPU 数量无关！）
带宽利用率：(N-1)/N
```

### 画图练习

画出 4 个 GPU 的 Ring AllReduce：
- 标注每步每个 GPU 发送/接收的数据块编号
- 标注 ReduceScatter 和 AllGather 的分界

---

## 日检（20 分钟）

关掉所有资料，完成以下任务：

1. **闭卷手写**（10min）：写出 V5 warp shuffle reduction 的完整 kernel
   - 通过标准：逻辑正确，包含 warp reduction + cross-warp reduction

2. **口述**（5min）：Bank conflict 是什么？在 reduction V1 中怎么产生的？V2 怎么解决的？
   - 通过标准：能说清 32 bank、stride 访问模式、sequential addressing 的改进

3. **口述**（5min）：Ring AllReduce 的总通信量是多少？为什么和 GPU 数量几乎无关？
   - 通过标准：能说出 2*(N-1)/N * D，解释每步只传 D/N

---

## 参考资料

- Mark Harris, "Optimizing Parallel Reduction in CUDA", NVIDIA
- CUDA Programming Guide Chapter: Warp Shuffle Functions
- leimao/CUDA-Reduction (GitHub)
