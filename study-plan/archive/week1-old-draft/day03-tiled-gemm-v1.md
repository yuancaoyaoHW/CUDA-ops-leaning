# Day 3：Tiled GEMM V1（Block Tiling + Shared Memory）

## 学习目标
- 实现 naive GEMM 和 tiled GEMM
- 理解 tiling 如何把 GEMM 从 memory-bound 变成 compute-bound
- 用 Nsight Compute 分析性能瓶颈
- 理解 Tensor Parallelism 的通信模式

---

## 上午（3h）- 实现

### Naive GEMM（基线）

```cuda
// 每个 thread 计算 C 的一个元素
// 问题：每个元素需要读 2K 个数据，计算 2K FLOPs → AI = 2K/(2K*4*2) ≈ 0.5 → memory-bound
__global__ void gemm_naive(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

### Tiled GEMM V1

```cuda
#define BM 32  // Block tile M
#define BN 32  // Block tile N
#define BK 32  // Block tile K

__global__ void gemm_tiled_v1(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // 这个 thread 负责的输出位置
    int row = by * BM + ty;
    int col = bx * BN + tx;

    float sum = 0.0f;

    // 沿 K 维度滑动
    for (int k_tile = 0; k_tile < K; k_tile += BK) {
        // 协作加载 A tile [BM x BK] 到 shared memory
        if (row < M && (k_tile + tx) < K)
            As[ty][tx] = A[row * K + k_tile + tx];
        else
            As[ty][tx] = 0.0f;

        // 协作加载 B tile [BK x BN] 到 shared memory
        if ((k_tile + ty) < K && col < N)
            Bs[ty][tx] = B[(k_tile + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        // 计算部分和
        for (int kk = 0; kk < BK; kk++) {
            sum += As[ty][kk] * Bs[kk][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}
```

### 为什么 Tiling 有效（面试核心问题）

```
Naive: 每个输出元素读 A 的一行 (K 个) + B 的一列 (K 个) = 2K 次 global memory 访问
       M*N 个输出元素 → 总读取 2*M*N*K 次
       但 A 只有 M*K 个元素，B 只有 K*N 个元素
       → 每个元素被重复读了 N 次（A）或 M 次（B）

Tiled: 每个 BM*BN 的输出 tile 需要 A 的 BM*K 和 B 的 K*BN
       但通过 shared memory，每个 BK 的 slice 只从 global memory 读一次
       被 BM 个 thread（A）或 BN 个 thread（B）共享
       → 数据复用率 = BM 或 BN

Arithmetic Intensity 变化:
  Naive: AI = 2K / (2K * sizeof * 2) ≈ 0.5 FLOP/byte (memory-bound)
  Tiled: AI = 2*BM*BN*BK / ((BM*BK + BK*BN) * sizeof) = 2*BM*BN / ((BM+BN) * sizeof)
         BM=BN=32, fp32: AI = 2*32*32 / ((32+32)*4) = 2048/256 = 8 FLOP/byte
         BM=BN=64, fp16: AI = 2*64*64 / ((64+64)*2) = 8192/256 = 32 FLOP/byte
         → 随 block size 增大，AI 线性增长 → 从 memory-bound 变成 compute-bound
```

### Launch Configuration

```cuda
dim3 block(BN, BM);  // 32x32 = 1024 threads per block
dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
gemm_tiled_v1<<<grid, block>>>(A, B, C, M, N, K);
```

### Benchmark

测试 M=N=K = 1024, 2048, 4096：
- 记录 GFLOPS = 2*M*N*K / time / 1e9
- 对比 cuBLAS（`cublasSgemm`）
- 计算 % of cuBLAS

---

## 下午（2h）- 性能分析

### GEMM 的 Roofline

```
理论计算量: 2*M*N*K FLOPs
理论数据量: (M*K + K*N + M*N) * sizeof bytes

M=N=K=2048, fp32:
  计算: 2 * 2048^3 = 17.18 GFLOP
  数据: (2048^2 * 3) * 4 = 50.3 MB
  AI = 17.18G / 50.3M = 341 FLOP/byte → compute-bound

M=N=K=2048, fp16:
  计算: 2 * 2048^3 = 17.18 GFLOP
  数据: (2048^2 * 3) * 2 = 25.2 MB
  AI = 17.18G / 25.2M = 682 FLOP/byte → 更加 compute-bound

但如果 M=1 (GEMV, decode attention 的情况):
  计算: 2 * 1 * N * K = 2NK FLOPs
  数据: (K + K*N + N) * 2 ≈ 2KN bytes (fp16)
  AI = 2NK / 2KN = 1 FLOP/byte → memory-bound!
```

### Nsight Compute 检查项

```bash
ncu --set full -o reports/gemm_v1 ./gemm_benchmark
```

关键指标：
- `sm__throughput.avg.pct_of_peak_sustained_elapsed`：compute 利用率
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`：memory 利用率
- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared`：shared memory bank conflict 数量
- `sm__warps_active.avg.pct_of_peak_sustained_active`：occupancy
- `launch__occupancy`：理论 occupancy（受 register/shared memory 限制）

### 性能瓶颈分析

V1 的典型问题：
1. **Bank conflict**：`Bs[kk][tx]` 当 kk 固定时，32 个 thread 访问 Bs 的同一列 → 可能有 conflict
2. **Low occupancy**：32x32 = 1024 threads/block，shared memory = 2*32*32*4 = 8KB → occupancy 受 thread 数限制
3. **No vectorized load**：每次只读 1 个 float，没有利用 128-bit load

---

## 晚上（1.5h）- 分布式：Tensor Parallelism

### 原理

把模型的每一层切分到多个 GPU 上并行计算。

### MLP 的 TP（Megatron 方案）

```
标准 MLP: Y = GeLU(X @ A) @ B
  A: [hidden, 4*hidden]  (第一个 linear)
  B: [4*hidden, hidden]  (第二个 linear)

Column Parallel（切 A 的列）:
  A = [A1 | A2]  切到 2 个 GPU
  GPU0: Y0 = GeLU(X @ A1)  shape: [batch*seq, 2*hidden]
  GPU1: Y1 = GeLU(X @ A2)  shape: [batch*seq, 2*hidden]
  → 不需要通信！因为 GeLU 是 element-wise，可以独立算

Row Parallel（切 B 的行）:
  B = [B1; B2]
  GPU0: Z0 = Y0 @ B1  shape: [batch*seq, hidden]
  GPU1: Z1 = Y1 @ B2  shape: [batch*seq, hidden]
  最终: Z = Z0 + Z1  → 需要 AllReduce!
```

### Attention 的 TP

```
Multi-Head Attention: 把 heads 分到不同 GPU
  GPU0: head 0,1,...,h/2-1
  GPU1: head h/2,...,h-1

  Q/K/V projection: Column Parallel（每个 GPU 只算自己 heads 的 Q/K/V）
  Output projection: Row Parallel → 需要 AllReduce
```

### 通信量分析

```
每个 Transformer layer:
  MLP: 1 次 AllReduce (Row Parallel output)
  Attention: 1 次 AllReduce (output projection)
  总计: 2 次 AllReduce per layer

每次 AllReduce 通信量:
  数据大小 = batch_size * seq_len * hidden_size * sizeof(fp16)
  AllReduce = 2 * (N-1)/N * data_size

例：batch=32, seq=2048, hidden=4096, fp16, TP=8:
  data = 32 * 2048 * 4096 * 2 = 512 MB
  AllReduce = 2 * 7/8 * 512 = 896 MB per AllReduce
  每层 2 次 = 1.75 GB per layer
  32 层 = 56 GB 总通信量

  NVLink 带宽 ~600 GB/s (A100 双向)
  通信时间 ≈ 56 / 600 ≈ 93 ms（可以和计算 overlap 一部分）
```

### 为什么 TP 要放 node 内

- TP 需要频繁 AllReduce（每层 2 次）
- AllReduce 需要高带宽、低延迟
- Node 内 NVLink: 600 GB/s, ~1us latency
- Node 间 InfiniBand: 50-100 GB/s, ~5us latency
- → TP 放 node 内（通常 TP=8 对应 8 GPU/node）

---

## 日检（20 分钟）

1. **闭卷手写**（10min）：写出 tiled GEMM 的核心循环（shared memory load + compute + syncthreads）
2. **口述**（5min）：Tiling 为什么能把 GEMM 从 memory-bound 变成 compute-bound？AI 怎么变化的？
3. **口述**（5min）：TP 中 AllReduce 插在哪里？每层通信几次？通信量公式？

---

## 参考资料

- PMPP Chapter 4-5: Memory and Tiling
- leimao/CUDA-GEMM-Optimization (GitHub)
- Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism (Shoeybi et al., 2019)
