# Day 4：GEMM V2（Thread Tiling + Vectorized Load + Double Buffering）

## 学习目标
- 实现 thread tiling（每个 thread 算多个输出）
- 理解 vectorized load 和 bank conflict 消除
- 理解 double buffering 隐藏 memory latency
- 理解 Pipeline Parallelism（GPipe vs 1F1B）

---

## 上午（3h）- Thread Tiling

### 为什么需要 Thread Tiling

V1 问题：每个 thread 只算 1 个输出元素
- 内循环每次从 shared memory 读 2 个值（As[ty][kk] 和 Bs[kk][tx]），算 1 次 FMA
- Compute/Load ratio = 1 FMA / 2 loads = 0.5 → shared memory bandwidth 成为瓶颈

Thread tiling：每个 thread 算 TM×TN 个输出（如 8×8）
- 内循环从 shared memory 读 TM+TN 个值，算 TM*TN 次 FMA
- Compute/Load ratio = TM*TN / (TM+TN) = 64/16 = 4 → 好得多

### 实现

```cuda
#define BM 128
#define BN 128
#define BK 8
#define TM 8   // 每个 thread 算 8 行
#define TN 8   // 每个 thread 算 8 列

__global__ void gemm_v2(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // Block 内 thread 布局: (BM/TM) x (BN/TN) = 16x16 = 256 threads
    int tx = threadIdx.x % (BN / TN);  // 0..15
    int ty = threadIdx.x / (BN / TN);  // 0..15

    // 这个 thread 负责的输出区域起始位置
    int row_start = blockIdx.y * BM + ty * TM;
    int col_start = blockIdx.x * BN + tx * TN;

    // Register 存储部分和
    float C_reg[TM][TN] = {0.0f};
    float A_reg[TM];
    float B_reg[TN];

    for (int k_tile = 0; k_tile < K; k_tile += BK) {
        // 协作加载 As 和 Bs（每个 thread 加载多个元素）
        // ... (省略加载逻辑)

        __syncthreads();

        // 计算：从 shared memory 加载到 register，做外积
        for (int kk = 0; kk < BK; kk++) {
            // 加载 A 的一列到 register
            for (int i = 0; i < TM; i++)
                A_reg[i] = As[ty * TM + i][kk];
            // 加载 B 的一行到 register
            for (int j = 0; j < TN; j++)
                B_reg[j] = Bs[kk][tx * TN + j];
            // 外积累加
            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++)
                    C_reg[i][j] += A_reg[i] * B_reg[j];
        }

        __syncthreads();
    }

    // 写回 C
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++)
            if (row_start + i < M && col_start + j < N)
                C[(row_start + i) * N + col_start + j] = C_reg[i][j];
}
```

### 性能分析

```
BM=128, BN=128, BK=8, TM=8, TN=8:
  Threads per block: (128/8) * (128/8) = 16 * 16 = 256
  Shared memory: (128*8 + 8*128) * 4 = 8 KB
  Registers per thread: TM*TN + TM + TN = 64 + 8 + 8 = 80 (还有其他变量)

  每个 BK 步的计算: BM * BN * BK * 2 = 128*128*8*2 = 262K FLOPs
  每个 BK 步的数据: (BM*BK + BK*BN) * 4 = (1024+1024)*4 = 8 KB from global
  AI (global): 262K / 8K = 32.8 FLOP/byte → compute-bound ✓
```

---

## 下午（2h）- Vectorized Load + Bank Conflict 消除

### Vectorized Load

```cuda
// 标准 load: 每次 4 bytes
float val = A[offset];  // 1 个 LDS.32 指令

// Vectorized load: 每次 16 bytes
float4 val = reinterpret_cast<float4*>(A)[offset/4];  // 1 个 LDS.128 指令

// 好处：
// 1. 减少 load instruction 数量（4x fewer）
// 2. 更好地利用 memory bandwidth（一次 transaction 搬更多数据）
// 3. 编译器可以更好地 schedule
```

### Bank Conflict 消除

```
问题：Bs[kk][tx*TN + j]
  当 BN=128, 32 个 thread 的 tx*TN 分别是 0,8,16,...,248
  bank = (tx*TN + j) % 32
  如果 TN=8: bank = (tx*8 + j) % 32
  tx=0: bank 0-7; tx=4: bank 0-7 → 4-way conflict!

解决方案 1: Padding
  __shared__ float Bs[BK][BN + PAD];  // PAD=1 或 4
  bank = (tx*TN + j) % 32 变成 (tx*TN + j + kk*PAD) % 32 → 打散

解决方案 2: Swizzle
  重新排列 shared memory 中的数据布局
  使得同一 warp 的 thread 访问不同 bank
  更复杂但更高效（CUTLASS 使用的方法）

解决方案 3: 转置 B
  把 B tile 转置后存入 shared memory: Bs[tx][kk] 而非 Bs[kk][tx]
  访问变成 Bs[tx*TN + j][kk] → 连续 thread 访问连续行 → 无 conflict
```

### Double Buffering

```cuda
// 思路：用 2 倍 shared memory，计算和加载 overlap
__shared__ float As[2][BM][BK];
__shared__ float Bs[2][BK][BN];

int buf = 0;

// 预加载第一个 tile
load_tile(As[0], Bs[0], k_tile=0);
__syncthreads();

for (int k_tile = 0; k_tile < K - BK; k_tile += BK) {
    // 异步加载下一个 tile 到另一个 buffer
    load_tile(As[1-buf], Bs[1-buf], k_tile + BK);

    // 用当前 buffer 计算
    compute(As[buf], Bs[buf], C_reg);

    buf = 1 - buf;
    __syncthreads();
}
// 处理最后一个 tile
compute(As[buf], Bs[buf], C_reg);
```

CUDA 的 `cp.async`（Ampere+）：
```cuda
// 异步从 global memory 拷贝到 shared memory，不占用计算单元
__pipeline_memcpy_async(&As[1-buf][ty][tx], &A[...], sizeof(float));
__pipeline_commit();
// ... 做计算 ...
__pipeline_wait_prior(0);  // 等待拷贝完成
```

### 时间线对比

```
无 Double Buffering:
  |--Load--|--Compute--|--Load--|--Compute--|--Load--|--Compute--|
  总时间 = N * (T_load + T_compute)

有 Double Buffering:
  |--Load--|
           |--Compute--|--Load--|
                       |--Compute--|--Load--|
                                   |--Compute--|
  总时间 ≈ T_load + N * max(T_load, T_compute)
  如果 compute-bound: 总时间 ≈ N * T_compute（load 完全隐藏）
```

---

## 晚上（1.5h）- 分布式：Pipeline Parallelism

### 基本概念

模型按层切分到不同 GPU（stage）：
```
Stage 0: Layer 0-7    (GPU 0)
Stage 1: Layer 8-15   (GPU 1)
Stage 2: Layer 16-23  (GPU 2)
Stage 3: Layer 24-31  (GPU 3)
```

通信：只需要相邻 stage 间传递 activation（P2P），通信量小。

### GPipe（All-Forward-All-Backward）

```
时间线（4 stages, 4 micro-batches）:

        Stage0  Stage1  Stage2  Stage3
Step 1: F1      -       -       -
Step 2: F2      F1      -       -
Step 3: F3      F2      F1      -
Step 4: F4      F3      F2      F1
Step 5: -       F4      F3      F2
Step 6: -       -       F4      F3
Step 7: -       -       -       F4
Step 8: B4      -       -       -      ← 开始 backward
...

Bubble = 空闲时间 / 总时间
       = (PP-1) * (T_f + T_b) / (num_microbatches * (T_f + T_b) + (PP-1) * (T_f + T_b))
       ≈ (PP-1) / (num_microbatches + PP-1)

例：PP=4, micro=4: bubble = 3/7 = 43%（很大！）
例：PP=4, micro=32: bubble = 3/35 = 8.6%（可接受）
```

### 1F1B（One Forward One Backward）

```
时间线（4 stages, 8 micro-batches）:

Warmup phase (PP-1 = 3 个 forward):
        Stage0  Stage1  Stage2  Stage3
Step 1: F1      -       -       -
Step 2: F2      F1      -       -
Step 3: F3      F2      F1      -

Steady state (交替 1F1B):
Step 4: F4,B1   F3      F2      F1
Step 5: F5,B2   F4,B1   F3      F2
...

Cooldown phase (PP-1 = 3 个 backward):
最后几步只做 backward

优势 vs GPipe:
  - Bubble ratio 相同
  - 但显存更低：同时只需要存 PP-1 个 micro-batch 的 activation
    GPipe: 需要存所有 micro-batch 的 activation 直到 backward
    1F1B: 最多存 PP-1 个
```

### Interleaved 1F1B（Megatron 的改进）

```
每个 stage 持有多个不连续的 layer chunks（如 stage 0 持有 layer 0-3 和 16-19）
好处：进一步减少 bubble（virtual pipeline stages 更多）
代价：增加通信次数（每个 chunk 之间需要 P2P）
```

### PP 的通信量

```
每次 P2P 传输: batch_size * seq_len * hidden_size * sizeof
例：batch=32, seq=2048, hidden=4096, fp16:
  = 32 * 2048 * 4096 * 2 = 512 MB per P2P

vs TP 的 AllReduce (896 MB per AllReduce, 每层 2 次):
  PP 通信量远小于 TP → PP 适合放 node 间
```

---

## 日检（20 分钟）

1. **闭卷画图**（5min）：画出 double buffering 的时间线（load 和 compute 的 overlap）
2. **口述**（5min）：Thread tiling 为什么能提高性能？Compute/Load ratio 怎么变化？
3. **口述**（5min）：1F1B 比 GPipe 好在哪里？Bubble ratio 一样吗？
4. **口述**（5min）：为什么 TP 放 node 内、PP 放 node 间？

---

## 参考资料

- leimao/CUDA-GEMM-Optimization (GitHub) - V00 到 V07 的完整优化路径
- CUTLASS documentation - Efficient GEMM design
- Efficient Large-Scale Language Model Training on GPU Clusters (Narayanan et al., 2021) - Megatron 3D parallelism
- GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism (Huang et al., 2019)
