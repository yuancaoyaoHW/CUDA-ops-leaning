# Tensor Core 与 GEMM

## 1. 学习目标

- 理解 Tensor Core 的硬件架构与支持的数据类型
- 掌握 GEMM（General Matrix Multiply，通用矩阵乘法）的数学定义与计算复杂度
- 理解 tiled GEMM 的分块策略与 double buffering
- 掌握 WMMA API 和 MMA PTX 指令的使用
- 能够分析 GEMM 的 roofline 位置与优化方向
- 理解 cuBLAS 和 CUTLASS 的设计思路

## 2. 前置知识

- GPU 内存层次（global → shared → register）
- Warp 执行模型
- 矩阵乘法的数学定义
- Arithmetic intensity 概念

## 3. 核心术语表

| 术语 | 英文全称 | 定义 |
|------|----------|------|
| Tensor Core | Tensor Core | 专用矩阵乘加硬件单元，执行 D = A×B + C |
| GEMM | General Matrix Multiply | C = α·A·B + β·C |
| WMMA | Warp Matrix Multiply Accumulate | CUDA C++ API for Tensor Core |
| MMA | Matrix Multiply Accumulate | PTX 级别的 Tensor Core 指令 |
| TFLOPS | Tera Floating-point Operations Per Second | 每秒万亿次浮点运算 |
| Arithmetic Intensity | Arithmetic Intensity | FLOPs / Bytes，计算密度 |
| Tiling | Tiling | 将大矩阵分块加载到 shared memory 的策略 |
| Double Buffering | Double Buffering | 使用两块 buffer 重叠加载与计算 |
| Epilogue | Epilogue | GEMM 后的融合操作（bias add、activation 等） |
| cuBLAS | CUDA Basic Linear Algebra Subroutines | NVIDIA 官方 BLAS 库 |
| CUTLASS | CUDA Templates for Linear Algebra Subroutines | NVIDIA 开源 GEMM 模板库 |

## 4. 动机

### 4.1 GEMM 在 LLM 中的地位

LLM 推理中 GEMM 占比：
- Attention: QKV projection (GEMM) + attention score (GEMM) + output projection (GEMM)
- FFN: up projection (GEMM) + gate projection (GEMM) + down projection (GEMM)
- 总计：每个 transformer layer 有 6-7 个 GEMM
- 对于 70B 模型，GEMM 占推理时间 > 80%

### 4.2 Tensor Core 的性能优势

| 硬件 | FP32 TFLOPS | FP16 Tensor Core TFLOPS | 加速比 |
|------|-------------|------------------------|--------|
| A100 | 19.5 | 312 | 16x |
| H100 | 67 | 989 (FP16) / 1979 (FP8) | 15-30x |

不使用 Tensor Core = 浪费 90%+ 的计算能力。

### 4.3 GEMM Shape 在 LLM 中的特点

| 阶段 | M | N | K | 特点 |
|------|---|---|---|------|
| Prefill (seq=2048) | 2048 | 4096 | 4096 | 大 M，compute-bound |
| Decode (batch=1) | 1 | 4096 | 4096 | M=1，memory-bound (GEMV) |
| Decode (batch=32) | 32 | 4096 | 4096 | 小 M，介于两者之间 |

## 5. 数学定义

### 5.1 GEMM 定义

```
C[M×N] = α · A[M×K] × B[K×N] + β · C[M×N]
```

计算量：`2 × M × N × K` FLOPs（每个输出元素需要 K 次乘加）

### 5.2 Arithmetic Intensity

```
AI = 2MNK / ((M×K + K×N + M×N) × sizeof(dtype))

// 对于方阵 M=N=K=n, FP16:
AI = 2n³ / (3n² × 2) = n/3

// n=4096: AI = 1365 FLOPs/Byte → 远超 roofline 拐点 → compute-bound
// n=1 (GEMV): AI = 2×1×N×K / ((K+N+N)×2) ≈ 1 → memory-bound
```

### 5.3 Tensor Core 操作

A100 Tensor Core 基本操作：
```
D[16×16] = A[16×16] × B[16×16] + C[16×16]  (FP16 → FP32)
```

一个 warp 执行一次 16×16×16 MMA：
- 输入：A (FP16, 16×16), B (FP16, 16×16)
- 输出：D (FP32, 16×16)
- 每个 thread 持有部分 fragment

## 6. 推导逻辑

### 6.1 Naive GEMM → Tiled GEMM

**Naive**：每个 thread 计算 C 的一个元素
```
for k in range(K):
    C[i][j] += A[i][k] * B[k][j]
```
问题：每个元素需要从 global memory 读 2K 个值，总读取 = 2MNK → 带宽不够

**Tiled**：将 A、B 分块加载到 shared memory
```
for tile in range(K / TILE_K):
    load A_tile[TILE_M × TILE_K] to shared memory
    load B_tile[TILE_K × TILE_N] to shared memory
    __syncthreads()
    compute partial C using shared memory
    __syncthreads()
```
数据复用：每个元素被 TILE_M 或 TILE_N 个 thread 共享

### 6.2 Double Buffering

```
Buffer A_smem[2][TILE_M][TILE_K]
Buffer B_smem[2][TILE_K][TILE_N]

// Iteration 0: load to buffer[0], compute nothing
load_global_to_shared(buffer[0], tile=0)

// Iteration i (i >= 1):
load_global_to_shared(buffer[i%2], tile=i)      // 加载下一块
compute_from_shared(buffer[(i-1)%2])             // 计算当前块
// 加载和计算重叠！
```

### 6.3 CUTLASS 层次结构

```
Grid Level:    Grid → Threadblock tiles (CTA tile)
Block Level:   Threadblock → Warp tiles
Warp Level:    Warp → MMA instruction tiles (16×16×16)
Thread Level:  Thread → Fragment (register)
```

典型配置（A100, FP16）：
- CTA tile: 128×256×64
- Warp tile: 64×64×64
- MMA tile: 16×8×16

## 7. 算子流程

### 7.1 WMMA API 使用

```cuda
#include <mma.h>
using namespace nvcuda;

__global__ void gemm_wmma(half* A, half* B, float* C, int M, int N, int K) {
    // 声明 fragment
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    
    // 初始化累加器
    wmma::fill_fragment(c_frag, 0.0f);
    
    // 计算 warp 负责的 tile 位置
    int warp_row = (blockIdx.y * blockDim.y + threadIdx.y) / 32 * 16;
    int warp_col = (blockIdx.x * blockDim.x + threadIdx.x) / 32 * 16;
    
    // 沿 K 维度累加
    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(a_frag, A + warp_row * K + k, K);
        wmma::load_matrix_sync(b_frag, B + k * N + warp_col, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // 写回
    wmma::store_matrix_sync(C + warp_row * N + warp_col, c_frag, N, wmma::mem_row_major);
}
```

### 7.2 cuBLAS 调用

```cuda
#include <cublas_v2.h>

cublasHandle_t handle;
cublasCreate(&handle);

float alpha = 1.0f, beta = 0.0f;
// 注意 cuBLAS 是 column-major
// C = A × B → cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, 
//                          &alpha, B, N, A, K, &beta, C, N)
cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
             N, M, K,
             &alpha,
             B, CUDA_R_16F, N,
             A, CUDA_R_16F, K,
             &beta,
             C, CUDA_R_32F, N,
             CUBLAS_COMPUTE_32F,
             CUBLAS_GEMM_DEFAULT_TENSOR_OP);
```

## 8. PyTorch baseline

```python
import torch
import torch.utils.benchmark as benchmark

def gemm_benchmark():
    shapes = [
        (1, 4096, 4096),      # decode batch=1 (GEMV)
        (32, 4096, 4096),     # decode batch=32
        (128, 4096, 4096),    # decode batch=128
        (2048, 4096, 4096),   # prefill seq=2048
        (4096, 4096, 4096),   # square
    ]
    
    for M, N, K in shapes:
        A = torch.randn(M, K, device='cuda', dtype=torch.float16)
        B = torch.randn(K, N, device='cuda', dtype=torch.float16)
        
        # Warmup
        for _ in range(10):
            C = torch.mm(A, B)
        torch.cuda.synchronize()
        
        # Benchmark
        timer = benchmark.Timer(
            stmt='torch.mm(A, B)',
            globals={'A': A, 'B': B, 'torch': torch}
        )
        result = timer.blocked_autorange()
        
        tflops = 2 * M * N * K / result.median / 1e12
        print(f"M={M:5d}, N={N}, K={K}: {result.median*1000:.3f} ms, {tflops:.1f} TFLOPS")

gemm_benchmark()
```

## 9. CUDA 实现思路

### 9.1 Naive GEMM（baseline）

```cuda
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

### 9.2 Tiled GEMM with Shared Memory

```cuda
#define TILE 32

__global__ void gemm_tiled(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;
    
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        // Cooperative loading
        if (row < M && t * TILE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;
            
        if (t * TILE + threadIdx.y < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        
        __syncthreads();
        
        for (int k = 0; k < TILE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        
        __syncthreads();
    }
    
    if (row < M && col < N)
        C[row * N + col] = sum;
}
```

### 9.3 优化层次

1. **Tiling** → 减少 global memory 访问
2. **Vectorized load** (float4) → 提高带宽利用
3. **Double buffering** → 重叠 load 和 compute
4. **Register tiling** → 每个 thread 计算多个输出元素
5. **Tensor Core (WMMA/MMA)** → 使用专用硬件
6. **Epilogue fusion** → 融合 bias/activation

## 10. Triton 实现思路

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 计算当前 block 的行列范围
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    
    # 指针
    A = a_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    B = b_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)
    
    # 累加器
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        a = tl.load(A, mask=...[truncated 7413 chars]