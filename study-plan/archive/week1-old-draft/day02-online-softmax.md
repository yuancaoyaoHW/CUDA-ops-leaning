# Day 2：Online Softmax

## 学习目标
- 手推 online softmax 的数学推导
- 实现 Triton + CUDA 两个版本
- 用 Roofline 分析证明 softmax 是 memory-bound
- 理解 ZeRO-1/2/3 的内存模型

---

## 上午（3h）- 实现

### 数学推导（必须能闭卷手推）

标准 softmax（三遍扫描）：
```
Pass 1: m = max(x_1, x_2, ..., x_N)
Pass 2: d = sum(exp(x_i - m))
Pass 3: y_i = exp(x_i - m) / d
```

Online softmax（一遍扫描）：
```
初始化: m = -inf, d = 0
对每个 x_i:
    m_new = max(m, x_i)
    d = d * exp(m - m_new) + exp(x_i - m_new)  // 关键：rescale 旧的 d
    m = m_new
最终: y_i = exp(x_i - m) / d
```

为什么 rescaling 正确：
```
旧的 d = sum_{j<i}(exp(x_j - m_old))
新的 d 应该 = sum_{j<=i}(exp(x_j - m_new))
     = sum_{j<i}(exp(x_j - m_new)) + exp(x_i - m_new)
     = sum_{j<i}(exp(x_j - m_old) * exp(m_old - m_new)) + exp(x_i - m_new)
     = d_old * exp(m_old - m_new) + exp(x_i - m_new)  ✓
```

### Triton 实现（row-wise softmax）

```python
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    n_cols,
    input_row_stride, output_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # 加载一行
    row = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=-float('inf'))

    # Online softmax: 求 max
    row_max = tl.max(row, axis=0)

    # 求 exp 和 sum
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)

    # 归一化
    output = numerator / denominator
    tl.store(output_ptr + row_idx * output_row_stride + col_offsets, output, mask=mask)
```

注意：上面是 Triton 的简化版（Triton 内部 `tl.max` 和 `tl.sum` 已经是高效 reduction）。真正的 online 版本在 FlashAttention 中更关键。

### CUDA 实现

```cuda
__global__ void softmax_kernel(float* input, float* output, int N, int D) {
    int row = blockIdx.x;
    extern __shared__ float smem[];

    float local_max = -INFINITY;
    float local_sum = 0.0f;

    // Pass 1+2: online max + sum
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float val = input[row * D + i];
        if (val > local_max) {
            local_sum = local_sum * expf(local_max - val) + 1.0f;
            local_max = val;
        } else {
            local_sum += expf(val - local_max);
        }
    }

    // Block-level reduction for max and sum (需要 warp shuffle)
    // ... (用 Day 1 学的 reduction 技巧)

    // Pass 3: normalize
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        output[row * D + i] = expf(input[row * D + i] - global_max) / global_sum;
    }
}
```

### Benchmark

测试配置：
- Shape: [batch, seq_len] 其中 batch=32
- seq_len = 128, 512, 2048, 8192
- 对比 torch.softmax

记录：执行时间 (us)、achieved bandwidth (GB/s)、vs torch 的加速比

---

## 下午（2h）- Roofline 分析

### Softmax 的 Arithmetic Intensity

```
输入: N 个 fp16 元素 → 读 2N bytes
输出: N 个 fp16 元素 → 写 2N bytes
总数据移动: 4N bytes

计算:
  - N 次 max 比较: N FLOPs
  - N 次 exp: ~N FLOPs (特殊函数，实际更贵但按 1 FLOP 算)
  - N 次加法 (sum): N FLOPs
  - N 次除法: N FLOPs
  总计: ~5N FLOPs (保守估计)

Arithmetic Intensity = 5N / 4N = 1.25 FLOP/byte
```

### RTX 4060 Laptop 的 Roofline

```
Memory Bandwidth: ~256 GB/s
FP32 Peak: ~11 TFLOPS
FP16 Peak (Tensor Core): ~176 TFLOPS

Ridge Point (FP16): 176000 / 256 ≈ 687 FLOP/byte
Ridge Point (FP32): 11000 / 256 ≈ 43 FLOP/byte

Softmax AI = 1.25 FLOP/byte
  → 远低于 ridge point
  → 严重 memory-bound
  → 理论峰值性能 = 1.25 * 256 = 320 GFLOPS (远低于 compute peak)
  → 优化方向：减少内存访问（fuse），而非增加计算效率
```

### Nsight Compute 验证

```bash
ncu --set full --target-processes all -o reports/softmax_ncu \
    python bench_softmax.py
```

检查指标：
- `dram__bytes_read.sum` + `dram__bytes_write.sum` → 实际数据移动
- `sm__throughput.avg.pct_of_peak_sustained_elapsed` → compute 利用率（应该很低）
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` → memory 利用率（应该接近 100%）

### 面试关键结论

> "Softmax 是 memory-bound kernel，AI 约 1.25 FLOP/byte，远低于 GPU 的 ridge point。
> 单独优化 softmax kernel 的意义有限（已经接近 bandwidth 峰值）。
> 真正的优化是把 softmax fuse 到 attention 里（FlashAttention），避免把 N×N 的 attention matrix 写回 HBM。"

---

## 晚上（1.5h）- 分布式：ZeRO 内存模型

### 混合精度训练的内存占用

假设模型参数量 = Φ（个数，不是 bytes）

不用任何优化时，每个 GPU 需要存储：
```
fp16 parameters:     2Φ bytes
fp16 gradients:      2Φ bytes
fp32 master params:  4Φ bytes  (optimizer 需要)
fp32 momentum:       4Φ bytes  (Adam)
fp32 variance:       4Φ bytes  (Adam)
─────────────────────────────
总计:                16Φ bytes

例：7B 模型 → 16 * 7G = 112 GB per GPU（单卡放不下）
```

### ZeRO 各阶段

```
ZeRO-1（分片 optimizer states）:
  每卡: 2Φ + 2Φ + 12Φ/N = 4Φ + 12Φ/N bytes
  通信: AllReduce gradients = 2*(N-1)/N * 2Φ bytes（和 DDP 一样）
  例：7B, 8 GPU → 4*7 + 12*7/8 = 28 + 10.5 = 38.5 GB

ZeRO-2（分片 optimizer + gradients）:
  每卡: 2Φ + (2Φ + 12Φ)/N bytes
  通信: ReduceScatter gradients = (N-1)/N * 2Φ bytes（比 DDP 少一半）
  例：7B, 8 GPU → 2*7 + 14*7/8 = 14 + 12.25 = 26.25 GB

ZeRO-3（全分片）:
  每卡: (2Φ + 2Φ + 12Φ)/N = 16Φ/N bytes
  通信: Forward AllGather + Backward AllGather + ReduceScatter ≈ 3*(N-1)/N * 2Φ
  例：7B, 8 GPU → 16*7/8 = 14 GB（可以放下！）
  代价：通信量约 1.5x DDP
```

### 关键 tradeoff（面试常问）

| 方案 | 内存 | 通信量 | 适用场景 |
|------|------|--------|----------|
| DDP | 16Φ | 2*(N-1)/N * 2Φ | 单卡能放下模型 |
| ZeRO-1 | 4Φ + 12Φ/N | 同 DDP | 模型大但不至于放不下 |
| ZeRO-2 | 2Φ + 14Φ/N | 0.5x DDP | 中等模型 |
| ZeRO-3 | 16Φ/N | 1.5x DDP | 超大模型，单卡放不下 |
| FSDP | ≈ ZeRO-3 | ≈ ZeRO-3 | PyTorch 原生方案 |

---

## 日检（20 分钟）

1. **闭卷手推**（5min）：写出 online softmax 的 rescaling 公式，解释为什么正确
2. **闭卷手写**（10min）：写出 Triton softmax kernel（row-wise）
3. **口述**（5min）：
   - Softmax 是 compute-bound 还是 memory-bound？AI 是多少？
   - 怎么优化？（答：fuse 到 attention 里）
   - ZeRO-3 比 DDP 通信量多多少？为什么值得？（答：1.5x，但内存从 16Φ 降到 16Φ/N）

---

## 参考资料

- Online normalizer calculation for softmax (Milakov & Gimelshein, 2018)
- Triton tutorial: Fused Softmax
- ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (Rajbhandari et al., 2020)
