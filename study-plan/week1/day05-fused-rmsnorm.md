# Day 5：Fused RMSNorm + Residual

## 学习目标
- 实现 fused RMSNorm + residual add kernel
- 理解 kernel fusion 对 memory-bound kernel 的加速原理
- 能量化分析 fuse 前后的内存访问次数
- 理解 FSDP 的通信模式

---

## 上午（3h）- 实现

### RMSNorm 数学

```
RMSNorm(x) = x * weight / sqrt(mean(x^2) + eps)

展开：
  rms = sqrt(sum(x_i^2) / n + eps)
  output_i = x_i * weight_i / rms
           = x_i * weight_i * rsqrt(sum(x_i^2) / n + eps)
```

### 为什么要 Fuse（面试必答题）

Transformer 中的典型模式：
```python
# 不 fuse（PyTorch 默认）
residual = hidden_states + residual          # kernel 1: element-wise add
hidden_states = rms_norm(residual) * weight  # kernel 2+3: reduce + normalize
```

不 fuse 的内存访问分析：
```
Kernel 1 (residual add):
  读: hidden_states (2N bytes) + residual (2N bytes)
  写: residual (2N bytes)
  → 6N bytes

Kernel 2 (RMSNorm):
  读: residual (2N bytes) + weight (2N bytes，但可以 cache)
  写: output (2N bytes)
  内部还需要: 读 residual 算 sum(x^2)，再读一次做 normalize
  → 至少 6N bytes

总计: ≥ 12N bytes per element（多次读写 global memory）
```

Fuse 后：
```
Fused kernel:
  读: hidden_states (2N) + old_residual (2N) + weight (2N)
  写: new_residual (2N) + output (2N)
  → 10N bytes

  但关键是：residual 只从 global memory 读一次！
  在 register 中完成: add → compute rms → normalize
  实际: 读 6N + 写 4N = 10N bytes（vs 不 fuse 的 ≥12N）

  更重要的是减少了 kernel launch overhead 和 memory round-trip
```

### Triton 实现

```python
import torch
import triton
import triton.language as tl

@triton.jit
def fused_rmsnorm_residual_kernel(
    # 输入指针
    X_ptr,           # 当前层输出 [M, N]
    Residual_ptr,    # 旧 residual [M, N]
    Weight_ptr,      # RMSNorm weight [N]
    # 输出指针
    Out_ptr,         # normalized output [M, N]
    NewResidual_ptr, # 新 residual [M, N]
    # 维度
    stride_m,        # row stride
    N: tl.constexpr, # hidden size
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # 每个 program 处理一行
    row = tl.program_id(0)
    row_offset = row * stride_m

    # 加载一行数据
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(X_ptr + row_offset + cols, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(Residual_ptr + row_offset + cols, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(Weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # Step 1: Residual add（在 register 中完成）
    hidden = x + residual

    # Step 2: 保存新 residual
    tl.store(NewResidual_ptr + row_offset + cols, hidden.to(tl.float16), mask=mask)

    # Step 3: RMSNorm（在 register 中完成）
    variance = tl.sum(hidden * hidden, axis=0) / N
    rrms = 1.0 / tl.sqrt(variance + eps)
    normed = hidden * rrms

    # Step 4: 乘 weight
    out = normed * weight

    # Step 5: 写回
    tl.store(Out_ptr + row_offset + cols, out.to(tl.float16), mask=mask)


def fused_rmsnorm_residual(x, residual, weight, eps=1e-6):
    M, N = x.shape
    out = torch.empty_like(x)
    new_residual = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(N)
    assert BLOCK_SIZE <= 8192, "hidden size too large for single-block kernel"

    grid = (M,)
    fused_rmsnorm_residual_kernel[grid](
        x, residual, weight, out, new_residual,
        x.stride(0), N, eps, BLOCK_SIZE,
    )
    return out, new_residual
```

### 测试正确性

```python
def test_fused_rmsnorm_residual():
    M, N = 2048, 4096
    x = torch.randn(M, N, device='cuda', dtype=torch.float16)
    residual = torch.randn(M, N, device='cuda', dtype=torch.float16)
    weight = torch.randn(N, device='cuda', dtype=torch.float16)

    # Reference
    ref_residual = x + residual
    ref_rms = torch.sqrt(ref_residual.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    ref_out = (ref_residual.float() / ref_rms * weight.float()).half()

    # Fused
    out, new_res = fused_rmsnorm_residual(x, residual, weight)

    torch.testing.assert_close(new_res, ref_residual.half(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)
```

---

## 下午（2h）- Benchmark + Profiling

### Benchmark 脚本

```python
import torch
import triton

def benchmark_unfused(x, residual, weight, eps=1e-6):
    new_residual = x + residual
    variance = new_residual.float().pow(2).mean(-1, keepdim=True)
    rrms = torch.rsqrt(variance + eps)
    out = (new_residual.float() * rrms * weight.float()).half()
    return out, new_residual

# 测试不同 hidden_size
for N in [1024, 2048, 4096, 8192]:
    M = 4096  # batch * seq_len
    x = torch.randn(M, N, device='cuda', dtype=torch.float16)
    residual = torch.randn(M, N, device='cuda', dtype=torch.float16)
    weight = torch.randn(N, device='cuda', dtype=torch.float16)

    # Benchmark
    t_fused = triton.testing.do_bench(lambda: fused_rmsnorm_residual(x, residual, weight))
    t_unfused = triton.testing.do_bench(lambda: benchmark_unfused(x, residual, weight))

    # 计算 achieved bandwidth
    # 读: x + residual + weight = (2M*N + 2M*N + 2N) bytes
    # 写: out + new_residual = (2M*N + 2M*N) bytes
    total_bytes = (4 * M * N + 2 * N + 4 * M * N) * 1  # 简化
    bw_fused = total_bytes / (t_fused * 1e-3) / 1e9  # GB/s

    print(f"N={N}: fused={t_fused:.2f}ms, unfused={t_unfused:.2f}ms, "
          f"speedup={t_unfused/t_fused:.2f}x, BW={bw_fused:.0f} GB/s")
```

### 预期结果

```
N=1024: speedup ≈ 1.3-1.5x
N=2048: speedup ≈ 1.3-1.5x
N=4096: speedup ≈ 1.4-1.6x
N=8192: speedup ≈ 1.4-1.6x

Achieved BW 应该接近 200-240 GB/s（4060 峰值 ~256 GB/s）
如果远低于峰值 → 检查是否有 bank conflict 或 occupancy 问题
```

### Nsight Compute 验证

```bash
ncu --set full -o reports/rmsnorm_fused python bench_rmsnorm.py --fused
ncu --set full -o reports/rmsnorm_unfused python bench_rmsnorm.py --unfused
```

对比：
- fused 的 `dram__bytes.sum` 应该明显小于 unfused
- fused 的 `dram__throughput` 应该更接近峰值

---

## 晚上（1.5h）- 分布式：FSDP

### FSDP 原理

FSDP (Fully Sharded Data Parallel) = PyTorch 原生的 ZeRO-3 实现。

```
核心思想：
  - 每个 GPU 只存 1/N 的 parameters + gradients + optimizer states
  - 需要用的时候 AllGather 拿到完整参数
  - 用完立即释放

Forward pass (对每个 layer):
  1. AllGather: 从其他 GPU 收集完整参数 → 通信 (N-1)/N * param_size
  2. Compute: 用完整参数做 forward
  3. 释放非本地参数（只保留 1/N）

Backward pass (对每个 layer):
  1. AllGather: 再次收集完整参数 → 通信 (N-1)/N * param_size
  2. Compute: 计算梯度
  3. ReduceScatter: 每个 GPU 只保留 1/N 的梯度 → 通信 (N-1)/N * grad_size
  4. 释放完整参数和非本地梯度
```

### 通信量对比

```
DDP:
  Backward: AllReduce gradients = 2 * (N-1)/N * 2Φ
  总计: 4*(N-1)/N * Φ bytes

FSDP:
  Forward: AllGather params = (N-1)/N * 2Φ per layer * L layers
  Backward: AllGather params + ReduceScatter grads = 2*(N-1)/N * 2Φ per layer * L layers
  总计: 3*(N-1)/N * 2Φ = 6*(N-1)/N * Φ bytes

FSDP / DDP = 6/4 = 1.5x 通信量
但内存从 16Φ 降到 16Φ/N
```

### FSDP vs DDP 选择（面试常问）

```
用 DDP: 模型能放进单卡（参数 + 梯度 + optimizer < GPU memory）
用 FSDP: 模型放不进单卡，需要分片
用 TP: 单层太大放不进单卡（hidden_size 很大），或需要减少 latency
用 PP: 模型很深，TP 的通信太频繁

实际组合（大模型训练）:
  TP=8 (node 内) + PP=4 (跨 node) + FSDP (跨 PP group)
  总 GPU = 8 * 4 * DP_size
```

---

## 日检（20 分钟）

1. **闭卷手写**（10min）：写出 fused RMSNorm + residual 的 Triton kernel 核心逻辑
   - 通过标准：包含 residual add、variance 计算、rsqrt、weight multiply

2. **口述**（5min）：Kernel fusion 为什么能加速 memory-bound kernel？具体省了哪些内存访问？
   - 期望答案：减少 global memory round-trip，数据在 register 中完成多步计算，从 12N→10N bytes

3. **口述**（5min）：FSDP 的通信量比 DDP 多多少？为什么值得？
   - 期望答案：1.5x，因为内存从 16Φ 降到 16Φ/N，使得超大模型可以训练

---

## 参考资料

- Triton tutorial: Layer Normalization
- vLLM 源码: `vllm/model_executor/layers/layernorm.py`
- PyTorch FSDP tutorial
- ZeRO paper (Rajbhandari et al., 2020)
