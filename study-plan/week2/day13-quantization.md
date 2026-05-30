# Day 13：量化基础（INT4 / FP8）

## 学习目标
- 理解 weight-only quantization 的完整流程
- 区分 GPTQ vs AWQ 的核心差异
- 理解 FP8 的两种格式及适用场景
- 实现 INT4 dequant kernel
- 理解 MoE Expert Parallelism

---

## 上午（3h）- 量化理论

### Weight-Only Quantization 流程

```
目标：把 fp16 权重压缩到 INT4/INT8，推理时 dequant 回 fp16 再计算

Step 1: Calibration
  用少量数据（128-256 samples）跑 forward
  收集每层 activation 的统计信息（用于 AWQ）
  或收集 Hessian 信息（用于 GPTQ）

Step 2: 计算 scale 和 zero_point
  Per-tensor: 整个权重矩阵一个 scale → 精度最差，速度最快
  Per-channel: 每个 output channel 一个 scale → 精度好
  Per-group (group_size=128): 每 128 个权重一个 scale → 精度最好

  对称量化: w_int = round(w / scale), scale = max(|w|) / (2^(bits-1) - 1)
  非对称量化: w_int = round((w - zero_point) / scale)

Step 3: Quantize
  w_int4 = clamp(round(w_fp16 / scale), -8, 7)  # INT4 范围 [-8, 7]

Step 4: Pack
  8 个 INT4 值 pack 到 1 个 INT32:
  packed = (w0 & 0xF) | ((w1 & 0xF) << 4) | ... | ((w7 & 0xF) << 28)

推理时 Dequant:
  w_fp16 = (unpack(packed) - zero_point) * scale
  output = input @ w_fp16  (实际在 kernel 内部 fuse)
```

### GPTQ vs AWQ

```
GPTQ (Generalized Post-Training Quantization):
  核心 idea: Layer-by-layer reconstruction
  对每层:
    1. 用 calibration data 计算该层的 Hessian: H = X^T @ X
    2. 按 Hessian 对角线排序（重要性），逐列量化
    3. 量化一列后，用 Hessian 信息补偿其他列的误差
    4. 最小化: ||W @ X - Q(W) @ X||^2

  优点: 精度高（有误差补偿）
  缺点:
    - 慢（需要逐列处理 + 矩阵运算）
    - 可能 overfit calibration data（reconstruction 针对特定输入优化）
    - 7B 模型量化需要 ~1 小时

AWQ (Activation-Aware Weight Quantization):
  核心 idea: 保护重要权重
  观察: 1% 的 "salient" 权重对精度影响巨大
  方法:
    1. 用 calibration data 计算 activation magnitude: s = mean(|X|, dim=0)
    2. s 大的 channel → 对应的权重更重要
    3. 对重要权重用更大的 scale 保护:
       w_scaled = w * s^α  (α 通过搜索确定，通常 0.5-1.0)
       量化 w_scaled，推理时除以 s^α 补偿
    4. 等价于给重要权重更细的量化粒度

  优点:
    - 快（只需要一次 forward 收集统计）
    - 泛化好（不 overfit calibration data）
    - 7B 模型量化 ~10 分钟
  缺点:
    - 极低 bit（2-bit）时精度不如 GPTQ

面试常问对比:
  | 维度 | GPTQ | AWQ |
  |------|------|-----|
  | 方法 | 逐层 reconstruction | Activation-aware scaling |
  | 速度 | 慢（~1h for 7B） | 快（~10min for 7B） |
  | 精度 | 略高（4-bit） | 略低但泛化好 |
  | Overfit | 可能 | 不会 |
  | 适用 | 固定 prompt 场景 | 通用场景 |
```

### FP8 量化

```
两种 FP8 格式:

E4M3 (4 bit exponent, 3 bit mantissa):
  范围: ±448
  精度: ~3.6 位有效数字
  用途: 推理时的权重和激活（分布集中，不需要大范围）

E5M2 (5 bit exponent, 2 bit mantissa):
  范围: ±57344
  精度: ~2.5 位有效数字
  用途: 训练时的梯度（需要大动态范围）

FP8 vs INT4 的本质区别:
  INT4: 存储 4-bit，计算时 dequant 到 fp16，Tensor Core 做 fp16 乘法
        → 省内存和带宽，但不省计算
  FP8:  存储 8-bit，Tensor Core 直接做 FP8 乘法（H100+）
        → 省内存、带宽、AND 计算（2x throughput vs fp16）

H100 的 FP8 支持:
  FP8 Tensor Core: 1978 TFLOPS (vs fp16 989 TFLOPS) → 2x!
  不需要 dequant → 没有 unpack overhead
  精度损失很小（< 0.1% perplexity increase for most models）
```

---

## 下午（2h）- INT4 Dequant Kernel

### Triton 实现

```python
import triton
import triton.language as tl

@triton.jit
def int4_dequant_gemv_kernel(
    # 输入
    X_ptr,        # [K] fp16 input vector
    W_ptr,        # [N, K//8] int32 packed weights
    Scale_ptr,    # [N, K//group_size] fp16 scales
    Zero_ptr,     # [N, K//group_size] fp16 zero points
    # 输出
    Y_ptr,        # [N] fp16 output
    # 维度
    N, K,
    group_size: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    n_offset = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offset < N

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offset = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offset < K

        # 加载 input vector
        x = tl.load(X_ptr + k_offset, mask=k_mask, other=0.0).to(tl.float32)

        # 加载 packed weights 并 unpack INT4
        # 每个 int32 包含 8 个 INT4 值
        pack_idx = k_offset // 8  # 哪个 int32
        bit_idx = (k_offset % 8) * 4  # 在 int32 中的 bit 位置

        packed = tl.load(
            W_ptr + n_offset[:, None] * (K // 8) + pack_idx[None, :],
            mask=n_mask[:, None] & k_mask[None, :],
            other=0
        )

        # Unpack: 取出 4 bits
        w_int4 = (packed >> bit_idx[None, :]) & 0xF  # [BLOCK_N, BLOCK_K]
        # 转为有符号: 0-7 → 0-7, 8-15 → -8 to -1
        w_int4 = tl.where(w_int4 > 7, w_int4 - 16, w_int4)

        # 加载 scale 和 zero_point
        group_idx = k_offset // group_size
        scale = tl.load(
            Scale_ptr + n_offset[:, None] * (K // group_size) + group_idx[None, :],
            mask=n_mask[:, None] & k_mask[None, :],
            other=1.0
        ).to(tl.float32)
        zero = tl.load(
            Zero_ptr + n_offset[:, None] * (K // group_size) + group_idx[None, :],
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0
        ).to(tl.float32)

        # Dequantize
        w_fp = (w_int4.to(tl.float32) - zero) * scale  # [BLOCK_N, BLOCK_K]

        # GEMV: acc += w_fp @ x
        acc += tl.sum(w_fp * x[None, :], axis=1)

    # 写回
    tl.store(Y_ptr + n_offset, acc.to(tl.float16), mask=n_mask)
```

### Benchmark

```python
# 对比: INT4 dequant GEMV vs fp16 GEMV
N, K = 4096, 4096  # 典型 LLM hidden size
x = torch.randn(K, device='cuda', dtype=torch.float16)
w_fp16 = torch.randn(N, K, device='cuda', dtype=torch.float16)

# 量化
scale = w_fp16.abs().max(dim=1, keepdim=True).values / 7.0
w_int4 = torch.clamp(torch.round(w_fp16 / scale), -8, 7).to(torch.int8)
# Pack...

# Benchmark
t_fp16 = triton.testing.do_bench(lambda: w_fp16 @ x)
t_int4 = triton.testing.do_bench(lambda: int4_dequant_gemv(x, w_packed, scale, zero))

# INT4 应该更快（读取数据量减少 4x）
# 但 dequant overhead 会吃掉一部分收益
# 实际加速: ~2-3x for memory-bound GEMV (decode)
```

---

## 晚上（1.5h）- MoE Expert Parallelism

### MoE 基本结构

```
标准 MLP: output = FFN(input)  → 所有 token 用同一个 FFN

MoE MLP:
  1. Router: scores = softmax(input @ router_weight)  → [num_tokens, num_experts]
  2. Top-K: 选择 top-k experts (通常 k=2)
  3. Dispatch: 把每个 token 发送到对应的 expert
  4. Compute: 每个 expert 独立计算
  5. Combine: 加权合并结果

优势: 参数量大（多个 expert），但每个 token 只激活 k 个 → 计算量不变
例: DeepSeek-V3: 671B 参数，但每个 token 只用 37B 激活参数
```

### Expert Parallelism

```
把 experts 分布到不同 GPU:
  GPU 0: Expert 0, 1, 2, 3
  GPU 1: Expert 4, 5, 6, 7
  ...

通信模式: All-to-All
  Dispatch (token → expert):
    每个 GPU 把需要发给其他 GPU 的 token 打包发送
    通信量 = num_tokens * hidden_size * sizeof * (被路由到其他 GPU 的比例)

  Combine (expert → token):
    每个 GPU 把计算结果发回原始 GPU
    通信量 = 同上

  总通信量 ≈ 2 * num_tokens * hidden_size * sizeof
  （假设 token 均匀分布到所有 GPU）

Load Balancing 问题:
  如果大部分 token 都路由到同一个 expert → 该 GPU 过载
  解决:
    1. Auxiliary loss: 训练时加 loss 鼓励均匀分配
    2. Capacity factor: 每个 expert 最多处理 CF * (N/E) 个 token
    3. 溢出 token 被丢弃或路由到其他 expert
```

### DeepSeek-V3 的 MoE 设计

```
256 个 routed experts + 1 个 shared expert
每个 token 选 top-8 experts
Expert Parallelism: 256 experts 分布到多个 GPU

特殊优化:
  - FP8 训练: 权重和激活都用 FP8 → 2x 训练速度
  - Auxiliary-loss-free load balancing: 不用 aux loss，用 bias 调节
  - Multi-Token Prediction (MTP): 一次预测多个 token
```

---

## 日检（20 分钟）

1. **口述**（5min）：GPTQ vs AWQ 的核心区别？各自适合什么场景？
2. **口述**（5min）：FP8 E4M3 vs E5M2 分别用在哪？为什么 FP8 能 2x throughput？
3. **闭卷写**（10min）：INT4 unpack 的逻辑（从 int32 中取出第 i 个 INT4 值）

---

## 参考资料

- GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers (Frantar et al., 2023)
- AWQ: Activation-aware Weight Quantization (Lin et al., 2023)
- FP8 Formats for Deep Learning (Micikevicius et al., 2022)
- DeepSeek-V3 Technical Report
