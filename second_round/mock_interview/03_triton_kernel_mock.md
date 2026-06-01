# 模拟面试 3：45 分钟 Triton Kernel 优化

## 面试官 Profile

- **公司**：某 AI 公司（对标 OpenAI / xAI），大量使用 Triton 做 custom kernel
- **级别**：Senior ML Infra Engineer，Triton 早期用户，贡献过 Triton compiler patches
- **风格**：学术+工程结合，喜欢讨论编程模型的设计哲学。会问"为什么 Triton 这样设计"而不只是"怎么用"。
- **偏好**：看重对 compiler 行为的理解，不只是写出能跑的代码。

---

## Opening Question

> "用 Triton 实现一个 fused softmax kernel，要求支持任意 sequence length。然后讨论：Triton 的 auto-tuning 机制是怎么工作的？你会怎么设置 tuning config？"

---

## Candidate Expected Answer（基于候选人真实经验）

**候选人诚实回应**：
"我对 Triton 的了解主要来自学习材料和 vLLM 中使用 Triton kernel 的经验，但我没有从零写过 production-level 的 Triton kernel。

我理解 Triton 的核心编程模型：
- Block-level programming：每个 program instance 处理一个数据 block
- 自动处理 memory coalescing 和 shared memory 管理
- 通过 `tl.load` / `tl.store` 做向量化内存访问
- Auto-tuning 通过 `@triton.autotune` decorator 搜索最优配置

Fused softmax 的 Triton 实现思路：
```python
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)
```

这个实现的限制是 BLOCK_SIZE 必须 >= n_cols，对于很长的 sequence 需要分块处理。"

---

## Weak Answer（常见错误）

"Triton 就是 Python 写 GPU kernel 的工具，比 CUDA 简单。直接用 `tl.softmax` 就行了... 等等，好像没有这个函数。那就用 numpy 的方式写，Triton 会自动优化。"

**为什么弱**：
- 不理解 Triton 的编程模型（不是 numpy）
- 不知道 Triton 没有内置 softmax
- 不理解 block-level programming 的含义
- 对 auto-tuning 完全不了解

---

## Strong Answer（工程深度 + 数据 + tradeoff）

"让我分几个层次来实现和讨论：

**1. 基础 Fused Softmax**

```python
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
    ],
    key=['n_cols'],
)
@triton.jit
def fused_softmax_kernel(
    output_ptr, input_ptr,
    input_row_stride, output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # 每个 program instance 处理一行
    row_idx = tl.program_id(0)
    row_start = input_ptr + row_idx * input_row_stride
    
    # 处理 n_cols > BLOCK_SIZE 的情况：多次迭代
    # Pass 1: online max
    m = tl.full([1], value=-float('inf'), dtype=tl.float32)
    for start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        x = tl.load(row_start + col_offsets, mask=mask, other=-float('inf'))
        m = tl.maximum(m, tl.max(x, axis=0))
    
    # Pass 2: sum of exp
    d = tl.zeros([1], dtype=tl.float32)
    for start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        x = tl.load(row_start + col_offsets, mask=mask, other=-float('inf'))
        d += tl.sum(tl.exp(x - m), axis=0)
    
    # Pass 3: normalize and store
    out_start = output_ptr + row_idx * output_row_stride
    for start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        x = tl.load(row_start + col_offsets, mask=mask, other=-float('inf'))
        result = tl.exp(x - m) / d
        tl.store(out_start + col_offsets, result, mask=mask)
```

**2. Online Softmax 优化（2-pass）**

```python
@triton.jit
def fused_softmax_online_kernel(
    output_ptr, input_ptr,
    input_row_stride, output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = input_ptr + row_idx * input_row_stride
    
    # Pass 1: online softmax reduction (max + sum in one pass)
    m = tl.full([1], value=-float('inf'), dtype=tl.float32)
    d = tl.zeros([1], dtype=tl.float32)
    
    for start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        x = tl.load(row_start + col_offsets, mask=mask, other=-float('inf'))
        block_max = tl.max(x, axis=0)
        new_m = tl.maximum(m, block_max)
        # 修正之前的 sum
        d = d * tl.exp(m - new_m) + tl.sum(tl.exp(x - new_m), axis=0)
        m = new_m
    
    # Pass 2: normalize
    out_start = output_ptr + row_idx * output_row_stride
    for start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        x = tl.load(row_start + col_offsets, mask=mask, other=-float('inf'))
        result = tl.exp(x - m) / d
        tl.store(out_start + col_offsets, result, mask=mask)
```

从 3-pass 优化到 2-pass：减少一次 global memory 读取，对 memory-bound kernel 约 33% 性能提升。

**3. Auto-tuning 机制**

Triton 的 auto-tuning 工作原理：
- `@triton.autotune` 在首次调用时，对每个 config 运行 kernel 并计时
- 选择最快的 config 缓存起来，后续调用直接使用
- `key` 参数指定哪些输入参数变化时需要重新 tune

Config 设计考虑：
- `BLOCK_SIZE`：影响每个 program 处理的数据量和 register 压力
  - 小 BLOCK_SIZE（256-512）：低 register 压力，高 occupancy，但循环次数多
  - 大 BLOCK_SIZE（2048-4096）：减少循环，但 register 压力大，可能降低 occupancy
- `num_warps`：影响并行度
  - 小 num_warps（4）：适合 BLOCK_SIZE 小的情况
  - 大 num_warps（16-32）：适合 BLOCK_SIZE 大的情况，更多并行
- `num_stages`：pipeline stages，影响 prefetch 深度

**4. Triton vs CUDA 的 tradeoff**

| 维度 | Triton | CUDA |
|------|--------|------|
| 开发效率 | 高（Python-like，自动处理很多细节） | 低（需要手动管理所有细节） |
| 性能上限 | 通常达到 CUDA 的 80-95% | 100%（手动优化极限） |
| Shared memory | 编译器自动管理 | 手动分配和管理 |
| Memory coalescing | 编译器自动优化 | 需要手动确保 |
| Warp-level control | 有限（无法直接用 shuffle） | 完全控制 |
| 适用场景 | Element-wise, reduction, attention | 需要极致优化的 GEMM, 复杂 memory pattern |

**什么时候必须用 CUDA**：
- 需要 warp-level primitives（如 cooperative groups）
- 需要精确控制 shared memory layout（如 swizzle for bank conflict）
- 需要 inline PTX（如 TMA on Hopper）
- 性能差距 > 10% 且是 critical path

**什么时候 Triton 更好**：
- 快速原型验证
- Fusion kernel（多个 element-wise ops 合并）
- 中等复杂度的 reduction/attention
- 需要跨硬件可移植性"

---

## Follow-up Chain（5 层递进追问）

### Follow-up 1：Triton Compiler 内部
> "Triton 编译器是怎么把你的 Python 代码变成高效 GPU code 的？中间经过哪些 pass？"

**期望回答**：
- Triton IR → TTIR（Triton Tensor IR）→ TTGIR（Triton GPU IR）→ LLVM IR → PTX → CUBIN
- 关键 optimization passes：
  - Coalesce：确保 memory access 是 coalesced 的
  - Pipeline：插入 async copy 做 prefetch
  - Allocation：分配 shared memory
  - Layout conversion：选择最优的数据 layout
- Triton 的核心创新：block-level 抽象让编译器有更多优化空间

### Follow-up 2：Auto-tuning 的局限
> "Auto-tuning 在 production 中有什么问题？首次调用的 warmup 时间怎么处理？"

**期望回答**：
- Warmup 问题：首次调用需要尝试所有 config，可能耗时数秒
- 解决方案：
  1. Offline tuning：提前在目标硬件上 tune，保存结果
  2. Cache persistence：将 tuning 结果写入文件，下次启动直接加载
  3. Shape bucketing：将相近 shape 归为一组，共享 tuning 结果
- 其他局限：
  - 搜索空间爆炸：config 组合太多时 tuning 时间过长
  - 硬件依赖：不同 GPU 型号需要重新 tune
  - 动态 shape：每个新 shape 都需要 tune

### Follow-up 3：Triton 实现 FlashAttention
> "能用 Triton 实现 FlashAttention 吗？和 CUDA 版本相比性能差多少？瓶颈在哪？"

**期望回答**：
- 可以实现，Triton 官方 tutorial 就有 FlashAttention 示例
- 性能对比：Triton FlashAttention 约为 CUDA FlashAttention-2 的 80-90%
- 性能差距来源：
  1. Triton 无法精确控制 shared memory layout → 可能有 bank conflict
  2. 无法使用 TMA（Tensor Memory Accelerator）on Hopper
  3. Warp specialization 在 Triton 中难以表达
  4. 编译器可能生成次优的 register allocation
- 但 Triton 版本的优势：代码量少 10x，易于修改和实验

### Follow-up 4：Kernel Fusion 策略
> "在 LLM inference 中，哪些 ops 适合用 Triton 做 fusion？fusion 的收益怎么估算？"

**期望回答**：
- 适合 fusion 的 pattern：
  1. LayerNorm + Residual Add：两个 element-wise ops，fusion 后只读写一次
  2. GELU/SiLU activation + 前后的 element-wise ops
  3. RoPE（Rotary Position Embedding）：涉及 sin/cos 计算 + element-wise multiply
  4. Attention 中的 softmax + dropout + matmul（FlashAttention 就是这个）
- 收益估算：
  - 如果 ops 是 memory-bound：fusion 收益 ≈ 减少的 memory read/write 次数
  - 例：LayerNorm（2 reads + 1 write）+ Add（1 read + 1 write）→ fused（2 reads + 1 write）
  - 节省 2 次 memory transaction → 约 40% 性能提升
- 不适合 fusion 的情况：
  - 两个 ops 之间有大的中间结果需要 materialize
  - 其中一个 op 是 compute-bound（如 GEMM）→ fusion 不减少瓶颈

### Follow-up 5：Triton 在不同硬件上的可移植性
> "Triton 号称支持多种 GPU backend。实际上 AMD GPU 和 NVIDIA GPU 上的性能差异大吗？"

**期望回答**：
- Triton 通过 LLVM 后端支持 AMD（ROCm）和 NVIDIA（CUDA）
- 实际性能差异：
  - 简单 kernel（element-wise, reduction）：差异 < 10%
  - 复杂 kernel（attention, GEMM）：AMD 上可能慢 20-40%
- 原因：
  - Triton 的优化 pass 主要针对 NVIDIA 架构设计
  - AMD 的 wavefront size 是 64（vs NVIDIA warp 32），需要不同的 tuning config
  - AMD 缺少某些硬件特性（如 TMA）
- 对候选人的意义：我在 Ascend NPU 上工作，理解跨硬件适配的挑战

---

## Pressure Follow-up（故意挑战候选人）

> "你说你理解 Triton 的编程模型，但你实际写过多少 Triton kernel？在 production 中用过吗？如果没有，你怎么证明你能胜任需要 Triton 开发的岗位？"

**期望应对**：
- 诚实承认：我没有在 production 中部署过自己写的 Triton kernel
- 但展示相关能力：
  1. 我理解 block-level programming 的核心概念
  2. 我在 vLLM 中使用过 Triton kernel（如 PagedAttention 的 Triton 实现）
  3. 我有跨硬件适配的经验（NPU），理解不同硬件的 programming model 差异
  4. Triton 的学习曲线比 CUDA 低很多，有 Python 基础可以快速上手
- 展示学习计划：我计划在 1 个月内完成 softmax、LayerNorm、RoPE 的 Triton 实现

---

## Debugging Scenario

> "你的 Triton softmax kernel 在 seq_len=8192 时比 PyTorch 的 `torch.softmax` 慢 2x。auto-tune 已经跑过了。怎么排查？"

**排查思路**：

1. **检查 auto-tune 结果**：
   - 看选中的 config 是否合理（BLOCK_SIZE, num_warps）
   - 可能 auto-tune 的搜索空间不够大，没有覆盖最优配置
   - 添加更多 config：`BLOCK_SIZE: [512, 1024, 2048, 4096, 8192]`

2. **检查 memory access pattern**：
   - seq_len=8192 时如果 BLOCK_SIZE < 8192，需要多次循环
   - 每次循环都是一次 global memory read → 总共 2-3 passes × 8192 × sizeof(float)
   - PyTorch 的 softmax 可能用了 cuDNN 的高度优化版本

3. **检查编译器生成的代码**：
   - `TRITON_PRINT_AUTOTUNING=1` 查看选中的 config
   - `MLIR_ENABLE_DUMP=1` 查看生成的 IR
   - 检查是否有不必要的 memory spill

4. **可能的原因**：
   - Triton 生成了 3-pass 而不是 2-pass（online softmax 没有被正确优化）
   - Register spilling 导致额外的 local memory 访问
   - PyTorch 用的是 cuDNN 的 persistent softmax（一次 load 到 shared memory）
   - 对于 seq_len=8192 可以放入 shared memory（8192×4=32KB < 164KB），但 Triton 可能没有这样做

5. **解决方案**：
   - 如果 seq_len 能放入 shared memory：设 BLOCK_SIZE=8192，一次 load 全部数据
   - 使用 online softmax 减少 pass 数
   - 增加 num_warps 到 32 以充分利用 SM
   - 如果仍然慢：可能需要回退到 CUDA 实现

---

## System Design Extension（扩展到更大规模）

> "设计一个 Triton kernel library 管理系统，支持：多种 kernel 的注册、auto-tune 结果缓存、跨 GPU 型号的 config 管理、CI 中的性能回归检测。"

**设计要点**：

1. **Kernel Registry**：
   - 每个 kernel 注册：name, input schema, output schema, tuning configs
   - 版本管理：kernel 代码变更时自动 invalidate cache
   - 依赖管理：kernel 之间的 fusion 关系

2. **Tuning Cache**：
   - Key: (kernel_name, kernel_version, gpu_model, input_shapes)
   - Value: optimal config + measured performance
   - Storage: SQLite for local, Redis for distributed
   - TTL: kernel 代码变更时 invalidate

3. **Cross-GPU Config Management**：
   - 每种 GPU 型号维护独立的 tuning 结果
   - 新 GPU 型号：先用相近型号的 config 作为 warmstart，再 fine-tune
   - Config migration：A100 → H100 时，调整 num_warps 和 BLOCK_SIZE

4. **Performance Regression CI**：
   - 每次 kernel 代码变更触发 benchmark
   - 对比 baseline（上一个 release）的性能
   - 回归阈值：> 5% 性能下降 → block merge
   - 报告：per-shape 性能对比表 + roofline 图

---

## Hire/No-Hire Evaluation

| 维度 | 权重 | 评分标准 |
|------|------|---------|
| Triton 编程能力 | 30% | 能否写出正确的 Triton kernel |
| 编译器理解 | 20% | 理解 Triton 编译流程和优化 pass |
| Auto-tuning | 15% | 理解 tuning 机制和 config 设计 |
| CUDA 对比 | 20% | 能否准确分析 Triton vs CUDA 的 tradeoff |
| 实战经验 | 15% | 是否有 production Triton kernel 经验 |

---

## Scorecard

| 维度 | 候选人预期得分 | 说明 |
|------|--------------|------|
| Triton 编程能力 | 3/10 | 能写基础 kernel 但未经 production 验证 |
| 编译器理解 | 2/10 | 了解大致流程但不深入 |
| Auto-tuning | 4/10 | 理解概念和基本用法 |
| CUDA 对比 | 5/10 | 能做定性分析，缺乏定量数据 |
| 实战经验 | 1/10 | 无 production Triton 开发经验 |
| **总分** | **3.0/10** | **No Hire for Triton-focused role** |

### 决策依据
- **No Hire 原因**：Triton 岗位需要实际的 kernel 开发和优化经验，候选人只有理论了解
- **候选人亮点**：理解 block-level programming 概念，有跨硬件适配思维
- **建议**：如果岗位允许 1-2 个月 ramp-up，且主要工作是 framework-level 而非 kernel-level，可以考虑
- **与 CUDA 面试对比**：Triton 学习曲线更低，候选人有 Python 基础，补强速度会比 CUDA 快
