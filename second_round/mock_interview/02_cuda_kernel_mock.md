# 模拟面试 2：45 分钟 CUDA Kernel 优化

## 面试官 Profile

- **公司**：NVIDIA GPU Computing 部门
- **级别**：Senior Engineer，10 年 CUDA 开发经验
- **风格**：直接、技术导向，会要求手写 kernel 代码。对"我了解概念但没写过"的回答会追问到底。
- **偏好**：看重实际动手能力，会给具体场景让候选人现场优化。

---

## Opening Question

> "给你一个 softmax kernel 的需求：输入是 [batch_size, seq_len] 的矩阵，对每一行做 softmax。先给我一个 naive 实现，然后逐步优化到高性能版本。边写边解释你的优化思路。"

---

## Candidate Expected Answer（基于候选人真实经验）

**候选人诚实回应**：
"我需要坦诚说明，我没有实际的 CUDA kernel 编写经验。我的工作主要在 Ascend NPU 上做推理优化，使用的是 vLLM 框架层面的优化而非底层 kernel 开发。但我可以基于我对 GPU 架构的理论理解来讨论这个问题。

Naive softmax 需要三次遍历：
1. 找 max（数值稳定性）
2. 计算 exp(x - max) 的 sum
3. 归一化 exp(x - max) / sum

优化方向我知道的有：
- Online softmax：一次遍历完成，维护 running max 和 running sum
- 利用 shared memory 减少 global memory 访问
- Warp-level reduction 用 `__shfl_down_sync`
- 向量化加载（float4）提高带宽利用率

但具体的 CUDA 代码实现，包括 grid/block 配置、shared memory 分配、bank conflict 避免等，我目前还在学习阶段。"

---

## Weak Answer（常见错误）

"Softmax 就是 exp(x) / sum(exp(x))，用 CUDA 的话每个 thread 处理一个元素，然后用 atomicAdd 求 sum..."

**为什么弱**：
- 没有考虑数值稳定性（减 max）
- atomicAdd 做 reduction 性能极差
- 没有利用 shared memory
- 没有考虑 warp-level 优化
- 对 CUDA 编程模型理解不足

---

## Strong Answer（工程深度 + 数据 + tradeoff）

"让我从 naive 到最优逐步优化：

**Step 1: Naive 3-pass（baseline）**
```cuda
// 每个 block 处理一行，blockDim.x = 256
__global__ void softmax_naive(float* input, float* output, int N) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    float* row_in = input + row * N;
    float* row_out = output + row * N;
    
    // Pass 1: find max
    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        local_max = fmaxf(local_max, row_in[i]);
    sdata[threadIdx.x] = local_max;
    __syncthreads();
    // tree reduction for max
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();
    
    // Pass 2: compute sum of exp
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        local_sum += expf(row_in[i] - row_max);
    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float row_sum = sdata[0];
    __syncthreads();
    
    // Pass 3: normalize
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        row_out[i] = expf(row_in[i] - row_max) / row_sum;
}
```

问题：3 次读 global memory，2 次 shared memory reduction 有 `__syncthreads` 开销。

**Step 2: Online Softmax（1-pass reduction）**
```cuda
// 使用 online algorithm: 维护 (max, sum) pair
// 当新元素 x 到来时:
//   new_max = max(old_max, x)
//   new_sum = old_sum * exp(old_max - new_max) + exp(x - new_max)
__device__ float2 online_softmax_reduce(float2 a, float2 b) {
    // a.x = max_a, a.y = sum_a; b.x = max_b, b.y = sum_b
    float new_max = fmaxf(a.x, b.x);
    float new_sum = a.y * expf(a.x - new_max) + b.y * expf(b.x - new_max);
    return make_float2(new_max, new_sum);
}
```

这样只需要 1 次读取数据做 reduction，再 1 次读取做 normalize = 2 passes over global memory。

**Step 3: Warp-level 优化**
```cuda
// 最后 32 个元素的 reduction 用 warp shuffle，避免 shared memory + __syncthreads
__device__ float2 warp_reduce(float2 val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float2 other;
        other.x = __shfl_down_sync(0xffffffff, val.x, offset);
        other.y = __shfl_down_sync(0xffffffff, val.y, offset);
        val = online_softmax_reduce(val, other);
    }
    return val;
}
```

**Step 4: Vectorized Load**
```cuda
// 用 float4 一次加载 128 bits，提高 memory bandwidth 利用率
float4 data = reinterpret_cast<float4*>(row_in)[threadIdx.x];
// 处理 4 个元素
```

**性能分析**：
- Naive 3-pass：3N reads + N writes = 4N memory transactions
- Online 2-pass：2N reads + N writes = 3N memory transactions → 25% 减少
- 加上 vectorized load：bandwidth utilization 从 ~60% 提升到 ~85%
- 理论带宽上限：A100 2TB/s，seq_len=4096, batch=1024, FP32 → 16MB data
- 理论最小时间：16MB / 2TB/s = 8μs
- 实际优化后约 12-15μs（75-85% bandwidth utilization）

**Step 5: 特殊情况优化**
- seq_len ≤ 32：一个 warp 处理一行，纯 register + shuffle，零 shared memory
- seq_len ≤ 1024：一个 block 处理一行，shared memory reduction
- seq_len > 1024：多个 block 处理一行，需要 global memory atomic 或两次 kernel launch"

---

## Follow-up Chain（5 层递进追问）

### Follow-up 1：Memory Coalescing
> "你的 kernel 中，如果 seq_len 不是 128 的倍数，float4 加载会怎样？边界怎么处理？"

**期望回答**：
- 最后不足 4 个元素的部分需要 fallback 到 scalar load
- 或者 padding 到 4 的倍数（浪费少量计算但保持 coalesced）
- 更好的方案：用 `__ldg` + 编译器自动向量化，让编译器处理边界
- 关键：确保 warp 内 32 个 thread 访问连续地址，否则会产生多次 memory transaction

### Follow-up 2：Bank Conflict
> "你的 shared memory reduction 中，`sdata[threadIdx.x]` 的访问模式有 bank conflict 吗？"

**期望回答**：
- 连续 thread 访问连续地址 → 无 bank conflict（stride=1, 每个 thread 访问不同 bank）
- 但如果用 2D shared memory 且列数是 32 的倍数 → 同列访问会冲突
- 解决：padding `sdata[BLOCK_SIZE + 1]` 或使用 swizzle pattern
- 在这个 1D reduction 场景中没有 bank conflict

### Follow-up 3：Occupancy vs Performance
> "你的 kernel 用了多少 shared memory？occupancy 是多少？occupancy 低一定性能差吗？"

**期望回答**：
- Shared memory 使用：blockDim.x × sizeof(float) = 256 × 4 = 1KB
- A100 每个 SM 最大 164KB shared memory → 可以同时运行很多 block
- Occupancy 受限因素：shared memory, registers, max blocks per SM
- 1KB shared memory → occupancy 不受 shared memory 限制
- Occupancy 低不一定差：如果 kernel 是 compute-bound，降低 occupancy 换取更多 register 可能更快
- 这个 softmax kernel 是 memory-bound → 高 occupancy 有利于隐藏 memory latency

### Follow-up 4：FlashAttention 中的 Softmax
> "FlashAttention 中的 softmax 和你写的有什么不同？为什么不能直接用你的 kernel？"

**期望回答**：
- FlashAttention 的 softmax 是 fused 在 attention 计算中的，不是独立 kernel
- 关键区别：FlashAttention 按 block 处理 K/V，每个 block 只看到部分数据
- 需要 online softmax 来处理"看不到全部数据"的情况
- 维护 running max 和 running sum，每处理一个 K block 就更新
- 最终结果数学上等价于看到全部数据的 softmax
- 这就是为什么 FlashAttention 能做到 O(N) SRAM 而不是 O(N²)

### Follow-up 5：GEMM Tiling 与 Softmax 的关系
> "在 attention 中，QK^T 是一个 GEMM。这个 GEMM 的 tiling 策略和标准 GEMM 有什么不同？"

**期望回答**：
- 标准 GEMM tiling：沿 M, N, K 三个维度分块
- Attention GEMM (QK^T)：Q [B, H, N, d] × K^T [B, H, d, N] → [B, H, N, N]
- 特殊性：输出矩阵 N×N 可能非常大（N=4096 → 64MB FP32），不能全部存在 SRAM
- FlashAttention 的策略：沿 N 维度（sequence）分块，每次只计算部分 QK^T
- 内层循环：load Q block → iterate over K blocks → online softmax → multiply V block
- 外层循环：iterate over Q blocks
- 这样 SRAM 只需要存 block_size × block_size 的中间结果

---

## Pressure Follow-up（故意挑战候选人）

> "你说你没有 CUDA 经验。那你怎么能做 LLM inference 优化？inference 的核心瓶颈就是 kernel 性能。你不会写 kernel，怎么知道瓶颈在哪？怎么优化？"

**期望应对**：
- 承认短板：确实，CUDA kernel 开发是我当前最大的技术 gap
- 但解释自己的价值：
  1. 系统级优化不全是 kernel：scheduling, batching, memory management 都是框架层面
  2. 我在 vLLM 上的工作是 proposer 逻辑和 KV cache 管理，不需要写 kernel
  3. Speculative decoding 的收益来自算法层面（减少 decode steps），不是 kernel 优化
- 学习计划：我正在系统学习 CUDA，已经完成了 vector add、reduction 的实现，计划 2-3 个月内完成 GEMM 和 attention kernel
- 不要 defensive：承认这是需要补强的方向，展示学习意愿和计划

---

## Debugging Scenario

> "你的 GEMM kernel 在 M=N=K=4096 时性能只有 cuBLAS 的 30%。NCU profiler 显示：Compute Throughput 45%, Memory Throughput 60%, Achieved Occupancy 25%。问题在哪？怎么优化？"

**排查思路**：

1. **分析 profiler 数据**：
   - Compute 45% + Memory 60% → 两者都不高，说明 kernel 既没有充分利用计算也没有充分利用带宽
   - Occupancy 25% → 很低，可能是 register 或 shared memory 使用过多
   - 这种"两低"模式通常意味着 latency-bound（stall 太多）

2. **可能的原因**：
   - Register spilling：每个 thread 用了太多 register → spill 到 local memory（慢）
   - Bank conflict：shared memory 访问模式有冲突
   - Warp stall：等待 memory 或 synchronization
   - 没有 double buffering：load 和 compute 没有 overlap

3. **优化方向**：
   - 检查 register 使用：`ncu --metrics launch__registers_per_thread`
   - 如果 register > 128 → 减少每个 thread 的工作量，或用 `__launch_bounds__` 限制
   - 增加 tile size 但减少 register blocking → 提高 occupancy
   - 添加 double buffering：用两块 shared memory 交替，一块 load 一块 compute
   - 检查 shared memory bank conflict：用 padding 或 swizzle

4. **目标**：
   - cuBLAS 在 4096×4096 上约 300 TFLOPS（A100 FP16 peak 312 TFLOPS）
   - 30% = 90 TFLOPS → 目标至少 80%+ = 250 TFLOPS
   - 需要：高 occupancy（50%+）、无 bank conflict、double buffering、vectorized load

**候选人诚实标注**：以上分析基于理论学习，我没有实际用 NCU 分析过 kernel 的经验。

---

## System Design Extension（扩展到更大规模）

> "如果要你设计一个 kernel auto-tuning 系统，自动为不同 shape 的 GEMM 选择最优的 tile size、block size、pipeline depth，你怎么设计？"

**设计要点**：

1. **搜索空间定义**：
   - Tile size: {32, 64, 128, 256} × {32, 64, 128, 256}
   - Block size: {64, 128, 256}
   - Pipeline stages: {1, 2, 3, 4}
   - Vectorize width: {1, 2, 4}
   - 总搜索空间：~1000 种配置

2. **Auto-tuning 策略**：
   - Exhaustive search for common shapes（预计算）
   - Bayesian optimization for unseen shapes
   - 用 hardware counters 作为 surrogate model 的 features

3. **Cache 机制**：
   - Shape → optimal config 的 lookup table
   - 按 (M, N, K, dtype) 索引
   - 支持 fuzzy matching（相近 shape 用相同 config）

4. **Runtime 集成**：
   - JIT compile optimal kernel for each shape
   - Warmup phase：前几次调用做 profiling，之后用 cached config
   - 类似 Triton 的 auto-tune decorator

---

## Hire/No-Hire Evaluation

| 维度 | 权重 | 评分标准 |
|------|------|---------|
| CUDA 编程能力 | 30% | 能否手写正确的 kernel，理解 thread/block/grid |
| Memory 优化 | 25% | 理解 coalescing, shared memory, bank conflict |
| 性能分析 | 20% | 能否解读 profiler 数据，定位瓶颈 |
| 架构理解 | 15% | 理解 GPU 硬件特性对 kernel 设计的影响 |
| 优化方法论 | 10% | 系统性的优化思路，而非 ad-hoc |

---

## Scorecard

| 维度 | 候选人预期得分 | 说明 |
|------|--------------|------|
| CUDA 编程能力 | 2/10 | 无实际 CUDA 编写经验，只能描述概念 |
| Memory 优化 | 3/10 | 理论知识有一些，但无法落地到代码 |
| 性能分析 | 2/10 | 没用过 NCU，无法解读实际 profiler 数据 |
| 架构理解 | 4/10 | 对 memory hierarchy 有基本理解 |
| 优化方法论 | 3/10 | 知道优化方向但缺乏系统性 |
| **总分** | **2.7/10** | **Strong No Hire for CUDA role** |

### 决策依据
- **No Hire 原因**：CUDA 岗位的核心要求是 kernel 开发能力，候选人完全没有
- **候选人亮点**：诚实承认短板，有学习计划，理论基础不为零
- **建议**：不适合 CUDA kernel 开发岗位，但如果岗位侧重 framework-level 优化（如 vLLM scheduler），可以考虑
- **成长路径**：需要 2-3 个月密集 CUDA 学习 + 项目实践才能达到面试门槛
