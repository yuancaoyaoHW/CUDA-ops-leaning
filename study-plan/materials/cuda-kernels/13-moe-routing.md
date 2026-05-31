# MoE Routing 与 Expert GEMM

## 1. 学习目标

- 理解 Mixture of Experts（MoE，混合专家模型）的架构与动机
- 掌握 Router（路由器）的 Top-K 选择机制与 load balancing
- 理解 Expert GEMM 的 grouped/permuted 实现策略
- 能够分析 MoE 的通信瓶颈（All-to-All）与计算特点
- 掌握 MoE 在推理中的显存与计算 tradeoff

## 2. 前置知识

- GEMM 与 Tensor Core
- Softmax 与 Top-K 选择
- Tensor Parallel / Expert Parallel 概念
- LLM FFN 层结构

## 3. 核心术语表

| 术语 | 英文全称 | 定义 |
|------|----------|------|
| MoE | Mixture of Experts | 混合专家模型，每个 token 只激活部分专家 |
| Router | Router / Gating Network | 决定每个 token 分配到哪些专家的网络 |
| Expert | Expert | 一个独立的 FFN 子网络 |
| Top-K | Top-K Routing | 每个 token 选择 K 个专家（通常 K=2） |
| Load Balancing | Load Balancing Loss | 鼓励 token 均匀分配到各专家的辅助 loss |
| Expert Parallel | Expert Parallelism | 将不同专家放在不同 GPU 上 |
| All-to-All | All-to-All Communication | MoE 中 token 到专家的重分配通信 |
| Capacity Factor | Capacity Factor | 每个专家最多处理的 token 数的上限系数 |
| Grouped GEMM | Grouped GEMM | 多个不同大小的 GEMM 打包执行 |
| Permutation | Token Permutation | 按专家分组重排 token 的操作 |

## 4. 动机

### 4.1 MoE 的核心优势

Dense model: 每个 token 经过所有参数 → 计算量 = 参数量
MoE model: 每个 token 只经过部分参数 → 计算量 << 参数量

```
Mixtral 8x7B:
- 总参数: 46.7B (8 个 7B expert FFN + shared attention)
- 每 token 激活参数: ~12.9B (2 experts activated)
- 推理速度接近 12B dense model
- 质量接近 70B dense model
```

### 4.2 MoE FFN 结构

```
Standard FFN:
  hidden = up_proj(x) * gate_proj(x)  # SwiGLU
  output = down_proj(hidden)

MoE FFN:
  router_logits = router(x)            # [batch×seq, num_experts]
  top_k_weights, top_k_indices = topk(softmax(router_logits), k=2)
  
  # 对每个 token，只计算被选中的 2 个 expert
  for expert_idx in top_k_indices:
      expert_output += weight × expert[expert_idx](x)
```

### 4.3 推理中的挑战

1. **Grouped GEMM**：不同 expert 处理不同数量的 token → 不规则 GEMM
2. **Token permutation**：需要按 expert 重排 token → 额外内存操作
3. **显存**：所有 expert 权重都需要加载（即使只激活 2 个）
4. **通信**：Expert Parallel 需要 All-to-All 通信

## 5. 数学定义

### 5.1 Router

```
router_logits = x × W_router    # [tokens, num_experts]
router_probs = softmax(router_logits, dim=-1)
top_k_weights, top_k_indices = topk(router_probs, k=K)
top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)  # renormalize
```

### 5.2 Expert 计算

```
output = Σ_{i ∈ top_k_indices} top_k_weights[i] × Expert_i(x)

Expert_i(x) = down_proj_i(silu(gate_proj_i(x)) × up_proj_i(x))
```

### 5.3 Load Balancing Loss

```
# 鼓励均匀分配
f_i = (tokens assigned to expert i) / total_tokens  # fraction
p_i = mean(router_probs[:, i])  # average probability

aux_loss = num_experts × Σ_i f_i × p_i
```

## 6. 推导逻辑

### 6.1 Grouped GEMM 策略

**问题**：8 个 expert，每个处理不同数量的 token
```
Expert 0: 150 tokens × [hidden, intermediate]
Expert 1: 80 tokens × [hidden, intermediate]
Expert 2: 200 tokens × [hidden, intermediate]
...
```

**策略 A：Padding + Batched GEMM**
- 将所有 expert 的 token 数 pad 到最大值
- 使用 batched GEMM（cuBLAS）
- 缺点：浪费计算（padding 部分无用）

**策略 B：Permute + Grouped GEMM**
- 按 expert 重排 token
- 使用 CUTLASS grouped GEMM（每个 group 不同 M）
- 优点：无浪费
- 缺点：permutation 有额外开销

**策略 C：Triton Grouped GEMM**
- 用 Triton 实现 variable-size GEMM
- 每个 program 处理一个 expert 的一个 tile

### 6.2 Token Permutation

```cuda
// Step 1: 计算每个 expert 的 token 数量
expert_counts[num_experts]  // histogram

// Step 2: 计算 prefix sum → 每个 expert 的起始位置
expert_offsets = cumsum(expert_counts)

// Step 3: Scatter tokens to expert-grouped layout
for each token t:
    for each selected expert e in top_k_indices[t]:
        permuted_input[expert_offsets[e] + local_idx] = input[t]
```

### 6.3 Expert Parallel

```
GPU 0: Expert 0, 1 (+ shared attention layers)
GPU 1: Expert 2, 3
GPU 2: Expert 4, 5
GPU 3: Expert 6, 7

All-to-All:
1. 每个 GPU 计算 router → 知道每个 token 去哪个 expert
2. All-to-All 将 token 发送到对应 GPU
3. 每个 GPU 计算本地 expert
4. All-to-All 将结果发回原 GPU
```

## 7. 算子流程

### 7.1 MoE Forward（单 GPU）

```cuda
// 1. Router
router_logits = linear(x, W_router);  // [tokens, num_experts]
top_k_weights, top_k_indices = topk_softmax(router_logits, K);

// 2. Permute tokens by expert
permuted_input, expert_offsets = permute_tokens(x, top_k_indices);

// 3. Grouped GEMM (gate + up projection)
gate_out = grouped_gemm(permuted_input, W_gate, expert_offsets);
up_out = grouped_gemm(permuted_input, W_up, expert_offsets);

// 4. Activation
hidden = silu(gate_out) * up_out;

// 5. Grouped GEMM (down projection)
expert_out = grouped_gemm(hidden, W_down, expert_offsets);

// 6. Unpermute and weighted sum
output = unpermute_and_reduce(expert_out, top_k_weights, top_k_indices);
```

### 7.2 Triton Grouped GEMM

```python
@triton.jit
def grouped_gemm_kernel(
    A, B, C,
    expert_offsets,  # [num_experts + 1]
    M_per_expert,    # [num_experts]
    N, K,
    num_experts,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # 确定当前 program 属于哪个 expert
    expert_id = determine_expert(pid, M_per_expert, BLOCK_M)
    
    # 获取当前 expert 的 A 起始位置和 B 权重
    a_offset = tl.load(expert_offsets + expert_id)
    local_pid = pid - get_start_pid(expert_id, M_per_expert, BLOCK_M)
    
    # 标准 tiled GEMM，但 A 的行数是 M_per_expert[expert_id]
    # B 的权重是 expert_id 对应的权重矩阵
    # ...
```

## 8. PyTorch baseline

```python
import torch
import torch.nn as nn

class MoELayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLUFFN(hidden_size, intermediate_size) for _ in range(num_experts)
        ])
    
    def forward(self, x):
        # x: [batch*seq, hidden_size]
        router_logits = self.router(x)  # [tokens, num_experts]
        router_probs = torch.softmax(router_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            # 找到分配给 expert i 的 token
            mask = (top_k_indices == i).any(dim=-1)
            if mask.any():
                expert_input = x[mask]
                expert_output = expert(expert_input)
                # 加权累加
                weights = top_k_weights[mask][top_k_indices[mask] == i]
                output[mask] += weights.unsqueeze(-1) * expert_output
        
        return output
```

## 9-20. 关键内容摘要

### Profiling 指标
- Expert load imbalance: max_tokens_per_expert / avg_tokens_per_expert
- Grouped GEMM utilization: 实际 TFLOPS / peak TFLOPS
- All-to-All communication time vs compute time ratio
- Token permutation overhead

### 关键习题
1. Mixtral 8x7B 的总参数量和每 token 激活参数量分别是多少？
2. 为什么 MoE 需要 load balancing loss？如果不加会怎样？
3. Grouped GEMM 的三种实现策略各有什么优缺点？
4. Expert Parallel 中 All-to-All 通信的数据量如何计算？
5. MoE 推理时所有 expert 权重都需要在 GPU 上吗？有什么优化方案？

### 关键答案
1. 总参数 ≈ 46.7B（8×7B FFN + shared attention/embedding）。每 token 激活 ≈ 12.9B（2 experts + shared）。
2. 不加 load balancing loss，router 可能将大部分 token 路由到少数 expert（"winner-take-all"），导致：(a) 大部分 expert 未被训练；(b) 推理时负载不均；(c) 模型退化为 dense。
3. Padding: 简单但浪费计算；Permute+Grouped: 无浪费但有 permutation 开销；Triton: 灵活但需要自定义 kernel。
4. All-to-All 数据量 = tokens × hidden_size × dtype_size × 2（发送+接收）。对于 4096 tokens × 4096 hidden × FP16 = 32MB × 2 = 64MB。
5. 是的，推理时所有 expert 权重必须在 GPU 上（因为不知道哪些 token 会路由到哪个 expert）。优化：Expert Parallel 分散到多 GPU；Offloading（预测性加载）；量化减小 expert 大小。
