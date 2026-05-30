# Day 12：Speculative Decoding + PagedAttention Kernel

## 学习目标
- 理解 speculative decoding 的算法和数学保证
- 能分析 acceptance rate 对加速比的影响
- 理解 PagedAttention kernel 的实现细节
- LeetCode 练习

---

## 上午（3h）- Speculative Decoding

### 动机

```
Decode 阶段的问题：
  每步只生成 1 个 token
  每步需要读取整个模型权重 + KV cache
  GPU 计算利用率极低（memory-bound）
  生成 100 个 token → 100 次 forward → 100 次读取权重

如果能一次生成多个 token → 减少 forward 次数 → 提高吞吐
但自回归模型天然是一个一个生成的...

Speculative Decoding 的 idea：
  用一个小模型（draft model）快速猜测多个 token
  用大模型（target model）并行验证这些猜测
  接受正确的，拒绝错误的
  → 一次 target forward 可能接受多个 token
```

### 算法详解

```
输入: target model p, draft model q, speculation length γ

Step 1: Draft（用小模型生成 γ 个 token）
  x_1 ~ q(·|context)
  x_2 ~ q(·|context, x_1)
  ...
  x_γ ~ q(·|context, x_1, ..., x_{γ-1})

Step 2: Verify（用大模型并行验证）
  一次 forward 计算 p(·|context), p(·|context,x_1), ..., p(·|context,x_1,...,x_γ)
  （因为 x_1...x_γ 已知，可以并行计算所有位置的概率）

Step 3: Accept/Reject（逐个位置判断）
  对每个位置 i = 1, ..., γ:
    采样 r ~ Uniform(0, 1)
    如果 r < min(1, p(x_i|...) / q(x_i|...)):
      接受 x_i，继续检查下一个
    否则:
      拒绝 x_i
      从 adjusted distribution 采样替代 token:
        x_i' ~ normalize(max(0, p(·|...) - q(·|...)))
      丢弃 x_{i+1}, ..., x_γ
      break

Step 4: Bonus token
  如果所有 γ 个都被接受，额外从 p(·|context, x_1,...,x_γ) 采样一个 token
  → 最多一次验证得到 γ+1 个 token
```

### 数学保证

```
关键定理：Speculative decoding 的输出分布 = target model 的分布

证明直觉：
  - 如果 q(x) ≤ p(x): 一定接受（概率 1）
  - 如果 q(x) > p(x): 以概率 p(x)/q(x) 接受
  - 拒绝时从 max(0, p-q) 采样 → 补偿被拒绝的概率质量
  - 总效果：每个 token 的边际分布 = p(x)

这意味着：
  - 输出质量完全不变（和直接用 target model 一样）
  - 只是加速，没有精度损失
  - 这是 speculative decoding 最大的优势
```

### Acceptance Rate 分析

```
定义: α = P(draft token 被接受)

Expected tokens per verification step:
  E[accepted] = (1 - α^(γ+1)) / (1 - α)

  α = 0.5, γ = 5: E = (1 - 0.5^6) / 0.5 = 1.97 tokens/step
  α = 0.7, γ = 5: E = (1 - 0.7^6) / 0.3 = 2.94 tokens/step
  α = 0.9, γ = 5: E = (1 - 0.9^6) / 0.1 = 4.69 tokens/step

Speedup 计算:
  设 target forward 时间 = T_target
  设 draft forward 时间 = T_draft (通常 T_draft << T_target)
  设 γ 个 draft token 的生成时间 = γ * T_draft

  每个 verification cycle:
    时间 = γ * T_draft + T_target (draft + verify)
    产出 = E[accepted] tokens

  Speedup = E[accepted] / (γ * T_draft/T_target + 1)

  例: T_draft = 0.1 * T_target, γ = 5, α = 0.8:
    E[accepted] = (1 - 0.8^6) / 0.2 = 3.69
    Speedup = 3.69 / (5*0.1 + 1) = 3.69 / 1.5 = 2.46x

什么时候不值得:
  - α < 0.5: 频繁拒绝，浪费 draft 计算
  - T_draft 太大: draft model 太慢，overhead 高
  - Batch size 大: decode 已经接近 compute-bound，加速空间小
```

### 实际应用

```
Draft model 选择:
  1. 小版本模型: 如 Llama-7B 做 target，Llama-68M 做 draft
  2. 同模型的浅层: 只用前 2 层做 draft（Medusa）
  3. N-gram model: 几乎零成本，但 α 低（~20%）
  4. 专门训练的 draft head: 在 target model 上加一个小 head

vLLM 中的实现:
  - 支持多种 draft model
  - 动态调整 γ（根据 acceptance rate）
  - 和 continuous batching 结合
```

---

## 下午（2h）- PagedAttention Kernel

### 和标准 Attention Kernel 的区别

```
标准 Attention:
  K, V 是连续的 tensor: K[seq_len, head_dim]
  直接做矩阵乘: Q @ K^T

PagedAttention:
  K, V 存储在不连续的 blocks 中
  需要通过 block_table 索引:
    block_table[seq_id] = [physical_block_0, physical_block_1, ...]
    每个 physical_block 存储 block_size 个 token 的 K/V

  Kernel 需要:
    1. 遍历 block_table 中的每个 block
    2. 从每个 physical block 读取 K/V
    3. 计算 attention scores
    4. 做 online softmax
```

### Kernel 伪代码

```cuda
// 每个 thread block 处理一个 (sequence, head) 对
__global__ void paged_attention_kernel(
    float* Q,           // [num_seqs, num_heads, head_dim]
    float* K_cache,     // [num_blocks, block_size, num_heads, head_dim]
    float* V_cache,     // [num_blocks, block_size, num_heads, head_dim]
    int* block_tables,  // [num_seqs, max_num_blocks]
    int* seq_lens,      // [num_seqs]
    float* output,      // [num_seqs, num_heads, head_dim]
    int block_size,
    int head_dim,
) {
    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int seq_len = seq_lens[seq_idx];
    int num_blocks = (seq_len + block_size - 1) / block_size;

    // 加载 Q (当前 token 的 query)
    float q[HEAD_DIM];
    load_q(Q, seq_idx, head_idx, q);

    // Online softmax 变量
    float m = -INFINITY;
    float l = 0.0f;
    float o[HEAD_DIM] = {0};

    // 遍历所有 KV blocks
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        // 通过 block_table 找到物理块
        int physical_block = block_tables[seq_idx * max_blocks + block_idx];

        // 从物理块读取 K 和 V
        for (int token_idx = 0; token_idx < block_size; token_idx++) {
            int global_token = block_idx * block_size + token_idx;
            if (global_token >= seq_len) break;

            // 计算 attention score
            float score = dot(q, K_cache[physical_block][token_idx][head_idx]);
            score *= scale;

            // Online softmax update
            float m_new = max(m, score);
            float exp_old = exp(m - m_new);
            float exp_new = exp(score - m_new);
            l = l * exp_old + exp_new;

            // Update output
            for (int d = 0; d < HEAD_DIM; d++) {
                o[d] = o[d] * exp_old + exp_new * V_cache[physical_block][token_idx][head_idx][d];
            }
            m = m_new;
        }
    }

    // Normalize
    for (int d = 0; d < HEAD_DIM; d++) {
        output[seq_idx * num_heads * head_dim + head_idx * head_dim + d] = o[d] / l;
    }
}
```

### 性能考虑

```
PagedAttention 的额外开销:
  1. Block table 间接寻址: 多一次 global memory 读取
  2. 非连续内存访问: K/V 不连续，无法 vectorized load 跨 block
  3. Block 边界处理: 最后一个 block 可能不满

优化:
  1. 把 block_table 放在 shared memory 或 constant memory
  2. Block 内的 K/V 是连续的 → block 内可以 vectorized load
  3. 预取下一个 block 的地址

实际影响:
  - 相比连续 KV cache，PagedAttention 有 ~5-10% 的性能开销
  - 但内存节省 5-8x → 可以服务更多并发 → 总吞吐量大幅提升
  - 这是一个 latency vs throughput 的 tradeoff
```

---

## 晚上（1.5h）- LeetCode

### Beam Search 实现

```python
def beam_search(model, input_ids, beam_width=4, max_len=50):
    # 每个 beam: (score, token_ids)
    beams = [(0.0, input_ids)]

    for step in range(max_len):
        all_candidates = []
        for score, seq in beams:
            if seq[-1] == EOS:
                all_candidates.append((score, seq))
                continue
            # 获取下一个 token 的概率分布
            logits = model(seq)
            log_probs = log_softmax(logits[-1])
            # 取 top-k candidates
            top_k = torch.topk(log_probs, beam_width)
            for log_p, token in zip(top_k.values, top_k.indices):
                all_candidates.append((score + log_p, seq + [token]))

        # 保留 top beam_width 个
        beams = sorted(all_candidates, key=lambda x: x[0], reverse=True)[:beam_width]

        # 所有 beam 都结束了
        if all(seq[-1] == EOS for _, seq in beams):
            break

    return beams[0][1]  # 返回最高分的序列
```

### Top-K Sampling (LeetCode 215 变体)

```python
# 用 quickselect 实现 O(n) 的 top-k
def top_k_sampling(logits, k, temperature=1.0):
    logits = logits / temperature
    top_k_logits, top_k_indices = torch.topk(logits, k)
    probs = torch.softmax(top_k_logits, dim=-1)
    sampled_idx = torch.multinomial(probs, 1)
    return top_k_indices[sampled_idx]
```

---

## 日检（20 分钟）

1. **口述**（7min）：Speculative decoding 的完整算法步骤？为什么输出分布不变？
2. **口述**（5min）：Acceptance rate = 0.7, γ = 5 时，expected tokens per step 是多少？
3. **口述**（5min）：PagedAttention kernel 和标准 attention kernel 的核心区别？
4. **口述**（3min）：什么时候 speculative decoding 不值得用？

---

## 参考资料

- Fast Inference from Transformers via Speculative Decoding (Leviathan et al., 2023)
- Accelerating LLM Inference with Staged Speculative Decoding (Spector et al., 2023)
- vLLM source: `csrc/attention/attention_kernels.cu`
