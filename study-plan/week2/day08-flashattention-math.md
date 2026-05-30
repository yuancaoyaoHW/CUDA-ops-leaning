# Day 8：FlashAttention 数学推导

## 学习目标
- 完整手推 FlashAttention 的 tiling + online softmax rescaling
- 理解 IO 复杂度为什么更低
- 理解 forward 和 backward 的区别
- 开始读 vLLM 论文

---

## 上午（3h）- 数学推导

### 标准 Attention 的问题

```
标准计算: O = softmax(QK^T / sqrt(d)) @ V

步骤：
  1. S = Q @ K^T          → shape [N, N], 需要 O(N^2) 内存
  2. P = softmax(S)       → shape [N, N], 需要 O(N^2) 内存
  3. O = P @ V            → shape [N, d]

内存问题：
  N = seq_len = 8192, fp16:
  S 矩阵大小 = 8192^2 * 2 = 128 MB（单个 head！）
  多 head: 32 heads * 128 MB = 4 GB → 显存爆炸

IO 问题：
  S 和 P 都要写回 HBM，再读回来
  总 HBM 读写 = O(N^2 * d) bytes
```

### FlashAttention 核心 Idea

```
不存储完整的 N×N attention matrix！
把 Q 分成 Tr 块（每块 Br 行），K/V 分成 Tc 块（每块 Bc 行）
用 online softmax 逐块计算，只在 SRAM 中存储当前块的 attention scores
```

### Algorithm（必须能手推）

```
输入: Q [N, d], K [N, d], V [N, d]
输出: O [N, d]

初始化:
  O = zeros [N, d]
  l = zeros [N]      (softmax 分母的 running sum)
  m = -inf * ones [N] (running max)

外循环: for j = 1 to Tc (遍历 K/V 的块)
  加载 K_j [Bc, d] 和 V_j [Bc, d] 到 SRAM

  内循环: for i = 1 to Tr (遍历 Q 的块)
    加载 Q_i [Br, d], O_i [Br, d], l_i [Br], m_i [Br] 到 SRAM

    // 计算当前块的 attention scores
    S_ij = Q_i @ K_j^T / sqrt(d)    → [Br, Bc]

    // 当前块的 row-wise max 和 exp sum
    m_ij = rowmax(S_ij)              → [Br]
    P_ij = exp(S_ij - m_ij)         → [Br, Bc]  (未归一化)
    l_ij = rowsum(P_ij)             → [Br]

    // Rescaling（核心！）
    m_new = max(m_i, m_ij)          → [Br]
    l_new = l_i * exp(m_i - m_new) + l_ij * exp(m_ij - m_new)  → [Br]

    // 更新 O
    O_i = O_i * (l_i * exp(m_i - m_new) / l_new)    // rescale 旧的 O
        + exp(m_ij - m_new) / l_new * P_ij @ V_j    // 加上新的贡献

    // 更新 running stats
    l_i = l_new
    m_i = m_new

    写回 O_i, l_i, m_i 到 HBM
```

### 为什么 Rescaling 正确

```
定义: 对于前 j 个 K/V 块，正确的 output 应该是:
  O_correct = softmax(Q @ K_{1:j}^T) @ V_{1:j}

归纳证明:
  假设处理完第 j-1 个块后，O_i 存储的是:
    O_i = sum_{k=1}^{j-1} exp(S_ik - m_i) @ V_k / l_i
    其中 m_i = max over all seen elements, l_i = sum of all exp(x - m_i)

  处理第 j 个块时:
    新的 max: m_new = max(m_i, m_ij)
    旧的 O 需要 rescale: 因为 max 变了，所有旧的 exp 值都要乘以 exp(m_old - m_new)
    O_new = O_old * (l_old * exp(m_old - m_new) / l_new) + new_contribution / l_new

  这保证了 O 始终等于 "到目前为止所有块的正确 softmax attention"
```

### IO 复杂度分析

```
标准 Attention:
  读 Q, K, V: 3 * N * d * sizeof = O(Nd) from HBM
  写 S: N^2 * sizeof = O(N^2) to HBM
  读 S (for softmax): O(N^2) from HBM
  写 P: O(N^2) to HBM
  读 P, V (for P@V): O(N^2 + Nd) from HBM
  总 HBM IO: O(N^2 + Nd) ← 被 N^2 主导

FlashAttention:
  外循环 Tc 次，内循环 Tr 次
  每次内循环: 读 Q_i (Br*d), K_j (Bc*d), V_j (Bc*d), O_i (Br*d)
  总读取: Tc * Tr * (Br*d + Bc*d + Bc*d + Br*d) * sizeof
        = (N/Bc) * (N/Br) * (2*Br*d + 2*Bc*d) * sizeof
        = O(N^2 * d / min(Br, Bc))

  当 SRAM 大小 M 足够放下 Br*d + Bc*d 时:
    Br = Bc = M / (4d)  (简化)
    IO = O(N^2 * d^2 / M)

  对比: O(N^2 * d^2 / M) vs O(N^2)
  当 M > d^2 时（通常 M=192KB, d=128 → d^2=16KB → M >> d^2）:
    FlashAttention IO << 标准 Attention IO
```

---

## 下午（2h）- FlashAttention v1 vs v2

### v2 的改进

```
1. 循环顺序反转:
   v1: 外循环 K/V，内循环 Q → 每个 Q 块被多次读写
   v2: 外循环 Q，内循环 K/V → 每个 Q 块只读一次，O 在 register 中累加
   好处: 减少 HBM 写回次数

2. Work partitioning:
   v1: 每个 thread block 处理一个 Q 块，遍历所有 K/V 块
   v2: 更好的 warp 分工，减少 shared memory 读写

3. 支持非 2 的幂的 head_dim

4. 性能: v2 比 v1 快约 2x
```

### FlashAttention 在推理中的应用

```
Prefill 阶段:
  Q = [seq_len, d], K = [seq_len, d] → 标准 FlashAttention
  compute-bound（大矩阵乘）→ FlashAttention 减少 IO，让 compute 成为唯一瓶颈

Decode 阶段:
  Q = [1, d], K = [seq_len, d] → 退化为 GEMV
  memory-bound → FlashAttention 的 tiling 意义不大
  → 用 FlashDecoding: 把 KV 按 seq_len 切分并行处理

FlashDecoding:
  把 KV cache 的 seq_len 维度切分到多个 thread blocks
  每个 block 处理一段 KV，计算 partial attention
  最后 reduce 所有 blocks 的结果（需要 rescaling）
  好处: 增加并行度，提高 GPU 利用率
```

---

## 晚上（1.5h）- vLLM 论文

### PagedAttention 核心概念

```
问题: KV cache 的内存管理
  - 每个 request 的 KV cache 大小不确定（不知道会生成多少 token）
  - 预分配 max_seq_len → 巨大浪费（平均浪费 60-80%）
  - 不同 request 的 KV cache 大小不同 → 外部碎片

解决: 借鉴 OS 的虚拟内存/分页机制
  - 把 KV cache 分成固定大小的 block（如 16 tokens/block）
  - 每个 block 存储 16 个 token 的 K 和 V
  - Block 不需要物理连续 → 消除外部碎片
  - 按需分配 block → 消除内部浪费（最多浪费 1 个 block）

Block Table:
  逻辑块号 → 物理块号 的映射表
  类似 OS 的页表
  每个 sequence 有自己的 block table

  例: sequence "Hello world, how are you?"
  逻辑块 0 (token 0-15) → 物理块 7
  逻辑块 1 (token 16-31) → 物理块 3
  逻辑块 2 (token 32-47) → 物理块 12

Copy-on-Write:
  Beam search 中多个 beam 共享前缀
  共享的 block 只存一份，引用计数 > 1
  当某个 beam 需要修改时才复制 → 节省内存
```

### 内存节省效果

```
传统方案（预分配 max_seq_len=2048）:
  每个 request: 2048 * num_layers * 2 * hidden_size * sizeof
  7B 模型, fp16: 2048 * 32 * 2 * 4096 * 2 = 1 GB per request
  80GB GPU: 最多 ~60 个并发 request（考虑模型参数占用）

PagedAttention（按需分配）:
  平均 output 长度 256 tokens → 实际只用 256/2048 = 12.5% 的内存
  → 并发提升 ~5-8x
  内存浪费 < 4%（只有最后一个 block 可能不满）
```

---

## 日检（20 分钟）

1. **闭卷手推**（10min）：写出 FlashAttention 的 rescaling 公式（m_new, l_new, O_new 的更新）
2. **口述**（5min）：FlashAttention 的 IO 复杂度是多少？为什么比标准 attention 低？
3. **口述**（5min）：PagedAttention 解决什么问题？Block table 是什么？

---

## 参考资料

- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao et al., 2022)
- FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning (Dao, 2023)
- Stephen Diehl, "The FlashAttention CUDA Kernel Line by Line"
- Efficient Memory Management for LLM Serving with PagedAttention (Kwon et al., 2023)
