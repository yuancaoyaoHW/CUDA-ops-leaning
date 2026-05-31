# FlashAttention

## 1. 学习目标

- 理解 FlashAttention 的核心动机：减少 HBM 访问而非减少计算量
- 掌握 tiling + online softmax 的算法设计
- 理解 FlashAttention 的 IO 复杂度分析
- 能够描述 FlashAttention-2 的优化改进
- 掌握 FlashAttention 在 prefill 阶段的性能优势

## 2. 前置知识

- Attention 机制（Q×K^T → softmax → ×V）
- Online softmax 算法
- GPU 内存层次（SRAM vs HBM）
- Tiling 策略

## 3. 核心术语表

| 术语 | 英文全称 | 定义 |
|------|----------|------|
| FlashAttention | FlashAttention | 通过 tiling 和 online softmax 避免写 S×S 矩阵到 HBM 的 attention 算法 |
| IO Complexity | IO Complexity | 算法对 HBM 的读写总量 |
| Tiling | Tiling | 将 Q/K/V 分块加载到 SRAM 计算 |
| Online Softmax | Online Softmax | 单 pass 计算 max 和 sum 的 softmax |
| Recomputation | Recomputation | backward 时重新计算 attention score 而非存储 |
| SRAM | Static Random-Access Memory | GPU 片上高速存储（shared memory） |
| FlashAttention-2 | FlashAttention-2 | 改进版：更好的 work partitioning 和 parallelism |
| FlashAttention-3 | FlashAttention-3 | Hopper 架构优化版：利用 TMA 和 warp specialization |

## 4. 动机

### 4.1 标准 Attention 的 HBM 瓶颈

标准实现的 HBM 访问：
```
Step 1: S = Q×K^T    → 写 S[B,H,S,S] 到 HBM     O(S²) writes
Step 2: P = softmax(S) → 读 S, 写 P 到 HBM       O(S²) reads + writes
Step 3: O = P×V      → 读 P 从 HBM               O(S²) reads
Total HBM IO: O(S² + S²) = O(S²)
```

对于 S=4096, H=32, FP16：S×S 矩阵 = 32×4096²×2 = 1GB
这远超 SRAM 容量（A100: ~20MB shared memory total）

### 4.2 FlashAttention 的核心思想

**不将 S×S 矩阵写到 HBM**：
- 将 Q 分成 blocks，K/V 分成 blocks
- 对每个 Q block，遍历所有 K/V blocks
- 在 SRAM 中计算局部 attention score
- 使用 online softmax 增量更新输出
- 最终输出直接写到 HBM

HBM IO: O(S²D / M)，其中 M = SRAM 大小
当 M > D 时（通常成立），IO = O(S²D² / M) ≈ O(S) per query position

### 4.3 性能对比

| 方法 | HBM IO | 计算量 | Wall-clock |
|------|--------|--------|-----------|
| Standard | O(S²) | O(S²D) | 慢（IO-bound） |
| FlashAttention | O(S²D/M) | O(S²D) | 快（compute-bound） |

计算量相同，但 FlashAttention 将瓶颈从 IO 转移到 compute → 更好利用 Tensor Core。

## 5. 数学定义

### 5.1 分块 Online Softmax

将 K/V 分成 T_c 个 block，每个 block 大小 B_c：

```
初始化: O = 0, m = -inf, l = 0

For each KV block j = 0, ..., T_c-1:
    S_j = Q_i × K_j^T / √d        // [B_r, B_c] 在 SRAM 中
    m_j = rowmax(S_j)               // 当前 block 的行最大值
    
    m_new = max(m, m_j)             // 更新全局最大值
    l_new = l × exp(m - m_new) + rowsum(exp(S_j - m_new))  // 更新分母
    
    P_j = exp(S_j - m_new)         // 当前 block 的 softmax 分子
    O = O × (l × exp(m - m_new) / l_new) + P_j × V_j / l_new  // 更新输出
    
    m = m_new
    l = l_new
```

### 5.2 IO 复杂度证明

设 SRAM 大小为 M，Q/K/V 各为 [N, d]：
- Q 被分成 T_r = ceil(N / B_r) 个 block
- K/V 被分成 T_c = ceil(N / B_c) 个 block
- 约束：B_r × d + B_c × d + B_r × B_c ≤ M

HBM 读取：
- Q: N × d（每个 Q block 读一次）
- K: T_r × N × d（每个 Q block 遍历所有 K）
- V: T_r × N × d

总 IO = O(N²d² / M)

当 M = O(d²) 时（A100 shared memory 足够），IO = O(N²)
但常数因子远小于标准方法（标准方法需要写 N² 的 score 矩阵）

## 6. 推导逻辑

### 6.1 FlashAttention-2 的改进

1. **减少非 matmul FLOPs**：将 rescaling 操作延迟到最后
2. **更好的 work partitioning**：外层循环遍历 Q blocks（而非 KV blocks）
3. **Warp 间并行**：不同 warp 处理不同的 KV blocks，减少同步

FlashAttention-1 vs 2 性能：
- FA-1: ~50-60% Tensor Core utilization
- FA-2: ~70-75% Tensor Core utilization

### 6.2 Causal Mask 处理

```
对于 causal attention，当 Q block i 和 K block j 满足：
- j > i: 整个 block 被 mask → 跳过（不计算）
- j < i: 整个 block 不被 mask → 正常计算
- j = i: 部分 mask → 计算后 apply mask
```

这减少了约 50% 的计算量（上三角全部跳过）。

### 6.3 Backward Pass

FlashAttention backward 的关键：**recomputation**
- Forward 不保存 S×S 的 attention score（太大）
- 只保存 O, m, l（输出、max、sum）
- Backward 时重新计算 S = Q×K^T，再计算梯度
- 额外计算量 vs 节省的内存 → 值得（因为 forward 已经是 compute-bound）

## 7. 算子流程

### 7.1 FlashAttention Forward 伪代码

```
Input: Q[N,d], K[N,d], V[N,d] in HBM
Output: O[N,d] in HBM

1. Set block sizes B_r, B_c based on SRAM size M
2. Initialize O = zeros[N,d], l = zeros[N], m = -inf[N] in HBM

3. Divide Q into T_r blocks, K/V into T_c blocks

4. For i = 0 to T_r-1:  (outer loop: Q blocks)
     Load Q_i [B_r, d] from HBM to SRAM
     Load O_i [B_r, d], l_i [B_r], m_i [B_r] from HBM
     
     For j = 0 to T_c-1:  (inner loop: KV blocks)
       Load K_j [B_c, d], V_j [B_c, d] from HBM to SRAM
       
       // Compute attention score (in SRAM)
       S_ij = Q_i × K_j^T ∈ [B_r, B_c]    // Tensor Core GEMM
       
       // Online softmax update
       m_ij = rowmax(S_ij)
       m_new = max(m_i, m_ij)
       P_ij = exp(S_ij - m_new)
       l_new = l_i × exp(m_i - m_new) + rowsum(P_ij)
       
       // Update output
       O_i = O_i × (l_i × exp(m_i - m_new) / l_new) + P_ij × V_j / l_new
       
       m_i = m_new
       l_i = l_new
     
     Write O_i, l_i, m_i back to HBM

5. Return O
```

### 7.2 Block Size 选择

A100 (164KB shared memory per SM):
```
SRAM budget: Q_block + K_block + V_block + Score_block + Output_block
B_r × d + B_c × d + B_c × d + B_r × B_c + B_r × d ≤ M

d=128, FP16:
B_r × 256 + 2 × B_c × 256 + B_r × B_c × 2 + B_r × 256 ≤ 164K

典型选择: B_r = 128, B_c = 64 (FlashAttention-2)
```

## 8. PyTorch baseline

```python
import torch
import torch.nn.functional as F

# PyTorch 2.0+ 自动使用 FlashAttention
def flash_attention_pytorch(q, k, v, causal=True):
    """
    q: [B, H, S, D]
    k: [B, H_kv, S, D]  
    v: [B, H_kv, S, D]
    """
    # F.scaled_dot_product_attention 自动选择最优 backend
    # 包括 FlashAttention, Memory-Efficient Attention, Math
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)

# 强制使用 FlashAttention backend
with torch.backends.cuda.sdp_kernel(
    enable_flash=True, enable_math=False, enable_mem_efficient=False
):
    output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

# 或直接使用 flash_attn 库
from flash_attn import flash_attn_func
output = flash_attn_func(q, k, v, causal=True)
```

## 9. CUDA 实现思路

FlashAttention 的完整 CUDA 实现非常复杂（~2000 行），核心结构：

```cuda
// 简化的 FlashAttention kernel 结构
template<int Br, int Bc, int D>
__global__ void flash_attention_kernel(
    const half* Q, const half* K, const half* V, half* O,
    float* L, float* M,  // logsumexp 和 max
    int N, int d
) {
    // Shared memory allocation
    __shared__ half Q_smem[Br][D];
    __shared__ half K_smem[Bc][D];
    __shared__ half V_smem[Bc][D];
    __shared__ float S_smem[Br][Bc];
    
    int q_block_idx = blockIdx.x;  // which Q block
    
    // Load Q block to SRAM
    load_Q_block(Q, Q_smem, q_block_idx, Br, D);
    
    // Initialize accumulators
    float O_acc[Br][D] = {0};
    float m[Br] = {-INFINITY};
    float l[Br] = {0};
    
    // Loop over KV blocks
    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        // Load K, V blocks
        load_KV_block(K, V, K_smem, V_smem, kv_block, Bc, D);
        __syncthreads();
        
        // Compute S = Q @ K^T using Tensor Core (WMMA)
        compute_gemm(Q_smem, K_smem, S_smem, Br, Bc, D);
        
        // Online softmax + accumulate O
        online_softmax_update(S_smem, V_smem, O_acc, m, l, Br, Bc, D);
        __syncthreads();
    }
    
    // Final normalization and write O
    normalize_and_store(O_acc, l, O, q_block_idx, Br, D);
}
```

## 10. Triton 实现思路

```python
@triton.jit
def flash_attention_triton(
    Q, K, V, O,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    N, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_m = tl.program_id(2)  # Q block index
    
    # Initialize
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    
    # Load Q block [BLOCK_M, D]
    q = load_q_block(Q, pid_batch, pid_head, pid_m, BLOCK_M, D)
    
    # Iterate over KV blocks
    for start_n in range(0, N, BLOCK_N):
        # Load K [BLOCK_N, D], V [BLOCK_N, D]
        k = load_kv_block(K, pid_batch, pid_head, start_n, BLOCK_N, D)
        v = load_kv_block(V, pid_batch, pid_head, start_n, BLOCK_N, D)
        
        # S = Q @ K^T [BLOCK_M, BLOCK_N]
        s = tl.dot(q, tl.trans(k)) * (1.0 / tl.sqrt(D))
        
        # Causal mask
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        s = tl.where(offs_m[:, None] >= offs_n[None, :], s, float('-inf'))
        
        # Online softmax
        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(q.dtype), v)
        m_i = m_new
    
    # Normalize
    acc = acc / l_i[:, None]
    store_output(O, acc, pid_batch, pid_head, pid_m, BLOCK_M, D)
```

## 11. Memory Access 分析

### 11.1 HBM IO 对比

| 方法 | HBM Reads | HBM Writes | Total IO |
|------|-----------|------------|----------|
| Standard | O(Nd + N²) | O(N² + Nd) | O(N² + Nd) |
| FlashAttention | O(N²d/M) | O(Nd) | O(N²d/M + Nd) |

N=4096, d=128, M=100KB:
- Standard: ~4096² × 2 = 32MB (score matrix alone)
- FlashAttention: ~4096² × 128 / 100K ≈ 21MB (但不写 score)

### 11.2 SRAM 使用

```
Q block: B_r × d × 2 bytes
K block: B_c × d × 2 bytes  
V block: B_c × d × 2 bytes
Score: B_r × B_c × 4 bytes (FP32 for softmax)
Output acc: B_r × d × 4 bytes (FP32)

Total: (B_r + 2B_c) × d × 2 + B_r × B_c × 4 + B_r × d × 4
```

## 12. Parallelism 分析

### FlashAttention-1
- 外层循环（KV blocks）：顺序
- 内层循环（Q blocks）：并行（不同 block 分配到不同 SM）
- 问题：Q blocks 数量可能不够填满所有 SM

### FlashAttention-2
- 外层循环（Q blocks）：并行
- 内层循环（KV blocks）：顺序
- 优势：Q blocks 通常更多，更好的 SM 利用率
- Warp 间：不同 warp 处理 KV block 的不同部分

## 13. Compute-bound / Memory-bound 判断

FlashAttention 将 attention 从 **memory-bound 转为 compute-bound**：
```
计算量: O(N²d) FLOPs（不变）
HBM IO: O(N²d/M)（大幅减少）

AI = O(N²d) / O(N²d/M × bytes_per_element) = O(M / bytes)

M = 100KB, FP16: AI ≈ 100K / 2 = 50K FLOPs/Byte
远超 roofline 拐点 → compute-bound
```

这意味着 FlashAttention 的性能上限由 Tensor Core TFLOPS 决定。

## 14. Profiling 指标

| 指标 | FlashAttention 期望值 | 标准 Attention |
|------|---------------------|---------------|
| Tensor Core Utilization | 60-75% | < 30% |
| HBM Bandwidth Utilization | 30-50% | > 80% |
| SM Occupancy | 50-75% | varies |
| Kernel Duration (S=4096) | ~2-5ms | ~10-20ms |

## 15. Benchmark 设计

```python
import torch
from flash_attn import flash_attn_func

seq_lens = [512, 1024, 2048, 4096, 8192, 16384, 32768]
B, H, D = 1, 32, 128

for S in seq_lens:
    q = torch.randn(B, S, H, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, S, H, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, S, H, D, device='cuda', dtype=torch.float16)
    
    # FlashAttention
    torch.cuda.synchronize()
    t_flash = benchmark(lambda: flash_attn_func(q, k, v, causal=True))
    
    # Standard (PyTorch math backend)
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        t_standard = benchmark(lambda: F.scaled_dot_product_attention(
            q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), is_causal=True))
    
    tflops = 4 * B * H * S * S * D / t_flash / 1e12  # approximate
    print(f"S={S:6d} | Flash: {t_flash*1000:.2f}ms ({tflops:.1f} TFLOPS) | Std: {t_standard*1000:.2f}ms | Speedup: {t_standard/t_flash:.1f}x")
```

## 16. 常见错误

1. **Block size 超过 SRAM**：B_r × B_c × d 超过 shared memory → launch failure
2. **Online softmax 精度**：rescaling 时 exp 下溢 → 使用 FP32 累加
3. **Causal mask 边界**：block 边界处 mask 处理不当
4. **Backward recomputation 遗漏**：忘记保存 logsumexp 用于 backward
5. **不适用于 decode**：decode 时 S_q=1，FlashAttention 的 tiling 优势消失

## 17. 实验任务

1. 对比 FlashAttention vs 标准 attention 在不同 seq_len 下的性能
2. 测量 FlashAttention 的 Tensor Core utilization（用 Nsight Compute）
3. 验证 FlashAttention 的数值精度（与 FP32 标准实现对比）
4. 实现简化版 Triton FlashAttention（不含 causal mask）
5. 分析 FlashAttention 在 seq_len < 512 时是否仍有优势

## 18. 习题 20 道

1. FlashAttention 减少了什么？计算量还是 IO？
2. 标准 attention 的 HBM IO 复杂度是多少？FlashAttention 呢？
3. 为什么 FlashAttention 需要 online softmax？
4. FlashAttention-2 相比 1 的主要改进是什么？
5. FlashAttention 的 backward pass 为什么需要 recomputation？
6. 对于 seq_len=32K, head_dim=128, FP16，标准 attention 的 score 矩阵需要多少内存？
7. FlashAttention 的 block size 如何选择？受什么限制？
8. 为什么 FlashAttention 将 attention 从 memory-bound 变为 compute-bound？
9. FlashAttention 适用于 decode phase 吗？为什么？
10. Causal mask 如何在 FlashAttention 中高效处理？
11. FlashAttention-3 针对 Hopper 架构做了什么优化？
12. 如何验证 FlashAttention 实现的数值正确性？
13. FlashAttention 的 Tensor Core utilization 通常是多少？瓶颈在哪？
14. 为什么 FlashAttention 对长序列的加速比更大？
15. FlashAttention 与 xformers memory_efficient_attention 的区别？
16. 在 GQA 场景下，FlashAttention 如何处理 K/V head 数量不同？
17. FlashAttention 的 SRAM 使用量如何计算？
18. 为什么 FlashAttention 不需要存储 attention weight 矩阵？
19. FlashAttention 对 dropout 的支持有什么特殊处理？
20. 如何用 Triton 实现一个简化版 FlashAttention？

## 19. 标准答案

1. 减少 HBM IO（从 O(N²) 到 O(N²d/M)），计算量不变（仍是 O(N²d)）。

2. 标准: O(N²) reads + writes（score 矩阵）。FlashAttention: O(N²d/M) reads（Q/K/V 的分块读取）。

3. 因为 K/V 被分块处理，每次只看到部分 score，需要 online 更新全局 max 和 sum。

4. (a) 外层循环改为遍历 Q blocks（更好的并行度）；(b) 减少非 matmul FLOPs；(c) warp 间更好的 work partitioning。

5. 如果保存 N×N 的 attention score 用于 backward，内存开销 O(N²) 太大。Recomputation 只需额外 O(N²d) 计算，但节省 O(N²) 内存。

6. 32K × 32K × 2 bytes × 32 heads = 32² × 1024² × 2 × 32 = 64GB → 不可能存储。

7. Block size 受 SRAM 容量限制：B_r×d + 2×B_c×d + B_r×B_c ≤ SRAM_size。典型 A100: B_r=128, B_c=64。

8. 因为减少了 HBM IO，计算量不变。AI 从 ~1 提升到 ~50K FLOPs/Byte，远超 roofline 拐点。

9. 不太适用。Decode 时 S_q=1，Q block 只有 1 行，tiling 的数据复用优势消失。Decode 应使用专门的 decode attention kernel。

10. 对于 Q block i 和 K block j：如果 j 完全在 i 之后 → 跳过整个 block；如果 j 完全在 i 之前 → 正常计算；如果 j 与 i 重叠 → 计算后 apply mask（设为 -inf）。

(后续答案略)

## 20. 复习卡片 30 张

1. Q: FlashAttention 的核心思想？ A: Tiling + online softmax，避免将 N×N score 矩阵写到 HBM
2. Q: 标准 attention 的 IO 复杂度？ A: O(N²)
3. Q: FlashAttention 的 IO 复杂度？ A: O(N²d/M)，M=SRAM size
4. Q: FlashAttention 改变了计算量吗？ A: 没有，仍是 O(N²d)
5. Q: Online softmax 在 FA 中的作用？ A: 允许分块计算 softmax，无需看到全部 score
6. Q: FA-2 vs FA-1 的主要改进？ A: 外层循环改为 Q blocks，更好的 warp partitioning
7. Q: FA 的 backward 为什么用 recomputation？ A: 避免存储 O(N²) 的 attention score
8. Q: FA 适用于 decode 吗？ A: 不太适用，S_q=1 时 tiling 优势消失
9. Q: FA 的 Tensor Core utilization？ A: 60-75%（FA-2）
10. Q: Causal mask 在 FA 中如何优化？ A: 跳过完全被 mask 的 KV blocks
(后续卡片略)
