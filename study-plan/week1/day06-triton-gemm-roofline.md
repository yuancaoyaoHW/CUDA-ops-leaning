# Day 6：Triton GEMM + Roofline 总结

## 学习目标
- 用 Triton 实现高性能 GEMM（利用 Tensor Core）
- 理解 Triton 的 autotune 机制
- 整理所有 kernel 的 roofline 分析表
- 开始 LeetCode 刷题

---

## 上午（3h）- Triton GEMM

### 跟官方 Tutorial 实现

```python
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8},
                      num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)

    # Swizzle for better L2 cache hit
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # 计算指针偏移
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # A 和 B 的起始指针
    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # 累加器
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 主循环
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)

        acc += tl.dot(a, b)  # 调用 Tensor Core!

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # 写回
    c = acc.to(tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + offs_cm[:, None] * stride_cn + offs_cn[None, :] * stride_cn
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)
```

### 关键概念

**`tl.dot(a, b)`**：
- 底层调用 Tensor Core（WMMA/WGMMA 指令）
- 要求 shape 是 16 的倍数（Tensor Core 的最小粒度）
- fp16 输入，fp32 累加 → 精度和速度兼顾

**`@triton.autotune`**：
- 自动搜索最优的 BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages
- `key=['M', 'N', 'K']`：当这些参数变化时重新搜索
- `num_stages`：pipeline stages 数量（对应 double/triple buffering）
- `num_warps`：每个 block 的 warp 数量

**GROUP_SIZE_M（L2 Cache Swizzle）**：
- 问题：如果按行优先遍历 output tiles，相邻 block 访问的 B 列不同 → L2 miss
- 解决：把 blocks 分组，组内的 blocks 访问相邻的 B 列 → L2 hit
- 效果：大矩阵时可以提升 10-20%

### Benchmark 对比

```python
import torch

M = N = K = 2048
A = torch.randn(M, K, device='cuda', dtype=torch.float16)
B = torch.randn(K, N, device='cuda', dtype=torch.float16)

# cuBLAS
t_cublas = triton.testing.do_bench(lambda: torch.mm(A, B))

# Triton
t_triton = triton.testing.do_bench(lambda: matmul(A, B))

# 你的 CUDA V2
t_cuda = triton.testing.do_bench(lambda: cuda_gemm_v2(A, B))

flops = 2 * M * N * K
print(f"cuBLAS: {flops/t_cublas/1e9:.1f} TFLOPS")
print(f"Triton: {flops/t_triton/1e9:.1f} TFLOPS ({t_cublas/t_triton*100:.0f}% of cuBLAS)")
print(f"CUDA V2: {flops/t_cuda/1e9:.1f} TFLOPS ({t_cublas/t_cuda*100:.0f}% of cuBLAS)")
```

### 为什么 Triton 比手写 CUDA 容易接近 cuBLAS

- Triton 自动处理 shared memory tiling 和 double buffering
- `tl.dot` 直接映射到 Tensor Core
- autotune 搜索最优配置
- 但仍然比 cuBLAS 慢 10-30%（cuBLAS 有手写 PTX、更精细的 warp scheduling）

---

## 下午（2h）- Roofline 总结

### 所有 Kernel 的性能数据表

| Kernel | Shape | AI (FLOP/B) | Bound | 优化方向 |
|--------|-------|-------------|-------|----------|
| Vector add | N=1M, fp16 | 0.25 | Memory | 已达峰值，无法优化 |
| Reduction | N=1M, fp32 | 0.125 | Memory | Warp shuffle 已接近峰值 |
| Softmax | [4096, 8192], fp16 | 1.25 | Memory | Fuse into attention |
| RMSNorm | [4096, 4096], fp16 | ~0.8 | Memory | Fuse with residual |
| RMSNorm fused | [4096, 4096], fp16 | ~0.6 | Memory | 已接近峰值 |
| GEMM (large) | [2048,2048,2048], fp16 | 682 | Compute | Tensor Core, tiling |
| GEMM (small M) | [1,4096,4096], fp16 | 1.0 | Memory | Batching |
| Attention (prefill) | seq=2048, d=128 | ~50+ | Compute | FlashAttention |
| Attention (decode) | batch=1, seq=2048 | ~1 | Memory | Batching, KV quantize |

### 面试中怎么用这个表

面试官问："给你一个 kernel，怎么判断优化方向？"

回答框架：
```
1. 计算 Arithmetic Intensity = FLOPs / Bytes
2. 和 GPU 的 ridge point 比较（A100: ~312, H100: ~530, 4060: ~687 for fp16）
3. AI < ridge point → memory-bound → 优化方向：
   - Kernel fusion（减少 global memory round-trip）
   - Vectorized load（提高 bandwidth utilization）
   - 数据压缩（FP8/INT4 减少数据量）
4. AI > ridge point → compute-bound → 优化方向：
   - Tensor Core（WMMA/WGMMA）
   - 更好的 tiling（提高 occupancy 和 ILP）
   - 算法优化（减少计算量）
```

### 关键洞察（面试加分）

```
LLM 推理中的两个阶段：
  Prefill: 大 batch GEMM → compute-bound → 用 Tensor Core 加速
  Decode: batch=1 的 GEMV → memory-bound → 用 batching + KV quantize 加速

这就是为什么：
  - FlashAttention 对 prefill 有效（减少 IO，让 compute 成为瓶颈）
  - Batching 对 decode 有效（amortize KV cache 读取）
  - KV cache FP8 对 decode 有效（减少数据量）
  - PD 分离有意义（两个阶段的优化方向完全不同）
```

---

## 晚上（1.5h）- LeetCode

### LRU Cache (146) - 必做

```python
class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.cache = {}  # key -> node
        self.head = Node(0, 0)  # dummy head (most recent)
        self.tail = Node(0, 0)  # dummy tail (least recent)
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, key: int) -> int:
        if key in self.cache:
            node = self.cache[key]
            self._remove(node)
            self._add_to_head(node)
            return node.val
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self._remove(self.cache[key])
        node = Node(key, value)
        self._add_to_head(node)
        self.cache[key] = node
        if len(self.cache) > self.cap:
            lru = self.tail.prev
            self._remove(lru)
            del self.cache[lru.key]
```

和 AI Infra 的关系：
- vLLM 的 block manager 用类似 LRU 策略 evict KV cache blocks
- PagedAttention 的物理块回收就是 LRU eviction

### 为什么要刷这题

面试中可能直接问：
- "设计一个 KV Cache 的 block allocator，支持 O(1) 的 allocate 和 free"
- "如果显存不够了，怎么决定 evict 哪个 request 的 KV cache？"

---

## 日检（20 分钟）

1. **口述**（5min）：给定一个 element-wise add kernel（N=10M, fp16），计算 AI，判断 bound 类型，说出理论峰值性能
   - 答：AI = 1 FLOP / 6 bytes ≈ 0.17, memory-bound, 峰值 = 0.17 * 256 GB/s = 43 GFLOPS

2. **口述**（5min）：Triton 的 `tl.dot` 底层调用什么？对 shape 有什么要求？
   - 答：Tensor Core (WMMA/WGMMA)，shape 必须是 16 的倍数

3. **口述**（5min）：LLM 推理的 prefill 和 decode 分别是什么 bound？为什么？
   - 答：Prefill compute-bound（大矩阵乘），Decode memory-bound（GEMV，读整个 KV cache）

4. **手写**（5min）：LRU Cache 的 get 和 put 的时间复杂度？用什么数据结构？
   - 答：O(1)，HashMap + Doubly Linked List

---

## 参考资料

- Triton official tutorial: Matrix Multiplication
- leimao/CUDA-GEMM-Optimization (GitHub)
- Roofline: An Insightful Visual Performance Model (Williams et al., 2009)
