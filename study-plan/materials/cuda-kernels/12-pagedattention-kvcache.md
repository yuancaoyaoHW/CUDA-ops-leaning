# PagedAttention 与 KV Cache

## 1. 学习目标

- 理解 KV Cache 的动机与内存占用计算
- 掌握 PagedAttention 的虚拟内存管理思想
- 理解 block table 的逻辑-物理映射机制
- 能够分析 KV cache 的内存碎片问题与 PagedAttention 的解决方案
- 掌握 KV cache 在 prefill/decode 中的不同使用模式

## 2. 前置知识

- Attention 机制（Q×K^T → softmax → ×V）
- LLM 自回归生成过程
- 操作系统虚拟内存与分页概念
- GPU 内存管理

## 3. 核心术语表

| 术语 | 英文全称 | 定义 |
|------|----------|------|
| KV Cache | Key-Value Cache | 缓存历史 token 的 K/V 向量，避免重复计算 |
| PagedAttention | Paged Attention | 将 KV cache 分页管理的 attention 算法 |
| Block Table | Block Table | 逻辑 block 到物理 block 的映射表 |
| Physical Block | Physical Block | GPU 显存中固定大小的 KV 存储单元 |
| Logical Block | Logical Block | 请求视角的连续 KV 序列分块 |
| Block Size | Block Size | 每个 block 存储的 token 数（通常 16） |
| Fragmentation | Memory Fragmentation | 内存碎片，导致显存利用率低 |
| Preemption | Preemption | 抢占低优先级请求的 KV cache 空间 |
| Swap | Swap | 将 KV cache 从 GPU 换出到 CPU |
| Copy-on-Write | Copy-on-Write | 共享 block 被修改时才复制 |

## 4. 动机

### 4.1 KV Cache 的必要性

自回归生成中，每个新 token 需要与所有历史 token 做 attention：
```
Without KV Cache:
  Token 1: compute K₁, V₁
  Token 2: compute K₁, K₂, V₁, V₂  (重复计算 K₁, V₁!)
  Token n: compute K₁...Kₙ, V₁...Vₙ  (O(n²) 重复计算)

With KV Cache:
  Token 1: compute K₁, V₁ → cache
  Token 2: compute K₂, V₂ → cache, read K₁V₁ from cache
  Token n: compute Kₙ, Vₙ → cache, read K₁...Kₙ₋₁, V₁...Vₙ₋₁ from cache
```

节省：从 O(n²) 计算降为 O(n) 计算（每步只计算新 token 的 K/V）

### 4.2 KV Cache 内存占用

```
KV Cache per token = 2 × num_layers × num_kv_heads × head_dim × sizeof(dtype)

LLaMA-2 70B (GQA-8):
= 2 × 80 × 8 × 128 × 2 bytes (FP16)
= 327,680 bytes ≈ 320 KB per token

对于 seq_len = 4096:
= 320 KB × 4096 = 1.28 GB per request

如果并发 32 个请求:
= 1.28 GB × 32 = 41 GB → 超过单卡 80GB 的一半！
```

### 4.3 传统 KV Cache 的问题

**预分配方式**：为每个请求预分配 max_seq_len 的 KV cache
```
问题 1: 内部碎片
  请求实际长度 500，预分配 4096 → 浪费 87.8%

问题 2: 外部碎片
  请求结束后释放的空间不连续，新请求无法使用

问题 3: 无法共享
  beam search 中多个 beam 的公共前缀无法共享 KV cache
```

### 4.4 PagedAttention 的解决方案

借鉴操作系统虚拟内存的分页机制：
- KV cache 被分成固定大小的 **block**（如 16 tokens）
- 每个请求维护一个 **block table**（逻辑→物理映射）
- Block 按需分配，不预分配
- 内部碎片最多浪费 1 个 block（最后一个 block 未满）
- 支持 copy-on-write 共享

## 5. 数学定义

### 5.1 KV Cache 大小计算

```
cache_size_per_token = 2 × L × H_kv × D × dtype_size

其中:
L = num_layers
H_kv = num_kv_heads
D = head_dim
dtype_size = 2 (FP16) 或 1 (INT8/FP8)
```

### 5.2 Block Table 映射

```
block_table[request_id][logical_block_idx] = physical_block_idx

physical_address(request_id, token_pos, layer, head, dim) =
    physical_block_pool[block_table[request_id][token_pos // block_size]]
    + (token_pos % block_size) × H_kv × D
    + head × D + dim
```

### 5.3 显存利用率

```
传统方式:
utilization = actual_tokens / (num_requests × max_seq_len)
典型值: 20-50%

PagedAttention:
utilization = actual_tokens / (allocated_blocks × block_size)
典型值: >95% (只有最后一个 block 有浪费)
```

## 6. 推导逻辑

### 6.1 PagedAttention Kernel 设计

标准 attention 假设 K/V 在内存中连续：
```
K[seq_pos] = base_ptr + seq_pos × stride
```

PagedAttention 中 K/V 不连续（分散在不同 physical block）：
```
K[seq_pos] = block_pool[block_table[seq_pos / block_size]] 
             + (seq_pos % block_size) × stride
```

Kernel 需要：
1. 查 block table 获取物理地址
2. 从不连续的 block 中 gather K/V
3. 计算 attention score
4. 写入新 token 的 K/V 到当前 block

### 6.2 vLLM Block Manager

```
BlockManager:
├── free_blocks: List[int]          # 空闲物理 block 列表
├── block_tables: Dict[req_id, List[int]]  # 每个请求的 block table
├── ref_count: Dict[int, int]       # 每个物理 block 的引用计数
│
├── allocate(req_id, num_tokens)    # 分配 block
├── free(req_id)                    # 释放 block
├── fork(src_req, dst_req)          # copy-on-write fork
└── swap_out/swap_in(req_id)        # GPU↔CPU 换入换出
```

### 6.3 Copy-on-Write（用于 beam search）

```
Beam 1: [block_A, block_B, block_C]  ref_count: A=2, B=2, C=1
Beam 2: [block_A, block_B, block_D]  ref_count: D=1

当 Beam 1 需要修改 block_B（写入新 token）:
1. 检查 ref_count[B] > 1 → 需要 copy
2. 分配新 block_E，复制 block_B 内容到 block_E
3. Beam 1 的 block_table[1] = E
4. ref_count[B] -= 1, ref_count[E] = 1
5. 写入新 token 到 block_E
```

## 7. 算子流程

### 7.1 PagedAttention Decode Kernel

```cuda
// vLLM 的 PagedAttention kernel (简化版)
__global__ void paged_attention_kernel(
    float* __restrict__ output,          // [num_heads, head_dim]
    const float* __restrict__ q,         // [num_heads, head_dim]
    const float* __restrict__ k_cache,   // [num_blocks, block_size, num_kv_heads, head_dim]
    const float* __restrict__ v_cache,   // [num_blocks, block_size, num_kv_heads, head_dim]
    const int* __restrict__ block_table, // [max_num_blocks_per_seq]
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    float scale
) {
    int head_idx = blockIdx.x;
    int kv_head_idx = head_idx / (num_heads / num_kv_heads);
    
    // Load q to registers/shared memory
    // ...
    
    // Online softmax over all KV blocks
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float acc[HEAD_DIM] = {0};
    
    int num_blocks = (seq_len + block_size - 1) / block_size;
    
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        // 查 block table 获取物理 block 地址
        int physical_block = block_table[block_idx];
        
        int block_start = block_idx * block_size;
        int block_end = min(block_start + block_size, seq_len);
        
        for (int pos = block_start; pos < block_end; pos++) {
            int offset_in_block = pos - block_start;
            
            // 从物理 block 读取 K
            const float* k_ptr = k_cache + 
                physical_block * block_size * num_kv_heads * head_dim +
                offset_in_block * num_kv_heads * head_dim +
                kv_head_idx * head_dim;
            
            // Compute dot product
            float score = dot_product(q_head, k_ptr, head_dim) * scale;
            
            // Online softmax update
            float new_max = fmaxf(max_score, score);
            float exp_diff = expf(max_score - new_max);
            float exp_score = expf(score - new_max);
            
            // Rescale accumulator
            for (int d = 0; d < head_dim; d++) {
                acc[d] *= exp_diff;
            }
            sum_exp = sum_exp * exp_diff + exp_score;
            max_score = new_max;
            
            // Accumulate V
            const float* v_ptr = v_cache + 
                physical_block * block_size * num_kv_heads * head_dim +
                offset_in_block * num_kv_heads * head_dim +
                kv_head_idx * head_dim;
            
            for (int d = 0; d < head_dim; d++) {
                acc[d] += v_ptr[d] * exp_score;
            }
        }
    }
    
    // Normalize and write output
    for (int d = 0; d < head_dim; d++) {
        output[head_idx * head_dim + d] = acc[d] / sum_exp;
    }
}
```

## 8. PyTorch baseline

```python
import torch

class KVCache:
    """简单的连续 KV Cache 实现"""
    def __init__(self, max_batch, max_seq_len, num_layers, num_kv_heads, head_dim, dtype=torch.float16):
        self.k_cache = torch.zeros(
            num_layers, max_batch, max_seq_len, num_kv_heads, head_dim,
            device='cuda', dtype=dtype)
        self.v_cache = torch.zeros_like(self.k_cache)
        self.seq_lens = torch.zeros(max_batch, dtype=torch.int32, device='cuda')
    
    def update(self, layer_idx, batch_idx, k, v, pos):
        """写入新 token 的 K/V"""
        self.k_cache[layer_idx, batch_idx, pos] = k
        self.v_cache[layer_idx, batch_idx, pos] = v
    
    def get(self, layer_idx, batch_idx, seq_len):
        """读取历史 K/V"""
        return (self.k_cache[layer_idx, batch_idx, :seq_len],
                self.v_cache[layer_idx, batch_idx, :seq_len])


class PagedKVCache:
    """PagedAttention 风格的 KV Cache"""
    def __init__(self, num_blocks, block_size, num_layers, num_kv_heads, head_dim, dtype=torch.float16):
        self.block_size = block_size
        self.k_cache = torch.zeros(
            num_layers, num_blocks, block_size, num_kv_heads, head_dim,
            device='cuda', dtype=dtype)
        self.v_cache = torch.zeros_like(self.k_cache)
        # block_tables[req_id] = list of physical block indices
        self.block_tables = {}
        self.free_blocks = list(range(num_blocks))
    
    def allocate_block(self):
        return self.free_blocks.pop()
    
    def free_request(self, req_id):
        for block in self.block_tables.pop(req_id, []):
            self.free_blocks.append(block)
```

## 9-20. (实验任务、习题、答案、复习卡片)

### 关键习题

1. 计算 LLaMA-3 8B (32 layers, 8 KV heads, head_dim=128, FP16) 的 KV cache per token 大小。
2. 如果 block_size=16，一个 seq_len=1000 的请求需要多少个 block？内部碎片是多少？
3. PagedAttention 相比连续 KV cache 的性能开销是什么？（提示：间接寻址）
4. Copy-on-Write 在 beam search 中如何节省内存？
5. 如果 GPU 有 80GB 显存，模型占 16GB，KV cache 最多能服务多少并发请求？（LLaMA-3 8B, avg_seq=2048）

### 关键答案

1. 2 × 32 × 8 × 128 × 2 = 131,072 bytes = 128 KB/token
2. ceil(1000/16) = 63 blocks。内部碎片 = (63×16 - 1000) × 128KB/token_in_block = 8 tokens × cache_per_token
3. 开销：每次访问 K/V 需要先查 block table（一次额外的 global memory read），且 K/V 不连续可能影响 coalescing。实际开销 < 5%。
4. 多个 beam 共享公共前缀的 block（ref_count > 1），只有分叉后的新 token 需要新 block。4 beam × 2048 tokens 从 4×2048 降为 ~2048 + 4×delta。
5. 可用显存 = 80 - 16 = 64GB。每请求 KV cache = 128KB × 2048 = 256MB。最大并发 = 64GB / 256MB = 256 请求。
