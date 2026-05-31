# Coding Mock 题集
## 学习目标
本章把 LLM inference/framework/CUDA 面试中的高频题压缩成可计时训练的 mock 题。练完后应能:

1. 在 30 分钟内写出 CUDA kernel baseline, 说明线程映射、mask、同步和验证方法。
2. 在 45 分钟内完成 inference 子系统设计, 覆盖 API、数据结构、并发、失败和指标。
3. 在 20 分钟内实现 sampling/search/cache 相关基础算法, 并说明复杂度。
4. 用评分标准检查答案, 先保证正确性, 再讨论性能和工程化。
## 前置知识
1. CUDA: grid/block/thread、warp、shared memory、barrier、coalesced access。
2. Kernel 验证: PyTorch reference、aligned 和 non-aligned size、tolerance。
3. LLM serving: prefill/decode、KV cache、continuous batching、streaming、SLO。
4. 系统设计: rate limit、admission control、backpressure、metrics、failure mode。
5. 算法: heap、beam search、prefix tree、token bucket、free list。
## 核心内容
题集分三类:

1. CUDA kernel: vector add、reduction、softmax、GEMM tiling。
2. 系统设计: rate limiter、KV cache allocator、batch scheduler。
3. 算法: top-k sampling、beam search、prefix tree。

答题顺序建议: 澄清输入和约束 -> 给 baseline -> 说明边界和测试 -> 分析复杂度/性能 -> 讨论优化和故障。
## 完整的问答/题目
### CUDA 题 1: Vector Add
题目: 给定 float32 向量 `a`、`b`, 长度 `n`, 输出 `c[i] = a[i] + b[i]`, 支持任意 `n`。
时间限制: 10 分钟。
参考实现:

```cpp
__global__ void vec_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

void launch(const float* a, const float* b, float* c, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    vec_add<<<grid, block>>>(a, b, c, n);
}
```
评分标准: `idx < n` mask 正确; 相邻线程访问相邻地址; launch config 可解释; 测 `n=1024` 和 `n=1000`。
常见错误: `grid = n / block` 丢尾部; 无 mask 越界; 只测整除长度; 线程映射说不清。
### CUDA 题 2: Block Reduction Sum
题目: 实现 `sum(x)` 第一阶段 reduction。每个 block 读一段 float32, shared memory 内归约, 输出 partial sum。
时间限制: 20 分钟。
参考实现:

```cpp
__global__ void reduce_sum_stage1(const float* x, float* partial, int n) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + tid;
    float v = 0.0f;
    if (idx < n) v += x[idx];
    if (idx + blockDim.x < n) v += x[idx + blockDim.x];
    smem[tid] = v;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = smem[0];
}
```
评分标准: tail mask 正确; 每轮 shared memory 读写后同步; 能说明第二阶段再归约 partial; 测浮点 tolerance。
常见错误: 忘记 barrier; shared memory 大小不匹配; 只支持 `n <= blockDim`; 不解释浮点归约误差。
### CUDA 题 3: Row-wise Softmax
题目: 给定 `x[rows, cols]`, 每行 softmax。假设 `cols <= 1024`, 一个 block 处理一行, 要求数值稳定。
时间限制: 25 分钟。
参考实现:

```cpp
__global__ void row_softmax(const float* x, float* y, int rows, int cols) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float v = -INFINITY;
    if (tid < cols) v = x[row * cols + tid];
    smem[tid] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    float m = smem[0];

    float e = 0.0f;
    if (tid < cols) e = expf(x[row * cols + tid] - m);
    smem[tid] = e;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid < cols) y[row * cols + tid] = e / smem[0];
}
```
评分标准: 先减 row max; padding lane 不写输出; `cols < blockDim` 正确; 能说明 `cols > 1024` 需要多 block 或 online softmax。
常见错误: 直接 `exp(x)` overflow; max 初始值用 0 导致全负行错误; denominator 未说明; 忘记 mask。
### CUDA 题 4: GEMM Tiling
题目: 简化 tiled GEMM, `C[M,N] = A[M,K] * B[K,N]`, float32 row-major, 用 shared memory tile 复用数据。
时间限制: 30 分钟。
参考实现:

```cpp
template<int TILE>
__global__ void gemm_tiled(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < K; t += TILE) {
        int a_col = t + threadIdx.x;
        int b_row = t + threadIdx.y;
        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();
        for (int i = 0; i < TILE; ++i) acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}
```
评分标准: A/B load 和 C store 都 mask; shared memory 复用讲清楚; 两个 barrier 都能解释; 能提到 coalescing、bank conflict、register tiling、tensor core。
常见错误: 只 mask store; 第二个 barrier 缺失; row/col 映射反; 直接跳 WMMA 但没有 baseline。
### 系统设计题 1: LLM Serving Rate Limiter
题目: 设计 LLM API rate limiter, 支持按用户限制 requests/min、input tokens/min、output tokens/min, 兼顾流式输出和多实例部署。
时间限制: 45 分钟。

参考实现:

```python
def admit(req):
    input_tokens = tokenizer.count(req.prompt)
    reserve_output = req.max_new_tokens
    keys = [
        ("rpm", req.org, 1),
        ("input_tpm", req.org, input_tokens),
        ("output_tpm", req.org, reserve_output),
    ]
    ok, wait_ms = redis_token_bucket_try_take(keys, clock.now_ms())
    if not ok:
        return Reject(429, wait_ms)
    return Permit(input_tokens, reserve_output)

def finish(req, permit, actual_output_tokens):
    refund = max(0, permit.reserve_output - actual_output_tokens)
    redis_token_bucket_refund(("output_tpm", req.org), refund)
```
评分标准: 三个维度都限制; output tokens 用 reservation/refund 或增量扣减; 分布式扣减原子; 429 带 retry 信息; Redis 故障有 fail-open/fail-closed 策略。
常见错误: 只限 QPS; 输出生成后才扣; 多实例用本地计数; 不处理 tokenizer 估算误差。
### 系统设计题 2: KV Cache Allocator
题目: 设计 KV cache allocator, 支持变长 sequence、释放、复用、碎片控制和 OOM。
时间限制: 45 分钟。

参考实现:

```python
class KVAllocator:
    def __init__(self, num_pages):
        self.free = list(range(num_pages))
        self.tables = {}

    def allocate_for_seq(self, seq_id, tokens, block_tokens):
        need = ceil_div(tokens, block_tokens)
        if len(self.free) < need:
            raise OOM("kv pages exhausted")
        pages = [self.free.pop() for _ in range(need)]
        self.tables[seq_id] = pages
        return pages

    def append_token(self, seq_id, pos, block_tokens):
        if pos % block_tokens != 0:
            return None
        if not self.free:
            raise OOM("cannot grow decode cache")
        page = self.free.pop()
        self.tables[seq_id].append(page)
        return page

    def free_seq(self, seq_id):
        self.free.extend(self.tables.pop(seq_id, []))
```
评分标准: block table 和 physical page 分离; 固定 page 降低外部碎片; prefill 批量分配, decode 边界追加; 完成/取消/超时释放; OOM 触发 admission/backpressure/evict。
常见错误: 每请求连续 malloc; sequence done 不归还 page; decode 增长失败无处理; 忽略 beam/prefix 共享的 copy-on-write。
### 系统设计题 3: Continuous Batch Scheduler
题目: 设计 batch scheduler 同时处理 prefill 和 decode, 目标是在满足 latency SLO 的同时提高 GPU token throughput。
时间限制: 45 分钟。

参考实现:

```python
def schedule_tick(waiting, running, token_budget):
    batch = []
    budget = token_budget
    for seq in running.by_deadline():
        if budget <= 0:
            break
        batch.append(DecodeStep(seq))
        budget -= 1
    while waiting and budget > 0 and kv_allocator.has_capacity():
        req = waiting.peek()
        chunk = min(req.remaining_prefill, budget, MAX_PREFILL_CHUNK)
        if chunk == 0:
            break
        waiting.pop_if_new(req)
        batch.append(PrefillChunk(req, chunk))
        budget -= chunk
    return batch
```
评分标准: 区分 prefill/decode; 用 token budget 和 KV capacity 做 admission; decode 优先但要公平; 长 prompt chunked prefill; metrics 覆盖 TTFT、ITL、queue delay、tokens/s、OOM。
常见错误: 固定 request batch size; 长 prefill 阻塞 decode; cancel 后 KV 泄漏; 只看 throughput 不看用户可见延迟。
### 算法题 1: Top-k Sampling
题目: 给定 logits、`k`、`temperature`, 实现 top-k sampling, 返回 token id。
时间限制: 20 分钟。
参考实现:

```python
def top_k_sample(logits, k, temperature=1.0):
    import heapq, math, random
    if not logits or k <= 0:
        raise ValueError("bad input")
    temperature = max(temperature, 1e-6)
    pairs = heapq.nlargest(min(k, len(logits)), enumerate(logits), key=lambda p: p[1])
    m = max(v for _, v in pairs)
    weights = [math.exp((v - m) / temperature) for _, v in pairs]
    r = random.random() * sum(weights)
    acc = 0.0
    for (idx, _), w in zip(pairs, weights):
        acc += w
        if acc >= r:
            return idx
    return pairs[-1][0]
```
评分标准: 只在 top-k 内采样; 减 max 保证稳定; heap 复杂度 `O(V log k)`; 处理 `k > vocab_size` 和非法输入。
常见错误: top-k 后取 argmax; 全 vocab softmax 后再截断; temperature 方向写反; 无随机种子测试。
### 算法题 2: Beam Search
题目: 实现简化 beam search。`step_fn(prefix) -> log_probs`, beam size 为 `B`, 最多 `max_len`, 遇 eos 可结束。
时间限制: 25 分钟。
参考实现:

```python
def beam_search(step_fn, bos, eos, beam_size, max_len, length_penalty=0.0):
    beams = [([bos], 0.0, False)]
    for _ in range(max_len):
        cand = []
        for tokens, score, done in beams:
            if done:
                cand.append((tokens, score, True))
                continue
            for tok, lp in top_items(step_fn(tokens), beam_size):
                cand.append((tokens + [tok], score + lp, tok == eos))
        def norm(item):
            tokens, score, _ = item
            return score / (len(tokens) ** length_penalty if length_penalty > 0 else 1.0)
        beams = sorted(cand, key=norm, reverse=True)[:beam_size]
        if all(done for _, _, done in beams):
            break
    return max(beams, key=lambda item: item[1])[0]
```
评分标准: 每步扩展所有 beam; eos 后不再扩展; log probability 累加; 能说明 length penalty 和复杂度。
常见错误: 退化成 greedy; 概率相乘 underflow; eos 后继续扩展; 最终排序不考虑长度偏置。
### 算法题 3: Prefix Tree
题目: 实现 prefix tree, 以 token id 序列为 key, 支持 insert 和 longest-prefix-match, 用于 prompt cache。
时间限制: 25 分钟。
参考实现:

```python
class Node:
    def __init__(self):
        self.children = {}
        self.value = None

class PrefixTree:
    def __init__(self):
        self.root = Node()

    def insert(self, tokens, cache_ref):
        cur = self.root
        for tok in tokens:
            cur = cur.children.setdefault(tok, Node())
        cur.value = cache_ref

    def longest_prefix(self, tokens):
        cur, best_len, best_value = self.root, 0, None
        for i, tok in enumerate(tokens):
            if tok not in cur.children:
                break
            cur = cur.children[tok]
            if cur.value is not None:
                best_len, best_value = i + 1, cur.value
        return best_len, best_value
```
评分标准: 返回最长已缓存前缀; insert/lookup `O(sequence_length)`; cache key 包含模型、tokenizer、adapter 版本; 有 LRU、引用计数和释放策略。
常见错误: 只支持完整命中; token ids 字符串拼接有歧义; 跨模型复用 KV; 删除时泄漏 cache_ref。
## 追问方向与深入点
1. Vector add 如何用 `float4`, 对齐条件是什么?
2. Reduction 如何用 warp shuffle 减少 shared memory 和 barrier?
3. Softmax 如何扩展到 attention online softmax?
4. GEMM baseline 到 tensor core 需要补哪些概念?
5. Rate limiter 在 Redis 故障时如何按租户等级降级?
6. KV allocator 如何支持 beam search shared prefix 和 copy-on-write?
7. Scheduler 如何动态平衡 throughput、TTFT 和 ITL?
8. Top-k、top-p、temperature、repetition penalty 的组合顺序如何设计?
9. Beam search 如何批量化 `step_fn`, 避免每个 beam 单独 forward?
10. Prefix tree 在多进程 serving 中如何同步、过期和回收?
## 评分标准
总分 100:

1. 正确性 35: mask、边界、数值稳定、资源释放。
2. 性能意识 25: 内存访问、同步、复杂度、GPU 利用率。
3. 工程完整性 20: API、错误处理、测试、metrics、并发。
4. 表达 10: 先 baseline 再优化, 术语准确, 数据流清楚。
5. 追问 10: 能承认限制, 给出下一层改进方向。

单题 1-5 分: 1 分方向错误; 2 分 baseline 有关键漏洞; 3 分正确但优化少; 4 分正确且能讲 tradeoff; 5 分能连接 LLM serving 场景并稳定应对追问。
## 复习卡片 15 张
1. Q: CUDA 任意长度数组最重要的边界是什么? A: 所有 global load/store 都要有 `idx < n` 或等价 mask。
2. Q: vector add 的主要性能条件是什么? A: contiguous coalesced load/store 和足够并行度。
3. Q: reduction 为什么要 barrier? A: 下一轮读取依赖上一轮所有线程写完 shared memory。
4. Q: reduction 为什么和 PyTorch 可能不完全一致? A: 浮点归约顺序不同导致舍入误差。
5. Q: softmax 为什么减 max? A: 防 overflow, 且 softmax 对平移不变。
6. Q: softmax max 初始值为什么不能用 0? A: 全负行会被错误抬高。
7. Q: tiled GEMM 的 shared memory 价值是什么? A: 复用 A/B tile, 降低 global memory 带宽压力。
8. Q: GEMM tail mask 应用在哪? A: A load、B load、C store。
9. Q: LLM rate limiter 为什么按 token 限制? A: 成本主要由 input/output tokens 和 decode 时长决定。
10. Q: output token 为什么预留? A: 生成前未知实际长度, 预留防止超卖, 结束后退还。
11. Q: KV cache 为什么分页? A: 支持变长复用, 降低连续分配碎片。
12. Q: continuous batching 为什么 decode 优先? A: decode 决定 streaming inter-token latency。
13. Q: top-k sampling 如何优化复杂度? A: heap/select 找 top-k, 避免全量排序。
14. Q: beam search 为什么用 log probability? A: 避免概率连乘 underflow, 方便累加比较。
15. Q: prompt cache key 包含什么? A: token ids、模型版本、tokenizer 版本和 adapter/LoRA 标识。
