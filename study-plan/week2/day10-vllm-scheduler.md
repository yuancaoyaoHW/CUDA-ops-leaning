# Day 10：vLLM Scheduler 源码阅读

## 学习目标
- 读懂 vLLM scheduler 的核心调度逻辑
- 理解 Block Manager 的物理块分配
- 画出完整的 request 生命周期流程图
- 输出分析文档

---

## 上午（3h）- 源码阅读

### 入口：LLMEngine.step()

文件：`vllm/engine/llm_engine.py`

```python
def step(self):
    # 1. Scheduler 决定本次 step 处理哪些 request
    scheduler_outputs = self.scheduler.schedule()

    # 2. 准备模型输入
    model_input = self.model_runner.prepare_input(scheduler_outputs)

    # 3. 执行模型 forward
    output = self.model_runner.execute_model(model_input)

    # 4. 处理输出（采样、判断是否完成）
    request_outputs = self.process_outputs(output, scheduler_outputs)

    return request_outputs
```

### Scheduler 核心逻辑

文件：`vllm/core/scheduler.py`

```python
def schedule(self):
    # 优先级：swap_in > prefill > 继续 decode

    # Step 1: 尝试 swap in（把之前换出的 request 换回来）
    while self.swapped and can_swap_in():
        seq = self.swapped.pop(0)
        self._swap_in(seq)
        self.running.append(seq)

    # Step 2: 尝试 prefill 新 request
    while self.waiting and can_prefill():
        seq = self.waiting.pop(0)
        self._allocate_blocks(seq)  # 分配 KV cache blocks
        self.running.append(seq)

    # Step 3: 如果显存不够，preempt running 中的 request
    while not enough_blocks_for_decode():
        victim = self.running.pop()  # FIFO or priority
        if self.swap_space_available():
            self._swap_out(victim)
            self.swapped.append(victim)
        else:
            self._recompute(victim)
            self.waiting.appendleft(victim)

    return SchedulerOutputs(
        scheduled_seq_groups=self.running,
        blocks_to_swap_in=...,
        blocks_to_swap_out=...,
        blocks_to_copy=...,
    )
```

### Block Manager

文件：`vllm/core/block_manager.py`

```python
class BlockManager:
    def __init__(self, block_size, num_gpu_blocks, num_cpu_blocks):
        self.block_size = block_size  # tokens per block (e.g., 16)
        self.gpu_allocator = BlockAllocator(num_gpu_blocks)
        self.cpu_allocator = BlockAllocator(num_cpu_blocks)
        self.block_tables = {}  # seq_id -> list of physical block ids

    def allocate(self, seq):
        """为新 sequence 分配初始 blocks"""
        num_blocks = ceil(seq.prompt_len / self.block_size)
        blocks = [self.gpu_allocator.allocate() for _ in range(num_blocks)]
        self.block_tables[seq.seq_id] = blocks

    def append_slot(self, seq):
        """Decode 时追加一个 token，可能需要新 block"""
        block_table = self.block_tables[seq.seq_id]
        last_block = block_table[-1]

        if last_block.is_full():
            # 需要分配新 block
            new_block = self.gpu_allocator.allocate()
            block_table.append(new_block)

    def swap_out(self, seq):
        """把 GPU blocks 换到 CPU"""
        gpu_blocks = self.block_tables[seq.seq_id]
        cpu_blocks = [self.cpu_allocator.allocate() for _ in gpu_blocks]
        # 异步拷贝 GPU → CPU
        for gpu_b, cpu_b in zip(gpu_blocks, cpu_blocks):
            self.cache_engine.swap_out(gpu_b, cpu_b)
        # 释放 GPU blocks
        for b in gpu_blocks:
            self.gpu_allocator.free(b)
        self.block_tables[seq.seq_id] = cpu_blocks

    def can_allocate(self, seq):
        """检查是否有足够的 free blocks"""
        required = ceil(seq.prompt_len / self.block_size)
        return self.gpu_allocator.get_num_free_blocks() >= required
```

### Preemption 策略

```
两种 preemption 方式：

1. Swap（换出到 CPU）:
   - 把 victim 的 KV cache blocks 从 GPU 拷贝到 CPU
   - 优点：恢复时不需要重新 prefill
   - 缺点：需要 CPU 内存，swap in/out 有延迟
   - 适用：CPU 内存充足，request 已经生成了很多 token

2. Recompute（丢弃重算）:
   - 丢弃 victim 的 KV cache，放回 waiting queue
   - 恢复时需要重新 prefill
   - 优点：不需要 CPU 内存
   - 缺点：浪费已经做过的计算
   - 适用：request 刚开始 decode（重算代价小）

选择策略：
  - 如果有 CPU swap space → 优先 swap
  - 如果没有 → recompute
  - 高级策略：根据已生成 token 数决定（多的 swap，少的 recompute）
```

---

## 下午（2h）- 画流程图 + 写文档

### Request 完整生命周期

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Request                             │
│  "Explain quantum computing in simple terms"                 │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Tokenizer: text → token_ids [1, 45, 892, 12, ...]          │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Waiting Queue (等待 prefill)                                │
│  排队等待 scheduler 选中                                      │
└─────────────────────┬───────────────────────────────────────┘
                      │ scheduler.schedule() 选中
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Prefill Phase                                               │
│  - BlockManager 分配 KV cache blocks                         │
│  - 一次 forward 处理所有 input tokens                         │
│  - 计算所有 input tokens 的 KV cache                          │
│  - 生成第一个 output token                                    │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Running Queue (Decode Loop)                                 │
│  每个 step:                                                   │
│  - 输入: 上一步生成的 token                                    │
│  - 读取 KV cache                                             │
│  - 计算 attention (decode attention, memory-bound)            │
│  - 生成下一个 token                                           │
│  - 追加 KV cache (可能需要新 block)                            │
│                                                              │
│  循环直到: EOS token 或 max_length                             │
└─────────────────────┬───────────────────────────────────────┘
                      │ 可能被 preempt
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  [如果被 preempt]                                            │
│  Swap: KV cache → CPU memory → Swapped Queue                │
│  或 Recompute: 丢弃 KV cache → Waiting Queue                 │
│  等待资源释放后恢复                                            │
└─────────────────────────────────────────────────────────────┘
                      │ 正常完成
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Output                                                      │
│  - Detokenize: token_ids → text                              │
│  - 释放 KV cache blocks                                      │
│  - 返回给 client                                             │
└─────────────────────────────────────────────────────────────┘
```

### Scheduler 决策流程图

```
schedule() 被调用
      │
      ▼
┌─ swapped queue 非空？─┐
│  YES                   │ NO
│  ▼                     │
│  有足够 GPU blocks     │
│  swap in？             │
│  YES → swap in         │
│  NO → 跳过             │
└────────┬───────────────┘
         │
         ▼
┌─ waiting queue 非空？─┐
│  YES                   │ NO
│  ▼                     │
│  有足够 GPU blocks     │
│  做 prefill？          │
│  YES → allocate +      │
│         prefill         │
│  NO → 跳过             │
└────────┬───────────────┘
         │
         ▼
┌─ running 中所有 seq    ─┐
│  都有足够 blocks        │
│  继续 decode？          │
│  YES → 正常 decode      │
│  NO → preempt victim    │
└─────────────────────────┘
```

### 输出文档

写入 `docs/vllm-scheduler-analysis.md`，包含：
1. 架构总览图
2. Request 生命周期
3. Scheduler 决策逻辑
4. Block Manager 的分配/释放
5. Preemption 策略对比

---

## 晚上（1.5h）- LeetCode

### Trie (208) - 实现前缀树

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True

    def search(self, word):
        node = self._find(word)
        return node is not None and node.is_end

    def startsWith(self, prefix):
        return self._find(prefix) is not None

    def _find(self, s):
        node = self.root
        for ch in s:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node
```

和 AI Infra 的关系：SGLang 的 RadixAttention 用 Radix Tree（压缩前缀树）管理 KV cache prefix。

### Top-K (215)

```python
import heapq
def findKthLargest(nums, k):
    # 维护大小为 k 的最小堆
    return heapq.nlargest(k, nums)[-1]
```

和 AI Infra 的关系：Top-K sampling、MoE 的 top-k routing。

---

## 日检（20 分钟）

1. **闭卷画图**（10min）：画出 vLLM 的 request 生命周期（从 client request 到 output）
2. **口述**（5min）：Scheduler 的三个队列分别是什么？什么时候 request 在它们之间转移？
3. **口述**（5min）：Preemption 的两种策略是什么？各自的优缺点？

---

## 参考资料

- vLLM 源码: https://github.com/vllm-project/vllm
- vLLM 论文: Efficient Memory Management for LLM Serving with PagedAttention
- Aleksa Gordić, "Inside vLLM: Anatomy of a High-Throughput LLM Inference System"
