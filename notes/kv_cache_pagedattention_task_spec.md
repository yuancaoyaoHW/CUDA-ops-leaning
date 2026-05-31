# KV Cache / PagedAttention Week 5 Task Specification

## Scope

Turn the Week 5 KV cache, PagedAttention, and prefill/decode materials into executable toy packages. The reader is a future implementer who should be able to build the local tasks without reading vLLM source or downloading model assets.

Constraints:

- Do not use a vLLM runtime.
- Do not download model weights.
- Do not clone large serving repositories.
- Use synthetic tensors and synthetic request traces only.
- Keep CUDA/Triton work optional until the PyTorch reference and metadata tests are correct.
- Do not edit `study-plan/progress.yaml` while implementing these packages unless a separate operator verification task explicitly requires it.

## Week 5 Mapping

| Day | Package focus | Output |
| --- | --- | --- |
| 29 | Contiguous KV cache layout | `kv_cache_append` toy, memory footprint note |
| 30 | Paged block table | logical-to-physical mapping tests, page-size tradeoff note |
| 31 | PagedAttention dataflow | address-mapping diagram and decode dataflow spec |
| 32 | Scheduler interaction | synthetic prefill/decode workload inputs for later scheduler toy |
| 33 | Troubleshooting metrics | TTFT, TPOT, KV usage, fragmentation estimates |
| 35 | Interview review | mock answers for KV cache, PagedAttention, and failure modes |

## Package 1: `kv_cache_append` Contiguous KV Toy

### Owned File Paths

Suggested implementation artifacts:

- `kernels/python/kv_cache_append/__init__.py`
- `kernels/python/kv_cache_append/contiguous.py`
- `tests/test_kv_cache_append.py`
- `benchmarks/bench_kv_cache_append.py`
- `notes/kv_cache_append.md`
- `reports/json/kv_cache_append_bench.json`

Do not require Triton for the first slice. A PyTorch implementation is enough to make shape, offset, and prefill/decode behavior executable.

### Reference Behavior

Implement a contiguous cache that stores K and V for each request in a preallocated max-length region.

Reference layout:

```text
k_cache[layer, batch, position, kv_head, dim]
v_cache[layer, batch, position, kv_head, dim]
shape = [num_layers, max_batch, max_seq_len, num_kv_heads, head_dim]
```

Required operations:

- `append(layer, batch, k_new, v_new)`: write one decode token at `seq_lens[batch]`, then increment the sequence length.
- `append_many(layer, batch, k_tokens, v_tokens)`: write a prefill chunk shaped `[tokens, num_kv_heads, head_dim]`.
- `read_prefix(layer, batch, length=None)`: return K/V for positions `[0, length)`.
- `reset(batch)`: set the request sequence length to zero without clearing all data.
- `memory_footprint_bytes(dtype_size)`: compute `2 * layers * max_batch * max_seq_len * num_kv_heads * head_dim * dtype_size`.

Reference checks should compare reads against PyTorch slicing and concatenation:

```python
expected_k = torch.cat(k_tokens_written_for_batch, dim=0)
actual_k, actual_v = cache.read_prefix(layer, batch)
torch.testing.assert_close(actual_k, expected_k)
```

### Shape and Layout Requirements

Default shape set:

| Case | layers | batch | max_seq_len | kv_heads | head_dim | Why |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| tiny | 1 | 2 | 8 | 2 | 4 | readable debugging |
| block boundary prep | 2 | 3 | 32 | 4 | 16 | aligns with paged block tests |
| GQA proxy | 4 | 4 | 128 | 2 | 32 | fewer KV heads than query heads |
| LLM proxy | 32 | 8 | 2048 | 8 | 128 | memory estimate only unless hardware allows |

Support dtype values that PyTorch CUDA supports locally: fp32, fp16, and bf16 if available. CPU tests may use fp32 for metadata and mapping checks.

### Tests

Required tests:

- Append one decode token and read it back.
- Append a prefill chunk, then append decode tokens.
- Use variable sequence length across batches.
- Reject overflow when `seq_lens[batch] + tokens > max_seq_len`.
- Keep layer data independent.
- Include aligned and non-aligned lengths such as 16, 17, 31, and 32 because the later paged package uses block masking.
- Validate GQA/MQA shape implications at the cache level: cache stores `num_kv_heads`, not `num_query_heads`; query-to-KV head grouping belongs in attention, not in append.

### Benchmark Plan

`benchmarks/bench_kv_cache_append.py` should measure synthetic append bandwidth.

Metrics:

- append tokens per second
- effective GB/s for writes: `2 * tokens * layers * kv_heads * head_dim * dtype_size / seconds`
- read-prefix GB/s for contiguous reads
- memory footprint per request and per batch

Workloads:

- decode-only: repeatedly append one token per active request.
- prefill-then-decode: append prompt chunks, then one token per step.
- variable length: random prompt/output lengths from a fixed seed.

JSON fields:

```json
{
  "operator": "kv_cache_append",
  "layout": "contiguous",
  "dtype": "float16",
  "layers": 32,
  "batch": 8,
  "max_seq_len": 2048,
  "num_kv_heads": 8,
  "head_dim": 128,
  "tokens_appended": 8192,
  "ms": 1.23,
  "effective_write_gbps": 123.4,
  "memory_footprint_bytes": 4294967296
}
```

### Note Sections

`notes/kv_cache_append.md` should include:

- Operator: why KV cache avoids recomputing old K/V during decode.
- Layout: contiguous address formula and shape.
- Reference: PyTorch slicing behavior.
- Tests: variable length, overflow, aligned and non-aligned positions.
- Benchmark: append bandwidth and read bandwidth.
- Limitation: contiguous preallocation wastes memory when actual lengths are far below `max_seq_len`.
- Bridge to paging: explain why contiguous offsets are simple but fragmentation and over-reservation are expensive.

## Package 2: Paged Block Table Mapping Toy

### Owned File Paths

Suggested implementation artifacts:

- `kernels/python/paged_kv/__init__.py`
- `kernels/python/paged_kv/block_table.py`
- `kernels/python/paged_kv/paged_cache.py`
- `tests/test_paged_kv_block_table.py`
- `benchmarks/bench_paged_kv.py`
- `notes/paged_kv_cache.md`
- `reports/json/paged_kv_bench.json`

The first implementation can be pure Python plus PyTorch tensors. The important behavior is metadata correctness and address mapping.

### Data Model

Use fixed-size physical blocks and per-request logical block tables.

```text
k_blocks[layer, physical_block, offset_in_block, kv_head, dim]
v_blocks[layer, physical_block, offset_in_block, kv_head, dim]
shape = [num_layers, num_blocks, block_size, num_kv_heads, head_dim]

block_table[request_id][logical_block] = physical_block
seq_lens[request_id] = number of valid tokens
free_blocks = stack or queue of unallocated physical blocks
ref_count[physical_block] = number of request tables pointing at the block
```

Logical-to-physical address mapping:

```text
logical_block = token_pos // block_size
offset_in_block = token_pos % block_size
physical_block = block_table[request_id][logical_block]

address = blocks[layer, physical_block, offset_in_block, kv_head, dim]
```

### Required Operations

- `allocate_request(request_id)`: create empty metadata for a request.
- `append(request_id, layer, k_new, v_new)`: allocate a new physical block when appending crosses a block boundary.
- `append_many(request_id, layer, k_tokens, v_tokens)`: support prefill chunks that span multiple blocks.
- `read_token(request_id, layer, token_pos)`: map logical token position to a physical block and return K/V.
- `read_prefix(request_id, layer, length=None)`: gather valid tokens in logical order even if physical blocks are non-contiguous.
- `free(request_id)`: decrement refcounts and return blocks with zero refs to the free list.
- `stats()`: return allocated blocks, free blocks, valid tokens, reserved token slots, internal fragmentation, and utilization.

Optional copy-on-write:

- `fork(src_request_id, dst_request_id)`: share the source block table and increment refcounts.
- On append into a shared last block, allocate a new block, copy valid entries from the shared block, update the destination table, and decrement the old block refcount.
- If the shared request is exactly at a block boundary, allocate a fresh block without copying.

### Fragmentation Requirements

The toy should make fragmentation visible with concrete numbers.

Definitions:

```text
allocated_token_slots = allocated_blocks * block_size
valid_tokens = sum(seq_lens.values())
internal_fragmentation_tokens = allocated_token_slots - valid_tokens
utilization = valid_tokens / allocated_token_slots
```

For active requests, PagedAttention-style allocation should waste at most one partially filled block per request. External fragmentation is represented by the free block list being reusable regardless of request identity because blocks are fixed-size.

### Tests

Required tests:

- logical block 0 maps to a physical block and offsets 0 through `block_size - 1` read back in order.
- Appending token `block_size` allocates a new physical block.
- Variable sequence lengths across requests read back correctly.
- Physical block order does not need to match logical order; force a non-contiguous allocation pattern and verify logical reads.
- Freeing a middle request returns its physical blocks and a later request can reuse them.
- Fragmentation stats match hand-computed values for lengths such as 1, 15, 16, 17, and 31 when `block_size=16`.
- Block boundary writes do not corrupt the last token of the previous block.
- GQA/MQA shape implication: the block pool stores `num_kv_heads`; attention maps `num_query_heads` to KV heads with `kv_head = query_head // (num_query_heads // num_kv_heads)` when supported.
- Optional copy-on-write tests: forked requests share prefix blocks, append to one request does not mutate the other, and refcounts/free behavior remains correct.

### Benchmark Plan

`benchmarks/bench_paged_kv.py` should measure metadata and gather overhead, not full model inference.

Metrics:

- append bandwidth for paged writes.
- lookup overhead per token for `read_token`.
- gather-prefix bandwidth for logical reads across non-contiguous physical blocks.
- block-table bytes read per decode step.
- internal fragmentation tokens and utilization for each workload.

Synthetic workloads:

- fixed length: many requests with identical prompt/output length.
- variable length: seeded distribution of prompt lengths and output lengths.
- churn: random request arrivals and frees to expose block reuse.
- forked beams, optional: shared prefix plus short divergent suffixes.

Compare against contiguous:

```text
contiguous_reserved_tokens = active_requests * max_seq_len
paged_reserved_tokens = allocated_blocks * block_size
saved_slots = contiguous_reserved_tokens - paged_reserved_tokens
```

## PagedAttention Decode Dataflow Spec

This package does not need a full optimized attention kernel. It must explain and optionally prototype the address mapping used by a decode attention step.

Inputs for one decode step:

```text
q: [num_query_heads, head_dim]
k_blocks: [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
v_blocks: [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
block_table: [num_logical_blocks_for_request]
seq_len: current KV length
num_query_heads, num_kv_heads, head_dim, block_size
scale = 1 / sqrt(head_dim)
```

Dataflow:

1. Decode computes K/V for the new token and appends it to the paged cache.
2. For each query head, compute `kv_head = query_head // group_size`, where `group_size = num_query_heads // num_kv_heads`.
3. Iterate logical token positions from 0 to `seq_len - 1`.
4. Convert each logical token to `(logical_block, offset_in_block)`.
5. Read `physical_block = block_table[logical_block]`.
6. Load K/V from the physical block pool.
7. Compute `score = dot(q[query_head], k) * scale`.
8. Use a numerically stable softmax over all historical tokens.
9. Accumulate `output[query_head] += probability * v`.

Reference pseudocode:

```python
def paged_decode_attention_ref(q, k_blocks, v_blocks, block_table, seq_len, layer, num_kv_heads, block_size):
    num_query_heads, head_dim = q.shape
    group_size = num_query_heads // num_kv_heads
    out = torch.empty_like(q)
    for qh in range(num_query_heads):
        kvh = qh // group_size
        keys = []
        values = []
        for pos in range(seq_len):
            logical = pos // block_size
            offset = pos % block_size
            physical = block_table[logical]
            keys.append(k_blocks[layer, physical, offset, kvh])
            values.append(v_blocks[layer, physical, offset, kvh])
        k = torch.stack(keys, dim=0)
        v = torch.stack(values, dim=0)
        scores = (k @ q[qh]) / math.sqrt(head_dim)
        probs = torch.softmax(scores.float(), dim=0).to(v.dtype)
        out[qh] = probs @ v
    return out
```

The implementation can compare this reference with a contiguous attention baseline by first gathering paged K/V into logical order.

## End-to-End Test Matrix

Use small deterministic tensors whose values encode their logical position, head, and dim so mapping mistakes are easy to diagnose.

| Test class | Cases |
| --- | --- |
| logical/physical mapping | sequential allocation, shuffled free-list allocation, reused physical block |
| variable sequence length | requests of length 0, 1, 7, 16, 17, 31, 32 |
| block boundary | append at positions 15, 16, and 17 for `block_size=16` |
| fragmentation | active lengths `[1, 16, 17, 31]`, then free one middle request |
| GQA/MQA | `num_query_heads=8, num_kv_heads=8`, `8/2` GQA, and `8/1` MQA if local materials support the shape |
| decode attention | paged output equals contiguous gather baseline for tiny shapes |
| copy-on-write optional | fork, append divergent suffix, free both requests |

## Benchmark and Workload Simulation Plan

### Append Bandwidth

Measure contiguous and paged append for the same synthetic token count.

Report:

- total appended tokens
- average append ms
- effective write GB/s
- allocation count
- block-boundary count

### Lookup Overhead

Measure logical-to-physical lookup cost separately from tensor math:

- Python metadata lookup for many random token positions.
- Tensor gather of K/V across block tables.
- Optional CUDA/Triton version later, only after the PyTorch reference is correct.

Report overhead as nanoseconds or microseconds per token lookup when CPU-bound, and as effective GB/s when gathering GPU tensors.

### TTFT, TPOT, and KV Usage Estimates

Use a synthetic request trace:

```text
request_id, arrival_step, prompt_len, output_len, num_query_heads, num_kv_heads
```

Estimate:

- TTFT proxy: prompt tokens processed during prefill before first generated token.
- TPOT proxy: decode step time from active sequence count, KV bytes read, and lookup overhead.
- KV usage: `2 * layers * valid_tokens * num_kv_heads * head_dim * dtype_size`.
- Paged reserved usage: `2 * layers * allocated_blocks * block_size * num_kv_heads * head_dim * dtype_size`.
- Contiguous reserved usage: `2 * layers * active_requests * max_seq_len * num_kv_heads * head_dim * dtype_size`.

Decode KV bytes per generated token:

```text
kv_read_bytes = 2 * layers * seq_len * num_kv_heads * head_dim * dtype_size
```

Paged metadata overhead per generated token:

```text
block_table_reads = layers * ceil(seq_len / block_size)
block_table_bytes = block_table_reads * sizeof(int32)
```

The simulator should make clear that TTFT is dominated by prefill work, while TPOT is sensitive to decode KV reads, active batch size, block-table lookup overhead, and scheduling interference.

## Interview Explanation

### What PagedAttention Solves

PagedAttention solves KV cache over-reservation and fragmentation. A traditional contiguous cache reserves `max_seq_len` slots per request even when the request is short. PagedAttention stores KV in fixed-size physical blocks and gives each request a logical block table, so memory is allocated on demand and freed in reusable blocks.

### Tradeoffs

- Benefit: much higher KV memory utilization for variable-length workloads.
- Benefit: larger dynamic batches because less memory is stranded in unused per-request regions.
- Benefit: copy-on-write can share prefix blocks for beam-like or forked sequences.
- Cost: attention kernels perform an extra block table lookup before loading K/V.
- Cost: K/V for one logical sequence is not physically contiguous, which can hurt locality or coalescing if the kernel is poorly designed.
- Cost: metadata correctness becomes part of serving correctness: refcounts, frees, block reuse, and optional swap/preemption must be right.

### Failure Modes

- Wrong logical-to-physical mapping returns another request's KV.
- Off-by-one errors at block boundaries corrupt the previous or next token.
- Refcount bugs free a shared block too early or leak blocks forever.
- Fragmentation metrics can look good while a scheduler still OOMs because too many long sequences are admitted.
- GQA/MQA head mapping can read the wrong KV head when `num_query_heads != num_kv_heads`.
- Long prefill can inflate TTFT and block decode, causing TPOT or ITL spikes even when KV allocation is efficient.
- Overly small block size increases block-table overhead; overly large block size increases internal fragmentation.

## Completion Checklist

- `kv_cache_append` has reference behavior, tests, benchmark, and note.
- Paged block table has append/read/free, logical-to-physical tests, fragmentation stats, and optional copy-on-write plan.
- PagedAttention decode dataflow explains address mapping from logical token to physical block.
- Tests cover variable length, block boundary, fragmentation, and GQA/MQA shape implications.
- Benchmark or simulator reports append bandwidth, lookup overhead, TTFT/TPOT proxies, and KV usage estimates.
- Notes explicitly state that no vLLM runtime, model downloads, or large repository clones are required.
