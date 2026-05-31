# FlashAttention Toy Implementation Spec

Operator: `flash_attention_toy`
Scope: forward-only toy FlashAttention-style scaled dot-product attention for small prefill shapes on a 4060 8GB GPU.

## Scope

- Implement a forward-only toy attention kernel with inputs `q`, `k`, `v` shaped `[batch, heads, seq, head_dim]` and output `[batch, heads, seq, head_dim]`.
- Support causal and non-causal attention in the wrapper and tests.
- Target small educational shapes only. The goal is correctness, address mapping, online softmax state, benchmarking, profiling, and notes.
- No backward pass, dropout, attention bias tensors, ALiBi, RoPE fusion, GQA/MQA, KV cache paging, production optimization, persistent kernels, Tensor Core tuning, or model integration.
- Keep the first implementation readable even if it is slower than PyTorch SDPA.

## PyTorch SDPA Reference

Primary reference:

```python
import torch
import torch.nn.functional as F

expected = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
```

Manual fallback for debugging, dtype control, and environments where SDPA backend selection obscures behavior:

```python
scale = q.shape[-1] ** -0.5
scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
if causal:
    s_q, s_k = q.shape[-2], k.shape[-2]
    row = torch.arange(s_q, device=q.device)[:, None]
    col = torch.arange(s_k, device=q.device)[None, :]
    scores = scores.masked_fill(col > row, float("-inf"))
probs = torch.softmax(scores, dim=-1)
expected = torch.matmul(probs, v.float()).to(q.dtype)
```

The wrapper should validate that `q`, `k`, and `v` are CUDA tensors, contiguous, 4D, same dtype, same device, and have matching `batch`, `heads`, `seq`, and `head_dim` for this first MHA-only toy.

## Shape Set for 4060 8GB

Start with fp16. Add fp32 only for very small debug cases if useful.

| name | batch | heads | seq | head_dim | causal | reason |
|---|---:|---:|---:|---:|---|---|
| tiny_debug | 1 | 1 | 16 | 32 | true/false | easy manual inspection |
| aligned_small | 1 | 4 | 128 | 64 | true/false | standard small attention |
| non_aligned_seq | 1 | 2 | 130 | 64 | true | tests block masks on seq |
| non_aligned_dim | 1 | 2 | 128 | 80 | false | tests head_dim masking if supported |
| medium_prefill | 1 | 8 | 512 | 64 | true | useful 4060 baseline |
| larger_prefill | 1 | 8 | 1024 | 64 | true | still reasonable on 8GB |
| dim128 | 1 | 4 | 512 | 128 | true | common LLM head_dim |
| batch2 | 2 | 4 | 256 | 64 | true | batch/head grid check |

Implementation may initially restrict `head_dim` to `{32, 64, 80, 128}` and `seq <= 1024`. If `head_dim=80` makes the Triton tile awkward, keep it as a shape-error test and use `seq=130, head_dim=64` for the required non-aligned correctness coverage.

## Tests

Correctness tests:

- Small shape correctness against PyTorch SDPA for `tiny_debug`, `aligned_small`, and `batch2`.
- causal mask correctness: compare causal output to SDPA/manual fallback and include a test where changing future `k/v` positions cannot affect earlier output rows.
- Non-causal correctness for at least one aligned shape.
- Non-aligned seq: `seq=130` or `seq=257` with block masks on Q rows and KV columns.
- Non-aligned head_dim where reasonable: either support `head_dim=80` with a dimension mask or explicitly reject it with a clear shape error.
- dtype tolerance: fp32 debug `rtol=1e-4, atol=1e-4`; fp16 `rtol=2e-2, atol=2e-2` versus SDPA/manual fp32 fallback cast to fp16.
- Finite output check for all valid non-empty inputs.

Shape error tests:

- Reject non-4D inputs with a message mentioning `4D`.
- Reject mismatched `batch`, `heads`, `seq`, or `head_dim` for `q/k/v`.
- Reject CPU tensors and non-contiguous tensors.
- Reject unsupported dtypes outside fp16 and optional fp32.
- Reject unsupported `head_dim` values if the kernel only specializes a fixed set.
- Reject zero `seq` or zero `head_dim`.

## Triton Implementation Outline

Use a simple one-program-per-`(batch, head, q_block)` layout:

```python
grid = (batch, heads, triton.cdiv(seq, BLOCK_M))
```

Program IDs and address mapping:

```python
pid_b = tl.program_id(0)
pid_h = tl.program_id(1)
pid_m = tl.program_id(2)
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = start_n + tl.arange(0, BLOCK_N)
offs_d = tl.arange(0, BLOCK_D)
```

For contiguous `[B, H, S, D]` tensors, element addresses are:

```python
q_ptr + ((pid_b * H + pid_h) * S + offs_m[:, None]) * D + offs_d[None, :]
k_ptr + ((pid_b * H + pid_h) * S + offs_n[:, None]) * D + offs_d[None, :]
v_ptr + ((pid_b * H + pid_h) * S + offs_n[:, None]) * D + offs_d[None, :]
o_ptr + ((pid_b * H + pid_h) * S + offs_m[:, None]) * D + offs_d[None, :]
```

Block loop:

1. Load a Q tile `[BLOCK_M, BLOCK_D]` once per program with row and head_dim masks.
2. Initialize online softmax state in fp32:
   - `m_i = -inf` for each Q row.
   - `l_i = 0` for each Q row.
   - `acc = 0` for `[BLOCK_M, BLOCK_D]`.
3. Loop `start_n` over KV blocks from `0` to `seq` in steps of `BLOCK_N`.
4. Load K and V tiles with `offs_n < seq` and `offs_d < head_dim` masks.
5. Compute `scores = tl.dot(q, tl.trans(k)) * scale`.
6. Apply padding mask for invalid KV columns.
7. Apply causal handling:
   - For non-causal, no triangular mask.
   - For causal, skip whole KV blocks where `start_n > max(offs_m)`.
   - For diagonal/overlapping blocks, use `tl.where(offs_n[None, :] <= offs_m[:, None], scores, -inf)`.
8. Update online softmax:
   - `m_block = tl.max(scores, axis=1)`
   - `m_new = tl.maximum(m_i, m_block)`
   - `alpha = tl.exp(m_i - m_new)`
   - `p = tl.exp(scores - m_new[:, None])`
   - `l_new = l_i * alpha + tl.sum(p, axis=1)`
   - `acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)`
   - `m_i = m_new`, `l_i = l_new`
9. Store `acc / l_i[:, None]` to O with Q row and head_dim masks.

Use `BLOCK_M=16 or 32`, `BLOCK_N=32 or 64`, and `BLOCK_D=next_power_of_2(head_dim)` for the toy. Accumulate `scores`, `m_i`, `l_i`, and `acc` in fp32.

## Benchmark Plan

Compare toy Triton against PyTorch SDPA using warmup, repeated timing, and `torch.cuda.synchronize()` around timed regions. Benchmark fp16 first:

- `(B=1, H=4, S=128, D=64)`, causal and non-causal.
- `(B=1, H=8, S=512, D=64)`, causal.
- `(B=1, H=8, S=1024, D=64)`, causal.
- `(B=1, H=4, S=512, D=128)`, causal.
- `(B=2, H=4, S=256, D=64)`, causal.

Approximate TFLOPS:

```python
flops = 4 * batch * heads * seq * seq * head_dim
if causal:
    flops *= 0.5
tflops = flops / (ms / 1e3) / 1e12
```

JSON schema:

```json
{
  "operator": "flash_attention_toy",
  "device": "NVIDIA GeForce RTX 4060 Laptop GPU",
  "dtype": "float16",
  "results": [
    {
      "impl": "triton_toy",
      "reference": "torch_sdpa",
      "batch": 1,
      "heads": 8,
      "seq": 512,
      "head_dim": 64,
      "causal": true,
      "ms": 0.123,
      "tflops": 1.09,
      "max_abs_err": 0.015625,
      "rtol": 0.02,
      "atol": 0.02
    }
  ]
}
```

## Profile Plan and Expected Bottlenecks

Profile one representative causal shape, then one small shape:

- Nsight Compute: `(B=1, H=8, S=512, D=64)`, fp16, causal.
- Optional Nsight Systems: benchmark script with `S=128` and `S=512` to separate launch overhead from kernel time.

Expected bottlenecks:

- The toy kernel is likely slower than PyTorch SDPA because it will not use production-grade tiling, warp specialization, or tuned Tensor Core pipelines.
- Small `seq` can be launch-bound and occupancy-limited.
- Larger prefill shapes should spend most time in `tl.dot`; if not, look for register pressure, inefficient masks, excess `tl.exp`, or poor block sizes.
- Non-aligned `seq` and `head_dim` masks add overhead and may reduce vectorization.
- Causal attention should skip or mask upper-triangular work; if causal runtime matches non-causal runtime exactly, block skipping is probably missing.

## Note Requirements

The final implementation note should include:

- Correctness: SDPA reference, manual fallback, shape coverage, causal behavior, dtype tolerance, and shape errors.
- IO complexity: standard attention writes/reads the `S x S` score/probability matrices in HBM, while FlashAttention keeps block scores on chip and stores only the final output plus online softmax state.
- Online softmax: explain why tracking row max `m` and denominator `l` lets the kernel stream KV blocks without materializing the full attention matrix.
- Prefill vs decode: prefill has many Q rows and benefits from tiled FlashAttention; decode has `S_q=1`, repeatedly reads the KV cache, and is usually memory-bound.
- Benchmark: JSON table with SDPA comparison, `ms`, approximate TFLOPS, and error metrics.
- Profile: Nsight finding or explicit reason profiling was skipped.
- Bottleneck: one concrete conclusion about launch, memory, math, masking, occupancy, or register pressure.
- `next_experiment`: one specific follow-up, such as adding block-level causal skip, trying `BLOCK_M/BLOCK_N` variants, or building a separate decode attention toy.

## Explicit Out of Scope

- Backward pass and recomputation logic.
- Dropout, arbitrary masks, attention bias, ALiBi, RoPE fusion, sliding-window attention, or variable-length packed sequences.
- GQA/MQA, KV cache layout, PagedAttention, Flash-Decoding, and decode-optimized kernels.
- Production optimization, persistent kernels, warp-specialized pipelines, autotuning, and parity with FlashAttention-2/3.
- Downloading model weights or integrating with an LLM runtime.
