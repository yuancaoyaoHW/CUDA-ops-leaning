Operator: row_softmax, y = softmax(x, dim=1)
Scope: implement a Triton row_softmax baseline first, then use it as the reference point for an online_softmax follow-up.

## Scope

- Baseline `row_softmax`: one Triton program handles one contiguous input row. It supports 2D CUDA tensors shaped `(rows, cols)` where `cols > 0` and `cols <= BLOCK_SIZE`.
- Baseline algorithm: safe softmax with a row max reduction, exponentiation, row sum reduction, and normalized store in one kernel launch for rows that fit in one block.
- Follow-up `online_softmax`: compute the row denominator state with online softmax update rules so larger rows can be processed in block chunks without separately materializing row max and row sum.
- Out of scope for the first implementation: masked attention softmax, causal masks, backward, non-contiguous strides, multi-block rows, and fused matmul or attention.

## PyTorch Reference

Use PyTorch as the correctness reference:

```python
expected = torch.softmax(x, dim=1)
```

Manual stable formula for tests and learning notes:

```python
m = x.max(dim=1, keepdim=True).values
numerator = torch.exp(x.float() - m.float())
expected = numerator / numerator.sum(dim=1, keepdim=True)
expected = expected.to(x.dtype)
```

Online softmax state for the follow-up:

```python
m = -torch.inf
d = 0.0
for x_i in row:
    m_new = max(m, x_i)
    d = d * exp(m - m_new) + exp(x_i - m_new)
    m = m_new
y_i = exp(x_i - m) / d
```

Two partial online states must combine as:

```python
m = max(m1, m2)
d = d1 * exp(m1 - m) + d2 * exp(m2 - m)
```

## Triton Address Mapping

Baseline mapping:

```python
row = tl.program_id(0)
offs = tl.arange(0, BLOCK_SIZE)
mask = offs < n_cols
in_ptrs = x_ptr + row * n_cols + offs
out_ptrs = y_ptr + row * n_cols + offs
vals = tl.load(in_ptrs, mask=mask, other=-float("inf"))
```

- Grid: `(rows,)`.
- `tl.program_id(0)` selects exactly one logical row.
- `row * n_cols` is the row offset for a contiguous `(rows, cols)` tensor.
- `offs` maps each lane in the Triton block to a column index in that row.
- `mask = offs < n_cols` protects non-aligned column counts when `BLOCK_SIZE` is the next power of two.
- Load masked elements as `-inf` so `tl.max(vals, axis=0)` ignores them.
- Store with the same mask so no out-of-row or out-of-allocation address is written.

Baseline computation:

```python
row_max = tl.max(vals, axis=0)
shifted = vals - row_max
num = tl.exp(shifted)
den = tl.sum(num, axis=0)
out = num / den
tl.store(out_ptrs, out, mask=mask)
```

For fp16 input, reductions and exponent math should be promoted to fp32 before the final cast to the output dtype.

Online follow-up mapping:

```python
row = tl.program_id(0)
for start in range(0, n_cols, BLOCK_SIZE):
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    vals = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=-float("inf"))
```

The follow-up uses the same row offset, block offsets, and mask pattern, but updates `(m, d)` per block with the online combine rule. A second pass over the row still writes `exp(x_i - m) / d` unless the next experiment fuses softmax into a consumer.

## Numerical Stability

- Always subtract the row max before exponentiation.
- Never compute `torch.exp(x)` or `tl.exp(vals)` directly on unshifted logits.
- For masked lanes, use `-inf`, not `0.0`; otherwise negative-only rows can get an incorrect max or denominator.
- Accumulate denominator in fp32 for fp32 and fp16 inputs.
- Handle large positive and negative logits, including values around `[-1000, 1000]`, without NaN or inf in valid outputs.
- Each valid output row should sum to approximately 1.0.

## Tests

Aligned shapes:

- `(128, 256)`
- `(512, 512)`
- `(1024, 1024)`
- `(4096, 2048)`

Non-aligned shapes that require the Triton mask:

- `(128, 257)`
- `(1, 257)`
- `(513, 64)`
- `(33, 1000)`

Correctness cases:

- fp32 random logits vs `torch.softmax(x, dim=1)`.
- fp16 random logits vs `torch.softmax(x, dim=1)`.
- negative-only logits to prove masked lanes use `-inf`.
- high-magnitude logits such as `x * 100` or explicit rows containing `1000`, `1001`, and `1002`.
- zero rows shaped `(0, 256)` should return shape `(0, 256)` if the wrapper can avoid launching an empty grid.

Shape error cases:

- Reject non-2D input with a `ValueError` mentioning `2D`.
- Reject non-contiguous input with a `ValueError` mentioning `contiguous`.
- Reject zero columns `(rows, 0)` with a `ValueError` mentioning at least one column.
- Reject unsupported dtypes outside fp32/fp16 for the first implementation.

## dtype and Tolerance Plan

- fp32: compare to `torch.softmax(x, dim=1)` with default `torch.testing.assert_close` tolerances, or explicitly `rtol=1e-5, atol=1e-6`.
- fp16: output dtype should match input dtype. Compare with `rtol=1e-2, atol=2e-2`, matching the existing row reduction tolerance style.
- For stability stress tests, compute a manual fp32 reference and cast only the final expected output to the input dtype.
- Also assert `torch.isfinite(actual).all()` for valid non-empty inputs.

## Benchmark Plan

Use warmup, repeated timing, and `torch.cuda.synchronize()` as in `bench_row_sum.py` and `bench_row_max.py`.

Initial shapes:

- `(128, 256)`
- `(128, 257)`
- `(512, 512)`
- `(1024, 1024)`
- `(2048, 1024)`
- `(4096, 2048)`

Benchmark both Triton and PyTorch for fp32 and fp16.

Effective bandwidth estimate:

```python
element_size = torch.empty((), dtype=dtype).element_size()
bytes_moved = rows * cols * element_size * 2
gbps = bytes_moved / (ms / 1e3) / 1e9
```

For the baseline one-block fused row_softmax, use read once plus write once as the idealized `GB/s` denominator. Mention that exp/reduction work means `gbps` is only an effective memory-traffic metric, not a complete roofline model.

JSON schema:

```json
{
  "operator": "row_softmax",
  "device": "NVIDIA GeForce ...",
  "results": [
    {
      "impl": "triton",
      "rows": 128,
      "cols": 257,
      "shape": [128, 257],
      "dtype": "float32",
      "ms": 0.0123,
      "gbps": 21.34
    },
    {
      "impl": "torch",
      "rows": 128,
      "cols": 257,
      "shape": [128, 257],
      "dtype": "float32",
      "ms": 0.0188,
      "gbps": 13.97
    }
  ]
}
```

## Nsight Profile Plan

- Profile one isolated Triton `row_softmax` launch with Nsight Compute on shape `(4096, 2048)`, fp32.
- Save reports under `reports/ncu/` using the same convention as prior row reduction notes.
- Also run one fp16 profile if fp32 and fp16 benchmark behavior diverges.

Expected metrics:

- Memory throughput should be high, but probably lower than `row_sum` and `row_max` because softmax adds `exp`, division, and two reductions.
- Compute throughput should be higher than row reductions but still not look like a dense GEMM workload.
- Warp stalls may include memory dependency and math pipeline pressure from `tl.exp`.
- Occupancy can be limited by registers for the row vector, especially with large `BLOCK_SIZE`.
- L2 hit rate is not expected to be the main explanation for one-pass rows because there is little row reuse.

## Learning Note Requirements

The final `notes/row_softmax.md` should include these sections:

- correctness: PyTorch reference, stable formula, shape coverage, dtype tolerances, and mask behavior.
- performance: benchmark table with `ms`, `GB/s` or `gbps`, shape, dtype, and PyTorch comparison.
- bottleneck: Nsight-backed conclusion, separating memory traffic, `exp` cost, reductions, launch overhead, and register pressure where possible.
- next_experiment: one specific experiment to run after the baseline.

## Recommended Next Experiment

After the baseline passes tests and has benchmark/profile data, implement `online_softmax` for rows larger than one Triton block. Compare it against the baseline on `cols=2048`, `4096`, and `8192`, and record whether reducing the max-plus-sum passes improves `ms` or whether extra `exp` work and register pressure dominate.
