# INT4 Dequant Task Specification

Operator: `int4_dequant`
Scope: Week 7 executable experiment for INT4 pack/unpack/dequant, with GPTQ, AWQ, FP8, and DeepSeek references used only to motivate design choices.

## Scope

- Implement a small, readable INT4 weight-only dequant experiment.
- Cover bit packing, unpacking, scale/zero-point math, PyTorch reference checks, Triton kernel correctness, benchmark, profile, and learning note.
- Target local synthetic tensors only. Do not download model weights.
- Do not implement full GPTQ, full AWQ, activation quantization, FP8 training, MoE routing, bitsandbytes parity, weight-only GEMV fusion, or model-runtime integration in this task.

## Relationship to Week 7 References

- GPTQ: motivates why transformer weights can be stored in low precision after post-training quantization, but this task only consumes already-quantized INT4 values and dequantizes them. Paper-derived details such as Hessian-aware row/block reconstruction and exact grouping conventions need later verification against `14_gptq_frantar_2023.pdf`.
- AWQ: motivates activation-aware scaling and preserving important channels, but this task only supports generic per-channel or per-group scales and optional zero points. Paper-derived details about salient weight/channel selection and calibration flow need later verification against `15_awq_lin_2023.pdf`.
- FP8 formats: provide contrast with INT4. FP8 is a numeric floating format family with exponent/mantissa tradeoffs; INT4 here is an integer code plus external scale/zero-point metadata. Exact E4M3/E5M2 behavior and special value handling need later verification against `16_fp8_formats_micikevicius_2022.pdf`.
- DeepSeek FP8/MoE: connects quantization to large-scale MoE training/inference pressure. The local task does not implement MoE, expert parallelism, or FP8. Paper-derived claims about DeepSeek-V3 FP8 recipes and MoE placement need later verification against `17_deepseek_v3_2024.pdf`.
- QLoRA/NF4 context: NF4 is a 4-bit codebook-oriented format for normally distributed weights. The local INT4 task uses linear affine dequantization, not NF4 codebook lookup or double quantization.

## Owned File Paths for Implementation Slice

The later implementation task should create or modify only INT4 dequant artifacts:

- `kernels/triton/int4_dequant/__init__.py`
- `kernels/triton/int4_dequant/int4_dequant.py`
- `tests/test_int4_dequant.py`
- `benchmarks/bench_int4_dequant.py`
- `reports/json/int4_dequant_bench.json`
- `notes/int4_dequant.md`
- `reports/ncu/int4_dequant_*` only when saving profiler output

Do not edit `study-plan/progress.yaml` in the task spec stage. If a later operator verification command updates progress, commit that diff only when the operator artifact task explicitly allows it.

## INT4 Data Layout Options

Start with one canonical layout, then test options through the PyTorch reference:

- Packed storage: two INT4 values per byte in a `torch.uint8` tensor.
- Nibble order: low nibble stores even logical element `2*i`; high nibble stores odd logical element `2*i + 1`. Document this explicitly in tests.
- Unsigned affine INT4: codes are `0..15`; dequant formula is `(code - zero_point) * scale`.
- Signed INT4 option: decode nibble as two's-complement `[-8, 7]`; dequant formula is `signed_code * scale`. This can be a reference/test option before the Triton kernel supports it.
- Scale layout:
  - Per-channel: `scale` shaped `[out_features]` or `[rows]`, one scale per row/channel.
  - Per-group: `scale` shaped `[rows, ceil(cols / group_size)]`, one scale per row and contiguous group of columns.
- Zero point layout mirrors scale:
  - `zero_point` can be scalar, per-channel, or per-group.
  - For symmetric signed INT4, `zero_point` should be omitted or fixed at zero.
- Group size: start with `group_size in {32, 64, 128}`. Include a non-divisible `cols` test so the last group uses masks.
- Shape convention: treat weights as `[rows, cols]`; packed tensor has shape `[rows, ceil(cols / 2)]`.

## PyTorch Reference

Use the PyTorch reference as the source of truth for tests and debugging.

```python
import torch

def pack_uint4(q: torch.Tensor) -> torch.Tensor:
    if q.dtype != torch.uint8:
        raise ValueError("q must be uint8 codes")
    if torch.any(q > 15):
        raise ValueError("INT4 codes must be in [0, 15]")
    flat = q.reshape(-1)
    if flat.numel() % 2:
        flat = torch.cat([flat, torch.zeros(1, dtype=torch.uint8, device=q.device)])
    lo = flat[0::2] & 0x0F
    hi = (flat[1::2] & 0x0F) << 4
    return (lo | hi).reshape(*q.shape[:-1], (q.shape[-1] + 1) // 2)

def unpack_uint4(packed: torch.Tensor, logical_cols: int) -> torch.Tensor:
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    out = torch.empty((*packed.shape[:-1], packed.shape[-1] * 2), dtype=torch.uint8, device=packed.device)
    out[..., 0::2] = lo
    out[..., 1::2] = hi
    return out[..., :logical_cols]

def dequant_uint4_affine(
    packed: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | int | float,
    rows: int,
    cols: int,
    group_size: int,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    codes = unpack_uint4(packed, cols).reshape(rows, cols).to(torch.float32)
    col_groups = torch.arange(cols, device=packed.device) // group_size
    scale_g = scale[:, col_groups].to(torch.float32)
    if isinstance(zero_point, torch.Tensor):
        zp_g = zero_point[:, col_groups].to(torch.float32)
    else:
        zp_g = float(zero_point)
    return ((codes - zp_g) * scale_g).to(out_dtype)
```

For signed INT4 reference coverage:

```python
def uint4_to_int4(codes: torch.Tensor) -> torch.Tensor:
    return torch.where(codes >= 8, codes.to(torch.int16) - 16, codes.to(torch.int16))
```

## Triton Dequant Kernel Outline

First Triton version: one program per row and column block, unpack two nibbles per byte, apply group scale and zero point, store fp16/bf16/fp32 output.

```python
@triton.jit
def int4_dequant_kernel(
    packed_ptr, scale_ptr, zp_ptr, out_ptr,
    rows: tl.constexpr, cols: tl.constexpr, packed_stride: tl.constexpr,
    scale_stride: tl.constexpr, out_stride: tl.constexpr,
    group_size: tl.constexpr, has_zp: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    cols_off = block * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    mask = cols_off < cols

    byte_off = cols_off // 2
    nibble_is_high = (cols_off & 1) == 1
    byte = tl.load(packed_ptr + row * packed_stride + byte_off, mask=mask, other=0)
    code = tl.where(nibble_is_high, (byte >> 4) & 0x0F, byte & 0x0F).to(tl.float32)

    group = cols_off // group_size
    scale = tl.load(scale_ptr + row * scale_stride + group, mask=mask, other=0.0).to(tl.float32)
    zp = tl.load(zp_ptr + row * scale_stride + group, mask=mask, other=0.0).to(tl.float32) if has_zp else 0.0
    out = (code - zp) * scale
    tl.store(out_ptr + row * out_stride + cols_off, out, mask=mask)
```

Implementation constraints:

- Require CUDA, contiguous tensors, 2D packed input, and valid logical `rows, cols`.
- Require `packed.shape == (rows, ceil(cols / 2))`.
- Require `scale.shape == (rows, ceil(cols / group_size))` for the baseline.
- Require `zero_point` to be absent or shaped like `scale` for the baseline.
- Use masks for odd `cols`, non-divisible group sizes, and non-aligned `BLOCK_COLS`.
- Keep dequant separate from GEMV/GEMM so the benchmark measures unpack/dequant overhead directly.

## Tests

Correctness tests:

- Bit packing round trip for random codes in `[0, 15]`.
- Explicit nibble order test: codes `[1, 2, 15, 0]` packs to bytes `[0x21, 0x0F]`.
- Odd logical sizes: `cols = 1, 3, 65, 129`; ensure padded high nibble is ignored.
- Even logical sizes: `cols = 2, 64, 128, 256`.
- Group scaling: `group_size = 32, 64, 128`, including `cols` not divisible by group size.
- Channel scaling: compare per-row scale behavior by using distinct scales for each row.
- Zero point: test scalar-like zero, per-group zero points, and edge codes `0` and `15`.
- dtype tolerance:
  - fp32 output: `rtol=1e-5, atol=1e-6`.
  - fp16 output: `rtol=1e-3, atol=1e-3` for simple scales; relax to `rtol=1e-2, atol=1e-2` for random scales.
  - bf16 output: include when CUDA/PyTorch supports it; use `rtol=2e-2, atol=2e-2`.
- Shape errors:
  - Reject non-2D packed tensors.
  - Reject `cols <= 0`, `rows <= 0`, or mismatched `rows`.
  - Reject `packed.shape[-1] != ceil(cols / 2)`.
  - Reject invalid `scale` or `zero_point` shapes.
  - Reject unsupported dtypes for output.
  - Reject CPU tensors and non-contiguous tensors in the Triton wrapper.

Suggested shape set:

| rows | cols | group_size | reason |
| ---: | ---: | ---: | --- |
| 1 | 1 | 32 | smallest odd shape |
| 1 | 64 | 32 | single aligned row |
| 4 | 65 | 32 | odd cols and partial group |
| 16 | 128 | 64 | common aligned group test |
| 128 | 1024 | 128 | medium benchmark/correctness |
| 512 | 4097 | 128 | non-aligned large mask test |

## Benchmark Plan

Benchmark the Triton dequant kernel against the PyTorch reference using warmup, repeated timing, CUDA events, and `torch.cuda.synchronize()`. Write results to `reports/json/int4_dequant_bench.json`.

Benchmark shapes should include:

- Small launch-bound: `[rows=1, cols=1024]`.
- Medium: `[rows=128, cols=1024]`, `[rows=128, cols=4096]`.
- Large memory-bound proxy: `[rows=1024, cols=4096]`.
- Non-aligned: `[rows=513, cols=4097]`.

Run output dtypes `float16`, `bfloat16` when supported, and `float32` for debug comparison.

Effective bytes moved:

```python
packed_bytes = rows * ((cols + 1) // 2)
groups = (cols + group_size - 1) // group_size
scale_bytes = rows * groups * scale.element_size()
zp_bytes = rows * groups * zero_point.element_size() if has_zero_point else 0
out_bytes = rows * cols * out_dtype_size
bytes_moved = packed_bytes + scale_bytes + zp_bytes + out_bytes
gbps = bytes_moved / (ms / 1e3) / 1e9
```

This `GB/s` or `gbps` value is an effective bandwidth metric. Scale and zero-point metadata may cache better than this denominator assumes.

`reports/json/int4_dequant_bench.json` schema:

```json
{
  "metadata": {
    "operator": "int4_dequant",
    "device": "NVIDIA ...",
    "torch_version": "...",
    "triton_version": "...",
    "warmup": 20,
    "iters": 100
  },
  "results": [
    {
      "impl": "triton",
      "reference": "torch_unpack_dequant",
      "shape": {"rows": 128, "cols": 4096},
      "packed_shape": [128, 2048],
      "dtype": "float16",
      "scale_dtype": "float16",
      "zero_point_dtype": "uint8",
      "group_size": 128,
      "signed": false,
      "has_zero_point": true,
      "ms": 0.045,
      "gbps": 92.3,
      "bytes": 4153344,
      "max_abs_err": 0.0,
      "rtol": 0.001,
      "atol": 0.001
    }
  ]
}
```

## Profile Plan and Expected Bottleneck

Profile one large aligned shape and one non-aligned shape:

- Nsight Compute: `[rows=1024, cols=4096, group_size=128, dtype=float16]`.
- Optional Nsight Systems: benchmark script with small and large shapes to separate launch overhead from kernel time.

Expected bottleneck:

- Large shapes should be memory-bandwidth-bound because each output element performs a few bit operations, one scale load, optional zero-point load, and one output store.
- Small shapes should be launch-bound.
- Non-aligned `cols` and partial groups add mask overhead but should not change correctness.
- Repeated scale loads can become visible for small `group_size`; a later optimization may reuse one scale per group instead of loading per element.

Record:

- Kernel time, achieved memory throughput, and whether the kernel is memory-bound in Speed of Light / Memory Workload Analysis.
- Occupancy and register pressure.
- Any difference between aligned and non-aligned shapes.
- Whether zero-point loads materially change runtime.

## Note Requirements

The final `notes/int4_dequant.md` learning note must include:

- `Operator`: INT4 pack/unpack/dequant scope and where weight-only dequant appears before GEMV/GEMM.
- `Reference`: PyTorch pack, unpack, affine dequant, signed/unsigned choices, scale, zero point, and group size.
- `Implementation`: Triton program IDs, address mapping, nibble order, mask behavior, dtype handling.
- `Tests`: bit packing round trip, odd/even sizes, group/channel scaling, zero point, dtype tolerance, and shape errors.
- `Benchmark`: table or JSON summary from `reports/json/int4_dequant_bench.json` with `ms`, `GB/s` or `gbps`, shape, dtype, and PyTorch comparison.
- `Profile`: command, shape, profiler report path, key metrics, and whether profiling was skipped.
- `Bottleneck`: one concrete conclusion about launch overhead, memory bandwidth, scale/zero-point loads, masking, or register pressure.
- `Relationship to GPTQ/AWQ/FP8/DeepSeek`: explain what those references motivate and what this task intentionally omits.
- `Next experiment`: one specific follow-up, such as fusing dequant with GEMV for decode or adding signed INT4 kernel support.

## Interview Questions

- Why does INT4 dequant need scale and usually a zero point, while FP8 stores exponent/mantissa information in each value?
- What is the difference between symmetric signed INT4 and unsigned affine INT4?
- How do per-channel and per-group scales trade metadata overhead against quantization error?
- Why does an INT4 dequant-only kernel often look memory-bound?
- What does GPTQ optimize that this local experiment does not implement?
- What does AWQ use activation information for, and why is that out of scope here?
- How is NF4 different from a linear INT4 code with scale and zero point?
- In a MoE model, why can quantizing expert weights help memory pressure even if only top-k experts are active per token?
- What bugs do odd `cols` and non-divisible `group_size` tests catch?
- How would fusing dequant with GEMV change the bytes moved and expected bottleneck?

## Explicit Constraints

- No model downloads.
- No bitsandbytes clone.
- No vLLM, llama.cpp, or other large repository clone.
- No package installation.
- No full GPTQ implementation.
- No full AWQ implementation.
- No FP8 training or DeepSeek reproduction.
- No full quantization algorithm or calibration pipeline.
- No model-specific runtime bundle.
