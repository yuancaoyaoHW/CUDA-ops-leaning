# GEMM / CUTLASS Week 3 Task Specification

## Scope

Turn the Week 3 GEMM/CUTLASS reading into executable local artifacts:

- `benchmarks/bench_gemm.py`: PyTorch/cuBLAS baseline using `torch.mm` or `torch.matmul`.
- `reports/json/gemm_bench.json`: machine-readable benchmark output.
- `notes/gemm_roofline.md`: roofline analysis note generated from benchmark results.
- Optional CUTLASS profiler comparison when a usable local CUTLASS profiler already exists.

Constraints:

- Do not clone CUTLASS or any other large external repository.
- Do not install dependencies.
- Do not download model weights or model-specific runtime bundles.
- Use the existing `llm-kernel-lab` conda environment.
- Keep all shapes 4060-safe by default; larger shapes must be opt-in.

## Day Mapping

| Day | Executable task | Output |
| --- | --- | --- |
| 15 | Build PyTorch/cuBLAS GEMM harness | `benchmarks/bench_gemm.py` |
| 16 | Run shape and dtype sweep | `reports/json/gemm_bench.json` |
| 17 | Try local CUTLASS profiler, or document limited environment | CUTLASS section in `notes/gemm_roofline.md` |
| 18 | Compute arithmetic intensity and roofline classification | roofline table |
| 19 | Note epilogue/fusion implications | short note section |
| 20 | Answer cuBLAS vs CUTLASS vs Triton questions | interview section |
| 21 | Review gaps and next experiment | Week 3 summary |

## `benchmarks/bench_gemm.py` Design

Use PyTorch matmul as the cuBLAS baseline. No external repository is needed because CUDA PyTorch dispatches GEMM to cuBLAS/cuBLASLt for normal CUDA tensors.

Required behavior:

- Accept shape presets: `--preset quick`, `--preset full`, and optional `--shape M,N,K`.
- Accept dtype list: `--dtypes fp32,tf32,fp16,bf16`.
- Use CUDA events for timing, with warmup and measured iterations.
- Synchronize before and after timing.
- Emit JSON to `reports/json/gemm_bench.json`.
- Print a compact table with shape, dtype, ms, TFLOPS, arithmetic intensity, and bound guess.
- Use deterministic random input allocation, but benchmark only `torch.mm(A, B)` or `torch.matmul(A, B)`.
- Avoid correctness comparisons for every timed iteration; one optional preflight check against a smaller fp32 result is enough if added.

Suggested CLI:

```bash
python benchmarks/bench_gemm.py --preset quick --dtypes fp32,tf32,fp16,bf16
python benchmarks/bench_gemm.py --preset full --output reports/json/gemm_bench.json
python benchmarks/bench_gemm.py --shape 128,2048,2048 --dtypes fp16,bf16
```

Suggested measurement logic:

```python
def gemm_flops(m, n, k):
    return 2 * m * n * k

def gemm_bytes(m, n, k, dtype_size):
    return (m * k + k * n + m * n) * dtype_size

def arithmetic_intensity(m, n, k, dtype_size):
    return gemm_flops(m, n, k) / gemm_bytes(m, n, k, dtype_size)

def tflops(m, n, k, ms):
    return gemm_flops(m, n, k) / (ms / 1000.0) / 1e12
```

For TF32:

- Use `torch.float32` tensors.
- Measure once with `torch.backends.cuda.matmul.allow_tf32 = False` as fp32.
- Measure once with `torch.backends.cuda.matmul.allow_tf32 = True` as tf32.
- Record dtype as `tf32` for the second case because storage is fp32 but Tensor Core math may be TF32.

## Shape Sweep

Default shapes should fit comfortably on an RTX 4060-class 8GB environment. The full preset can include larger LLM-like shapes but should still avoid model assets.

| Category | M | N | K | Include in `quick` | Include in `full` | Why it matters |
| --- | ---: | ---: | ---: | --- | --- | --- |
| decode GEMV-like | 1 | 1024 | 1024 | yes | yes | Low arithmetic intensity; memory/launch overhead dominates. |
| decode GEMV-like | 1 | 4096 | 4096 | no | yes | LLM decode projection shape; expected memory-bound. |
| small-batch decode | 8 | 2048 | 2048 | yes | yes | Tests transition from GEMV-like to small GEMM. |
| small-batch decode | 32 | 4096 | 4096 | no | yes | Higher reuse but still far below large prefill AI. |
| prefill | 128 | 2048 | 2048 | yes | yes | 4060-safe prefill proxy. |
| prefill | 512 | 4096 | 4096 | no | yes | Larger prefill proxy, opt-in via full preset. |
| square GEMM | 1024 | 1024 | 1024 | yes | yes | Standard compute benchmark. |
| square GEMM | 2048 | 2048 | 2048 | yes | yes | Tensor Core utilization should improve. |
| square GEMM | 4096 | 4096 | 4096 | no | yes | Large compute-bound reference if memory allows. |
| 4060-safe reduced | 16 | 1024 | 4096 | yes | yes | Small decode-like M with large K. |
| 4060-safe reduced | 128 | 1024 | 4096 | yes | yes | Moderate activation batch against projection weights. |
| 4060-safe reduced | 256 | 2048 | 1024 | yes | yes | Non-square GEMM with moderate footprint. |

## Dtype Plan

| Requested dtype | Tensor dtype | Math mode | Run condition | Notes |
| --- | --- | --- | --- | --- |
| `fp32` | `torch.float32` | TF32 disabled | always if CUDA is available | Baseline CUDA core/cuBLAS path. |
| `tf32` | `torch.float32` | `allow_tf32=True` | CUDA device with TF32 support | Record separately because storage bytes are fp32 but math differs. |
| `fp16` | `torch.float16` | Tensor Core eligible | always if CUDA is available | Primary GEMM throughput path. |
| `bf16` | `torch.bfloat16` | Tensor Core eligible | only if supported | Skip with `"skipped": true` and a reason if unsupported. |

Implementation notes:

- Detect bf16 with `torch.cuda.is_bf16_supported()` when available.
- Do not fail the whole benchmark if one dtype is unsupported.
- For arithmetic intensity, use storage bytes: fp32/tf32 = 4, fp16/bf16 = 2.

## JSON Schema

Write `reports/json/gemm_bench.json` as a stable list of result objects plus metadata.

```json
{
  "metadata": {
    "benchmark": "gemm",
    "backend": "torch.mm/cublas",
    "device": "NVIDIA ...",
    "torch_version": "...",
    "cuda_version": "...",
    "preset": "quick",
    "warmup": 10,
    "iters": 50
  },
  "results": [
    {
      "name": "square_1024",
      "shape": {"m": 1024, "n": 1024, "k": 1024},
      "dtype": "fp16",
      "dtype_size_bytes": 2,
      "ms": 0.123,
      "tflops": 17.45,
      "flops": 2147483648,
      "bytes": 6291456,
      "arithmetic_intensity": 341.3333333333,
      "bound_guess": "compute",
      "skipped": false,
      "skip_reason": null
    }
  ]
}
```

Required result fields:

- `shape`: object with `m`, `n`, `k`.
- `dtype`: one of `fp32`, `tf32`, `fp16`, `bf16`.
- `ms`: median or average measured milliseconds per GEMM.
- `tflops`: `2*M*N*K / seconds / 1e12`.
- `arithmetic_intensity`: `2*M*N*K / ((M*K + K*N + M*N) * dtype_size_bytes)`.
- `bound_guess`: compare arithmetic intensity to a configurable ridge point.

## Roofline Calculation Template

Use the basic roofline model:

```text
FLOPs = 2 * M * N * K
Bytes = (M*K + K*N + M*N) * dtype_size_bytes
Arithmetic intensity = FLOPs / Bytes
Achieved TFLOPS = FLOPs / seconds / 1e12
Ridge point = peak_tflops * 1e12 / peak_bandwidth_bytes_per_second
Attainable TFLOPS = min(peak_tflops, peak_bandwidth_TBps * arithmetic_intensity)
Efficiency = achieved_tflops / attainable_tflops
```

Default local parameters should be configurable instead of hard-coded:

```text
peak_fp32_tflops = user_supplied_or_note_unknown
peak_tf32_tflops = user_supplied_or_note_unknown
peak_fp16_tflops = user_supplied_or_note_unknown
peak_bf16_tflops = user_supplied_or_note_unknown
peak_mem_bandwidth_gbs = user_supplied_or_note_unknown
```

If exact RTX 4060 laptop/desktop peak numbers are unknown, still compute arithmetic intensity and achieved TFLOPS, then mark theoretical roofline efficiency as `unknown`.

## `notes/gemm_roofline.md` Template

```markdown
# GEMM Roofline Notes

## Environment

- GPU:
- PyTorch:
- CUDA:
- Benchmark command:
- JSON source: `reports/json/gemm_bench.json`

## Roofline Parameters

| dtype | peak TFLOPS | bandwidth GB/s | ridge point FLOPs/Byte | source |
| --- | ---: | ---: | ---: | --- |
| fp32 | TBD | TBD | TBD | local spec / unknown |
| tf32 | TBD | TBD | TBD | local spec / unknown |
| fp16 | TBD | TBD | TBD | local spec / unknown |
| bf16 | TBD | TBD | TBD | local spec / unknown |

## GEMM Results

| shape | dtype | ms | TFLOPS | arithmetic intensity | bound guess | note |
| --- | --- | ---: | ---: | ---: | --- | --- |
| M=1,N=1024,K=1024 | fp16 | TBD | TBD | TBD | memory | decode GEMV-like |

## Interpretation

- Decode GEMV-like shapes:
- Small-batch decode shapes:
- Prefill shapes:
- Square GEMM shapes:
- 4060-safe reduced shapes:

## CUTLASS Profiler

- Availability:
- Command:
- Result:
- If unavailable: record limited environment note instead of installing or cloning.

## Epilogue / Fusion Note

- Bias, activation, and residual epilogues can reduce extra global memory traffic.
- For this week, keep the benchmark unfused so cuBLAS baseline remains simple.

## Next Experiment

- One shape:
- One dtype:
- One hypothesis:
```

## CUTLASS Profiler Plan

Only use CUTLASS if a profiler binary already exists locally. Do not clone CUTLASS and do not install dependencies.

Discovery commands:

```bash
find . /home/ycy -path '*cutlass*profiler*' -type f -executable 2>/dev/null | head
which cutlass_profiler || true
```

If a profiler is found, run one small 4060-safe case first:

```bash
cutlass_profiler --operation=Gemm --m=1024 --n=1024 --k=1024
cutlass_profiler --operation=Gemm --m=128 --n=2048 --k=2048
cutlass_profiler --operation=Gemm --m=1 --n=4096 --k=4096
```

Record:

- CUTLASS profiler path and version if available.
- Shape, dtype/layout, kernel name, runtime ms, and reported GFLOPS/TFLOPS.
- Whether the selected kernel uses Tensor Core.
- Comparison against `torch.mm` for the same shape and dtype.

Limited environment fallback:

```markdown
CUTLASS profiler was not run because no local profiler binary was available.
Per repo constraints, this Week 3 task does not clone CUTLASS, install dependencies,
or build large external profiler artifacts. The learning output is limited to
PyTorch/cuBLAS GEMM measurements, arithmetic intensity, roofline classification,
and conceptual CUTLASS comparison.
```

## Interview Questions

1. Why is `torch.mm` a valid cuBLAS/cuBLASLt baseline for GEMM?
2. When would you choose cuBLAS over CUTLASS?
3. When would you choose CUTLASS over cuBLAS?
4. When would you choose Triton over CUTLASS for an LLM kernel?
5. Why is `M=1,N=4096,K=4096` GEMV-like and usually memory-bound?
6. Why does increasing `M` move GEMM toward compute-bound behavior?
7. How do Tensor Cores change the roofline ridge point for fp16/bf16/tf32?
8. Why can a shape with high arithmetic intensity still report poor TFLOPS?
9. What does an epilogue fuse, and why can it improve effective bandwidth?
10. What evidence would convince you that a GEMM is compute-bound vs memory-bound?

Expected answer themes:

- cuBLAS: best default for standard GEMM, minimal maintenance, strong vendor tuning.
- CUTLASS: useful for learning GEMM internals, custom layouts, epilogues, and kernel specialization.
- Triton: useful for Python-level custom kernels and fast iteration, especially when the operation is not just plain GEMM.
- Memory-bound: low arithmetic intensity, bandwidth/launch overhead dominates, TFLOPS remains low despite Tensor Core availability.
- Compute-bound: high arithmetic intensity, Tensor Core utilization matters, achieved TFLOPS approaches a meaningful fraction of peak.

## Verification

After writing this spec, run:

```bash
rg -n "TFLOPS|tflops|roofline|CUTLASS|shape|arithmetic intensity" notes/gemm_cutlass_task_spec.md
```
