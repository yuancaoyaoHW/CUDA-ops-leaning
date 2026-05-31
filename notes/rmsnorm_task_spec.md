Operator family: RMSNorm, fused add RMSNorm, and a minimal PyTorch C++/CUDA RMSNorm extension.
Scope: turn the week1 day05-day06 and week2 day08/day11-day12 plan items into executable task packages with clear ownership, correctness, benchmark, profile, and note requirements.

## Package 1: `rmsnorm` Triton Baseline

### Owned File Paths

Create or modify only the RMSNorm artifacts for this package:

- `kernels/triton/rmsnorm/__init__.py`
- `kernels/triton/rmsnorm/rmsnorm.py`
- `tests/test_rmsnorm.py`
- `benchmarks/bench_rmsnorm.py`
- `notes/rmsnorm.md`
- `reports/ncu/rmsnorm_*` only when saving Nsight Compute output

Do not edit `study-plan/progress.yaml` as part of the implementation. If progress verification later changes it, commit that diff only when the operator artifact task explicitly allows it.

### PyTorch Reference Formula

Use PyTorch as the correctness reference:

```python
def torch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inv_rms = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x.float() * inv_rms * weight.float()).to(x.dtype)
```

For a 2D learning slice, treat `x` as `(rows, hidden_size)`. A later wrapper can flatten `(batch, seq, hidden_size)` to `(batch * seq, hidden_size)` before launching the same kernel.

### Implementation Constraints

- Implement one Triton program per row, matching the `row_sum` and `row_max` style.
- Require CUDA, contiguous, 2D input for the first package.
- Require `weight` to be 1D, contiguous, CUDA, and shaped `(hidden_size,)`.
- Use `BLOCK_SIZE = triton.next_power_of_2(hidden_size)` and mask all loads/stores for non-aligned hidden sizes.
- Accumulate `sum(x * x)` in fp32 and compute `rsqrt(mean + eps)` in fp32.
- Output shape and dtype must match `x`.
- Do not add backward support, non-contiguous strides, multi-block rows, or vectorized pack experiments to the baseline.
- Keep the baseline readable before optimizing. A later depth experiment can compare naive element loads with a vectorized fp16-friendly layout and report `GB/s`.

Triton address mapping:

```python
row = tl.program_id(0)
offs = tl.arange(0, BLOCK_SIZE)
mask = offs < hidden_size
x_vals = tl.load(x_ptr + row * hidden_size + offs, mask=mask, other=0.0)
w_vals = tl.load(weight_ptr + offs, mask=mask, other=0.0)
sum_sq = tl.sum(x_vals.to(tl.float32) * x_vals.to(tl.float32), axis=0)
inv_rms = tl.rsqrt(sum_sq / hidden_size + eps)
y = x_vals.to(tl.float32) * inv_rms * w_vals.to(tl.float32)
tl.store(out_ptr + row * hidden_size + offs, y, mask=mask)
```

### Shape Set

Aligned hidden sizes:

- `(1, 512)`
- `(128, 1024)`
- `(512, 2048)`
- `(2048, 4096)`

Non-aligned hidden sizes:

- `(1, 513)`
- `(33, 1000)`
- `(128, 1025)`
- `(513, 4097)`

Also test zero rows shaped `(0, 1024)` by returning an empty output without launching. Reject zero hidden size `(rows, 0)` for the first implementation because the RMS denominator is undefined for an empty row.

### dtype and Tolerance Plan

- fp32: compare with `rtol=1e-5, atol=1e-6`.
- fp16: output dtype is fp16; compare with `rtol=1e-2, atol=2e-2`.
- bf16: include if the local GPU and PyTorch build support bf16 CUDA tensors; compare with `rtol=2e-2, atol=3e-2`.
- For fp16 and bf16, reference math must use fp32 accumulation and cast only the final expected output to the input dtype.
- Include edge-value coverage with zeros, small random values, and moderately large values to prove fp32 accumulation avoids fp16 overflow paths.

### Shape Error Tests

- Reject non-2D `x` with a `ValueError` mentioning `2D`.
- Reject non-CUDA `x` or `weight` with a `ValueError` mentioning `CUDA`.
- Reject non-contiguous `x` or `weight` with a `ValueError` mentioning `contiguous`.
- Reject `weight.dim() != 1` with a `ValueError` mentioning `1D`.
- Reject `weight.numel() != hidden_size` with a `ValueError` mentioning `hidden`.
- Reject unsupported dtypes outside fp32/fp16/bf16 with a clear error.
- Reject zero hidden size with a clear error.

### Benchmark JSON Fields and GB/s

Benchmark PyTorch and Triton for fp32/fp16/bf16 where supported. Use warmup, repeated timing, and `torch.cuda.synchronize()` like `bench_row_sum.py`.

Idealized RMSNorm bytes moved:

```python
element_size = torch.empty((), dtype=dtype).element_size()
bytes_moved = (rows * hidden_size * 2 + hidden_size) * element_size
gbps = bytes_moved / (ms / 1e3) / 1e9
```

This counts one read of `x`, one write of `y`, and one read of `weight`. State that `weight` may be cached across rows, so `GB/s` is an effective denominator rather than exact HBM traffic.

JSON schema:

```json
{
  "operator": "rmsnorm",
  "device": "NVIDIA GeForce ...",
  "results": [
    {
      "impl": "triton",
      "rows": 128,
      "hidden_size": 1025,
      "shape": [128, 1025],
      "dtype": "float16",
      "eps": 1e-6,
      "ms": 0.0123,
      "gbps": 41.23
    }
  ]
}
```

### Nsight Profile Plan and Expected Bottleneck

- Profile one isolated Triton launch with Nsight Compute on `(2048, 4096)`, fp16, and save under `reports/ncu/`.
- If fp32 benchmark behavior differs materially, profile `(2048, 4096)`, fp32 as a second run.
- Inspect Speed of Light, Memory Workload Analysis, LaunchStats, and Occupancy sections.
- Expected bottleneck: memory-bound with high memory throughput, low compute utilization, and possible occupancy pressure from a large row vector/register footprint.
- Record achieved occupancy, register pressure clues, and whether non-aligned masking changes runtime enough to matter.

### Learning Note Required Sections

The final `notes/rmsnorm.md` must include:

- `Operator`: formula, input/output shapes, and where RMSNorm appears in LLM blocks.
- `Reference`: PyTorch formula with fp32 accumulation.
- `Implementation`: Triton row mapping, mask behavior for non-aligned hidden sizes, and dtype handling.
- `Tests`: aligned/non-aligned shape list, dtype/tolerance table, shape error tests, and edge cases.
- `Benchmark`: table or JSON summary with `ms`, `GB/s`, shape, dtype, and PyTorch comparison.
- `Profile`: Nsight command, shape, report path, and key metrics.
- `Bottleneck`: memory traffic, launch overhead for small rows, register/occupancy observations for large rows.
- `Next experiment`: one concrete follow-up, such as fused residual add or vectorized fp16 load/store.

## Package 2: `fused_add_rmsnorm` Triton Follow-Up

### Owned File Paths

Create or modify only the fused-add artifacts for this package:

- `kernels/triton/fused_add_rmsnorm/__init__.py`
- `kernels/triton/fused_add_rmsnorm/fused_add_rmsnorm.py`
- `tests/test_fused_add_rmsnorm.py`
- `benchmarks/bench_fused_add_rmsnorm.py`
- `notes/fused_add_rmsnorm.md`
- `reports/ncu/fused_add_rmsnorm_*` only when saving Nsight Compute output

The package may import `triton_rmsnorm` from the baseline package for unfused comparison, but it should not mutate baseline files unless the comparison exposes a shared bug.

### PyTorch Reference Formula

Use an out-of-place reference first:

```python
def torch_fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    added = x.float() + residual.float()
    inv_rms = torch.rsqrt(added.pow(2).mean(dim=-1, keepdim=True) + eps)
    norm_out = (added * inv_rms * weight.float()).to(x.dtype)
    residual_out = added.to(residual.dtype)
    return norm_out, residual_out
```

The Triton wrapper should return `(norm_out, residual_out)` or update a documented residual tensor in place. Tests must make the mutation contract explicit.

### Implementation Constraints

- Implement one Triton program per row.
- Require `x` and `residual` to have identical 2D shape, dtype, device, and contiguous layout.
- Require `weight.shape == (hidden_size,)`.
- Compute `added = x + residual`, accumulate `added * added` in fp32, and use fp32 for `inv_rms`.
- Store both normalized output and residual output in one kernel launch.
- Use masks for every load/store to support non-aligned hidden sizes.
- Avoid a shared-memory-heavy design in Triton unless a benchmark justifies it. The first Triton version can keep the row vector in registers for the normalize store.
- Do not implement backward, non-contiguous strides, quantized paths, or attention fusion in this package.
- Compare against the unfused sequence: residual add plus baseline RMSNorm, including kernel count and launch overhead notes.

### Shape Set

Aligned hidden sizes:

- `(1, 512)`
- `(128, 1024)`
- `(512, 2048)`
- `(2048, 4096)`

Non-aligned hidden sizes:

- `(1, 513)`
- `(33, 1000)`
- `(128, 1025)`
- `(513, 4097)`

Use at least one small-row shape to expose launch overhead and one large-row shape to expose memory bandwidth. Include zero rows `(0, 1024)` if the wrapper can avoid launching.

### dtype and Tolerance Plan

- fp32: `rtol=1e-5, atol=1e-6` for `norm_out`; exact or close comparison for `residual_out`.
- fp16: `rtol=1e-2, atol=2e-2` for `norm_out`; `rtol=1e-3, atol=1e-3` or exact where representable for `residual_out`.
- bf16: include when supported; `rtol=2e-2, atol=3e-2` for `norm_out`.
- Reference must compute `added`, sum of squares, and `inv_rms` in fp32.
- Include inputs with positive and negative residuals to catch accidental `x - residual` or stale residual behavior.

### Shape Error Tests

- Reject non-2D `x` or `residual`.
- Reject mismatched `x.shape` and `residual.shape`.
- Reject mismatched devices or dtypes between `x` and `residual`.
- Reject non-CUDA or non-contiguous tensors.
- Reject invalid `weight` rank, device, dtype, or hidden-size length.
- Reject zero hidden size.
- Assert that unsupported dtypes fail clearly rather than silently running with a wrong dispatch.

### Benchmark JSON Fields and GB/s

Benchmark fused Triton, unfused Triton (`x + residual` then `triton_rmsnorm`), and PyTorch reference where practical.

Idealized fused bytes moved:

```python
element_size = torch.empty((), dtype=dtype).element_size()
bytes_moved = (rows * hidden_size * 4 + hidden_size) * element_size
gbps = bytes_moved / (ms / 1e3) / 1e9
```

This counts reads of `x`, `residual`, and `weight`, plus writes of `residual_out` and `norm_out`. Mention that `weight` cache reuse makes this an effective `GB/s` estimate.

JSON schema:

```json
{
  "operator": "fused_add_rmsnorm",
  "device": "NVIDIA GeForce ...",
  "results": [
    {
      "impl": "triton_fused",
      "rows": 512,
      "hidden_size": 2048,
      "shape": [512, 2048],
      "dtype": "float16",
      "eps": 1e-6,
      "ms": 0.0456,
      "gbps": 184.0,
      "outputs": ["norm_out", "residual_out"]
    }
  ]
}
```

### Nsight Profile Plan and Expected Bottleneck

- Profile fused and unfused paths on `(2048, 4096)`, fp16.
- Save fused and unfused reports under `reports/ncu/` with names that identify the implementation.
- Record kernel count and timing; if possible, use Nsight Systems for launch overhead and Nsight Compute for the fused kernel bottleneck.
- Expected bottleneck: memory bandwidth and launch overhead improvement versus unfused. Register pressure can rise because the fused kernel holds `added`, `x`, `residual`, `weight`, and normalized values live.
- Compare achieved occupancy, memory throughput, register use indicators, and whether fusion improves `ms` despite more per-kernel work.

### Learning Note Required Sections

The final `notes/fused_add_rmsnorm.md` must include:

- `Operator`: residual-add plus RMSNorm formula and mutation/return contract.
- `Reference`: PyTorch reference and unfused comparison path.
- `Implementation`: Triton row mapping, non-aligned mask behavior, residual writeback, and dtype plan.
- `Tests`: aligned/non-aligned shapes, dtype/tolerance table, shape errors, and mutation checks.
- `Benchmark`: fused vs unfused vs PyTorch table with `ms`, `GB/s`, shape, dtype, and speedup.
- `Profile`: Nsight report paths, kernel count, launch timing if captured, and occupancy/memory metrics.
- `Bottleneck`: whether fusion is limited by memory traffic, launch overhead, register pressure, or occupancy.
- `Next experiment`: one follow-up, such as comparing with a CUDA extension, trying a vectorized path, or checking framework implementations.

## Package 3: `rmsnorm_cuda_ext` Minimal PyTorch C++/CUDA Extension

### Owned File Paths

Create or modify only the CUDA extension artifacts for this package:

- `kernels/cuda_ext/rmsnorm_cuda_ext/README.md`
- `kernels/cuda_ext/rmsnorm_cuda_ext/setup.py` or `kernels/cuda_ext/rmsnorm_cuda_ext/build.py`
- `kernels/cuda_ext/rmsnorm_cuda_ext/rmsnorm.cpp`
- `kernels/cuda_ext/rmsnorm_cuda_ext/rmsnorm_kernel.cu`
- `kernels/cuda_ext/rmsnorm_cuda_ext/__init__.py`
- `tests/test_rmsnorm_cuda_ext.py`
- `benchmarks/bench_rmsnorm_cuda_ext.py`
- `notes/rmsnorm_cuda_ext.md`
- `reports/ncu/rmsnorm_cuda_ext_*` only when saving Nsight Compute output

Do not install system packages, do not run `sudo apt install`, and do not download model weights or large repositories. The build must use the existing local conda environment and whatever CUDA toolkit/PyTorch extension support is already present.

### PyTorch Reference Formula

Use the same fp32-accumulation reference as the Triton baseline:

```python
def torch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inv_rms = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x.float() * inv_rms * weight.float()).to(x.dtype)
```

### Implementation Constraints

- Build the smallest PyTorch C++/CUDA extension that exposes `rmsnorm_forward(x, weight, eps)`.
- Use C++ `TORCH_CHECK` validation for device, dtype, rank, contiguity, hidden-size match, and non-empty hidden size.
- Include dtype dispatch for fp32, fp16, and bf16 where supported. It is acceptable to start with fp32/fp16 and mark bf16 unsupported with a clear `TORCH_CHECK` if the local toolchain blocks it, but the package must document the decision.
- CUDA kernel should use fp32 accumulation for `sum_sq` and `inv_rms`.
- Keep the CUDA kernel minimal: one block per row with block reduction; no backward pass, no vectorized half2 requirement, no persistent kernel.
- Prefer a local `setup.py`/JIT load path that builds inside the package directory or a documented torch extension cache. Document build path risks: stale cached objects, wrong `CUDA_HOME`, WSL2 CUDA toolkit mismatch, PyTorch CUDA version mismatch, and compiler errors hidden by cache reuse.
- Never fix build failures by installing system packages unless the user explicitly asks. Diagnose with existing `python`, `torch`, `nvcc`/CUDA availability checks only.
- If using `torch.utils.cpp_extension.load`, keep generated build artifacts out of source-controlled paths or document the build directory.

### Shape Set

Aligned hidden sizes:

- `(1, 512)`
- `(128, 1024)`
- `(512, 2048)`
- `(2048, 4096)`

Non-aligned hidden sizes:

- `(1, 513)`
- `(33, 1000)`
- `(128, 1025)`
- `(513, 4097)`

Also include zero rows `(0, 1024)` if the C++ wrapper can return an empty tensor without launching. Reject zero hidden size.

### dtype and Tolerance Plan

- fp32: `rtol=1e-5, atol=1e-6`.
- fp16: `rtol=1e-2, atol=2e-2`; compute reductions in fp32 and cast final output to fp16.
- bf16: target `rtol=2e-2, atol=3e-2` when dtype dispatch supports it on the local CUDA/PyTorch build.
- Tests must assert the output dtype matches `x.dtype`.
- Include a dtype dispatch test that proves fp32 and fp16 take supported paths, and that unsupported dtype errors mention dtype.

### Shape Error Tests

- C++ `TORCH_CHECK`: `x.is_cuda()` and `weight.is_cuda()`.
- C++ `TORCH_CHECK`: `x.dim() == 2` and `weight.dim() == 1`.
- C++ `TORCH_CHECK`: `x.is_contiguous()` and `weight.is_contiguous()`.
- C++ `TORCH_CHECK`: `weight.numel() == x.size(1)`.
- C++ `TORCH_CHECK`: `x.size(1) > 0`.
- C++ `TORCH_CHECK`: supported dtype and matching dtype or an explicitly documented cast rule.
- Python tests should expect `RuntimeError` and match clear fragments such as `CUDA`, `contiguous`, `2D`, `hidden`, and `dtype`.

### Benchmark JSON Fields and GB/s

Benchmark CUDA extension, Triton baseline, and PyTorch for the same shape/dtype matrix.

Idealized bytes moved:

```python
element_size = torch.empty((), dtype=dtype).element_size()
bytes_moved = (rows * hidden_size * 2 + hidden_size) * element_size
gbps = bytes_moved / (ms / 1e3) / 1e9
```

Use the same denominator as `rmsnorm` so extension vs Triton comparison is readable.

JSON schema:

```json
{
  "operator": "rmsnorm_cuda_ext",
  "device": "NVIDIA GeForce ...",
  "extension_build": {
    "mode": "setup.py",
    "torch_cuda": "12.x",
    "build_dir": "kernels/cuda_ext/rmsnorm_cuda_ext/build"
  },
  "results": [
    {
      "impl": "cuda_ext",
      "rows": 2048,
      "hidden_size": 4096,
      "shape": [2048, 4096],
      "dtype": "float16",
      "eps": 1e-6,
      "ms": 0.0678,
      "gbps": 247.5
    }
  ]
}
```

### Nsight Profile Plan and Expected Bottleneck

- Profile one extension kernel launch on `(2048, 4096)`, fp16, and compare with the Triton baseline profile for the same shape.
- Use Nsight Compute Occupancy and LaunchStats sections. Also collect compiler register output when practical with `--ptxas-options=-v` or equivalent extension build flags.
- Expected bottleneck: memory-bound. The minimal CUDA extension may underperform Triton if reduction code has lower occupancy, poor memory coalescing, higher register pressure, or no vectorized loads.
- Record achieved occupancy, registers per thread if available, shared memory per block, memory throughput, and whether non-aligned shapes add branch/mask overhead.
- Document build/profile failures separately from kernel correctness failures.

### Learning Note Required Sections

The final `notes/rmsnorm_cuda_ext.md` must include:

- `Operator`: RMSNorm formula and shape contract.
- `Build`: exact build command, build directory, `CUDA_HOME`/PyTorch CUDA observations, and build path risks.
- `Validation`: every `TORCH_CHECK`, dtype dispatch behavior, and no system package installation policy.
- `Reference`: PyTorch formula and dtype/tolerance table.
- `Implementation`: C++ binding, CUDA launch config, block reduction strategy, fp32 accumulation, and output casting.
- `Tests`: aligned/non-aligned shapes, dtype coverage, shape error tests, and unsupported dtype behavior.
- `Benchmark`: extension vs Triton vs PyTorch with `ms`, `GB/s`, shape, dtype, and build mode.
- `Profile`: Nsight report path, occupancy/register observations, and memory throughput.
- `Bottleneck`: memory bandwidth vs occupancy/register pressure, plus any build overhead or cache issue.
- `Next experiment`: one concrete follow-up, such as half2 vectorized loads, launch bounds, or fused add RMSNorm in CUDA.

## Common Verification Commands

Smallest relevant checks after implementing each package:

```bash
pytest tests/test_rmsnorm.py
pytest tests/test_fused_add_rmsnorm.py
pytest tests/test_rmsnorm_cuda_ext.py
python benchmarks/bench_rmsnorm.py
python benchmarks/bench_fused_add_rmsnorm.py
python benchmarks/bench_rmsnorm_cuda_ext.py
```

Full operator verification, when package artifacts are changed and the environment is available:

```bash
bash scripts/04_verify_all.sh
python study-plan/progress.py verify --operator rmsnorm
python study-plan/progress.py verify --operator fused_add_rmsnorm
python study-plan/progress.py verify --operator rmsnorm_cuda_ext
```

For this task-spec-only change, the narrow verification is:

```bash
rg -n "rmsnorm_cuda_ext|fused_add_rmsnorm|GB/s|TORCH_CHECK|dtype dispatch|non-aligned" notes/rmsnorm_task_spec.md
```
