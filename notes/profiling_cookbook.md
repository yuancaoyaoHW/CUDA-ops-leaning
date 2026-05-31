# Profiling Cookbook

This cookbook is for the LLM kernel lab on a WSL2 machine with an RTX 4060 Laptop GPU and 8GB VRAM. Keep profiling runs small, reproducible, and tied to one operator hypothesis.

## Tool Choice

Use CUDA events for benchmark timing. They answer "how fast is this kernel or benchmark loop?" with low overhead after warmup. Record median or mean, GB/s for memory-bound kernels, and TFLOPS for GEMM or attention-like kernels.

Use PyTorch profiler when Python, ATen, allocation, or operator composition may be the bottleneck. It answers "which PyTorch operator or Python region owns time or memory?" before dropping to kernel-level tools.

Use `nsys` when the problem is timeline-level. It answers "is the GPU busy, are kernels separated by launch gaps, is Python or synchronization blocking, are transfers on the critical path?" This is the first Nsight tool for launch-bound behavior.

Use `ncu` when one kernel is already identified. It answers "why is this kernel slow?" through occupancy, warp stall, L2, DRAM throughput, memory workload, scheduler, and Speed of Light metrics. Avoid running full benchmark loops under `ncu` unless you intentionally want many repeated kernel reports.

## Environment Pattern

Use the lab conda environment explicitly when profiling from scripts:

```bash
PYTHON=/home/ycy/miniconda3/envs/llm-kernel-lab/bin/python bash scripts/run_nsys.sh benchmarks/bench_vector_add.py
PYTHON=/home/ycy/miniconda3/envs/llm-kernel-lab/bin/python bash scripts/run_ncu.sh scripts/profile_row_sum_one.py
```

The wrappers write reports under `reports/nsys/` and `reports/ncu/`. Override tools only when they are installed outside `PATH`:

```bash
NCU=/opt/nvidia/nsight-compute/2024.3.2/ncu PYTHON=/home/ycy/miniconda3/envs/llm-kernel-lab/bin/python \
  bash scripts/run_ncu.sh scripts/profile_row_max_one.py
```

## Full-Benchmark Profiling

Use full-benchmark profiling to compare shape sweeps, PyTorch vs Triton, and launch behavior across many repeats.

```bash
PYTHON=/home/ycy/miniconda3/envs/llm-kernel-lab/bin/python bash scripts/run_nsys.sh benchmarks/bench_vector_add.py
PYTHON=/home/ycy/miniconda3/envs/llm-kernel-lab/bin/python bash scripts/run_nsys.sh benchmarks/bench_row_sum.py
PYTHON=/home/ycy/miniconda3/envs/llm-kernel-lab/bin/python bash scripts/run_nsys.sh benchmarks/bench_row_max.py
```

Use `nsys stats` after collection:

```bash
nsys stats --report cuda_gpu_kern_sum reports/nsys/<report>.nsys-rep
nsys stats --report cuda_api_sum reports/nsys/<report>.nsys-rep
```

If the timeline shows many tiny kernels with visible gaps, classify the shape as launch-bound before investigating memory or compute metrics. If the timeline shows one dominant kernel with little idle space, switch to `ncu`.

## Single-Kernel Profiling

Use a single-kernel script for `ncu`: one warmup or compile launch, synchronize, then one measured launch. The current scripts follow that pattern:

```bash
PYTHON=/home/ycy/miniconda3/envs/llm-kernel-lab/bin/python bash scripts/run_ncu.sh scripts/profile_row_sum_one.py
PYTHON=/home/ycy/miniconda3/envs/llm-kernel-lab/bin/python bash scripts/run_ncu.sh scripts/profile_row_max_one.py
```

For stricter capture, run `ncu` directly and skip the Triton compile/warmup launch:

```bash
ncu --set full --target-processes all --launch-skip 1 --launch-count 1 \
  -o reports/ncu/row_sum_one_kernel \
  /home/ycy/miniconda3/envs/llm-kernel-lab/bin/python scripts/profile_row_sum_one.py

ncu --set full --target-processes all --launch-skip 1 --launch-count 1 \
  -o reports/ncu/row_max_one_kernel \
  /home/ycy/miniconda3/envs/llm-kernel-lab/bin/python scripts/profile_row_max_one.py
```

Useful focused `ncu` passes:

```bash
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis --target-processes all \
  /home/ycy/miniconda3/envs/llm-kernel-lab/bin/python scripts/profile_row_sum_one.py

ncu --section Occupancy --section WarpStateStatistics --section SchedulerStatistics --target-processes all \
  /home/ycy/miniconda3/envs/llm-kernel-lab/bin/python scripts/profile_row_max_one.py
```

## Operator Commands

Vector add:

```bash
PYTHON=/home/ycy/miniconda3/envs/llm-kernel-lab/bin/python bash scripts/run_nsys.sh benchmarks/bench_vector_add.py
PYTHON=/home/ycy/miniconda3/envs/llm-kernel-lab/bin/python bash scripts/run_ncu.sh benchmarks/bench_vector_add.py
```

`vector_add` is usually memory-bound for large `n` and launch-bound for small `n`. Record effective bandwidth as `3 * n_elements * element_size / time`.

Row sum:

```bash
PYTHON=/home/ycy/miniconda3/envs/llm-kernel-lab/bin/python bash scripts/run_nsys.sh benchmarks/bench_row_sum.py
PYTHON=/home/ycy/miniconda3/envs/llm-kernel-lab/bin/python bash scripts/run_ncu.sh scripts/profile_row_sum_one.py
```

Row max:

```bash
PYTHON=/home/ycy/miniconda3/envs/llm-kernel-lab/bin/python bash scripts/run_nsys.sh benchmarks/bench_row_max.py
PYTHON=/home/ycy/miniconda3/envs/llm-kernel-lab/bin/python bash scripts/run_ncu.sh scripts/profile_row_max_one.py
```

Future row softmax:

```bash
PYTHON=/home/ycy/miniconda3/envs/llm-kernel-lab/bin/python bash scripts/run_nsys.sh benchmarks/bench_row_softmax.py
ncu --set full --target-processes all --launch-skip 1 --launch-count 1 \
  -o reports/ncu/row_softmax_one_kernel \
  /home/ycy/miniconda3/envs/llm-kernel-lab/bin/python scripts/profile_row_softmax_one.py
```

Expect row softmax to be memory-bound unless the implementation repeatedly reloads rows or uses expensive exponentials poorly. Check DRAM bytes, L2 hit rate, warp stalls, and numerical stability paths.

Future rmsnorm:

```bash
PYTHON=/home/ycy/miniconda3/envs/llm-kernel-lab/bin/python bash scripts/run_nsys.sh benchmarks/bench_rmsnorm.py
ncu --set full --target-processes all --launch-skip 1 --launch-count 1 \
  -o reports/ncu/rmsnorm_one_kernel \
  /home/ycy/miniconda3/envs/llm-kernel-lab/bin/python scripts/profile_rmsnorm_one.py
```

Expect rmsnorm to be memory-bound for normal hidden sizes. Record bytes for input, weight, and output, and watch whether reduction plus normalization are fused.

GEMM:

```bash
PYTHON=/home/ycy/miniconda3/envs/llm-kernel-lab/bin/python bash scripts/run_nsys.sh benchmarks/bench_gemm.py
ncu --set full --target-processes all --launch-skip 1 --launch-count 1 \
  -o reports/ncu/gemm_one_kernel \
  /home/ycy/miniconda3/envs/llm-kernel-lab/bin/python scripts/profile_gemm_one.py
```

For large square GEMM, expect compute-bound behavior and record TFLOPS. For decode-like `M=1` or small batch GEMM, expect memory-bound behavior despite using matrix multiply.

FlashAttention toy:

```bash
PYTHON=/home/ycy/miniconda3/envs/llm-kernel-lab/bin/python bash scripts/run_nsys.sh benchmarks/bench_flash_attention_toy.py
ncu --set full --target-processes all --launch-skip 1 --launch-count 1 \
  -o reports/ncu/flash_attention_toy_one_kernel \
  /home/ycy/miniconda3/envs/llm-kernel-lab/bin/python scripts/profile_flash_attention_toy_one.py
```

For prefill-like toy attention, check compute throughput, L2 reuse, and shared-memory effects. For decode-like attention, expect memory-bound behavior from K/V reads.

## Metrics Interpretation

Launch-bound:

- Evidence: `nsys` shows short kernels separated by visible launch gaps, CUDA API time is large relative to GPU time, and small shapes do not improve bandwidth.
- Usual fixes: larger problem sizes for measurement, fusion, CUDA Graphs, fewer Python dispatches.

Memory-bound:

- Evidence: low arithmetic intensity, Memory SOL or DRAM throughput is high relative to SM utilization, and extra math has little effect on time.
- For vector add, row_sum, row_max, row_softmax, and rmsnorm, start with this assumption for large tensors.
- Usual fixes: coalesced loads, fewer global reads/writes, fusion, vectorized access, lower precision where correct.

Compute-bound:

- Evidence: SM or tensor-pipe utilization is high, DRAM throughput is not saturated, and Speed of Light points to compute.
- GEMM and prefill-like FlashAttention toy can be compute-bound at large enough shapes.
- Usual fixes: Tensor Core usage, tiling, instruction mix, ILP, reducing divergence.

Occupancy:

- Occupancy is active warps divided by maximum resident warps. Low occupancy can hurt latency hiding, but 100% occupancy is not required.
- Check register use, shared memory per block, block size, and whether the grid has enough blocks to fill the RTX 4060 Laptop GPU.
- If occupancy rises but time gets worse, look for register spilling, lower cache locality, or extra memory traffic.

Warp stall:

- Long scoreboard usually means global memory latency is exposed.
- Barrier stalls point at synchronization or reduction structure.
- Math pipe throttle can be normal for compute-bound kernels.
- Near-zero eligible warps means the scheduler lacks work; inspect occupancy, dependency chains, and memory access.

L2:

- High L2 hit rate helps repeated reads and tiled kernels. Low L2 hit rate is normal for streaming vector add but suspicious for intended data reuse.
- If L2 throughput and DRAM throughput are both low, the kernel may be compute-bound or stalled elsewhere.
- On 8GB VRAM, avoid oversized working-set experiments that page, OOM, or heat the laptop GPU.

DRAM throughput:

- Compare effective benchmark GB/s with `ncu` DRAM throughput. Effective bandwidth counts useful algorithm bytes; DRAM throughput counts actual hardware bytes.
- High DRAM throughput plus low useful GB/s means wasted traffic, poor sector efficiency, uncoalesced access, or extra reads/writes.
- Low DRAM throughput for tiny shapes often means launch-bound behavior or data served from L2, not a memory-system success.

## WSL2 / RTX 4060 Laptop 8GB Constraints

- Keep shapes modest: hidden `512/1024/2048`, sequence `128/512/1024`, head dim `64/128`, batch `1/2/4`.
- Leave memory headroom. Do not use model weights, 7B/8B assets, long-context inference, or large external repos for profiling this lab.
- Laptop clocks, temperature, and power limits can dominate results. Record `nvidia-smi` temperature and power if a benchmark changes unexpectedly.
- WSL2 may expose Nsight command-line tools while GUI analysis happens on the Windows host. The wrappers print `nsys-ui` or `ncu-ui` report paths for host-side viewing.
- Some hardware counters may be unavailable because of driver, WSL2, or permissions. Do not install system packages to fix this during lab work.

## Failure and Permission Handling

If `ncu` is missing:

```bash
command -v ncu || test -x /opt/nvidia/nsight-compute/2024.3.2/ncu
```

If it exists outside `PATH`, set `NCU=/path/to/ncu`. If it is not installed, record that Nsight Compute was unavailable and use CUDA events plus `nsys` if available.

If `nsys` is missing, record that Nsight Systems was unavailable and continue with CUDA event benchmarks or PyTorch profiler. Do not install packages unless explicitly asked.

If `ncu` reports permission or counter access errors, retry a lighter section:

```bash
ncu --section SpeedOfLight --target-processes all \
  /home/ycy/miniconda3/envs/llm-kernel-lab/bin/python scripts/profile_row_sum_one.py
```

If counters are still blocked, record the exact error and fall back to timing, effective GB/s or TFLOPS, PyTorch profiler tables, and `nsys` timeline evidence. Do not run `sudo apt install`, pipe-to-shell installers, driver installers, or system-level changes.

## What To Record In Operator Notes

For each operator note, record:

- Operator, implementation, dtype, shape family, and exact command.
- Correctness status and tolerance versus PyTorch.
- Benchmark timing after warmup, repeat count, and effective GB/s or TFLOPS.
- Whether the result is launch-bound, memory-bound, compute-bound, or still unclear.
- Nsight Systems evidence: GPU idle, launch gap, sync, transfers, and dominant kernels.
- Nsight Compute evidence: Speed of Light, occupancy, main warp stall reason, L2 hit rate, DRAM throughput, and memory workload notes.
- WSL2 / RTX 4060 Laptop 8GB caveats: thermals, power, unavailable counters, OOM limits, or skipped full profiling.
- One next experiment with a specific prediction.
