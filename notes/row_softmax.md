# row_softmax

Operator: row_softmax, `y = softmax(x, dim=1)`.
Scope: 2D contiguous CUDA tensors shaped `[rows, cols]`.
Reference: PyTorch stable softmax formula in `pytorch_row_softmax`.
Implementation: one Triton program handles one row.
Kernel mapping: `pid = tl.program_id(0)` selects the row.
Addressing: `pid * cols + offsets` maps lanes to contiguous columns.
Masking: `offsets < cols` protects non-power-of-two column counts.
Masked load value: `-inf`, so padding never changes the row max.
Masked store: only valid columns are written.
Numerical stability: subtract the row max before `tl.exp`.
dtype support: fp32 and fp16 inputs.
Accumulation: values are promoted to fp32 before max, exp, and sum.
Output dtype: Triton store casts back to the output tensor dtype.
Zero rows: returns an empty tensor with the same shape.
Zero columns: rejected because a softmax row needs at least one value.
Non-contiguous inputs: rejected; this baseline assumes contiguous rows.
Unsupported dtypes: rejected for this first learning slice.

## correctness

Tests compare Triton output with `torch.softmax(x, dim=1)`.
fp32 tolerance: `rtol=1e-5`, `atol=1e-6`.
fp16 tolerance: `rtol=1e-2`, `atol=2e-2`.
Aligned shapes tested: `(128, 256)` and related power-of-two columns.
Non-aligned shapes tested: `(128, 257)`, `(1, 257)`, and `(33, 1000)`.
Negative-only rows verify that masked lanes use `-inf`, not `0.0`.
Large-logit rows include values around `1000` and `-1000`.
The large-logit test checks finite outputs and row normalization behavior.
Shape tests cover non-2D input, zero columns, unsupported dtype, and non-contiguous input.
Narrow verification result: `17 passed` for `tests/test_row_softmax.py`.

## performance

Device: NVIDIA GeForce RTX 4060 Laptop GPU.
Metric: effective GB/s assumes one read plus one write.
This GB/s is not a full roofline model because softmax also does exp, division, and reductions.
Benchmark JSON: `reports/json/row_softmax_bench.json`.
fp32 `(128, 256)`: Triton 0.043 ms, 6.17 GB/s; PyTorch 0.020 ms.
fp32 `(128, 257)`: Triton 0.044 ms, 5.95 GB/s; PyTorch 0.013 ms.
fp32 `(512, 512)`: Triton 0.040 ms, 52.02 GB/s; PyTorch 0.013 ms.
fp32 `(1024, 1024)`: Triton 0.041 ms, 206.31 GB/s; PyTorch 0.013 ms.
fp32 `(2048, 1024)`: Triton 0.040 ms, 418.69 GB/s; PyTorch 0.021 ms.
fp32 `(4096, 2048)`: Triton 0.342 ms, 196.44 GB/s; PyTorch 0.335 ms.
fp16 `(128, 256)`: Triton 0.032 ms, 4.06 GB/s; PyTorch 0.016 ms.
fp16 `(128, 257)`: Triton 0.032 ms, 4.10 GB/s; PyTorch 0.016 ms.
fp16 `(512, 512)`: Triton 0.034 ms, 31.26 GB/s; PyTorch 0.011 ms.
fp16 `(1024, 1024)`: Triton 0.032 ms, 132.42 GB/s; PyTorch 0.020 ms.
fp16 `(2048, 1024)`: Triton 0.031 ms, 271.27 GB/s; PyTorch 0.017 ms.
fp16 `(4096, 2048)`: Triton 0.036 ms, 927.63 GB/s; PyTorch 0.068 ms.
Small rows are launch-bound and PyTorch wins.
The largest fp32 case is close to PyTorch.
The largest fp16 case is faster than PyTorch in this run.

## profile

Nsight Compute report: `reports/ncu/row_softmax_one_kernel.ncu-rep`.
Profile shape: `(4096, 2048)`, fp32.
Kernel duration: 232.32 us.
Memory throughput: 205.17 Gbyte/s.
DRAM throughput: 91.69 percent of peak according to Nsight Compute.
Compute throughput: 10.40 percent.
Registers per thread: 34.
Achieved occupancy: 89.56 percent.
Top stall reported: scoreboard dependency on L1TEX data access.

## bottleneck

The baseline is primarily memory-throughput limited for the profiled large fp32 row.
The kernel also pays nontrivial softmax math cost from `tl.exp`, division, and two reductions.
Small shapes are dominated by launch overhead, which explains poor effective GB/s.
Large rows expose more memory traffic and reach higher effective bandwidth.
Register use is moderate, but not the first bottleneck in the one-kernel profile.
The profile does not suggest dense compute saturation; compute throughput stayed near 10 percent.

## next_experiment

Next experiment: implement `online_softmax` for larger rows and compare cols `2048`, `4096`, and `8192`.
The hypothesis is that online state combination helps when rows exceed one block.
Measure whether fewer logical passes over row state beats the extra exp and register pressure.
Keep the same correctness tests for masking, dtype behavior, and large logits.
