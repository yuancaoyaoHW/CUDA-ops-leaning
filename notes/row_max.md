 Operator: row_max, y = x.max(dim=1).values
  Implementation: one Triton program per row, tl.max within block
  Key correctness point: masked values must use -inf, not 0.0
  Tests:
  Benchmark:

  Profile: Nsight Compute isolated one Triton row_max_kernel launch on shape (4096, 2048), float32.
  Duration: 166.34 us.
  Compute throughput: 6.63%.
  Memory throughput: 95.77%.
  Bottleneck: memory-bound. The kernel is dominated by reading input rows from global memory; max comparison work is light.
  Next experiment: combine row_max and row_sum into a softmax learning slice.
