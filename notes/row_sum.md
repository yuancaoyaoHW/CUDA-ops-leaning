
  Operator: row_sum, y = x.sum(dim=1)
  Shape/dtype: float32, (128,256) ... (4096,2048)
  Reference: PyTorch x.sum(dim=1)
  Implementation: one Triton program per row, tl.sum within block
  Tests: 9 passed, includes aligned/non-aligned cols, float16, empty dims, shape errors
  Benchmark: small shapes PyTorch faster; large shapes Triton close/slightly faster
  Profile: Nsight Compute isolated one Triton row_sum_kernel launch on shape (4096, 2048), float32.
  Duration: 166.05 us.
  Compute throughput: 6.64%.
  Memory throughput: 96.32%.
  Bottleneck: memory-bound. The kernel spends most of its time moving input rows from global memory; arithmetic utilization is low.
  Next experiment: compare cols=1024/2048/4096 or try row_max before softmax.

