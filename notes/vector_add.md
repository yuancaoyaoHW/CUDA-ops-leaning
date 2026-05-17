  # vector_add

  Operator: z = x + y
  Shape/dtype:
  Reference: PyTorch x + y
  Implementation: Triton, 1D blocks, masked tail
  Tests: all pass
  Benchmark:
    device: NVIDIA GeForce RTX 4060 Laptop GPU
    
    n=       1,024 | Triton:   0.029 ms | PyTorch:   0.010 ms | ratio: 2.80
    n=       4,096 | Triton:   0.030 ms | PyTorch:   0.011 ms | ratio: 2.85
    n=      65,536 | Triton:   0.028 ms | PyTorch:   0.011 ms | ratio: 2.54
    n=   1,048,576 | Triton:   0.032 ms | PyTorch:   0.011 ms | ratio: 2.84
    n=  16,777,216 | Triton:   0.868 ms | PyTorch:   0.862 ms | ratio: 1.01
  Profile:
  What I learned: triton 先计算分多少块，再在每个块中并行执行add操作。
  What is still unclear: 
  Next experiment:
