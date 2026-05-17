# BLOCK_SIZE=256
(op4060) ycy@LAPTOP-3IEV3ODP:~/code/llm-kernel-lab$ python benchmarks/bench_vector_add.py
device: NVIDIA GeForce RTX 4060 Laptop GPU

n=       1,024 | Triton:   0.030 ms |    0.41 GB/s | PyTorch:   0.011 ms |    1.12 GB/s | ratio: 2.72
n=       4,096 | Triton:   0.029 ms |    1.71 GB/s | PyTorch:   0.009 ms |    5.18 GB/s | ratio: 3.04
n=      65,536 | Triton:   0.029 ms |   27.44 GB/s | PyTorch:   0.011 ms |   72.25 GB/s | ratio: 2.63
n=   1,048,576 | Triton:   0.032 ms |  393.21 GB/s | PyTorch:   0.012 ms | 1066.98 GB/s | ratio: 2.71
n=  16,777,216 | Triton:   0.832 ms |  241.96 GB/s | PyTorch:   0.826 ms |  243.65 GB/s | ratio: 1.01

# BLOCK_SIZE=512
(op4060) ycy@LAPTOP-3IEV3ODP:~/code/llm-kernel-lab$ python benchmarks/bench_vector_add.py
device: NVIDIA GeForce RTX 4060 Laptop GPU

n=       1,024 | Triton:   0.030 ms |    0.41 GB/s | PyTorch:   0.011 ms |    1.16 GB/s | ratio: 2.85
n=       4,096 | Triton:   0.031 ms |    1.57 GB/s | PyTorch:   0.012 ms |    3.96 GB/s | ratio: 2.52
n=      65,536 | Triton:   0.031 ms |   25.60 GB/s | PyTorch:   0.014 ms |   55.51 GB/s | ratio: 2.17
n=   1,048,576 | Triton:   0.031 ms |  407.02 GB/s | PyTorch:   0.013 ms | 1002.12 GB/s | ratio: 2.46
n=  16,777,216 | Triton:   0.867 ms |  232.18 GB/s | PyTorch:   0.861 ms |  233.80 GB/s | ratio: 1.01

# BLOCK_SIZE=1024
(op4060) ycy@LAPTOP-3IEV3ODP:~/code/llm-kernel-lab$ python benchmarks/bench_vector_add.py
device: NVIDIA GeForce RTX 4060 Laptop GPU

n=       1,024 | Triton:   0.032 ms |    0.39 GB/s | PyTorch:   0.011 ms |    1.16 GB/s | ratio: 2.97
n=       4,096 | Triton:   0.032 ms |    1.54 GB/s | PyTorch:   0.010 ms |    4.82 GB/s | ratio: 3.13
n=      65,536 | Triton:   0.049 ms |   16.00 GB/s | PyTorch:   0.010 ms |   77.85 GB/s | ratio: 4.86
n=   1,048,576 | Triton:   0.030 ms |  425.30 GB/s | PyTorch:   0.012 ms | 1084.07 GB/s | ratio: 2.55
n=  16,777,216 | Triton:   0.865 ms |  232.65 GB/s | PyTorch:   0.858 ms |  234.52 GB/s | ratio: 1.01

# BLOCK_SIZE=2048
(op4060) ycy@LAPTOP-3IEV3ODP:~/code/llm-kernel-lab$ python benchmarks/bench_vector_add.py
device: NVIDIA GeForce RTX 4060 Laptop GPU

n=       1,024 | Triton:   0.031 ms |    0.40 GB/s | PyTorch:   0.010 ms |    1.24 GB/s | ratio: 3.07
n=       4,096 | Triton:   0.033 ms |    1.48 GB/s | PyTorch:   0.011 ms |    4.30 GB/s | ratio: 2.90
n=      65,536 | Triton:   0.033 ms |   23.87 GB/s | PyTorch:   0.010 ms |   78.16 GB/s | ratio: 3.27
n=   1,048,576 | Triton:   0.029 ms |  429.38 GB/s | PyTorch:   0.013 ms |  966.65 GB/s | ratio: 2.25
n=  16,777,216 | Triton:   0.866 ms |  232.52 GB/s | PyTorch:   0.860 ms |  234.01 GB/s | ratio: 1.01
