# llm-kernel-lab 环境配置说明

## 当前状态

| 组件 | 状态 |
|------|------|
| GPU | RTX 4060 Laptop, 8GB, sm_89 |
| PyTorch | 2.12.0+cu126 |
| Triton | 3.7.0 |
| graphviz | 已安装 |
| jq | 已安装 |
| CUDA Toolkit (nvcc) | 未安装 |

## 快速验证

```bash
# 在项目目录运行
bash scripts/04_verify_all.sh

# 或使用完整 Python 路径（不激活 conda）
~/miniconda3/envs/llm-kernel-lab/bin/python scripts/verify_triton.py
```

## 安装系统 CUDA Toolkit（可选）

用于 CUTLASS 编译、Nsight profiling、PyTorch CUDA extension：

```bash
bash scripts/install_cuda_toolkit.sh
source ~/.bashrc
nvcc --version
```

安装后可验证 CUDA extension：

```bash
~/miniconda3/envs/llm-kernel-lab/bin/python scripts/verify_cuda_ext.py
```

## 目录结构

```
kernels/
  triton/        # Triton kernels
  cuda/          # CUDA C++ kernels (需要 nvcc)
tests/           # pytest 测试
benchmarks/     # 性能基准测试
reports/
  nsys/          # Nsight Systems 报告
  ncu/           # Nsight Compute 报告
  json/          # 其他数据
scripts/
  verify_triton.py      # Triton 验证
  verify_cuda_ext.py    # CUDA extension 验证
  run_nsys.sh           # Nsight Systems profiling
  run_ncu.sh            # Nsight Compute profiling
  install_cuda_toolkit.sh # 安装 CUDA toolkit
```

## 使用 Nsight

安装 CUDA toolkit 后：

```bash
# 系统级 trace
bash scripts/run_nsys.sh benchmarks/bench_vector_add.py

# 单 kernel profiling
bash scripts/run_ncu.sh benchmarks/bench_vector_add.py

# 在 Windows 上打开报告
# nsys-ui reports/nsys/bench_vector_add_*.nsys-rep
# ncu-ui reports/ncu/bench_vector_add_*.ncu-rep
```