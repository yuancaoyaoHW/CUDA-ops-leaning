# llm-kernel-lab

Learning repo for LLM inference framework internals and GPU kernel development on WSL2 Ubuntu.

The target machine is an RTX 4060 Laptop GPU with about 8 GB VRAM and 16 GB system memory. The goal is not to run large models. The goal is to build a small, repeatable lab for learning:

1. Triton kernels
2. PyTorch C++/CUDA extensions
3. vLLM runtime concepts
4. llama.cpp / ggml internals

This stage does not install vLLM, does not clone llama.cpp, and does not download any model weights.

## Repository Layout

```text
.
├── AGENTS.md
├── README.md
├── docs/
│   ├── cuda_extension.md
│   ├── llama_cpp_ggml.md
│   └── vllm_runtime.md
├── kernels/
│   ├── cuda_extension/
│   │   └── README.md
│   └── triton/
│       └── vector_add/
│           └── vector_add.py
├── scripts/
│   ├── 00_check_system.sh
│   ├── 01_setup_conda_env.sh
│   ├── 02_install_pytorch_triton.sh
│   └── 04_verify_all.sh
└── tests/
    └── test_vector_add.py
```

## Step-by-Step Setup

Run each step separately. Do not skip verification between steps.

### 0. Check the WSL2 system

```bash
bash scripts/00_check_system.sh
```

What it modifies: nothing. This is a read-only check.

It prints Linux, Ubuntu, compiler, build tool, conda, Python, and `nvidia-smi` information. It does not install drivers or system packages.

### 1. Create the conda environment

```bash
bash scripts/01_setup_conda_env.sh
```

What it modifies: your conda installation by creating an environment named `llm-kernel-lab`.

It does not use `sudo`, does not modify system Python, and does not install PyTorch.

Activate the environment:

```bash
conda activate llm-kernel-lab
```

### 2. Install Python dependencies

```bash
bash scripts/02_install_pytorch_triton.sh
```

What it modifies: only the active `llm-kernel-lab` conda environment.

It installs:

- PyTorch CUDA wheel
- Triton
- pytest
- numpy

It does not install a Windows NVIDIA driver, does not install system CUDA toolkit, and does not download models.

By default the script uses the PyTorch CUDA 12.1 wheel index:

```text
https://download.pytorch.org/whl/cu121
```

You can override it when needed:

```bash
PYTORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu124 bash scripts/02_install_pytorch_triton.sh
```

### 3. Verify everything

```bash
bash scripts/04_verify_all.sh
```

What it modifies: nothing important in the repository. It may create normal Python cache files.

It verifies:

- The active conda environment is `llm-kernel-lab`
- PyTorch imports successfully
- `torch.cuda.is_available()` is true
- The GPU name can be printed
- Triton imports successfully
- The minimal Triton vector add kernel runs
- `pytest tests/` passes

## Learning Route

### 1. Triton kernel

Start with `kernels/triton/vector_add/vector_add.py`. Learn how a simple elementwise kernel maps program IDs to blocks, applies masks, and launches from Python.

Next exercises:

- Add vector multiply
- Add scalar scale-and-add
- Benchmark against PyTorch
- Implement row-wise reductions

The next learning slice is `axpy` (`z = alpha * x + y`), which keeps the same block/mask structure but adds a scalar parameter.

### 2. CUDA extension

Use `kernels/cuda_extension/` for PyTorch C++/CUDA extension experiments.

Recommended sequence:

- Write a CPU-only C++ extension first
- Add a minimal CUDA kernel
- Use `torch.utils.cpp_extension`
- Compare extension output against PyTorch reference ops

### 3. vLLM runtime

Use `docs/vllm_runtime.md` for notes before installing vLLM.

Focus areas:

- KV cache layout
- PagedAttention
- scheduler behavior
- batching and continuous batching
- how custom kernels fit into runtime design

Do not install vLLM until the PyTorch/Triton baseline is verified.

### 4. llama.cpp / ggml

Use `docs/llama_cpp_ggml.md` for source-reading notes.

For this stage, do not clone llama.cpp. Later, read it separately with attention to:

- tensor representation
- quantization formats
- ggml graph execution
- CPU/GPU backend boundaries

## Constraints

- Do not run `sudo apt install` from repo scripts.
- Do not use `curl | sh` or similar pipe-to-shell installers.
- Do not install Windows NVIDIA drivers from WSL.
- Do not download model weights.
- Do not install 7B/8B model assets or model-specific dependencies.
- Keep each setup step independently runnable and independently verifiable.
