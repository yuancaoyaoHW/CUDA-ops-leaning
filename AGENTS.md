# Agent Instructions

This repository is a learning lab for LLM inference framework internals, Triton kernels, and PyTorch C++/CUDA extensions.

## Safety Rules

- Do not run `sudo apt install` or other system-level package installation commands unless the user explicitly asks.
- Do not run `curl | sh`, `wget | sh`, or any pipe-to-shell installer.
- Do not install Windows NVIDIA drivers.
- Do not download model weights.
- Do not install 7B/8B model assets or model-specific runtime bundles.
- Do not clone vLLM, llama.cpp, or other large external repositories unless the user explicitly asks.

## Environment Rules

- Use the conda environment named `llm-kernel-lab` by default.
- Do not assume PyTorch CUDA is working until `scripts/04_verify_all.sh` passes.
- Keep setup steps separately runnable and separately verifiable.

## Required Verification

Before changing code, identify the smallest relevant verification command.

After changing Triton kernels or tests, run:

```bash
bash scripts/04_verify_all.sh
```

If the full verification is too expensive or unavailable, run the narrowest relevant command and state what was not run:

```bash
pytest tests/test_vector_add.py
```

After changing setup scripts, run or ask the user to run the affected script directly. Explain what the script modifies before running it.

## Code Style

- Shell scripts must use `set -euo pipefail`.
- Keep kernels small and readable.
- Prefer reference checks against PyTorch operations.
- Add tests for both aligned and non-aligned tensor sizes when a kernel uses block masking.
