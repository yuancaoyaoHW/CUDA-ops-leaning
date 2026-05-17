#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-llm-kernel-lab}"

conda run -n "$ENV_NAME" python -c '
import torch

print(f"torch.__version__: {torch.__version__}")
print(f"torch.version.cuda: {torch.version.cuda}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable to PyTorch.")

print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
print(f"torch.cuda.get_device_capability(0): {torch.cuda.get_device_capability(0)}")
x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
y = x @ x
print(f"matmul ok: {tuple(y.shape)} {y.dtype}")
'
