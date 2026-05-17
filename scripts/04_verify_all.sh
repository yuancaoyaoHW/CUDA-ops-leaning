#!/usr/bin/env bash
# Verify PyTorch, CUDA, and Triton are working.
# Usage: bash scripts/04_verify_all.sh [--quick]
set -euo pipefail

QUICK_MODE=false
[[ "${1:-}" == "--quick" ]] && QUICK_MODE=true
ENV_NAME="${ENV_NAME:-llm-kernel-lab}"

echo "=== Verifying ${ENV_NAME} environment ==="

# Determine Python executable.
PYTHON="$HOME/miniconda3/envs/${ENV_NAME}/bin/python"

if [[ -n "${CONDA_PREFIX:-}" ]] && [[ "${CONDA_DEFAULT_ENV:-}" == "$ENV_NAME" ]]; then
    PYTHON="python"
fi

if [[ ! -x "$PYTHON" ]] && [[ "$PYTHON" != "python" ]]; then
    echo "ERROR: Python not found for conda env: ${ENV_NAME}" >&2
    echo "Run: ENV_NAME=${ENV_NAME} bash scripts/01_setup_conda_env.sh" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[1/6] Python version:"
"$PYTHON" --version

echo ""
echo "[2/6] PyTorch + CUDA:"
"$PYTHON" -c "
import torch
print('  torch:', torch.__version__)
print('  cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('  device:', torch.cuda.get_device_name(0))
    print('  capability:', torch.cuda.get_device_capability(0))
    x = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
    y = x @ x
    print('  matmul ok:', y.shape, y.dtype)
"

echo ""
echo "[3/6] Triton:"
"$PYTHON" -c "
import triton
print('  triton:', triton.__version__)
"

echo ""
echo "[4/6] Triton kernel test:"
"$PYTHON" "$SCRIPT_DIR/verify_triton.py"

if $QUICK_MODE; then
    echo ""
    echo "[5/6] pytest: SKIPPED (quick mode)"
    echo "[6/6] CUDA extension: SKIPPED (requires system nvcc)"
else
    echo ""
    echo "[5/6] pytest:"
    "$PYTHON" -m pytest "$SCRIPT_DIR/../tests" -v

    echo ""
    echo "[6/6] CUDA extension:"
    if command -v nvcc &>/dev/null; then
        "$PYTHON" "$SCRIPT_DIR/verify_cuda_ext.py"
    else
        echo "  SKIPPED: nvcc not found. Install CUDA toolkit:"
        echo "    bash scripts/install_cuda_toolkit.sh"
    fi
fi

echo ""
echo "=== Verification complete ==="
if ! command -v nvcc &>/dev/null; then
    echo "Note: Install system CUDA toolkit for CUTLASS/Nsight support:"
    echo "  bash scripts/install_cuda_toolkit.sh"
fi
