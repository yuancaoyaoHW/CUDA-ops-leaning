#!/usr/bin/env bash
# Setup CUDA_HOME for PyTorch's bundled CUDA (for CUDA extension compilation)
# Use this if you don't want to install system CUDA toolkit
set -euo pipefail

echo "=== Setting up CUDA_HOME for PyTorch bundled CUDA ==="

# Get PyTorch's CUDA path
CUDA_LIB_DIR=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path.replace('/share/cmake/Torch', ''))" 2>/dev/null || echo "")

if [[ -z "$CUDA_LIB_DIR" ]]; then
    echo "ERROR: Could not find PyTorch CUDA path"
    exit 1
fi

# Check for nvcc in PyTorch's CUDA
PT_CUDA_HOME=""
for d in "$CUDA_LIB_DIR" "$CUDA_LIB_DIR/../"; do
    if [[ -x "$d/bin/nvcc" ]]; then
        PT_CUDA_HOME=$(cd "$d" && pwd)
        break
    fi
done

if [[ -z "$PT_CUDA_HOME" ]]; then
    echo "PyTorch's bundled CUDA does not include nvcc."
    echo "You need to install system CUDA toolkit for CUDA extension compilation."
    echo "Run: bash scripts/install_cuda_toolkit.sh"
    exit 1
fi

echo "Found PyTorch CUDA at: $PT_CUDA_HOME"

# Add to bashrc if not present
if ! grep -q "CUDA_HOME.*llm-kernel-lab" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# CUDA for llm-kernel-lab (PyTorch bundled)" >> ~/.bashrc
    echo "export CUDA_HOME=$PT_CUDA_HOME" >> ~/.bashrc
    echo "export PATH=\$CUDA_HOME/bin:\$PATH" >> ~/.bashrc
    echo "Added to ~/.bashrc"
fi

echo ""
echo "Run: source ~/.bashrc"
echo "Then verify CUDA extension: python scripts/verify_cuda_ext.py"