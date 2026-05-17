#!/usr/bin/env bash
# Install NVIDIA CUDA Toolkit 12.6 on WSL2 Ubuntu
# Run this script manually: bash scripts/install_cuda_toolkit.sh
set -euo pipefail

echo "=== Installing NVIDIA CUDA Toolkit 12.6 on WSL2 ==="

# 1. Add NVIDIA repository keyring
echo "[1/3] Adding NVIDIA apt repository..."
wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
sudo dpkg -i /tmp/cuda-keyring.deb
rm -f /tmp/cuda-keyring.deb

# 2. Update and install CUDA toolkit (no driver, WSL uses Windows driver)
echo "[2/3] Installing cuda-toolkit-12-6..."
sudo apt update
sudo apt install -y cuda-toolkit-12-6

# 3. Add to PATH
echo "[3/3] Configuring PATH..."
CUDA_PATH="/usr/local/cuda-12.6"
if ! grep -q "CUDA_PATH" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# CUDA Toolkit" >> ~/.bashrc
    echo "export CUDA_HOME=/usr/local/cuda-12.6" >> ~/.bashrc
    echo "export PATH=\$CUDA_HOME/bin:\$PATH" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
    echo "Added CUDA environment variables to ~/.bashrc"
fi

echo ""
echo "=== Installation complete ==="
echo "Run: source ~/.bashrc"
echo "Verify: nvcc --version"