#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-llm-kernel-lab}"
PYTORCH_CUDA_INDEX_URL="${PYTORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
OP_DEV_PACKAGES=(
  triton
  pytest
  hypothesis
  numpy
  scipy
  einops
  rich
  pandas
  matplotlib
  ninja
  cmake
  packaging
  psutil
  pyyaml
)

if ! command -v conda >/dev/null 2>&1; then
  printf 'ERROR: conda was not found on PATH.\n' >&2
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -Fx "$ENV_NAME" >/dev/null 2>&1; then
  printf 'ERROR: conda environment does not exist: %s\n' "$ENV_NAME" >&2
  printf 'Run: bash scripts/01_create_conda_env.sh\n' >&2
  exit 1
fi

printf 'Installing into conda environment: %s\n' "$ENV_NAME"
printf 'Using PyTorch CUDA wheel index: %s\n' "$PYTORCH_CUDA_INDEX_URL"

# Official PyTorch selector shape:
#   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
# Keep inference-framework packages in separate envs; this script installs only operator-dev tooling.
conda run -n "$ENV_NAME" python -m pip install --upgrade pip setuptools wheel
conda run -n "$ENV_NAME" python -m pip install \
  --trusted-host download.pytorch.org \
  --index-url "$PYTORCH_CUDA_INDEX_URL" \
  torch torchvision torchaudio
conda run -n "$ENV_NAME" python -m pip install "${OP_DEV_PACKAGES[@]}"

printf '\nInstalled package versions:\n'
conda run -n "$ENV_NAME" python -c '
import importlib.metadata as metadata

for package in (
    "torch",
    "torchvision",
    "torchaudio",
    "triton",
    "pytest",
    "hypothesis",
    "numpy",
    "scipy",
    "einops",
    "rich",
    "pandas",
    "matplotlib",
    "ninja",
    "cmake",
    "packaging",
    "psutil",
    "pyyaml",
):
    try:
        print(f"{package}: {metadata.version(package)}")
    except metadata.PackageNotFoundError:
        print(f"{package}: NOT INSTALLED")
'
