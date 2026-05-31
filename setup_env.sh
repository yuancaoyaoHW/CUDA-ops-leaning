#!/usr/bin/env bash
# ============================================================
# llm-kernel-lab 一键环境部署脚本
#
# 用法:
#   bash setup_env.sh           # 完整安装（推荐首次使用）
#   bash setup_env.sh --check   # 仅检查当前环境状态
#   bash setup_env.sh --minimal # 仅安装 conda 环境 + PyTorch/Triton
#   bash setup_env.sh --full    # 完整安装 + CUDA toolkit + 资料下载
#
# 前置条件:
#   - Ubuntu/WSL2
#   - conda (miniconda/anaconda) 已安装
#   - NVIDIA GPU driver 已安装（WSL2 使用 Windows 驱动）
#   - 网络连接
#
# 安装内容:
#   1. conda 环境 (llm-kernel-lab): Python 3.11 + PyTorch + Triton + 开发工具
#   2. conda 环境 (llm-vllm-lab): vLLM 框架（可选）
#   3. CUDA Toolkit 12.6（可选，用于 CUTLASS/Nsight/CUDA extension）
#   4. 系统工具: jq, graphviz, wget, git
#   5. 学习资料下载（可选）
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

# ---- Colors ----
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

info()    { printf "${GREEN}[✓]${NC} %s\n" "$1"; }
warn()    { printf "${YELLOW}[!]${NC} %s\n" "$1"; }
error()   { printf "${RED}[✗]${NC} %s\n" "$1"; }
section() { printf "\n${BLUE}${BOLD}=== %s ===${NC}\n\n" "$1"; }

# ---- Parse args ----
MODE="default"
for arg in "${@:-}"; do
  case "$arg" in
    --check)   MODE="check" ;;
    --minimal) MODE="minimal" ;;
    --full)    MODE="full" ;;
    --help|-h) MODE="help" ;;
  esac
done

if [[ "$MODE" == "help" ]]; then
  echo "用法: bash setup_env.sh [--check|--minimal|--full]"
  echo ""
  echo "  (无参数)   安装 conda 环境 + PyTorch/Triton + 系统工具"
  echo "  --check    仅检查环境状态"
  echo "  --minimal  仅安装 conda 环境 + PyTorch/Triton"
  echo "  --full     完整安装 + CUDA toolkit + vLLM 环境 + 资料下载"
  exit 0
fi

# ============================================================
# 环境检查
# ============================================================
check_environment() {
  section "环境检查"

  # OS
  if grep -qi "microsoft" /proc/version 2>/dev/null; then
    info "WSL2 detected"
  else
    info "Native Linux detected"
  fi

  # GPU
  if command -v nvidia-smi &>/dev/null; then
    local gpu_name
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    local driver_ver
    driver_ver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    info "GPU: $gpu_name (Driver: $driver_ver)"
  else
    error "nvidia-smi not found — GPU driver not installed"
    return 1
  fi

  # Conda
  if command -v conda &>/dev/null; then
    info "Conda: $(conda --version 2>&1)"
  else
    error "Conda not found"
    return 1
  fi

  # Conda envs
  if conda env list | awk '{print $1}' | grep -Fx "llm-kernel-lab" &>/dev/null; then
    info "Conda env 'llm-kernel-lab' exists"
    # Check PyTorch
    local torch_ver
    torch_ver=$(conda run -n llm-kernel-lab python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "NOT INSTALLED")
    info "  PyTorch: $torch_ver"
    # Check Triton
    local triton_ver
    triton_ver=$(conda run -n llm-kernel-lab python -c "import triton; print(triton.__version__)" 2>/dev/null || echo "NOT INSTALLED")
    info "  Triton: $triton_ver"
    # Check CUDA available
    local cuda_avail
    cuda_avail=$(conda run -n llm-kernel-lab python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
    if [[ "$cuda_avail" == "True" ]]; then
      info "  CUDA available: True"
    else
      warn "  CUDA available: False"
    fi
  else
    warn "Conda env 'llm-kernel-lab' not found"
  fi

  if conda env list | awk '{print $1}' | grep -Fx "llm-vllm-lab" &>/dev/null; then
    info "Conda env 'llm-vllm-lab' exists"
  else
    warn "Conda env 'llm-vllm-lab' not found (optional)"
  fi

  # CUDA Toolkit
  if command -v nvcc &>/dev/null; then
    info "CUDA Toolkit: $(nvcc --version | grep release | awk '{print $6}')"
  else
    warn "CUDA Toolkit (nvcc) not installed (optional, needed for CUTLASS/Nsight)"
  fi

  # System tools
  for tool in git wget jq dot; do
    if command -v "$tool" &>/dev/null; then
      info "$tool: installed"
    else
      warn "$tool: not found"
    fi
  done

  # Resources
  if [[ -d "$PROJECT_DIR/resources/papers" ]] && [[ $(ls "$PROJECT_DIR/resources/papers/"*.pdf 2>/dev/null | wc -l) -gt 0 ]]; then
    local paper_count
    paper_count=$(ls "$PROJECT_DIR/resources/papers/"*.pdf 2>/dev/null | wc -l)
    info "Papers downloaded: $paper_count"
  else
    warn "Papers not downloaded yet"
  fi

  if [[ -d "$PROJECT_DIR/resources/repos" ]] && [[ $(ls -d "$PROJECT_DIR/resources/repos/"*/ 2>/dev/null | wc -l) -gt 0 ]]; then
    local repo_count
    repo_count=$(ls -d "$PROJECT_DIR/resources/repos/"*/ 2>/dev/null | wc -l)
    info "Repos cloned: $repo_count"
  else
    warn "Repos not cloned yet"
  fi
}

if [[ "$MODE" == "check" ]]; then
  check_environment
  exit 0
fi

# ============================================================
# Step 1: 系统依赖
# ============================================================
install_system_deps() {
  section "Step 1: 安装系统依赖"

  local packages_to_install=()
  for pkg in wget curl git jq graphviz; do
    if ! command -v "$pkg" &>/dev/null; then
      packages_to_install+=("$pkg")
    fi
  done

  if [[ ${#packages_to_install[@]} -gt 0 ]]; then
    info "Installing: ${packages_to_install[*]}"
    sudo apt update -qq
    sudo apt install -y -qq "${packages_to_install[@]}"
  else
    info "All system dependencies already installed"
  fi
}

# ============================================================
# Step 1.5: 自动安装 Miniconda（如果 conda 不存在）
# ============================================================
ensure_conda() {
  if command -v conda &>/dev/null; then
    info "Conda already installed: $(conda --version 2>&1)"
    return 0
  fi

  section "Step 1.5: 自动安装 Miniconda"

  local MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
  local INSTALLER="/tmp/miniconda_installer.sh"

  info "Downloading Miniconda installer..."
  wget -q "$MINICONDA_URL" -O "$INSTALLER"
  chmod +x "$INSTALLER"

  info "Installing Miniconda to ~/miniconda3..."
  bash "$INSTALLER" -b -p "$HOME/miniconda3"
  rm -f "$INSTALLER"

  # Initialize conda for current shell
  eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"

  # Add to .bashrc if not already there
  if ! grep -q "miniconda3" ~/.bashrc 2>/dev/null; then
    "$HOME/miniconda3/bin/conda" init bash
    info "Conda init added to ~/.bashrc"
  fi

  # Verify
  if command -v conda &>/dev/null || [[ -x "$HOME/miniconda3/bin/conda" ]]; then
    export PATH="$HOME/miniconda3/bin:$PATH"
    info "Miniconda installed: $(conda --version 2>&1)"
  else
    error "Miniconda installation failed"
    exit 1
  fi
}

# ============================================================
# Step 2: Conda 环境 (llm-kernel-lab)
# ============================================================
setup_kernel_env() {
  section "Step 2: 创建 conda 环境 (llm-kernel-lab)"

  ensure_conda

  local ENV_NAME="llm-kernel-lab"

  if conda env list | awk '{print $1}' | grep -Fx "$ENV_NAME" &>/dev/null; then
    info "Conda env '$ENV_NAME' already exists"
  else
    info "Creating conda env '$ENV_NAME'..."
    conda env create -f "$PROJECT_DIR/environment.yml"
  fi

  # Install PyTorch + Triton + dev packages
  info "Installing/updating PyTorch + Triton + dev packages..."
  local PYTORCH_CUDA_INDEX_URL="https://download.pytorch.org/whl/cu126"
  local DEV_PACKAGES=(
    triton pytest hypothesis numpy scipy einops rich pandas matplotlib
    ninja cmake packaging psutil pyyaml tqdm tabulate
  )

  conda run -n "$ENV_NAME" python -m pip install --upgrade pip setuptools wheel -q
  conda run -n "$ENV_NAME" python -m pip install --index-url "$PYTORCH_CUDA_INDEX_URL" \
    torch torchvision torchaudio -q
  conda run -n "$ENV_NAME" python -m pip install "${DEV_PACKAGES[@]}" -q

  info "Verifying installation..."
  conda run -n "$ENV_NAME" python -c "
import torch
import triton
print(f'  PyTorch {torch.__version__}')
print(f'  Triton  {triton.__version__}')
print(f'  CUDA    {torch.version.cuda}')
print(f'  GPU     {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NOT AVAILABLE\"}')
"
  info "Conda env '$ENV_NAME' ready"
}

# ============================================================
# Step 3: CUDA Toolkit (optional)
# ============================================================
setup_cuda_toolkit() {
  section "Step 3: 安装 CUDA Toolkit 12.6"

  if command -v nvcc &>/dev/null; then
    info "CUDA Toolkit already installed: $(nvcc --version | grep release | awk '{print $6}')"
    return 0
  fi

  info "Installing CUDA Toolkit 12.6..."

  # Add NVIDIA repository
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
  sudo dpkg -i /tmp/cuda-keyring.deb
  rm -f /tmp/cuda-keyring.deb

  sudo apt update -qq
  sudo apt install -y -qq cuda-toolkit-12-6

  # Configure PATH
  local CUDA_PATH="/usr/local/cuda-12.6"
  if ! grep -q "CUDA_HOME" ~/.bashrc; then
    cat >> ~/.bashrc << 'EOF'

# CUDA Toolkit (added by llm-kernel-lab setup)
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
EOF
  fi

  export CUDA_HOME="$CUDA_PATH"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

  info "CUDA Toolkit installed: $($CUDA_HOME/bin/nvcc --version | grep release | awk '{print $6}')"
}

# ============================================================
# Step 4: vLLM 环境 (optional)
# ============================================================
setup_vllm_env() {
  section "Step 4: 创建 conda 环境 (llm-vllm-lab)"

  local ENV_NAME="llm-vllm-lab"

  if conda env list | awk '{print $1}' | grep -Fx "$ENV_NAME" &>/dev/null; then
    info "Conda env '$ENV_NAME' already exists"
  else
    info "Creating conda env '$ENV_NAME'..."
    conda env create -f "$PROJECT_DIR/environment-vllm.yml"
  fi

  info "Installing vLLM..."
  conda run -n "$ENV_NAME" python -m pip install --upgrade pip -q
  conda run -n "$ENV_NAME" python -m pip install vllm -q

  info "Verifying vLLM..."
  conda run -n "$ENV_NAME" python -c "import vllm; print(f'  vLLM {vllm.__version__}')" 2>/dev/null || \
    warn "vLLM import failed (may need specific CUDA version)"

  info "Conda env '$ENV_NAME' ready"
}

# ============================================================
# Step 5: 下载学习资料
# ============================================================
download_resources() {
  section "Step 5: 下载学习资料"

  if [[ -f "$PROJECT_DIR/resources/download_all.sh" ]]; then
    bash "$PROJECT_DIR/resources/download_all.sh"
  else
    warn "resources/download_all.sh not found, skipping"
  fi
}

# ============================================================
# Step 6: 验证
# ============================================================
final_verification() {
  section "最终验证"

  local ENV_NAME="llm-kernel-lab"
  local all_pass=true

  # Run existing verification
  if [[ -f "$PROJECT_DIR/scripts/04_verify_all.sh" ]]; then
    info "Running project verification..."
    bash "$PROJECT_DIR/scripts/04_verify_all.sh" || true
  fi

  # Quick smoke test
  info "Running smoke test..."
  conda run -n "$ENV_NAME" python -c "
import torch
import triton
import triton.language as tl

# Minimal Triton kernel test
@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

n = 1024
x = torch.randn(n, device='cuda')
y = torch.randn(n, device='cuda')
out = torch.empty(n, device='cuda')
_add_kernel[(n // 256,)](x, y, out, n, BLOCK=256)
assert torch.allclose(out, x + y), 'Triton kernel test failed!'
print('  ✓ Triton kernel smoke test passed')
" && info "Smoke test passed" || { error "Smoke test failed"; all_pass=false; }

  echo ""
  if $all_pass; then
    info "🎉 环境部署完成！"
  else
    warn "部分组件未通过验证，请检查上方输出"
  fi

  echo ""
  echo "使用方式:"
  echo "  conda activate llm-kernel-lab"
  echo "  cd $PROJECT_DIR"
  echo "  pytest tests/ -v"
  echo ""
  echo "查看环境状态:"
  echo "  bash setup_env.sh --check"
}

# ============================================================
# 执行
# ============================================================
echo ""
printf "${BOLD}llm-kernel-lab 一键环境部署${NC}\n"
printf "Mode: ${BLUE}$MODE${NC}\n"
echo ""

case "$MODE" in
  minimal)
    install_system_deps
    setup_kernel_env
    final_verification
    ;;
  full)
    install_system_deps
    setup_kernel_env
    setup_cuda_toolkit
    setup_vllm_env
    download_resources
    final_verification
    ;;
  default)
    install_system_deps
    setup_kernel_env
    final_verification
    ;;
esac
