#!/usr/bin/env bash
# ============================================================
# 一键下载所有学习资料
# 用法: bash resources/download_all.sh [--papers] [--repos] [--tutorials]
#       不带参数则下载全部
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPERS_DIR="$SCRIPT_DIR/papers"
REPOS_DIR="$SCRIPT_DIR/repos"
TUTORIALS_DIR="$SCRIPT_DIR/tutorials"
MANUALS_DIR="$SCRIPT_DIR/manuals"

# Parse args
DO_PAPERS=false
DO_REPOS=false
DO_TUTORIALS=false
if [[ $# -eq 0 ]]; then
  DO_PAPERS=true; DO_REPOS=true; DO_TUTORIALS=true
else
  for arg in "$@"; do
    case "$arg" in
      --papers) DO_PAPERS=true ;;
      --repos) DO_REPOS=true ;;
      --tutorials) DO_TUTORIALS=true ;;
      *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
  done
fi

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { printf "${GREEN}[INFO]${NC} %s\n" "$1"; }
warn()  { printf "${YELLOW}[WARN]${NC} %s\n" "$1"; }
error() { printf "${RED}[ERROR]${NC} %s\n" "$1"; }

download_paper() {
  local url="$1"
  local filename="$2"
  local target="$PAPERS_DIR/$filename"
  if [[ -f "$target" ]]; then
    info "Already exists: $filename"
    return 0
  fi
  info "Downloading: $filename"
  if wget -q --timeout=30 --tries=2 -O "$target" "$url" 2>/dev/null; then
    info "  ✓ $filename"
  else
    warn "  ✗ Failed: $filename (try manually: $url)"
    rm -f "$target"
  fi
}

clone_repo() {
  local url="$1"
  local dirname="$2"
  local depth="${3:-1}"
  local target="$REPOS_DIR/$dirname"
  if [[ -d "$target" ]]; then
    info "Already exists: $dirname"
    return 0
  fi
  info "Cloning: $dirname"
  if git clone --depth "$depth" "$url" "$target" 2>/dev/null; then
    info "  ✓ $dirname"
  else
    warn "  ✗ Failed to clone: $dirname"
  fi
}

# ============================================================
# PAPERS
# ============================================================
if $DO_PAPERS; then
  info "=== Downloading Papers ==="
  mkdir -p "$PAPERS_DIR"

  # Week 1-2: 算子基础
  download_paper \
    "https://arxiv.org/pdf/1805.02867" \
    "01_online_softmax_milakov_2018.pdf"

  download_paper \
    "https://arxiv.org/pdf/1910.02054" \
    "02_zero_rajbhandari_2020.pdf"

  # Week 3-4: GEMM / Attention
  download_paper \
    "https://arxiv.org/pdf/2205.14135" \
    "03_flashattention_dao_2022.pdf"

  download_paper \
    "https://arxiv.org/pdf/2307.08691" \
    "04_flashattention2_dao_2023.pdf"

  # Week 5: KV Cache / Scheduler / Speculative Decoding
  download_paper \
    "https://arxiv.org/pdf/2309.06180" \
    "06_pagedattention_vllm_kwon_2023.pdf"

  download_paper \
    "https://arxiv.org/pdf/2206.01821" \
    "07_orca_yu_2022.pdf"

  download_paper \
    "https://arxiv.org/pdf/2211.17192" \
    "08_speculative_decoding_leviathan_2023.pdf"

  download_paper \
    "https://arxiv.org/pdf/2308.04623" \
    "09_staged_speculative_decoding_spector_2023.pdf"

  # Week 6: 推理框架 / 分布式
  download_paper \
    "https://arxiv.org/pdf/2312.07104" \
    "10_sglang_zheng_2023.pdf"

  download_paper \
    "https://arxiv.org/pdf/2407.00079" \
    "11_mooncake_kvcache_disaggregated_2024.pdf"

  download_paper \
    "https://arxiv.org/pdf/2310.01889" \
    "12_ring_attention_liu_2023.pdf"

  # Week 7: 量化
  download_paper \
    "https://arxiv.org/pdf/2210.17323" \
    "14_gptq_frantar_2023.pdf"

  download_paper \
    "https://arxiv.org/pdf/2306.00978" \
    "15_awq_lin_2023.pdf"

  download_paper \
    "https://arxiv.org/pdf/2209.05433" \
    "16_fp8_formats_micikevicius_2022.pdf"

  download_paper \
    "https://arxiv.org/pdf/2412.19437" \
    "17_deepseek_v3_2024.pdf"

  # Week 8: 模型基础
  download_paper \
    "https://arxiv.org/pdf/1706.03762" \
    "18_attention_is_all_you_need_2017.pdf"

  download_paper \
    "https://arxiv.org/pdf/2104.09864" \
    "19_roformer_rope_su_2021.pdf"

  download_paper \
    "https://arxiv.org/pdf/2305.13245" \
    "20_gqa_ainslie_2023.pdf"

  download_paper \
    "https://arxiv.org/pdf/2106.09685" \
    "21_lora_hu_2021.pdf"

  download_paper \
    "https://arxiv.org/pdf/2305.14314" \
    "22_qlora_dettmers_2023.pdf"

  # Week 5/8: GQA/MQA/MLA 补充
  download_paper \
    "https://arxiv.org/pdf/1911.02150" \
    "23_mqa_shazeer_2019.pdf"

  download_paper \
    "https://arxiv.org/pdf/2405.04434" \
    "24_deepseek_v2_mla_2024.pdf"

  # Week 8: RLHF/DPO/PPO 模型基础兜底
  download_paper \
    "https://arxiv.org/pdf/2203.02155" \
    "25_instructgpt_rlhf_ouyang_2022.pdf"

  download_paper \
    "https://arxiv.org/pdf/2305.18290" \
    "26_dpo_rafailov_2023.pdf"

  download_paper \
    "https://arxiv.org/pdf/1707.06347" \
    "27_ppo_schulman_2017.pdf"

  # Week 8: RAG
  download_paper \
    "https://arxiv.org/pdf/2005.11401" \
    "28_rag_lewis_2020.pdf"

  # Week 8: BERT
  download_paper \
    "https://arxiv.org/pdf/1810.04805" \
    "29_bert_devlin_2018.pdf"

  # Week 6: TensorRT-LLM (NVIDIA blog/whitepaper)
  download_paper \
    "https://arxiv.org/pdf/2401.14112" \
    "13_tensorrt_llm_2024.pdf"

  info "=== Papers download complete ==="
  echo ""
fi

# ============================================================
# REPOS
# ============================================================
if $DO_REPOS; then
  info "=== Cloning Repositories ==="
  mkdir -p "$REPOS_DIR"

  # Triton (tutorials are inside)
  clone_repo "https://github.com/triton-lang/triton.git" "triton" 1

  # CUTLASS (already in project root, but keep a reference copy for docs)
  # Skip if cutlass already exists at project root
  if [[ -d "$SCRIPT_DIR/../cutlass" ]]; then
    info "CUTLASS already at project root, skipping"
  else
    clone_repo "https://github.com/NVIDIA/cutlass.git" "cutlass" 1
  fi

  # CUDA optimization references
  clone_repo "https://github.com/leimao/CUDA-Reduction.git" "CUDA-Reduction" 1
  clone_repo "https://github.com/leimao/CUDA-GEMM-Optimization.git" "CUDA-GEMM-Optimization" 1

  # Inference frameworks
  clone_repo "https://github.com/vllm-project/vllm.git" "vllm" 1
  clone_repo "https://github.com/sgl-project/sglang.git" "sglang" 1
  clone_repo "https://github.com/NVIDIA/TensorRT-LLM.git" "TensorRT-LLM" 1

  # FlashAttention
  clone_repo "https://github.com/Dao-AILab/flash-attention.git" "flash-attention" 1

  # Quantization
  clone_repo "https://github.com/IST-DASLab/gptq.git" "gptq" 1
  clone_repo "https://github.com/mit-han-lab/llm-awq.git" "llm-awq" 1
  clone_repo "https://github.com/bitsandbytes-foundation/bitsandbytes.git" "bitsandbytes" 1

  info "=== Repos clone complete ==="
  echo ""
fi

# ============================================================
# TUTORIALS & MANUALS
# ============================================================
if $DO_TUTORIALS; then
  info "=== Downloading Tutorials & Manuals ==="
  mkdir -p "$TUTORIALS_DIR" "$MANUALS_DIR"

  # Triton tutorials (copy from cloned repo if available)
  if [[ -d "$REPOS_DIR/triton/python/tutorials" ]]; then
    info "Linking Triton tutorials from cloned repo"
    ln -sfn "$REPOS_DIR/triton/python/tutorials" "$TUTORIALS_DIR/triton-tutorials"
  else
    info "Triton tutorials: clone triton repo first (--repos)"
  fi

  # NVIDIA CUDA Programming Guide (PDF)
  download_paper \
    "https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf" \
    "../manuals/CUDA_C_Programming_Guide.pdf"

  # Nsight Compute documentation
  download_paper \
    "https://docs.nvidia.com/nsight-compute/pdf/NsightCompute.pdf" \
    "../manuals/NsightCompute.pdf"

  # Nsight Systems documentation
  download_paper \
    "https://docs.nvidia.com/nsight-systems/pdf/NsightSystems.pdf" \
    "../manuals/NsightSystems.pdf"

  # CUTLASS documentation (from cloned repo or project root)
  if [[ -d "$SCRIPT_DIR/../cutlass/media/docs" ]]; then
    ln -sfn "$SCRIPT_DIR/../cutlass/media/docs" "$MANUALS_DIR/cutlass-docs"
    info "Linked CUTLASS docs from project root"
  elif [[ -d "$REPOS_DIR/cutlass/media/docs" ]]; then
    ln -sfn "$REPOS_DIR/cutlass/media/docs" "$MANUALS_DIR/cutlass-docs"
    info "Linked CUTLASS docs from repos"
  fi

  # PyTorch CUDA Extension tutorial (save as markdown)
  info "Saving PyTorch CUDA Extension tutorial reference"
  cat > "$TUTORIALS_DIR/pytorch_cuda_extension.md" << 'EOF'
# PyTorch Custom C++/CUDA Extensions

Official tutorial: https://pytorch.org/tutorials/advanced/cpp_extension.html

## Key Steps

1. Write CUDA kernel (.cu file)
2. Write C++ wrapper with pybind11
3. Use `torch.utils.cpp_extension.load()` or `setup.py` with `CUDAExtension`

## JIT Compilation (recommended for dev)

```python
from torch.utils.cpp_extension import load

my_op = load(
    name='my_op',
    sources=['my_op.cpp', 'my_op_kernel.cu'],
    extra_cuda_cflags=['-O2']
)
```

## setup.py (for distribution)

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='my_op',
    ext_modules=[
        CUDAExtension('my_op', [
            'my_op.cpp',
            'my_op_kernel.cu',
        ])
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

## Reference
- https://pytorch.org/tutorials/advanced/cpp_extension.html
- https://pytorch.org/docs/stable/cpp_extension.html
EOF

  # vLLM architecture guide
  info "Saving vLLM architecture reference"
  cat > "$TUTORIALS_DIR/vllm_architecture.md" << 'EOF'
# vLLM Architecture Guide

Source: https://docs.vllm.ai/en/latest/design/arch_overview.html

## Key Source Files to Read

- `vllm/engine/llm_engine.py` - Main engine loop
- `vllm/core/scheduler.py` - Request scheduling (waiting/running/swapped queues)
- `vllm/core/block_manager.py` - Physical block allocation for KV cache
- `vllm/worker/model_runner.py` - Model execution
- `vllm/attention/backends/flash_attn.py` - FlashAttention backend
- `csrc/attention/attention_kernels.cu` - PagedAttention CUDA kernel

## Request Lifecycle

1. Client sends request → LLMEngine.add_request()
2. Request enters waiting queue
3. Scheduler.schedule() → allocate blocks → move to running
4. ModelRunner.execute_model() → forward pass
5. Process output → check stop condition
6. If done → return result; else → next decode step

## Key Concepts

- Block Table: logical block → physical block mapping (like OS page table)
- Preemption: swap-out (to CPU) or recompute (discard KV, re-prefill later)
- Copy-on-Write: shared prefix blocks for beam search
EOF

  # SGLang architecture guide
  info "Saving SGLang architecture reference"
  cat > "$TUTORIALS_DIR/sglang_architecture.md" << 'EOF'
# SGLang Architecture Guide

Source: https://github.com/sgl-project/sglang

## Key Concepts

- RadixAttention: Radix tree for KV cache prefix sharing
- Constrained Decoding: Grammar-guided generation
- Frontend Language: Python DSL for structured LLM programs

## Key Source Files

- `python/sglang/srt/server.py` - Server entry
- `python/sglang/srt/managers/scheduler.py` - Scheduler
- `python/sglang/srt/mem_cache/radix_cache.py` - RadixAttention cache

## vs vLLM

| Feature | vLLM | SGLang |
|---------|------|--------|
| KV Cache | PagedAttention (block table) | RadixAttention (radix tree) |
| Prefix Sharing | Copy-on-Write | Automatic via radix tree |
| Structured Output | Limited | Native (grammar, regex) |
| Multi-turn | Re-prefill or prefix cache | Automatic prefix reuse |
EOF

  # FlashDecoding blog reference
  info "Saving FlashDecoding reference"
  cat > "$TUTORIALS_DIR/flash_decoding.md" << 'EOF'
# Flash-Decoding for Long-Context LLM Inference

Source: https://crfm.stanford.edu/2023/10/12/flashdecoding.html

## Problem

Decode attention: Q=[1,d], K=[seq_len,d] → GEMV, memory-bound
Standard: 1 thread block per head → only num_heads blocks → low GPU utilization

## Solution: Split KV across thread blocks

1. Split KV cache along seq_len dimension into chunks
2. Each thread block computes partial attention for its chunk
3. Final reduce kernel combines partial results with online softmax rescaling

## Parallelism

- Standard: num_heads thread blocks
- FlashDecoding: num_heads × ceil(seq_len / chunk_size) thread blocks
- Even batch=1 can saturate GPU for long sequences

## When to use

- Long sequences (seq_len > 4096)
- Small batch sizes
- Decode phase (not prefill)
EOF

  info "=== Tutorials & Manuals download complete ==="
  echo ""
fi

# ============================================================
# Summary
# ============================================================
info "=== Download Summary ==="
if $DO_PAPERS; then
  PAPER_COUNT=$(find "$PAPERS_DIR" -name "*.pdf" 2>/dev/null | wc -l)
  info "Papers: $PAPER_COUNT PDFs in $PAPERS_DIR"
fi
if $DO_REPOS; then
  REPO_COUNT=$(find "$REPOS_DIR" -maxdepth 1 -type d 2>/dev/null | wc -l)
  REPO_COUNT=$((REPO_COUNT - 1))
  info "Repos: $REPO_COUNT repositories in $REPOS_DIR"
fi
if $DO_TUTORIALS; then
  info "Tutorials: $TUTORIALS_DIR"
  info "Manuals: $MANUALS_DIR"
fi
echo ""
info "Done! 资料已下载到 resources/ 目录"
