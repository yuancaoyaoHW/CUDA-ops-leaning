#!/usr/bin/env bash
set -euo pipefail

section() {
  printf '\n== %s ==\n' "$1"
}

check_cmd() {
  local name="$1"
  if command -v "$name" >/dev/null 2>&1; then
    printf '%s: %s\n' "$name" "$(command -v "$name")"
    "$name" --version 2>/dev/null | head -n 3 || true
  else
    printf '%s: NOT FOUND\n' "$name"
  fi
}

section "Kernel"
uname -a

section "Distribution"
if command -v lsb_release >/dev/null 2>&1; then
  lsb_release -a
else
  printf 'lsb_release: NOT FOUND\n'
  if [ -r /etc/os-release ]; then
    cat /etc/os-release
  fi
fi

section "Python"
check_cmd python
check_cmd python3

section "Build Tools"
check_cmd gcc
check_cmd g++
check_cmd cmake
check_cmd ninja

section "Conda"
check_cmd conda

section "NVIDIA GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  if ! nvidia-smi; then
    printf 'nvidia-smi exists but failed. WSL2 GPU access may be blocked by the operating system or driver setup.\n'
  fi
else
  printf 'nvidia-smi: NOT FOUND\n'
  printf 'WSL2 GPU support requires a working Windows NVIDIA driver with WSL support.\n'
fi
