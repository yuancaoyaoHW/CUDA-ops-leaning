#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-llm-kernel-lab}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

if ! command -v conda >/dev/null 2>&1; then
  printf 'ERROR: conda was not found on PATH.\n' >&2
  printf 'Install Miniconda/Anaconda first, then rerun this script.\n' >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -Fx "$ENV_NAME" >/dev/null 2>&1; then
  printf 'Conda environment already exists: %s\n' "$ENV_NAME"
  printf 'No changes made.\n'
else
  printf 'Creating conda environment: %s with Python %s\n' "$ENV_NAME" "$PYTHON_VERSION"
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi

printf '\nActivate it with:\n'
printf '  conda activate %s\n' "$ENV_NAME"
