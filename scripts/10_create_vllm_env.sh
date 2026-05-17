#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="llm-vllm-lab"
ENV_FILE="environment-vllm.yml"

if ! command -v conda >/dev/null 2>&1; then
  printf 'ERROR: conda was not found on PATH.\n' >&2
  printf 'Install conda first, then rerun this script.\n' >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -Fx "$ENV_NAME" >/dev/null 2>&1; then
  printf 'Conda environment already exists: %s\n' "$ENV_NAME"
  printf 'No changes made.\n'
  exit 0
fi

printf 'Creating conda environment from %s: %s\n' "$ENV_FILE" "$ENV_NAME"
conda env create -f "$ENV_FILE"
