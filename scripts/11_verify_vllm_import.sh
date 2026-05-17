#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="llm-vllm-lab"

conda run -n "$ENV_NAME" python -c '
import importlib.metadata as metadata

import vllm

try:
    version = metadata.version("vllm")
except metadata.PackageNotFoundError:
    version = getattr(vllm, "__version__", "unknown")

print(f"vllm: {version}")
'
