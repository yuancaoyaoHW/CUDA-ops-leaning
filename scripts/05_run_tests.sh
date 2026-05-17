#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-llm-kernel-lab}"

conda run -n "$ENV_NAME" python -m pytest tests
