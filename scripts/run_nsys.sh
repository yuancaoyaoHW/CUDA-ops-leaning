#!/usr/bin/env bash
# Profile with Nsight Systems
# Usage: bash scripts/run_nsys.sh <python_script> [args...]
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <python_script> [args...]"
    echo "Example: $0 benchmarks/bench_vector_add.py"
    exit 1
fi

SCRIPT="$1"
shift

OUTDIR="reports/nsys"
mkdir -p "$OUTDIR"

BASENAME=$(basename "$SCRIPT" .py)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="$OUTDIR/${BASENAME}_${TIMESTAMP}"
PYTHON="${PYTHON:-python}"

echo "Profiling: $SCRIPT"
echo "Output: ${OUTFILE}.nsys-rep"

nsys profile \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    -o "$OUTFILE" \
    "$PYTHON" "$SCRIPT" "$@"

echo ""
echo "View report:"
echo "  nsys-ui ${OUTFILE}.nsys-rep  # on Windows host"
