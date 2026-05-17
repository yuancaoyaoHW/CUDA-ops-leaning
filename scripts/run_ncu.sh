#!/usr/bin/env bash
# Profile with Nsight Compute (single kernel analysis)
# Usage: bash scripts/run_ncu.sh <python_script> [args...]
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <python_script> [args...]"
    echo "Example: $0 benchmarks/bench_vector_add.py"
    exit 1
fi

SCRIPT="$1"
shift

OUTDIR="reports/ncu"
mkdir -p "$OUTDIR"

BASENAME=$(basename "$SCRIPT" .py)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="$OUTDIR/${BASENAME}_${TIMESTAMP}"
PYTHON="${PYTHON:-python}"
NCU="${NCU:-ncu}"

if ! command -v "$NCU" >/dev/null 2>&1; then
    if [[ -x /opt/nvidia/nsight-compute/2024.3.2/ncu ]]; then
        NCU="/opt/nvidia/nsight-compute/2024.3.2/ncu"
    else
        echo "ERROR: ncu not found on PATH."
        echo "Set NCU=/path/to/ncu or install Nsight Compute."
        exit 1
    fi
fi

echo "Profiling: $SCRIPT"
echo "Output: ${OUTFILE}.ncu-rep"

"$NCU" \
    --set full \
    --target-processes all \
    -o "$OUTFILE" \
    "$PYTHON" "$SCRIPT" "$@"

echo ""
echo "View report:"
echo "  ncu-ui ${OUTFILE}.ncu-rep  # on Windows host"
