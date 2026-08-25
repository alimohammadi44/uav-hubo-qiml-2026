#!/bin/bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv-m2max"
RESULTS_DIR="$PROJECT_DIR/results"
LOG_DIR="$RESULTS_DIR/m2max_logs"
MPL_DIR="$(mktemp -d "${TMPDIR:-/tmp}/qiml-mpl.XXXXXX")"

cleanup() {
    rm -rf "$MPL_DIR"
}
trap cleanup EXIT

cd "$PROJECT_DIR"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "Creating isolated Python environment: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi

PYTHON="$VENV_DIR/bin/python"
export MPLCONFIGDIR="$MPL_DIR"
export PYTHONHASHSEED=0

echo "Installing benchmark requirements..."
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -r requirements.txt

echo
echo "Stage 1/3: A*, 40-trial RRT-best, and QUAV-style QAOA"
"$PYTHON" src/quav_style_baseline.py 2>&1 | tee "$LOG_DIR/quav_style_baseline.log"

echo
echo "Stage 2/3: exact CP-SAT baseline (8 workers)"
"$PYTHON" src/cp_milp_exact_baselines.py \
    --mode cpsat --workers 8 --time-limit 300 2>&1 | \
    tee "$LOG_DIR/cpsat_exact.log"

echo
echo "Stage 3/3: 50-read neal and 50-run trajectory-SA"
echo "This is the long stage and may take approximately 30--65 minutes."
"$PYTHON" src/task3_polished_v3_1.py 2>&1 | tee "$LOG_DIR/neal_and_sa.log"

echo
echo "Collecting M2 Max timing results..."
"$PYTHON" src/collect_runtime_benchmark.py \
    --label "Apple M2 Max" \
    --output-dir "$RESULTS_DIR"

echo
echo "Benchmark complete. Please send these files to the paper maintainer:"
echo "  $RESULTS_DIR/m2max_runtime_benchmark.json"
echo "  $RESULTS_DIR/m2max_runtime_benchmark.csv"
echo "  $RESULTS_DIR/unified_comparison_m2max.csv"

