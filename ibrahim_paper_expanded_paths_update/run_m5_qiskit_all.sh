#!/usr/bin/env bash
set +e

ROOT="$(pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="outputs/m5_qiskit_fixed_run_${STAMP}"
mkdir -p "$RUN_DIR"

echo "============================================================"
echo "Ibrahim QIML M5/local Qiskit run"
echo "Root: $ROOT"
echo "Run dir: $RUN_DIR"
echo "QAOA_REPS=${QAOA_REPS:-1}"
echo "QAOA_MAXITER=${QAOA_MAXITER:-5}"
echo "============================================================"

{
  echo "===== ENVIRONMENT ====="
  which python
  python -c "import sys; print('sys.executable:', sys.executable)"
  python -c "import platform; print('platform:', platform.platform())"
  python -c "import numpy; print('numpy:', numpy.__version__)"
  python -c "import matplotlib; print('matplotlib:', matplotlib.__version__)"
  python -c "import qiskit; print('qiskit:', qiskit.__version__)"
  python -c "import qiskit_optimization; print('qiskit_optimization:', qiskit_optimization.__version__)"
  python -c "import qiskit_algorithms; print('qiskit_algorithms:', qiskit_algorithms.__version__)"
  python -c "import dimod; print('dimod:', dimod.__version__)"
  python -c "import neal; print('neal:', neal.__version__)"
} > "$RUN_DIR/environment_check.txt" 2>&1

echo "[1/2] Running HUBO/QUBO-to-Pauli mapping..."
python src/qiskit_hubo_pauli_mapping.py \
  > "$RUN_DIR/qiskit_pauli_mapping_stdout.txt" \
  2> "$RUN_DIR/qiskit_pauli_mapping_stderr.txt"
MAP_EXIT=$?
echo "Mapping exit code: $MAP_EXIT"

if [ -f outputs/qiskit_pauli_mapping_summary.json ]; then
  cp outputs/qiskit_pauli_mapping_summary.json "$RUN_DIR/"
fi

echo "[2/2] Running fast Qiskit QAOA COBYLA..."
QAOA_REPS="${QAOA_REPS:-1}" QAOA_MAXITER="${QAOA_MAXITER:-5}" \
python src/task3_qiskit_qaoa_cobyla_fast.py \
  > "$RUN_DIR/qaoa_stdout.txt" \
  2> "$RUN_DIR/qaoa_stderr.txt"
QAOA_EXIT=$?
echo "QAOA exit code: $QAOA_EXIT"

# Copy likely generated result files if they exist
find outputs -maxdepth 3 -type f \( -name "*.json" -o -name "*.csv" -o -name "*.png" -o -name "*.md" -o -name "*.txt" \) \
  -not -path "$RUN_DIR/*" \
  -exec cp {} "$RUN_DIR/" \; 2>/dev/null

{
  echo "mapping_exit_code=$MAP_EXIT"
  echo "qaoa_exit_code=$QAOA_EXIT"
  echo "qaoa_reps=${QAOA_REPS:-1}"
  echo "qaoa_maxiter=${QAOA_MAXITER:-5}"
  echo "run_dir=$RUN_DIR"
} > "$RUN_DIR/run_summary.txt"

ZIP_FILE="outputs/M5_QISKIT_RESULTS_${STAMP}.zip"
zip -r "$ZIP_FILE" "$RUN_DIR" >/dev/null

echo "============================================================"
echo "Created zip:"
echo "$ZIP_FILE"
echo "============================================================"

exit 0
