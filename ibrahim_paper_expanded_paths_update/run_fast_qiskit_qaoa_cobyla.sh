#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="outputs/qiskit_fast_cobyla_run_${STAMP}"
mkdir -p "$RUN_DIR"
echo "Running fast Qiskit QAOA COBYLA check"
echo "Root: $ROOT" | tee "$RUN_DIR/run_summary.txt"
echo "Run dir: $RUN_DIR" | tee -a "$RUN_DIR/run_summary.txt"
python src/task3_qiskit_qaoa_cobyla_fast.py > "$RUN_DIR/task3_qiskit_qaoa_cobyla_stdout.txt" 2> "$RUN_DIR/task3_qiskit_qaoa_cobyla_stderr.txt"
EC=$?
echo "QAOA COBYLA exit code: $EC" | tee -a "$RUN_DIR/run_summary.txt"
if [ -d src/outputs/task3_qiskit ]; then
  cp -R src/outputs/task3_qiskit "$RUN_DIR/"
fi
zip -r "outputs/QISKIT_FAST_COBYLA_RESULTS_${STAMP}.zip" "$RUN_DIR" >/dev/null
if [ $EC -eq 0 ]; then
  echo "Created: outputs/QISKIT_FAST_COBYLA_RESULTS_${STAMP}.zip"
else
  echo "Run failed/interrupted. Logs are in: $RUN_DIR"
  echo "Zip still created: outputs/QISKIT_FAST_COBYLA_RESULTS_${STAMP}.zip"
fi
exit $EC
