#!/usr/bin/env bash
set -u

# Run all Qiskit/Hamiltonian checks for Ibrahim QIML paper and collect results.
# Usage from inside the unzipped package root:
#   bash run_all_qiskit_for_ibrahim_qiml.sh
# or pass the package root:
#   bash run_all_qiskit_for_ibrahim_qiml.sh "/path/to/ibrahim_paper_expanded_paths_update"

ROOT="${1:-$(pwd)}"
cd "$ROOT" || { echo "Cannot cd to $ROOT"; exit 1; }

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="outputs/qiskit_full_run_${STAMP}"
mkdir -p "$RUN_DIR"

echo "============================================================" | tee "$RUN_DIR/run_summary.txt"
echo "Ibrahim QIML Qiskit/Hamiltonian full run" | tee -a "$RUN_DIR/run_summary.txt"
echo "Root: $ROOT" | tee -a "$RUN_DIR/run_summary.txt"
echo "Run dir: $RUN_DIR" | tee -a "$RUN_DIR/run_summary.txt"
echo "Date: $(date)" | tee -a "$RUN_DIR/run_summary.txt"
echo "============================================================" | tee -a "$RUN_DIR/run_summary.txt"

# Save environment details.
{
  echo "Python executable: $(which python)"
  python --version
  echo ""
  echo "Installed package versions, if available:"
  python - <<'PY'
import importlib.metadata as md
pkgs = ["qiskit", "qiskit-aer", "qiskit-optimization", "qiskit-algorithms", "dimod", "dwave-neal", "matplotlib", "numpy"]
for p in pkgs:
    try:
        print(f"{p}: {md.version(p)}")
    except Exception:
        print(f"{p}: NOT INSTALLED")
PY
} > "$RUN_DIR/environment.txt" 2>&1

# 1. HUBO/QUBO -> Pauli-Z Hamiltonian mapping.
echo "[1/2] Running qiskit_hubo_pauli_mapping.py ..." | tee -a "$RUN_DIR/run_summary.txt"
python src/qiskit_hubo_pauli_mapping.py > "$RUN_DIR/qiskit_pauli_mapping_stdout.txt" 2> "$RUN_DIR/qiskit_pauli_mapping_stderr.txt"
MAP_STATUS=$?
echo "Mapping exit code: $MAP_STATUS" | tee -a "$RUN_DIR/run_summary.txt"
if [ -f outputs/qiskit_pauli_mapping_summary.json ]; then
  cp outputs/qiskit_pauli_mapping_summary.json "$RUN_DIR/"
fi

# 2. Tiny three-way QAOA comparison: SA vs neal vs Qiskit QAOA.
echo "[2/2] Running task3_qiskit_qaoa.py ..." | tee -a "$RUN_DIR/run_summary.txt"
python src/task3_qiskit_qaoa.py > "$RUN_DIR/task3_qiskit_qaoa_stdout.txt" 2> "$RUN_DIR/task3_qiskit_qaoa_stderr.txt"
QAOA_STATUS=$?
echo "QAOA exit code: $QAOA_STATUS" | tee -a "$RUN_DIR/run_summary.txt"

# The QAOA script currently writes to src/outputs/task3_qiskit.
if [ -d src/outputs/task3_qiskit ]; then
  mkdir -p "$RUN_DIR/task3_qiskit"
  cp -R src/outputs/task3_qiskit/* "$RUN_DIR/task3_qiskit/" 2>/dev/null || true
fi

# Also collect any root-level task output if future versions write there.
if [ -d outputs/task3_qiskit ]; then
  mkdir -p "$RUN_DIR/task3_qiskit_root_outputs"
  cp -R outputs/task3_qiskit/* "$RUN_DIR/task3_qiskit_root_outputs/" 2>/dev/null || true
fi

# Build compact combined paper table if both JSONs exist.
python - <<'PY' "$RUN_DIR" > "$RUN_DIR/combined_summary_stdout.txt" 2> "$RUN_DIR/combined_summary_stderr.txt"
import csv, json, pathlib, sys
run_dir = pathlib.Path(sys.argv[1])
rows = []
map_json = run_dir / "qiskit_pauli_mapping_summary.json"
if map_json.exists():
    d = json.loads(map_json.read_text())
    tiny = d.get("tiny_instance", {})
    rows.append({"result":"Hamiltonian mapping grid", "value": str(tiny.get("grid"))})
    rows.append({"result":"Hamiltonian mapping L", "value": str(tiny.get("L"))})
    rows.append({"result":"Hamiltonian mapping qubits", "value": str(tiny.get("num_qubits"))})
    rows.append({"result":"HUBO/QUBO terms", "value": str(d.get("hubo_terms"))})
    rows.append({"result":"Pauli terms incl. identity", "value": str(d.get("pauli_terms_including_constant"))})
    rows.append({"result":"Qiskit SparsePauliOp available", "value": str(d.get("qiskit_available_in_this_run"))})
qaoa_json = run_dir / "task3_qiskit" / "benchmark_results.json"
if qaoa_json.exists():
    q = json.loads(qaoa_json.read_text())
    for key, label in [("classical_sa","Classical SA"),("dwave_neal","D-Wave neal"),("qiskit_qaoa","Qiskit QAOA")]:
        r = q.get(key, {})
        f = r.get("feasibility", {})
        rows.append({"result":f"{label} best energy", "value": str(r.get("best_energy"))})
        rows.append({"result":f"{label} runtime_s", "value": str(r.get("runtime_s"))})
        rows.append({"result":f"{label} feasible", "value": str(f.get("all_hard_satisfied"))})
        rows.append({"result":f"{label} reaches target", "value": str(f.get("reaches_target"))})
if rows:
    with open(run_dir / "paper_ready_qiskit_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["result","value"])
        w.writeheader(); w.writerows(rows)
    print("Wrote", run_dir / "paper_ready_qiskit_summary.csv")
else:
    print("No JSON result files found for combined summary.")
PY

# Zip everything from this run.
ZIP_NAME="QISKIT_RESULTS_FOR_QIML_${STAMP}.zip"
(cd outputs && zip -qr "../$ZIP_NAME" "$(basename "$RUN_DIR")")
mv "$ZIP_NAME" "$RUN_DIR/../$ZIP_NAME" 2>/dev/null || true

ZIP_PATH="outputs/$ZIP_NAME"
echo "============================================================" | tee -a "$RUN_DIR/run_summary.txt"
echo "DONE" | tee -a "$RUN_DIR/run_summary.txt"
echo "Mapping status: $MAP_STATUS" | tee -a "$RUN_DIR/run_summary.txt"
echo "QAOA status: $QAOA_STATUS" | tee -a "$RUN_DIR/run_summary.txt"
echo "Results folder: $RUN_DIR" | tee -a "$RUN_DIR/run_summary.txt"
echo "Zip file: $ZIP_PATH" | tee -a "$RUN_DIR/run_summary.txt"
echo "============================================================" | tee -a "$RUN_DIR/run_summary.txt"

if [ "$QAOA_STATUS" -ne 0 ]; then
  echo "QAOA did not finish. Check: $RUN_DIR/task3_qiskit_qaoa_stderr.txt"
  exit $QAOA_STATUS
fi
