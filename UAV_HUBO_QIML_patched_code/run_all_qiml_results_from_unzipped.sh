#!/usr/bin/env bash
set -euo pipefail

# ================================================================
# Run all upgraded UAV HUBO / QIML experiments from an ALREADY
# unzipped patched package.
# ================================================================
# Usage:
#   1) Open Terminal.
#   2) cd into either:
#        - the unzipped package root that contains code_pkg/
#        OR
#        - the code_pkg/ directory itself.
#   3) Run:
#        conda activate uav_hubo_qiml
#        bash run_all_qiml_results_from_unzipped.sh
#
# Optional:
#   ENV_NAME=my_env bash run_all_qiml_results_from_unzipped.sh
#   SKIP_INSTALL=1 bash run_all_qiml_results_from_unzipped.sh
# ================================================================

ENV_NAME="${ENV_NAME:-uav_hubo_qiml}"
START_DIR="$(pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"

printf "==============================================================\n"
printf "UAV HUBO / QIML final experiment runner\n"
printf "Running from already-unzipped package\n"
printf "==============================================================\n"
printf "Start directory : %s\n" "$START_DIR"
printf "Conda env       : %s\n" "$ENV_NAME"
printf "Timestamp       : %s\n" "$STAMP"
printf "==============================================================\n"

# ---------------------------------------------------------------
# Detect code_pkg directory.
# ---------------------------------------------------------------
if [ -f "src/task2_grid_hubo.py" ] && [ -f "src/task3_polished_v3_1.py" ]; then
  CODE_DIR="$START_DIR"
elif [ -d "code_pkg" ] && [ -f "code_pkg/src/task2_grid_hubo.py" ]; then
  CODE_DIR="$START_DIR/code_pkg"
else
  FOUND_TASK2="$(find "$START_DIR" -maxdepth 4 -type f -name task2_grid_hubo.py | head -n 1 || true)"
  if [ -n "$FOUND_TASK2" ]; then
    SRC_DIR="$(dirname "$FOUND_TASK2")"
    CODE_DIR="$(dirname "$SRC_DIR")"
  else
    echo "ERROR: Could not find src/task2_grid_hubo.py."
    echo "Please run this script from the unzipped package root or from code_pkg/."
    echo "Current folder contents:"
    ls -la
    exit 1
  fi
fi

FINAL_ZIP="$CODE_DIR/UAV_HUBO_QIML_FINAL_RESULTS_${STAMP}.zip"

printf "Detected code directory: %s\n" "$CODE_DIR"
printf "Final results zip      : %s\n" "$FINAL_ZIP"
printf "==============================================================\n"

# ---------------------------------------------------------------
# Make conda available to non-interactive shell scripts.
# ---------------------------------------------------------------
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  echo "ERROR: conda command not found. Open a terminal where conda works, then rerun."
  exit 1
fi

printf "\n[1/8] Activating conda environment: %s\n" "$ENV_NAME"
conda activate "$ENV_NAME"

cd "$CODE_DIR"
export MPLBACKEND=Agg

# ---------------------------------------------------------------
# Install/update requirements unless skipped.
# ---------------------------------------------------------------
printf "\n[2/8] Checking/installing required Python packages...\n"
python -m pip install --upgrade pip setuptools wheel

if [ "${SKIP_INSTALL:-0}" = "1" ]; then
  echo "SKIP_INSTALL=1, so package installation is skipped."
elif [ -f requirements.txt ]; then
  python -m pip install -r requirements.txt
else
  echo "WARNING: requirements.txt not found. Installing expected packages manually."
  python -m pip install numpy scipy pandas matplotlib pillow networkx tqdm dimod==0.12.22 dwave-neal==0.6.0 ortools qiskit==1.4.6 qiskit-optimization qiskit-algorithms qiskit-aer
fi

mkdir -p results
mkdir -p src/outputs

# ---------------------------------------------------------------
# Save reproducibility details.
# ---------------------------------------------------------------
printf "\n[3/8] Saving environment details...\n"
python -m pip freeze > results/final_environment_qiml_2026.txt
python - <<'PY' | tee results/final_import_check.txt
import sys
print("Python:", sys.version)
modules = [
    "numpy",
    "matplotlib",
    "dimod",
    "neal",
    "ortools",
    "qiskit",
    "qiskit_optimization",
    "qiskit_algorithms",
    "qiskit_aer",
]
for name in modules:
    mod = __import__(name)
    print(f"{name}: {getattr(mod, '__version__', 'import OK')}")
print("All imports OK.")
PY

cd "$CODE_DIR/src"

# ---------------------------------------------------------------
# Run all experiments needed for final paper revision.
# ---------------------------------------------------------------
printf "\n[4/8] Running Task 2 HUBO builder / formulation counts...\n"
python task2_grid_hubo.py 2>&1 | tee ../results/final_task2_hubo_counts.txt

printf "\n[5/8] Running A* baseline...\n"
python astar_baseline.py 2>&1 | tee ../results/final_astar_baseline_log.txt

printf "\n[6/8] Running main SA vs neal benchmark. This may take a long time...\n"
python task3_polished_v3_1.py 2>&1 | tee ../results/final_main_benchmark_log.txt

printf "\n[7/8] Running QAOA tiny test...\n"
python task3_qiskit_qaoa.py 2>&1 | tee ../results/final_qaoa_tiny_log.txt

# ---------------------------------------------------------------
# Collect outputs and zip final package.
# ---------------------------------------------------------------
printf "\n[8/8] Collecting and zipping final results...\n"
cd "$CODE_DIR"

# Copy key output files into results for easy review.
if [ -d src/outputs ]; then
  find src/outputs -maxdepth 5 -type f \( -name "*.json" -o -name "*.csv" -o -name "*.txt" -o -name "*.md" \) -print -exec cp {} results/ \; 2>/dev/null || true
fi

{
  echo "UAV HUBO QIML final run"
  echo "Timestamp: $STAMP"
  echo "Start directory: $START_DIR"
  echo "Code directory: $CODE_DIR"
  echo "Conda env: $ENV_NAME"
  echo ""
  echo "Important log files:"
  echo "- results/final_task2_hubo_counts.txt"
  echo "- results/final_astar_baseline_log.txt"
  echo "- results/final_main_benchmark_log.txt"
  echo "- results/final_qaoa_tiny_log.txt"
  echo "- results/final_environment_qiml_2026.txt"
  echo "- results/final_import_check.txt"
  echo ""
  echo "Output folders:"
  if [ -d src/outputs ]; then
    find src/outputs -maxdepth 3 -type d | sort
  fi
} > results/RUN_MANIFEST.txt

rm -f "$FINAL_ZIP"

ZIP_ITEMS=(src results)
[ -f README.md ] && ZIP_ITEMS+=(README.md)
[ -f requirements.txt ] && ZIP_ITEMS+=(requirements.txt)
[ -f environment_qiml_2026.txt ] && ZIP_ITEMS+=(environment_qiml_2026.txt)

zip -qr "$FINAL_ZIP" "${ZIP_ITEMS[@]}"

printf "\n==============================================================\n"
printf "DONE\n"
printf "==============================================================\n"
printf "Final results zip: %s\n" "$FINAL_ZIP"
printf "Upload this zip file to ChatGPT so the paper can be updated with exact final numbers.\n"
printf "==============================================================\n"
