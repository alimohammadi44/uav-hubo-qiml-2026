#!/usr/bin/env bash
set -euo pipefail

# Run from package root or code_pkg or src.
HERE="$(pwd)"
if [[ -d "src" && -f "src/task2_grid_hubo.py" ]]; then
  CODE_PKG="$HERE"
elif [[ -f "task2_grid_hubo.py" ]]; then
  CODE_PKG="$(dirname "$HERE")"
else
  echo "ERROR: Run this from code_pkg or code_pkg/src, or copy quav_style_baseline.py into src first."
  exit 1
fi

cd "$CODE_PKG/src"
python quav_style_baseline.py --max-candidates 12 --qaoa-layers 1 --qaoa-steps 60 2>&1 | tee ../results/quav_style_baseline_log.txt

echo ""
echo "QUAV-style baseline complete. Outputs:"
echo "  src/outputs/quav_style_baseline/quav_style_results.json"
echo "  src/outputs/quav_style_baseline/quav_style_report.md"
echo "  src/outputs/quav_style_baseline/quav_style_paths.png"
echo "  src/outputs/quav_style_baseline/quav_style_metrics.png"
echo "  src/outputs/quav_style_baseline/quav_style_loss.png"
echo "  results/quav_style_baseline_log.txt"
