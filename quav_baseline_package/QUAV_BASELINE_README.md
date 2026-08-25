# QUAV-style baseline for Ibrahim UAV-HUBO project

This package adds a **QUAV-style reproduction baseline** to Ibrahim's UAV-HUBO code. It is **not the official QUAV implementation**, because I did not find a public official QUAV GitHub repository. It follows the published QUAV workflow at a reproducible benchmark level:

1. Grid/graph and obstacle preprocessing.
2. Candidate path generation with A* and RRT-style randomized planning.
3. Path-segment/path-candidate cost assignment.
4. QAOA-style quantum-assisted selection using a small state-vector simulator.
5. Feasibility-aware comparison with A* and RRT.

## Install/copy

Copy `src/quav_style_baseline.py` into your existing:

```bash
UAV_HUBO_QIML_patched_code/code_pkg/src/
```

Copy `run_quav_style_baseline.sh` into either `code_pkg/` or `code_pkg/src/`.

## Run

```bash
conda activate uav_hubo_qiml
cd /path/to/UAV_HUBO_QIML_patched_code/code_pkg
bash run_quav_style_baseline.sh
```

Or manually:

```bash
cd /path/to/UAV_HUBO_QIML_patched_code/code_pkg/src
python quav_style_baseline.py --max-candidates 12 --qaoa-layers 1 --qaoa-steps 60
```

## Outputs

```text
src/outputs/quav_style_baseline/quav_style_results.json
src/outputs/quav_style_baseline/quav_style_report.md
src/outputs/quav_style_baseline/quav_style_paths.png
src/outputs/quav_style_baseline/quav_style_metrics.png
src/outputs/quav_style_baseline/quav_style_loss.png
results/quav_style_baseline_log.txt
```

## How to cite/use in the paper

Use careful wording:

> Because no public official QUAV code was found, we implemented a QUAV-style baseline following the published algorithmic description. This baseline is used only as a reference implementation, not as a claim of exact reproduction of the authors' implementation.

## Why this helps

This lets the paper compare against the closest published QAOA/UAV obstacle-avoidance idea, while preserving the main contribution of our work: feasibility-aware HUBO/QUBO solver analysis.
