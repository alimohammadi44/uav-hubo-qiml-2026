# Feasibility-Aware HUBO/QUBO UAV Evaluation for QIML 2026

This repository contains the code, figures, result table, and paper for a feasibility-aware HUBO/QUBO evaluation protocol for UAV obstacle-avoidance and visibility-aware grid planning. The protocol is demonstrated on one fully specified `8 x 8`, `L = 20` instance; the repository does not claim a general instance-suite benchmark or quantum speedup.

The evaluation compares decoded-path feasibility, native HUBO energy, and solver behavior across classical, quantum-inspired, and quantum-compatible workflows.

## Camera-ready paper

The accepted QIML 2026 paper is available at [`paper/QIML_2026_UAV_HUBO_Submission.pdf`](paper/QIML_2026_UAV_HUBO_Submission.pdf). The buildable AAAI-27 LaTeX source is stored in `paper/`.

## Contents

```text
paper/      AAAI-27 camera-ready source, paper PDF, and stand-alone abstract
figures/    Figures used in the paper
src/        Reproducible Python implementation
results/    Main comparison table CSV
```

## Demonstration instance

The paper evaluates one `8 x 8` grid with horizon `L = 20`, obstacle cells, visibility cost, and a cubic temporal buffer-risk term. The compared methods are:

- A* baseline
- RRT-best baseline
- QUAV-style QAOA candidate-path selector
- trajectory-level simulated annealing over native HUBO energy
- HUBO-to-QUBO/BQM sampling with `neal`
- exact CP-SAT reference

## Recommended setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Apple M2 Max timing workflow

Run the publication-timing workflow from the repository root:

```bash
./run_m2max_benchmark.command
```

The first run creates an isolated `.venv-m2max` environment and installs the requirements. The complete workflow may take 30--65 minutes because it runs the full 50-run trajectory-SA experiment. It records solver-level wall-clock times while excluding plotting and file-serialization time.

The command produces:

```text
results/m2max_runtime_benchmark.json
results/m2max_runtime_benchmark.csv
results/unified_comparison_m2max.csv
```

## Configurable maps and multi-instance runs

Grid geometry is not hard-coded in the multi-instance workflow. Edit
[`configs/multi_instance_plan.json`](configs/multi_instance_plan.json) to set,
for every run:

- `grid_size` (an `N x N` grid)
- `horizon`
- `start` and `target` as `[row, column]`
- `obstacles` as a list of `[row, column]` cells
- optional per-run solver seeds and budgets under `solver`

Validate a plan without running any solver:

```bash
.venv-m2max/bin/python src/run_multi_instance_m2max.py \
  --plan configs/multi_instance_plan.json --validate-only
```

Run every configured map, or select one named run:

```bash
.venv-m2max/bin/python src/run_multi_instance_m2max.py \
  --plan configs/multi_instance_plan.json

.venv-m2max/bin/python src/run_multi_instance_m2max.py \
  --plan configs/multi_instance_plan.json \
  --run random_2_obstacles
```

The loader rejects out-of-grid coordinates, duplicate obstacles, obstacles on
the start/target, unreachable targets, and horizons too short for the shortest
feasible path. See
[`configs/example_custom_plan.json`](configs/example_custom_plan.json) for maps
with different grid sizes, targets, obstacle sets, and run budgets.

## Notes for reviewers and authors

- `neal` is a classical simulated-annealing sampler, not quantum hardware.
- The QUAV-style QAOA row is a candidate-path selector, not an official QUAV implementation.
- The paper uses CP-SAT as its exact correctness anchor.
- Feasibility, goal arrival, and full-task success are reported separately.
- Multi-instance evaluation is future work.
