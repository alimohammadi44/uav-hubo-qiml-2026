# Feasibility-Aware HUBO/QUBO UAV Evaluation for QIML 2026

This repository contains the code, figures, result tables, and camera-ready paper for a feasibility-aware HUBO/QUBO evaluation protocol for UAV obstacle-avoidance and visibility-aware grid planning. The protocol is evaluated on three fixed `8 x 8`, `L = 20` instances: one primary six-obstacle instance and two obstacle-layout sensitivity instances with two and three obstacles. These cases do not constitute a general instance-suite benchmark, and the paper does not claim quantum speedup.

The evaluation compares decoded-path feasibility, native HUBO energy, and solver behavior across classical, quantum-inspired, and quantum-compatible workflows.

The publication workflow is anchored by `src/run_multi_instance_m2max.py`,
`src/task2_grid_hubo.py`, `src/quav_style_baseline.py`,
`src/task3_polished_v3_1.py`, and the CP-SAT path in
`src/cp_milp_exact_baselines.py`. Other retained scripts are exploratory or
stand-alone development modules and are not sources for camera-ready claims.

## Camera-ready paper

The accepted QIML 2026 paper is available at [`paper/QIML_2026_UAV_HUBO_Submission.pdf`](paper/QIML_2026_UAV_HUBO_Submission.pdf). The buildable AAAI-27 LaTeX source is stored in `paper/`.

## Contents

```text
paper/      AAAI-27 camera-ready source, paper PDF, and stand-alone abstract
figures/    Figures used in the paper
src/        Reproducible Python implementation
results/    Primary comparison CSV and three-instance M2 Max result logs
```

## Publication instances

All three publication cases use an `8 x 8` grid, horizon `L = 20`, start `(0, 0)`, target `(7, 7)`, the same objective weights, visibility cost, and cubic temporal buffer-risk term. They differ only in obstacle count and placement. The compared methods are:

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

The publication workflow intentionally accepts only `8 x 8` maps. Edit
[`configs/multi_instance_plan.json`](configs/multi_instance_plan.json) to set,
for every run:

- `grid_size` (must be `8`)
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

The committed publication plan runs the primary instance plus both sensitivity
instances and requires an Apple M2 Max. The loader rejects any grid size other
than `8`, as well as out-of-grid
coordinates, duplicate obstacles, obstacles on
the start/target, unreachable targets, and horizons too short for the shortest
feasible path. See
[`configs/example_custom_plan.json`](configs/example_custom_plan.json) for an
alternate `8 x 8` map with a different obstacle set and run budget. It is a
configuration example only and is not reported as a paper result.

Publication data are mapped as follows:

- `results/unified_comparison_with_cpsat.csv` is the source for the primary-instance comparison in Table 2.
- `results/multi_instance_m2max/` contains the clean-revision logs for all three fixed instances.
- `paper/generated_primary_table.tex` and `paper/generated_sensitivity_table.tex` are generated directly from those result files by `src/generate_publication_tables.py`.

The primary CSV retains aggregate RRT, SA, and `neal` wall-clock times. The
table generator divides them by 40 trials, 50 runs, and 50 reads respectively,
so the paper consistently reports mean time/run (or mean time/read).

Verify the generated table and configuration tests with:

```bash
python3 src/generate_publication_tables.py --check
python3 -m unittest discover -s tests -v
```

## Notes for reviewers and authors

- `neal` is a classical simulated-annealing sampler, not quantum hardware.
- The QUAV-style QAOA row is a candidate-path selector, not an official QUAV implementation.
- CP-SAT certifies the optimum of the implemented integer-scaled hard-feasible objective.
- The optional scipy/HiGHS MILP helper is development code and is not a reported paper method.
- Feasibility, goal arrival, and full-task success are reported separately.
- Only `neal` samples the full penalized QUBO; the other workflows construct or enforce some or all feasibility conditions before scoring.
- The two added maps are fixed sensitivity cases, not distribution-level evidence or a broad benchmark suite.
