# UAV HUBO/QUBO Feasibility-Aware Benchmark for QIML 2026

This repository contains the code, figures, result table, and paper draft for a feasibility-aware HUBO/QUBO benchmark for UAV obstacle-avoidance and visibility-aware grid planning.

The paper does **not** claim quantum speedup. Its purpose is to compare decoded-path feasibility, native HUBO energy, and solver behavior across classical, quantum-inspired, and quantum-compatible workflows.

## Contents

```text
paper/      LaTeX source and compiled PDF
figures/    Figures used in the paper
src/        Reproducible Python code for the benchmark
results/    Main comparison table CSV
```

## Main benchmark

The main benchmark is an `8 x 8` grid with horizon `L = 20`, obstacle cells, visibility cost, and a cubic temporal buffer-risk term.

The compared methods are:

- A* baseline
- RRT-best baseline
- QUAV-style QAOA candidate-path selector
- trajectory-level simulated annealing over native HUBO energy
- HUBO-to-QUBO/BQM sampling with `neal`
- exact CP-SAT/MILP baseline

## Recommended setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Apple M2 Max runtime benchmark

Run the complete publication-timing workflow on the Apple M2 Max from the
repository root:

```bash
./run_m2max_benchmark.command
```

The first run creates an isolated `.venv-m2max` environment and installs the
requirements. The complete workflow may take 30--65 minutes because it runs
the full 50-run trajectory-SA experiment. It records solver-level wall-clock
times while excluding plotting and file-serialization time.

The command produces:

```text
results/m2max_runtime_benchmark.json
results/m2max_runtime_benchmark.csv
results/unified_comparison_m2max.csv
```

Send these three files to the paper maintainer before replacing the runtime
column in Table II. The existing paper PDF is intentionally not changed by the
benchmark command.

## Notes for reviewers/authors

- `neal` is a classical simulated annealing sampler, not hardware.
- The QUAV-style QAOA row is a candidate-path selector and is not an official QUAV implementation.
- CP-SAT/MILP are included as exact correctness anchors.
- Feasibility, goal arrival, and full-task success are intentionally reported separately.
