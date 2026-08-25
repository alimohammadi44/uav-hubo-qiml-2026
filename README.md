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

## Notes for reviewers/authors

- `neal` is a classical simulated annealing sampler, not hardware.
- The QUAV-style QAOA row is a candidate-path selector and is not an official QUAV implementation.
- CP-SAT/MILP are included as exact correctness anchors.
- Feasibility, goal arrival, and full-task success are intentionally reported separately.
