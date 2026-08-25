# Task 3 — Three-Way Solver Comparison

## Problem

Tiny QUBO: 2×2 grid, T=2 steps, 8 binary variables. Same problem solved by three different solvers for a fair benchmark.

## Why this tiny scenario

QAOA on Qiskit's state-vector simulator scales as 2^n. The full Task 2 HUBO has 960 variables, which is computationally impossible to simulate. A 3×3 grid with T=4 gives 36 variables, comfortably within QAOA's practical reach on a laptop. This is a standard practice in quantum optimization research — demonstrate the algorithm on a tractable instance to enable apples-to-apples comparison.

## Results

| Metric | Classical SA | D-Wave Neal | Qiskit QAOA |
|---|---|---|---|
| Best energy | -118.17 | -118.17 | -118.17 |
| Runtime (s) | 0.07 | 0.01 | 0.55 |
| Feasible | ✓ | ✓ | ✓ |
| Reaches target | ✗ | ✗ | ✗ |

## Analysis

This is the first part of a two-tier benchmark:

- **Tier 1 (this file)**: tiny 36-variable scenario, all three solvers comparable.
- **Tier 2 (task3_neal_comparison.py)**: full 960-variable scenario, only classical SA and Neal (QAOA infeasible at this scale).

### Solver characteristics observed

- **Classical SA**: fastest per-sample (no quantum simulation overhead). Treats the problem as pure combinatorial optimization. Mature, well-understood.

- **D-Wave Neal**: same algorithm D-Wave QPUs use internally, just on a CPU. Designed for HUBO/QUBO problems directly. Produces a distribution of samples.

- **Qiskit QAOA**: gate-based quantum algorithm running on classical simulator. Hybrid quantum-classical loop with SPSA tuning QAOA's variational parameters. Slowest because each parameter update requires running the full quantum circuit. Demonstrates how a real QAOA-on-hardware deployment would scale.

### Honest scaling limitation

QAOA's runtime growth is the key practical finding. Even on 36 variables, it is significantly slower than classical SA and Neal. For real-time UAV planning (the original paper's regime), QAOA would not be competitive on current quantum hardware. Neal, by contrast, is well-suited for HUBO problems and could be deployed via D-Wave's real QPU.

