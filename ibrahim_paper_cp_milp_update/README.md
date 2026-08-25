# Ibrahim UAV HUBO/QUBO paper update: QUAV-style + CP/MILP

This package integrates the final benchmark results into the Ibrahim paper and adds exact optimization baselines.

## New code
- `src/quav_style_baseline.py`: same-platform QUAV-style QAOA candidate-path baseline (not official QUAV code).
- `src/cp_milp_exact_baselines.py`: exact MILP baseline using SciPy/HiGHS and optional CP-SAT baseline using OR-Tools.

## How to run
From this folder:
```bash
conda activate uav_hubo_qiml
bash run_quav_style_baseline.sh
bash run_milp_only.sh
# Optional, if OR-Tools is installed:
bash run_cp_milp_exact_baselines.sh
```

For CP-SAT install OR-Tools if needed:
```bash
python -m pip install ortools
```

The MILP model enforces one-hot occupancy, start, target arrival, obstacle avoidance, and legal moves as hard constraints. It minimizes the native HUBO soft objective including the cubic proximity term using auxiliary binary variables.
