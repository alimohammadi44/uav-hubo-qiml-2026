# Hard-LOS Grid HUBO Starter for Masnavi-Style Comparison

This folder contains a first corrected code baseline for comparing a discrete HUBO planner with Masnavi-style occlusion-free MPC.

## Main point

The original grid scenario in the poster code has start=(0,0), target=(7,7), and the central obstacle block at (3,3),(3,4),(4,3),(4,4). Under hard per-step line-of-sight visibility, that original scenario is infeasible because the start cell is already occluded from the target. The script therefore supports two cases:

- `original`: reproduces the original grid/horizon and checks feasibility. It reports no hard-LOS path.
- `visible_start`: keeps the same target and obstacles, but uses start=(3,0) and horizon=12 so that a 100% LOS-visible path exists.

## Run

```bash
python hard_los_grid_hubo.py --scenario original --check-only
python hard_los_grid_hubo.py --scenario visible_start --runs 10 --sweeps 200
```

## Outputs

The script writes:

- `metrics.json`
- `astar_hard_los_trajectory.png`
- `sa_hard_los_trajectory.png`
- `sa_energy_history.png`
- `astar_trajectory.csv`

## Modeling changes from the previous code

1. LOS visibility is now a hard penalty term `H_los`, not just a soft cost.
2. The independent feasibility checker includes LOS violations.
3. A* searches only over cells that are both obstacle-free and LOS-visible.
4. SA move proposals are restricted to obstacle-free and LOS-visible cells.
5. The terminal target condition is hard in this starter script.

## Current smoke-test result

For `visible_start`, the A* and SA smoke test found a hard-feasible trajectory with 100% visibility. For `original`, the script correctly reports infeasibility because the start is LOS-occluded.
