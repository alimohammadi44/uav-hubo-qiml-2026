# Hard-LOS HUBO vs. Masnavi-Style Continuous MPC Comparison

This package runs a fair first comparison on one Python platform:

1. **Discrete hard-LOS A\*** baseline.
2. **Discrete hard-LOS trajectory-level SA/HUBO-style** baseline.
3. **Masnavi-style continuous optimizer** implemented from the mathematical formulation in the paper.

## Why this package exists

The original Masnavi repository contains ROS/Gazebo C++ code. That is useful for reproducing their full simulator, but it is not a drop-in Python benchmark in this sandbox. This package therefore implements the Masnavi **formulation** in Python for the same map used by the discrete HUBO benchmark.

## Masnavi-style continuous formulation used here

For each UAV trajectory point `p(t)` and each sampled point `u` on the line of sight to the target,

```text
p_los(t,u) = (1-u) p(t) + u p_target
||p_los(t,u) - p_obstacle||^2 / r_obstacle^2 >= 1
```

This is Eq. (13)-(14) of the Masnavi paper reduced to a 2-D circular-obstacle case. The objective minimizes squared acceleration smoothness.

## Important scenario note

The old poster scenario start `(0,0)` to target `(7,7)` is infeasible under hard LOS because the start is occluded. This package supports:

```bash
python compare_hard_los_hubo_vs_masnavi.py --scenario original --check-only
```

For an actual hard-LOS comparison, it uses the same obstacle map and target but moves the start to `(3,0)`, which is visible:

```bash
python compare_hard_los_hubo_vs_masnavi.py --scenario visible_start --runs 20 --sweeps 400
```

## Outputs

The script writes:

- `comparison_results.json`
- `metrics.csv`
- `comparison_paths.png`
- `comparison_metrics.png`
- per-method trajectory CSV files

## Interpretation

This is the correct direction for revising the paper if we want to compare with Masnavi. The current poster results should not be called Masnavi-equivalent because the earlier HUBO code treated LOS as a soft cost. This package treats LOS as hard in both the discrete and continuous baselines.
