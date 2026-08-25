# Soft-occlusion Masnavi-scenario comparison

This package modifies the previous hard-LOS comparison so that **obstacle/collision avoidance remains hard** but **LOS occlusion is soft**.

## Scenario

The default scenario is `masnavi_static_3cyl`, derived from the uploaded Masnavi repository:

- `target_tracker/launch/3_cylinder_0.35_world.launch`
- `target_tracker/params/3_obs_params.yaml`

Numerical values used:

| Quantity | Value | Source |
|---|---:|---|
| workspace | x in [-4, 4], y in [-3, 3] | `3_obs_params.yaml` |
| UAV start | (-2, 0) | `3_obs_params.yaml`, launch file |
| terminal/reference point | (3, 1) | `3_obs_params.yaml` |
| observed target for LOS | (5, 0) | `3_cylinder_0.35_world.launch` |
| obstacles | (1, -2.5), (-0.5, 1.5), (1, 1) | `3_cylinder_0.35_world.launch` |
| obstacle radius | 0.35 m | launch/URDF filename and cylinder URDF |

Important: the Masnavi IEEE Access paper states that static-environment target trajectories were created by teleoperation and replayed, but the paper does not list exact numerical target trajectories. Therefore this package uses the numerical scenario available in the uploaded repository, not an unrecoverable paper-only trajectory.

## Modeling change

Previous hard-LOS model:

```text
structural feasibility = start + terminal + collision-free + motion collision-free + LOS-visible
```

This soft-occlusion model:

```text
structural feasibility = start + terminal + collision-free + motion collision-free
visibility = separate trajectory-quality metric
soft LOS cost = sum max(0, -LOS clearance)
```

So a trajectory may be structurally feasible even if LOS is partially occluded.

## Methods compared

1. `A* soft-LOS`: graph search over collision-free grid nodes with soft LOS cost in edge/node cost.
2. `SA soft-LOS HUBO`: trajectory-level simulated annealing with obstacle/motion hard checks and soft LOS energy.
3. `Masnavi-style continuous soft-LOS`: continuous smoothness optimizer with hard collision constraints and soft LOS penalty.

## Run

```bash
python compare_soft_occlusion_masnavi_scenario.py --out-dir outputs_soft_occ --grid-step 1.0 --lambda-occ 8.0 --runs 5 --sweeps 50
```

Outputs:

- `outputs_soft_occ/comparison_paths.png`
- `outputs_soft_occ/comparison_metrics.png`
- `outputs_soft_occ/metrics.csv`
- `outputs_soft_occ/comparison_results.json`

## Current observation

For the numerical `3_cylinder_0.35` repository setup used here, all three methods find structurally feasible and 100% visible paths. This means the available static numerical scenario is not a strong occlusion-stress benchmark. For a stronger Masnavi-like comparison, the next code step should use a moving target trajectory or the wall-world setup from Fig. 9 / `3_wall_world_dynamic.launch`.
