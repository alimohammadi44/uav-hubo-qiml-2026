# Masnavi Code Inspection Notes

Uploaded repository: `multi-convex-mpc-target-tarcker-main.zip`.

## Repository structure

The uploaded repository contains two main parts:

1. `target_tracker/`: a ROS/Gazebo C++ package for the multi-convex MPC target tracker. This is the more relevant implementation of the Masnavi paper, but it requires ROS, Gazebo/Bebop simulator, Eigen QuadProg, and related dependencies.
2. `Nageli_implementation/`: an ACADO-based implementation and a precompiled `test` binary. This binary runs in the current sandbox, but it is not yet a fair comparison platform because it uses its own continuous scenario and output format.

## What can run in this sandbox

The precompiled ACADO demo can run:

```bash
cd Nageli_implementation
./test
```

It produces `x.txt`, `y.txt`, `obsx*.txt`, `obsy*.txt`. However, this should be treated as a smoke test only, not a final fair benchmark.

## Fair comparison requirement

For a defensible comparison, both planners must run on the same scenario and be evaluated with the same metrics:

- Same target position/trajectory.
- Same obstacle geometry after mapping grid obstacles to continuous disks/ellipses or continuous obstacles to grid cells.
- Same horizon or equivalent physical time window.
- Same hard LOS definition: no obstacle intersection along the UAV-target line segment.
- Same collision metric.
- Same trajectory-quality metrics: path length/smoothness, visibility, runtime, and feasibility rate.

## Important finding

The original discrete grid scenario used in the poster code is infeasible under hard LOS from t=0 because the start cell is occluded. A fair Masnavi comparison therefore requires either:

1. Change the discrete benchmark start/obstacles so that hard LOS is feasible, or
2. Define a startup phase where LOS is not required until the tracker has acquired the target, but this would no longer be a strict occlusion-free comparison.
