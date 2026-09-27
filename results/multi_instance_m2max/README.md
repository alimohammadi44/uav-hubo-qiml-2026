# Three fixed publication instances

This directory contains the clean-revision Apple M2 Max run used for the
camera-ready obstacle-layout sensitivity analysis. The run contains the
primary six-obstacle instance and two fixed sensitivity instances with two and
three obstacles. All cases use an `8 x 8` grid, `L = 20`, start `(0, 0)`, and
target `(7, 7)`.

The recorded source revision is
`061d246f3abc6129f078f82d1467d55af72fbded`. `manifest.json` records the
hardware, Python and solver-package versions, UTC timestamps, command, and
clean-source status. `plan_used.json` is an exact snapshot of the experiment
plan.

Files:

- `primary_6_obstacles.json`: rerun of the primary Table 2 geometry and budgets
- `random_2_obstacles.json`: fixed two-obstacle sensitivity instance
- `random_3_obstacles.json`: fixed nested three-obstacle sensitivity instance
- `multi_instance_summary.csv`: one row per case and method
- `configured_maps.png`: plotted geometry of all three cases
- `manifest.json`: run provenance
- `plan_used.json`: complete configuration snapshot

The paper's three-instance table is generated from these JSON files. The
primary Table 2 itself is generated from `../unified_comparison_with_cpsat.csv`;
its original same-platform timing values are therefore not silently replaced
by normal timing variation in this rerun. No non-`8 x 8` result is part of the
publication workflow.
