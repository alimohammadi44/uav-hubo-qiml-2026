# Hard-LOS UAV comparison platform — corrected obstacle model

This package compares a discrete hard-LOS benchmark with a Python continuous
Masnavi-style baseline on the same map.

Correction in this version:

- The first continuous baseline used small circular obstacles (`r=0.45`) while
  the figure showed full black obstacle cells. That allowed the continuous path
  to cut through the visual obstacle cells.
- This corrected version represents each obstacle cell by its circumscribed
  circle (`r = sqrt(2)/2`). Therefore, any continuous trajectory outside the
  circle is also outside the whole black cell.
- The continuous optimizer also enforces obstacle clearance along the plotted
  motion segments between trajectory samples, not only at the trajectory sample
  points.
- LOS visibility is still hard: sampled points on the LOS segment to the target
  must remain outside every obstacle.

Run:

```bash
python compare_hard_los_hubo_vs_masnavi.py --scenario visible_start --runs 20 --sweeps 400 --out-dir outputs_compare_fixed
```

The original start `(0,0)` remains infeasible under hard LOS.

```bash
python compare_hard_los_hubo_vs_masnavi.py --scenario original --check-only --out-dir outputs_original_check
```
