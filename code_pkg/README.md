# UAV HUBO — Code and Results

Reference implementation for the occlusion-aware grid path-planning HUBO
benchmark. Python 3.12.

    pip install -r requirements.txt

## Layout

    src/     source
    results/ benchmark JSON + figures used in the paper

## Source files

| File | Role |
|---|---|
| `task2_grid_hubo.py` | **Instance builder + native-HUBO solver.** Defines `Scenario` and `GridHUBO`; builds all seven Hamiltonian terms; `trajectory_sa()` is the constrained SA that evaluates the HUBO natively. |
| `task2_basis_hubo.py` | Alternative basis-coefficient encoding |
| `task2_comparison.py` | Compares the two encodings |
| `task3_polished_v3_1.py` | **The run reported in the paper.** SA vs `neal`, feasibility audit, penalty sweep. |
| `task3_qiskit_qaoa.py` | Tier-2 QAOA on the reduced instance |
| `astar_baseline.py` | Classical A* baseline |
| `task3_neal_comparison.py`, `task3_polished_v3.py` | Earlier iterations, kept for history |

## Reproducing the reported numbers

    cd src
    python task2_grid_hubo.py        # builds the instance, prints counts
    python astar_baseline.py         # A* baseline
    python task3_polished_v3_1.py    # main benchmark (long: SA ~40 min)
    python task3_qiskit_qaoa.py      # Tier-2 QAOA

Instance counts can be checked directly:

```python
from task2_grid_hubo import Scenario, GridHUBO
g = GridHUBO(Scenario()); g.build()
print(g.T, g.V, g.num_vars, len(g.H_prox()))
# 15 64 960 3172
```

## Key implementation points a reviewer will want

**Motion relation** — `GridHUBO.neighbors()` (~line 145) returns the four
cardinal neighbours **plus the cell itself**, so hovering is permitted.

**Cubic term** — `H_prox()` (~line 263) sums
`x[t-1,u]·x[t,v]·x[t+1,w]` over connected triples of buffer cells with
`v ∈ N(u)`, `w ∈ N(v)`. It is *not* a single repeated cell. Raw
coefficient η = 3 per triple, scaled by `lambda_prox = 5`, so each cubic
coefficient is 15. Of 3,172 cubic monomials, 312 are same-cell and 2,860
involve distinct cells.

**Native HUBO evaluation** — `trajectory_sa()` (~line 491) multiplies all
variables in each monomial regardless of degree. This is the only solver
here that evaluates the HUBO without quadratization.

**`neal` is quadratic-only** — `task3_polished_v3_1.py` (~line 167) wraps
`neal.SimulatedAnnealingSampler` in `dimod.HigherOrderComposite`, which
quadratizes the polynomial before sampling. `neal` therefore receives a
1,542-variable QUBO (582 product variables added), not the HUBO.
`sample_poly` is called **without** an explicit `penalty_strength`, so
dimod's default of 1.0 applies — a known limitation discussed in the paper.

**QAOA is quadratic-only** — `task3_qiskit_qaoa.py` (~line 446) raises an
error on any monomial of degree > 2. The Tier-2 instance is obstacle-free,
so it has no buffer cells and no cubic terms.

## Verified instance parameters

| Quantity | Value |
|---|---|
| Grid / layers | 8×8, L = 15 |
| Variables | 15 × 64 = 960 |
| Monomials | 87,684 (960 linear / 83,552 quadratic / 3,172 cubic) |
| Buffer cells | 24 |
| Max coefficient by degree | 1975 / 2000 / 15 |
| After quadratization | 1,542 variables (+582), 88,470 interactions |
| Weights | λ_obs 80, λ_occ 10, λ_prox 5, λ_goal 8, λ_term 50, λ_struct 1000 |

## Known gaps (also stated in the paper)

1. Per-sample trajectories were not retained, so visibility for SA/`neal`
   and the full-task-success intersection cannot be recomputed without a rerun.
2. `penalty_strength` was left at dimod's default 1.0; a sweep is needed
   before attributing `neal`'s behaviour to quadratization. Auditing the
   50-read run with `keep_penalty_variables=True` gives 44/50 product-consistent.
3. Exact Qiskit component versions for the Tier-2 run were not recorded.
4. L = 15 permits exactly 14 transitions for a Manhattan distance of 14,
   so every goal-reaching path is a shortest monotone path.
