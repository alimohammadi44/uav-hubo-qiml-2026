"""Exact CP-SAT baseline for the QIML UAV HUBO/QUBO benchmark.

This is the exact reference used by the paper. It solves the hard-feasible
trajectory problem with OR-Tools CP-SAT and minimizes the native HUBO soft
objective (visibility + goal + cubic sustained-proximity + terminal cost).
"""
from __future__ import annotations

import json, math, os, sys, time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
from ortools.sat.python import cp_model

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from task2_grid_hubo import Scenario, GridHUBO

Cell = Tuple[int, int]
PathT = List[Cell]
OUT_DIR = THIS_DIR / "outputs" / "cp_sat_exact_baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _soft_linear_coeff(t: int, v: int, s: Scenario) -> float:
    r, c = s.index_to_cell(v)
    tr, tc = s.target
    dist = math.hypot(r - tr, c - tc)
    coeff = s.lambda_occ * s.occlusion_count(v)
    coeff += s.lambda_goal * dist * (t + 1) / s.horizon
    if t == s.horizon - 1 and v != s.cell_index(*s.target):
        coeff += s.lambda_terminal
    return float(coeff)


def _metrics(path: PathT, s: Scenario, grid: GridHUBO, hubo) -> Dict[str, object]:
    idx = [s.cell_index(*c) for c in path]
    rep = grid.feasibility_report(idx)
    feasible = (not rep["start_violation"] and rep["move_violations"] == 0 and rep["obstacle_collisions"] == 0)
    goal = bool(path and path[-1] == s.target)
    energy = float(grid.eval_trajectory(hubo, idx))
    hard_constant = -s.lambda_uniq * s.horizon - s.lambda_start
    dirs = []
    for a, b in zip(path[:-1], path[1:]):
        d = (b[0]-a[0], b[1]-a[1])
        if d != (0, 0): dirs.append(d)
    turns = sum(dirs[i] != dirs[i-1] for i in range(1, len(dirs)))
    return {
        "feasible": feasible,
        "goal_arrival": goal,
        "full_success": feasible and goal,
        "path_length": sum(a != b for a, b in zip(path[:-1], path[1:])),
        "turns": int(turns),
        "buffer_steps": sum(s.cell_index(*c) in s.buffer_indices for c in path),
        "visibility_rate": sum(s.occlusion_count(s.cell_index(*c)) == 0 for c in path) / len(path),
        "native_hubo_energy": energy,
        "soft_cost": energy - hard_constant,
        "trajectory": [list(c) for c in path],
    }


def solve_exact_cpsat(time_limit_s: float = 300.0, workers: int = 8) -> Dict[str, object]:
    s = Scenario()
    grid = GridHUBO(s)
    hubo = grid.build()
    T, V = grid.T, grid.V
    free = [v for v in range(V) if v not in s.obstacle_indices]
    free_set = set(free)
    start_v = s.cell_index(*s.start)
    target_v = s.cell_index(*s.target)

    model = cp_model.CpModel()
    x = {(t, v): model.NewBoolVar(f"x_{t}_{v}") for t in range(T) for v in range(V)}
    for t in range(T):
        model.Add(sum(x[t, v] for v in free) == 1)
        for v in s.obstacle_indices:
            model.Add(x[t, v] == 0)
    model.Add(x[0, start_v] == 1)
    model.Add(x[T-1, target_v] == 1)

    for t in range(T-1):
        for u in free:
            model.Add(x[t, u] <= sum(x[t+1, v] for v in s.neighbors(u) if v in free_set))

    scale = 1000
    objective = []
    for t in range(T):
        for v in range(V):
            c = int(round(scale * _soft_linear_coeff(t, v, s)))
            if c:
                objective.append(c * x[t, v])

    prox_count = 0
    for key, coeff in grid.H_prox().items():
        vars3 = []
        for idx in key:
            tt, vv = divmod(idx, V)
            vars3.append(x[tt, vv])
        y = model.NewBoolVar(f"prox_{prox_count}")
        prox_count += 1
        for z in vars3:
            model.Add(y <= z)
        model.Add(y >= sum(vars3) - 2)
        objective.append(int(round(scale * float(coeff) * s.lambda_prox)) * y)

    model.Minimize(sum(objective))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = int(workers)
    t0 = time.perf_counter()
    status = solver.Solve(model)
    runtime = time.perf_counter() - t0

    status_name = solver.StatusName(status)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"method": "CP-SAT exact baseline", "status": status_name, "runtime_s": runtime}

    traj = []
    for t in range(T):
        v = next(v for v in range(V) if solver.Value(x[t, v]))
        traj.append(s.index_to_cell(v))
    m = _metrics(traj, s, grid, hubo)
    result = {
        "method": "CP-SAT exact baseline (OR-Tools)",
        "status": status_name,
        "runtime_s": runtime,
        "objective_soft_cost_scaled": solver.ObjectiveValue(),
        "proximity_aux_variables": prox_count,
        "metrics": m,
    }
    return result


def plot_path(path: Sequence[Sequence[int]], s: Scenario):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-0.5, s.n-0.5); ax.set_ylim(s.n-0.5, -0.5)
    ax.set_xticks(range(s.n)); ax.set_yticks(range(s.n)); ax.grid(True)
    for r, c in s.obstacles:
        ax.add_patch(plt.Rectangle((c-.5, r-.5), 1, 1, alpha=.75))
    xy = [(c, r) for r, c in path]
    ax.plot([p[0] for p in xy], [p[1] for p in xy], marker='o')
    ax.scatter([s.start[1]], [s.start[0]], s=120, marker='o', label='Start')
    ax.scatter([s.target[1]], [s.target[0]], s=160, marker='*', label='Target')
    ax.set_title('Exact CP-SAT path')
    ax.legend()
    fig.tight_layout()
    return fig


def main():
    result = solve_exact_cpsat()
    print(json.dumps(result, indent=2))
    (OUT_DIR / "cp_sat_result.json").write_text(json.dumps(result, indent=2))
    if "metrics" in result:
        s = Scenario()
        fig = plot_path(result["metrics"]["trajectory"], s)
        fig.savefig(OUT_DIR / "cp_sat_exact_path.png", dpi=180)
        plt.show()


if __name__ == "__main__":
    main()
