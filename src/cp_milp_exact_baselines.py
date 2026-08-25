"""
Exact CP-SAT / MILP baselines for the Ibrahim UAV HUBO/QUBO benchmark.

This script adds exact optimization baselines for the same 8x8, T=20 UAV
obstacle-avoidance/visibility benchmark used in the QIML paper.

Two models are provided:
  1) MILP solved by scipy.optimize.milp/HiGHS (run by default).
  2) CP-SAT solved by OR-Tools if ortools is installed (optional).

The exact model enforces hard path constraints directly:
  - exactly one cell at each time step
  - fixed start
  - fixed terminal target arrival
  - obstacle cells forbidden
  - only 4-connected/self-loop moves allowed

The objective minimizes the native HUBO soft objective over feasible paths:
  H_occ + H_goal + H_prox + H_terminal.
Because all hard constraints are enforced, native HUBO energy is reported as
hard constant (-lambda_uniq*T - lambda_start) plus the optimized soft cost.

Run from code_pkg/src after copying this file there:
    python cp_milp_exact_baselines.py --mode milp
    python cp_milp_exact_baselines.py --mode cpsat   # requires ortools
    python cp_milp_exact_baselines.py --mode both

Outputs:
    outputs/cp_milp_exact_baselines/exact_baseline_results.json
    outputs/cp_milp_exact_baselines/exact_baseline_summary.csv
    outputs/cp_milp_exact_baselines/exact_baseline_trajectory.csv
    outputs/cp_milp_exact_baselines/exact_baseline_path.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except Exception:
    plt = None
    mpatches = None

# Import Ibrahim benchmark formulation from the same src folder.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from task2_grid_hubo import Scenario, GridHUBO

Cell = Tuple[int, int]
Path = List[Cell]

OUT_DIR = os.path.join(THIS_DIR, "outputs", "cp_milp_exact_baselines")
os.makedirs(OUT_DIR, exist_ok=True)


def cell_path_to_indices(path: Path, s: Scenario) -> List[int]:
    return [s.cell_index(r, c) for r, c in path]


def indices_to_cell_path(traj: Sequence[int], s: Scenario) -> Path:
    return [s.index_to_cell(int(v)) for v in traj]


def path_length(path: Path) -> int:
    return sum(1 for a, b in zip(path[:-1], path[1:]) if a != b)


def turn_count(path: Path) -> int:
    dirs = []
    for a, b in zip(path[:-1], path[1:]):
        dr, dc = b[0] - a[0], b[1] - a[1]
        if (dr, dc) != (0, 0):
            dirs.append((dr, dc))
    return sum(1 for i in range(1, len(dirs)) if dirs[i] != dirs[i - 1])


def buffer_steps(path: Path, s: Scenario) -> int:
    return sum(1 for cell in path if s.cell_index(*cell) in s.buffer_indices)


def visibility_rate(path: Path, s: Scenario) -> float:
    if not path:
        return 0.0
    visible = 0
    for cell in path:
        if s.occlusion_count(s.cell_index(*cell)) == 0:
            visible += 1
    return visible / len(path)


def valid_motion(path: Path, s: Scenario) -> bool:
    idx = cell_path_to_indices(path, s)
    return all(idx[t + 1] in s.neighbors(idx[t]) for t in range(len(idx) - 1))


def obstacle_collisions(path: Path, s: Scenario) -> int:
    obs = set(s.obstacles)
    return sum(1 for cell in path if cell in obs)


def evaluate_path(path: Path, s: Scenario, grid: GridHUBO, hubo: Dict[Tuple[int, ...], float]) -> Dict[str, object]:
    traj_idx = cell_path_to_indices(path, s)
    hard_report = grid.feasibility_report(traj_idx)
    reaches = bool(path and path[-1] == s.target)
    feasible = (
        not hard_report["start_violation"]
        and hard_report["move_violations"] == 0
        and hard_report["obstacle_collisions"] == 0
    )
    energy = float(grid.eval_trajectory(hubo, traj_idx))
    hard_constant = -s.lambda_uniq * s.horizon - s.lambda_start if feasible else None
    soft_cost = energy - hard_constant if hard_constant is not None else None
    return {
        "feasible": bool(feasible),
        "goal_arrival": bool(reaches),
        "full_success": bool(feasible and reaches),
        "path_length": int(path_length(path)),
        "turns": int(turn_count(path)),
        "buffer_steps": int(buffer_steps(path, s)),
        "visibility_rate": float(visibility_rate(path, s)),
        "obstacle_collisions": int(obstacle_collisions(path, s)),
        "native_hubo_energy": energy,
        "hard_constant_if_feasible": hard_constant,
        "soft_cost_if_feasible": soft_cost,
        "trajectory": [list(c) for c in path],
    }


def _soft_linear_coeff_for_x(t: int, v: int, s: Scenario, grid: GridHUBO) -> float:
    """Native HUBO soft linear terms on x[t,v]."""
    r, c = s.index_to_cell(v)
    tr, tc = s.target
    dist = math.sqrt((r - tr) ** 2 + (c - tc) ** 2)
    coeff = 0.0
    coeff += s.lambda_occ * s.occlusion_count(v)
    coeff += s.lambda_goal * dist * (t + 1) / s.horizon
    # H_terminal is zero for the target and +lambda_terminal for non-target at final time.
    if t == s.horizon - 1 and v != s.cell_index(*s.target):
        coeff += s.lambda_terminal
    return float(coeff)


def solve_exact_milp(time_limit_s: float = 300.0, mip_rel_gap: float = 0.0, verbose: bool = False) -> Dict[str, object]:
    """Solve exact feasible-path problem with scipy.optimize.milp/HiGHS."""
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import coo_matrix
    except Exception as exc:
        return {"status": "not_available", "error": f"scipy.optimize.milp unavailable: {exc}"}

    s = Scenario()
    grid = GridHUBO(s)
    hubo = grid.build()
    T, V = grid.T, grid.V
    free = [v for v in range(V) if v not in s.obstacle_indices]
    free_set = set(free)
    start_v = s.cell_index(*s.start)
    target_v = s.cell_index(*s.target)

    # Variable layout: all x[t,v], then allowed transition e[t,u,v], then prox triple y terms.
    n_x = T * V
    x_var = lambda t, v: t * V + v

    trans: List[Tuple[int, int, int]] = []
    for t in range(T - 1):
        for u in free:
            for v in sorted(s.neighbors(u)):
                if v in free_set:
                    trans.append((t, u, v))
    trans_offset = n_x
    e_var: Dict[Tuple[int, int, int], int] = {key: trans_offset + i for i, key in enumerate(trans)}

    prox_terms = []
    for key, coeff in grid.H_prox().items():
        # H_prox unweighted coefficient is 3.0; weighted coefficient is lambda_prox*3.0.
        prox_terms.append((tuple(key), float(coeff) * s.lambda_prox))
    prox_offset = trans_offset + len(trans)
    y_var = {key: prox_offset + i for i, (key, _) in enumerate(prox_terms)}

    n_vars = n_x + len(trans) + len(prox_terms)

    c = np.zeros(n_vars, dtype=float)
    for t in range(T):
        for v in range(V):
            c[x_var(t, v)] = _soft_linear_coeff_for_x(t, v, s, grid)
    for i, (key, weighted_coeff) in enumerate(prox_terms):
        c[prox_offset + i] = weighted_coeff

    lb = np.zeros(n_vars, dtype=float)
    ub = np.ones(n_vars, dtype=float)

    # Forbid obstacles by bounds.
    for t in range(T):
        for v in s.obstacle_indices:
            ub[x_var(t, v)] = 0.0

    # Fix start and final goal by bounds.
    lb[x_var(0, start_v)] = ub[x_var(0, start_v)] = 1.0
    lb[x_var(T - 1, target_v)] = ub[x_var(T - 1, target_v)] = 1.0

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    con_lb: List[float] = []
    con_ub: List[float] = []
    row = 0

    def add_row(coeffs: List[Tuple[int, float]], low: float, high: float):
        nonlocal row
        for j, val in coeffs:
            rows.append(row)
            cols.append(j)
            data.append(float(val))
        con_lb.append(float(low))
        con_ub.append(float(high))
        row += 1

    # Exactly one free cell per time step.
    for t in range(T):
        add_row([(x_var(t, v), 1.0) for v in free], 1.0, 1.0)

    # Transition-flow coupling: outgoing and incoming allowed moves.
    out_by_tu: Dict[Tuple[int, int], List[int]] = {}
    in_by_tv: Dict[Tuple[int, int], List[int]] = {}
    for (t, u, v), idx in e_var.items():
        out_by_tu.setdefault((t, u), []).append(idx)
        in_by_tv.setdefault((t, v), []).append(idx)

    for t in range(T - 1):
        for u in free:
            add_row([(idx, 1.0) for idx in out_by_tu.get((t, u), [])] + [(x_var(t, u), -1.0)], 0.0, 0.0)
        for v in free:
            add_row([(idx, 1.0) for idx in in_by_tv.get((t, v), [])] + [(x_var(t + 1, v), -1.0)], 0.0, 0.0)

    # Linearize y = x_a * x_b * x_c for the cubic proximity terms.
    # y <= each x; y >= x_a + x_b + x_c - 2.
    for key, _weighted_coeff in prox_terms:
        y = y_var[key]
        for x in key:
            add_row([(y, 1.0), (x, -1.0)], -np.inf, 0.0)
        add_row([(y, 1.0)] + [(x, -1.0) for x in key], -2.0, np.inf)

    A = coo_matrix((data, (rows, cols)), shape=(row, n_vars)).tocsr()
    constraints = LinearConstraint(A, np.array(con_lb), np.array(con_ub))
    bounds = Bounds(lb, ub)
    integrality = np.ones(n_vars, dtype=int)

    options = {"time_limit": float(time_limit_s), "mip_rel_gap": float(mip_rel_gap), "disp": bool(verbose)}
    t0 = time.perf_counter()
    res = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality, options=options)
    runtime = time.perf_counter() - t0

    out = {
        "method": "MILP exact baseline (scipy HiGHS)",
        "status": int(res.status),
        "message": str(res.message),
        "success": bool(res.success),
        "runtime_s": float(runtime),
        "objective_soft_cost": None,
        "native_hubo_energy": None,
        "mip_gap": getattr(res, "mip_gap", None),
        "num_variables": int(n_vars),
        "num_binary_x": int(n_x),
        "num_transition_variables": int(len(trans)),
        "num_proximity_aux_variables": int(len(prox_terms)),
        "num_constraints": int(row),
    }

    if not res.success or res.x is None:
        return out

    xsol = res.x[:n_x]
    traj_idx: List[int] = []
    for t in range(T):
        vals = [(xsol[x_var(t, v)], v) for v in range(V)]
        v_best = max(vals)[1]
        traj_idx.append(int(v_best))
    path = indices_to_cell_path(traj_idx, s)
    metrics = evaluate_path(path, s, grid, hubo)
    out.update({
        "objective_soft_cost": float(res.fun),
        "native_hubo_energy": float(metrics["native_hubo_energy"]),
        "metrics": metrics,
    })
    return out


def solve_exact_cpsat(time_limit_s: float = 300.0, workers: int = 8, verbose: bool = False) -> Dict[str, object]:
    """Solve exact feasible-path problem with OR-Tools CP-SAT, if available."""
    try:
        from ortools.sat.python import cp_model
    except Exception as exc:
        return {"method": "CP-SAT exact baseline (OR-Tools)", "status": "not_available", "error": str(exc)}

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
        model.Add(sum(x[(t, v)] for v in free) == 1)
        for v in s.obstacle_indices:
            model.Add(x[(t, v)] == 0)
    model.Add(x[(0, start_v)] == 1)
    model.Add(x[(T - 1, target_v)] == 1)

    # Allowed transition variables and flow constraints.
    e = {}
    for t in range(T - 1):
        for u in free:
            allowed_vs = [v for v in sorted(s.neighbors(u)) if v in free_set]
            for v in allowed_vs:
                e[(t, u, v)] = model.NewBoolVar(f"e_{t}_{u}_{v}")
            model.Add(sum(e[(t, u, v)] for v in allowed_vs) == x[(t, u)])
        for v in free:
            allowed_us = [u for u in free if v in s.neighbors(u)]
            model.Add(sum(e[(t, u, v)] for u in allowed_us) == x[(t + 1, v)])

    # Scale float costs to integer for CP-SAT.
    scale = 1000
    obj_terms = []
    for t in range(T):
        for v in range(V):
            coeff = int(round(scale * _soft_linear_coeff_for_x(t, v, s, grid)))
            if coeff:
                obj_terms.append(coeff * x[(t, v)])

    prox_count = 0
    for key, coeff in grid.H_prox().items():
        # key contains variable indices t*V+v.
        elems = []
        for idx in key:
            tt, vv = divmod(idx, V)
            elems.append(x[(tt, vv)])
        y = model.NewBoolVar(f"prox_{prox_count}")
        prox_count += 1
        model.AddMultiplicationEquality(y, elems)
        obj_terms.append(int(round(scale * coeff * s.lambda_prox)) * y)

    model.Minimize(sum(obj_terms))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = int(workers)
    solver.parameters.log_search_progress = bool(verbose)

    t0 = time.perf_counter()
    status = solver.Solve(model)
    runtime = time.perf_counter() - t0

    status_name = solver.StatusName(status)
    out = {
        "method": "CP-SAT exact baseline (OR-Tools)",
        "status": status_name,
        "success": status in (cp_model.OPTIMAL, cp_model.FEASIBLE),
        "is_optimal": status == cp_model.OPTIMAL,
        "runtime_s": float(runtime),
        "objective_soft_cost_scaled": float(solver.ObjectiveValue()),
        "objective_soft_cost": float(solver.ObjectiveValue()) / scale if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None,
        "best_objective_bound": float(solver.BestObjectiveBound()) / scale if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None,
        "num_conflicts": int(solver.NumConflicts()),
        "num_branches": int(solver.NumBranches()),
        "num_proximity_aux_variables": int(prox_count),
    }
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return out

    traj_idx = []
    for t in range(T):
        active = [v for v in range(V) if solver.BooleanValue(x[(t, v)])]
        traj_idx.append(active[0])
    path = indices_to_cell_path(traj_idx, s)
    metrics = evaluate_path(path, s, grid, hubo)
    out.update({
        "native_hubo_energy": float(metrics["native_hubo_energy"]),
        "metrics": metrics,
    })
    return out


def save_outputs(results: Dict[str, object]) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "exact_baseline_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    rows = []
    for key, res in results.items():
        if not isinstance(res, dict):
            continue
        metrics = res.get("metrics") or {}
        rows.append({
            "method": res.get("method", key),
            "status": res.get("status", ""),
            "success": res.get("success", ""),
            "runtime_s": res.get("runtime_s", ""),
            "objective_soft_cost": res.get("objective_soft_cost", ""),
            "native_hubo_energy": res.get("native_hubo_energy", ""),
            "feasible": metrics.get("feasible", ""),
            "goal_arrival": metrics.get("goal_arrival", ""),
            "full_success": metrics.get("full_success", ""),
            "path_length": metrics.get("path_length", ""),
            "turns": metrics.get("turns", ""),
            "buffer_steps": metrics.get("buffer_steps", ""),
            "visibility_rate": metrics.get("visibility_rate", ""),
        })
    if rows:
        with open(os.path.join(OUT_DIR, "exact_baseline_summary.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    # Save first available trajectory as CSV and figure.
    s = Scenario()
    first = None
    for res in results.values():
        if isinstance(res, dict) and res.get("metrics", {}).get("trajectory"):
            first = res
            break
    if first:
        traj = first["metrics"]["trajectory"]
        with open(os.path.join(OUT_DIR, "exact_baseline_trajectory.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "row", "col", "occlusion_count", "in_buffer", "obstacle", "at_target"])
            for t, (r, c) in enumerate(traj):
                v = s.cell_index(r, c)
                writer.writerow([t, r, c, s.occlusion_count(v), int(v in s.buffer_indices), int(v in s.obstacle_indices), int((r, c) == s.target)])
        plot_path([tuple(c) for c in traj], s, os.path.join(OUT_DIR, "exact_baseline_path.png"), title=first.get("method", "Exact baseline"))


def plot_path(path: Path, s: Scenario, out_path: str, title: str) -> None:
    if plt is None:
        return
    gs = s.grid_size
    img = np.zeros((gs, gs))
    for v in s.buffer_indices:
        r, c = s.index_to_cell(v)
        img[r, c] = 1
    for r, c in s.obstacles:
        img[r, c] = 2
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap="Reds", origin="upper", vmin=0, vmax=2, alpha=0.55)
    ys = [r for r, _ in path]
    xs = [c for _, c in path]
    ax.plot(xs, ys, "o-", linewidth=2.5, markersize=6, label="Exact baseline path")
    ax.plot(s.start[1], s.start[0], "go", markersize=14, markeredgecolor="black", label="Start")
    ax.plot(s.target[1], s.target[0], "r*", markersize=18, markeredgecolor="black", label="Target")
    p_obs = mpatches.Patch(color=plt.cm.Reds(0.9), alpha=0.6, label="Obstacle")
    p_buf = mpatches.Patch(color=plt.cm.Reds(0.45), alpha=0.6, label="Buffer")
    h, l = ax.get_legend_handles_labels()
    ax.legend(handles=h + [p_obs, p_buf], loc="upper left", fontsize=8)
    ax.set_xticks(range(gs))
    ax.set_yticks(range(gs))
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, gs - 0.5)
    ax.set_ylim(gs - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["milp", "cpsat", "both"], default="milp")
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--mip-gap", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    results = {}
    if args.mode in ("milp", "both"):
        print("Running MILP exact baseline...")
        results["milp"] = solve_exact_milp(time_limit_s=args.time_limit, mip_rel_gap=args.mip_gap, verbose=args.verbose)
        print(json.dumps({k: v for k, v in results["milp"].items() if k != "metrics"}, indent=2))
    if args.mode in ("cpsat", "both"):
        print("Running CP-SAT exact baseline...")
        results["cpsat"] = solve_exact_cpsat(time_limit_s=args.time_limit, workers=args.workers, verbose=args.verbose)
        print(json.dumps({k: v for k, v in results["cpsat"].items() if k != "metrics"}, indent=2))
    save_outputs(results)
    print(f"Saved outputs under: {OUT_DIR}")


if __name__ == "__main__":
    main()
