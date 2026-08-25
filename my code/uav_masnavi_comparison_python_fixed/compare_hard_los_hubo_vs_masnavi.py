"""
Hard-LOS discrete HUBO benchmark vs. Masnavi-style continuous MPC baseline.

This script creates a fair single-platform comparison by running both methods
on the same 2-D obstacle map, same start, same target, same hard LOS definition,
and same trajectory length.

Important modeling note
-----------------------
This is not the ROS/Gazebo runtime from Masnavi's repository. It is a Python
implementation of the mathematical occlusion/collision constraints used by
Masnavi et al. in the 2-D circular-obstacle case:

    p_los(t,u) = (1-u) p_uav(t) + u p_target(t)
    ||p_los(t,u) - p_obstacle||^2 / r_obstacle^2 >= 1

This is Eq. (13)-(14) of the Masnavi paper reduced to 2-D circles. The objective
is the same class of smoothness objective: sum of squared accelerations.

The discrete method treats LOS as a hard feasibility rule by allowing only cells
whose line segment to the target has positive clearance from every obstacle.

Outputs are written to outputs_compare/ by default.

Run:
    python compare_hard_los_hubo_vs_masnavi.py --scenario visible_start --runs 20 --sweeps 400
    python compare_hard_los_hubo_vs_masnavi.py --scenario original --check-only
"""
from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.optimize import minimize

Cell = Tuple[int, int]  # (row, col)
Point = Tuple[float, float]  # (x, y)


@dataclass
class Scenario:
    grid_size: int = 8
    horizon_points: int = 12  # number of trajectory samples, including start and final target
    start: Cell = (3, 0)
    target: Cell = (7, 7)
    obstacles: List[Cell] = field(default_factory=lambda: [
        (3, 3), (3, 4), (4, 3), (4, 4),
        (2, 6), (5, 1),
    ])
    obstacle_radius: float = math.sqrt(2.0) / 2.0  # circumscribed circle for one occupied grid cell
    buffer_distance_cells: float = 1.25

    def cell_index(self, r: int, c: int) -> int:
        return r * self.grid_size + c

    def index_to_cell(self, v: int) -> Cell:
        return divmod(v, self.grid_size)

    def cell_center(self, cell_or_index: Cell | int) -> np.ndarray:
        if isinstance(cell_or_index, int):
            r, c = self.index_to_cell(cell_or_index)
        else:
            r, c = cell_or_index
        return np.array([float(c), float(r)], dtype=float)

    @property
    def start_index(self) -> int:
        return self.cell_index(*self.start)

    @property
    def target_index(self) -> int:
        return self.cell_index(*self.target)

    @property
    def start_xy(self) -> np.ndarray:
        return self.cell_center(self.start)

    @property
    def target_xy(self) -> np.ndarray:
        return self.cell_center(self.target)

    @property
    def obstacle_indices(self) -> Set[int]:
        return {self.cell_index(r, c) for r, c in self.obstacles}

    @property
    def obstacle_centers(self) -> np.ndarray:
        return np.array([self.cell_center(cell) for cell in self.obstacles], dtype=float)

    @property
    def num_cells(self) -> int:
        return self.grid_size * self.grid_size

    def neighbors(self, v: int, allow_hover: bool = True) -> Set[int]:
        r, c = self.index_to_cell(v)
        out: Set[int] = {v} if allow_hover else set()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                out.add(self.cell_index(nr, nc))
        return out

    def point_obstacle_clearance(self, p: np.ndarray) -> float:
        return float(np.min(np.linalg.norm(self.obstacle_centers - p[None, :], axis=1) - self.obstacle_radius))

    def segment_obstacle_clearance(self, p: np.ndarray, q: np.ndarray) -> float:
        """Minimum clearance from segment p--q to any circular obstacle.

        The circle radius is chosen as the circumscribed radius of a grid cell,
        so the continuous optimizer cannot cut through black obstacle cells in
        the figure.
        """
        v = q - p
        l2 = float(v @ v)
        if l2 <= 1e-12:
            return self.point_obstacle_clearance(p)
        best = math.inf
        for c in self.obstacle_centers:
            tau = float(((c - p) @ v) / l2)
            tau = max(0.0, min(1.0, tau))
            closest = p + tau * v
            clearance = float(np.linalg.norm(closest - c) - self.obstacle_radius)
            best = min(best, clearance)
        return best

    def los_clearance(self, p: np.ndarray, target: Optional[np.ndarray] = None) -> float:
        """Exact minimum clearance from segment p--target to any circular obstacle."""
        q = self.target_xy if target is None else target
        return self.segment_obstacle_clearance(p, q)

    def is_point_visible(self, p: np.ndarray, tol: float = 1e-9) -> bool:
        return self.los_clearance(p) >= -tol

    def is_cell_visible(self, v: int) -> bool:
        return self.is_point_visible(self.cell_center(v))

    @property
    def visible_indices(self) -> Set[int]:
        return {v for v in range(self.num_cells) if self.is_cell_visible(v)}

    @property
    def free_indices(self) -> Set[int]:
        out: Set[int] = set()
        for v in range(self.num_cells):
            p = self.cell_center(v)
            if self.point_obstacle_clearance(p) >= -1e-9:
                out.add(v)
        return out - self.obstacle_indices

    @property
    def free_visible_indices(self) -> Set[int]:
        return self.free_indices & self.visible_indices

    @property
    def buffer_indices(self) -> Set[int]:
        out: Set[int] = set()
        for v in self.free_indices:
            p = self.cell_center(v)
            d_to_obs_center = float(np.min(np.linalg.norm(self.obstacle_centers - p[None, :], axis=1)))
            if d_to_obs_center <= self.buffer_distance_cells:
                out.add(v)
        return out

    def describe(self) -> Dict[str, object]:
        return {
            "grid_size": self.grid_size,
            "horizon_points": self.horizon_points,
            "start": self.start,
            "target": self.target,
            "obstacles": self.obstacles,
            "obstacle_radius": self.obstacle_radius,
            "num_cells": self.num_cells,
            "num_obstacles": len(self.obstacles),
            "num_free_cells": len(self.free_indices),
            "num_visible_cells": len(self.visible_indices),
            "num_free_visible_cells": len(self.free_visible_indices),
            "start_visible": self.is_cell_visible(self.start_index),
            "target_visible": self.is_cell_visible(self.target_index),
            "start_los_clearance": self.los_clearance(self.start_xy),
        }


def build_scenario(name: str) -> Scenario:
    if name == "original":
        # Original paper grid start/target: infeasible under hard per-step LOS because start is occluded.
        return Scenario(start=(0, 0), target=(7, 7), horizon_points=15)
    if name == "visible_start":
        # Same map/target, but start is moved to a visible cell for a valid hard-LOS comparison under full-cell/circumscribed obstacles.
        return Scenario(start=(6, 0), target=(7, 7), horizon_points=16)
    raise ValueError(name)


def astar_hard_los(s: Scenario) -> Optional[List[int]]:
    allowed = s.free_visible_indices
    if s.start_index not in allowed or s.target_index not in allowed:
        return None

    def h(v: int) -> float:
        r, c = s.index_to_cell(v)
        rt, ct = s.target
        return abs(r - rt) + abs(c - ct)

    pq: List[Tuple[float, int]] = [(h(s.start_index), s.start_index)]
    g: Dict[int, float] = {s.start_index: 0.0}
    parent: Dict[int, int] = {}
    while pq:
        _, u = heapq.heappop(pq)
        if u == s.target_index:
            path = [u]
            while path[-1] != s.start_index:
                path.append(parent[path[-1]])
            path.reverse()
            if len(path) <= s.horizon_points:
                path = path + [s.target_index] * (s.horizon_points - len(path))
            return path
        for v in s.neighbors(u, allow_hover=False) & allowed:
            step_cost = 1.0 + (1.0 if v in s.buffer_indices else 0.0)
            ng = g[u] + step_cost
            if ng < g.get(v, math.inf):
                g[v] = ng
                parent[v] = u
                heapq.heappush(pq, (ng + h(v), v))
    return None


def trajectory_points_from_cells(s: Scenario, traj: Sequence[int]) -> np.ndarray:
    return np.vstack([s.cell_center(v) for v in traj])


def trajectory_metrics(s: Scenario, points: np.ndarray, runtime_s: float, name: str) -> Dict[str, object]:
    diffs = np.diff(points, axis=0)
    path_length = float(np.sum(np.linalg.norm(diffs, axis=1)))
    acc = points[2:] - 2 * points[1:-1] + points[:-2]
    smoothness = float(np.sum(np.sum(acc * acc, axis=1))) if len(points) >= 3 else 0.0
    point_clearances = np.array([s.point_obstacle_clearance(p) for p in points])
    los_clearances = np.array([s.los_clearance(p) for p in points])
    motion_clearances = np.array([s.segment_obstacle_clearance(points[i], points[i + 1]) for i in range(len(points) - 1)]) if len(points) >= 2 else point_clearances
    visible = los_clearances >= -1e-7
    collision_free = point_clearances >= -1e-7
    motion_collision_free = motion_clearances >= -1e-7
    terminal_error = float(np.linalg.norm(points[-1] - s.target_xy))
    start_error = float(np.linalg.norm(points[0] - s.start_xy))
    return {
        "method": name,
        "runtime_s": float(runtime_s),
        "path_length": path_length,
        "smoothness_sum_squared_acceleration": smoothness,
        "min_point_obstacle_clearance": float(point_clearances.min()),
        "min_motion_obstacle_clearance": float(motion_clearances.min()),
        "min_los_clearance": float(los_clearances.min()),
        "visibility_percent": float(np.mean(visible) * 100.0),
        "collision_free_percent": float(np.mean(collision_free) * 100.0),
        "motion_collision_free_percent": float(np.mean(motion_collision_free) * 100.0),
        "start_error": start_error,
        "terminal_error": terminal_error,
        "hard_feasible": bool(visible.all() and collision_free.all() and motion_collision_free.all() and start_error <= 1e-6 and terminal_error <= 1e-6),
    }


def discrete_energy(s: Scenario, traj: Sequence[int]) -> float:
    """A simple trajectory-level energy used by the SA baseline."""
    total = 0.0
    pts = trajectory_points_from_cells(s, traj)
    total += float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    # Soft proximity/risk term only; LOS and obstacle are hard through allowed moves.
    B = s.buffer_indices
    for t in range(1, len(traj) - 1):
        if traj[t - 1] in B and traj[t] in B and traj[t + 1] in B:
            total += 5.0
    return total


def trajectory_sa_hard_los(s: Scenario, init_path: Sequence[int], runs: int, sweeps: int, seed: int) -> Dict[str, object]:
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    allowed = s.free_visible_indices
    N = s.horizon_points
    best_global = list(init_path)
    best_global_e = discrete_energy(s, best_global)
    history: List[float] = []
    acc_count = 0
    eval_count = 0

    for _ in range(runs):
        traj = list(init_path)
        e = discrete_energy(s, traj)
        best = list(traj)
        best_e = e
        T_start, T_end = 10.0, 0.01
        cooling = (T_end / T_start) ** (1.0 / max(1, sweeps))
        temp = T_start
        for _sw in range(sweeps):
            for t in rng.permutation(np.arange(1, N - 1)):
                prev_v = traj[t - 1]
                next_v = traj[t + 1]
                cands = s.neighbors(prev_v) & s.neighbors(next_v) & allowed
                if not cands:
                    continue
                new_v = int(rng.choice(list(cands)))
                if new_v == traj[t]:
                    continue
                old = traj[t]
                before = discrete_energy(s, traj)
                traj[t] = new_v
                after = discrete_energy(s, traj)
                delta = after - before
                eval_count += 1
                if delta <= 0 or rng.random() < math.exp(-delta / max(temp, 1e-9)):
                    e = after
                    acc_count += 1
                    if e < best_e:
                        best_e = e
                        best = list(traj)
                else:
                    traj[t] = old
            history.append(e)
            temp *= cooling
        if best_e < best_global_e:
            best_global_e = best_e
            best_global = best

    return {
        "trajectory": best_global,
        "energy": best_global_e,
        "runtime_s": time.perf_counter() - start,
        "acceptance_rate": acc_count / max(1, eval_count),
        "runs": runs,
        "sweeps": sweeps,
        "history": history,
    }


def masnavi_style_continuous_optimize(
    s: Scenario,
    init_points: np.ndarray,
    los_samples: int = 31,
    maxiter: int = 800,
) -> Dict[str, object]:
    """Continuous 2-D Masnavi-style constrained smoothness optimizer.

    Variables are the interior trajectory points. Start and final target are hard.
    The LOS/collision constraints use the sampled LOS ellipsoid/circle condition
    from Eq. (13)-(14) of Masnavi et al., reduced to 2-D circles.
    """
    start = time.perf_counter()
    N = s.horizon_points
    if init_points.shape != (N, 2):
        raise ValueError(f"init_points must have shape {(N, 2)}")

    centers = s.obstacle_centers
    r = s.obstacle_radius
    target = s.target_xy
    bounds = [(-0.25, s.grid_size - 0.75), (-0.25, s.grid_size - 0.75)] * (N - 2)

    def unpack(z: np.ndarray) -> np.ndarray:
        pts = np.zeros((N, 2), dtype=float)
        pts[0] = s.start_xy
        pts[-1] = s.target_xy
        pts[1:-1] = z.reshape(N - 2, 2)
        return pts

    def objective(z: np.ndarray) -> float:
        pts = unpack(z)
        acc = pts[2:] - 2.0 * pts[1:-1] + pts[:-2]
        # Acceleration smoothness is the Masnavi-style core objective. A tiny
        # length regularizer stabilizes the nonlinear solve without changing the
        # hard LOS/collision constraints.
        length = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
        return float(np.sum(acc * acc) + 1e-3 * length)

    def segment_circle_margins(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        v = q - p
        l2 = float(v @ v)
        if l2 <= 1e-12:
            diff = p[None, :] - centers
            d2 = np.sum(diff * diff, axis=1)
            return d2 / (r * r) - 1.0
        taus = ((centers - p[None, :]) @ v) / l2
        taus = np.clip(taus, 0.0, 1.0)
        closest = p[None, :] + taus[:, None] * v[None, :]
        diff = closest - centers
        d2 = np.sum(diff * diff, axis=1)
        return d2 / (r * r) - 1.0

    def constraint_margins(z: np.ndarray) -> np.ndarray:
        pts = unpack(z)
        margins: List[float] = []
        # Exact segment-to-circle version of the Masnavi LOS/collision rule
        # in 2-D: every LOS segment p(t)--target must remain outside obstacles.
        for p in pts:
            margins.extend(segment_circle_margins(p, target).tolist())
        # Also enforce that the displayed motion segment between consecutive
        # samples cannot pass through an obstacle.
        for i in range(len(pts) - 1):
            margins.extend(segment_circle_margins(pts[i], pts[i + 1]).tolist())
        return np.array(margins, dtype=float)

    z0 = init_points[1:-1].reshape(-1).copy()
    # Make sure the initialization is inside the variable bounds.
    for i, (lo, hi) in enumerate(bounds):
        z0[i] = min(max(z0[i], lo + 1e-5), hi - 1e-5)

    res = minimize(
        objective,
        z0,
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "ineq", "fun": constraint_margins}],
        options={"maxiter": maxiter, "ftol": 1e-8, "disp": False},
    )

    pts = unpack(res.x)
    runtime = time.perf_counter() - start
    return {
        "points": pts,
        "runtime_s": runtime,
        "success": bool(res.success),
        "message": str(res.message),
        "objective": float(res.fun),
        "nit": int(getattr(res, "nit", -1)),
        "min_sampled_constraint_margin": float(np.min(constraint_margins(res.x))),
        "los_samples": los_samples,
    }


def plot_comparison(s: Scenario, paths: Dict[str, np.ndarray], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    visible = s.visible_indices
    free = s.free_indices
    buffer = s.buffer_indices
    for r in range(s.grid_size):
        for c in range(s.grid_size):
            v = s.cell_index(r, c)
            if v in s.obstacle_indices:
                face, alpha = "black", 0.85
            elif v not in free:
                face, alpha = "gray", 0.45
            elif v not in visible:
                face, alpha = "lightcoral", 0.55
            elif v in buffer:
                face, alpha = "khaki", 0.50
            else:
                face, alpha = "white", 1.0
            ax.add_patch(patches.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                           facecolor=face, edgecolor="gray", alpha=alpha))
    # obstacle circles used by continuous optimizer
    for c in s.obstacle_centers:
        ax.add_patch(patches.Circle((c[0], c[1]), s.obstacle_radius, fill=False, linewidth=2.0))

    for name, pts in paths.items():
        ax.plot(pts[:, 0], pts[:, 1], "o-", linewidth=2.2, markersize=4.5, label=name)

    ax.plot(s.start_xy[0], s.start_xy[1], "go", markersize=13, markeredgecolor="black", label="Start")
    ax.plot(s.target_xy[0], s.target_xy[1], "r*", markersize=17, markeredgecolor="black", label="Target")
    ax.set_xlim(-0.5, s.grid_size - 0.5)
    ax.set_ylim(s.grid_size - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(s.grid_size))
    ax.set_yticks(range(s.grid_size))
    ax.grid(True, alpha=0.25)
    ax.set_title("Hard-LOS comparison on shared map")
    legend_patches = [
        patches.Patch(facecolor="black", alpha=0.85, label="Obstacle cell"),
        patches.Patch(facecolor="lightcoral", alpha=0.55, label="LOS-forbidden cell"),
        patches.Patch(facecolor="khaki", alpha=0.50, label="Buffer cell"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + legend_patches, loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_metrics(metrics: List[Dict[str, object]], out_path: str) -> None:
    names = [str(m["method"]) for m in metrics]
    keys = ["runtime_s", "path_length", "smoothness_sum_squared_acceleration", "min_los_clearance"]
    titles = ["Runtime (s)", "Path length", "Smoothness: sum ||acc||²", "Minimum LOS clearance"]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.0))
    for ax, key, title in zip(axes.flatten(), keys, titles):
        vals = [float(m[key]) for m in metrics]
        ax.bar(names, vals)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_outputs(out_dir: str, s: Scenario, metrics: List[Dict[str, object]], paths: Dict[str, np.ndarray], extra: Dict[str, object]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    plot_comparison(s, paths, os.path.join(out_dir, "comparison_paths.png"))
    plot_metrics(metrics, os.path.join(out_dir, "comparison_metrics.png"))
    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as f:
        if metrics:
            writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
            writer.writeheader()
            for row in metrics:
                writer.writerow(row)
    for name, pts in paths.items():
        safe = name.lower().replace(" ", "_").replace("*", "astar").replace("/", "_")
        with open(os.path.join(out_dir, f"{safe}_trajectory.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "x", "y", "los_clearance", "point_clearance"])
            for t, p in enumerate(pts):
                writer.writerow([t, p[0], p[1], s.los_clearance(p), s.point_obstacle_clearance(p)])
    with open(os.path.join(out_dir, "comparison_results.json"), "w") as f:
        json.dump({"scenario": s.describe(), "metrics": metrics, "extra": extra}, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["visible_start", "original"], default="visible_start")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--sweeps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", default="outputs_compare")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    s = build_scenario(args.scenario)
    print("Scenario:")
    print(json.dumps(s.describe(), indent=2))

    astar_path = astar_hard_los(s)
    if astar_path is None:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "infeasible_reason.json"), "w") as f:
            json.dump({"scenario": s.describe(), "reason": "No hard-LOS path found. Start may be occluded or target unreachable."}, f, indent=2)
        print("No hard-LOS path exists for this scenario. Results written to infeasible_reason.json")
        return
    if args.check_only:
        print("A* hard-LOS path exists:", [s.index_to_cell(v) for v in astar_path])
        return

    astar_pts = trajectory_points_from_cells(s, astar_path)
    t0 = time.perf_counter()
    # A* runtime for metrics; run again just for fair timing.
    _ = astar_hard_los(s)
    astar_runtime = time.perf_counter() - t0

    sa = trajectory_sa_hard_los(s, astar_path, runs=args.runs, sweeps=args.sweeps, seed=args.seed)
    sa_pts = trajectory_points_from_cells(s, sa["trajectory"])

    cont = masnavi_style_continuous_optimize(s, astar_pts, los_samples=0, maxiter=600)
    cont_pts = cont["points"]

    metrics = [
        trajectory_metrics(s, astar_pts, astar_runtime, "A* hard-LOS"),
        trajectory_metrics(s, sa_pts, float(sa["runtime_s"]), "SA hard-LOS HUBO"),
        trajectory_metrics(s, cont_pts, float(cont["runtime_s"]), "Masnavi-style continuous"),
    ]

    paths = {
        "A* hard-LOS": astar_pts,
        "SA hard-LOS HUBO": sa_pts,
        "Masnavi-style continuous": cont_pts,
    }
    extra = {
        "sa": {k: v for k, v in sa.items() if k not in {"trajectory", "history"}},
        "masnavi_style_continuous": {k: v for k, v in cont.items() if k != "points"},
        "modeling_note": "Continuous baseline implements Masnavi Eq. (13)-(14) in 2-D circles and minimizes acceleration smoothness with hard start/terminal constraints.",
    }
    write_outputs(args.out_dir, s, metrics, paths, extra)
    print("Metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"Outputs written to {args.out_dir}")


if __name__ == "__main__":
    main()
