"""
Soft-occlusion discrete HUBO-style benchmark vs. Masnavi-style continuous baseline
on a Masnavi repository scenario.

This script is meant to replace the earlier hard-LOS comparison. Here:
  * obstacle/collision avoidance remains a hard feasibility constraint;
  * line-of-sight (LOS) occlusion is a soft cost/penalty;
  * feasibility-aware evaluation checks only structural hard constraints;
  * visibility is reported separately as a quality metric.

Scenario used here: `masnavi_static_3cyl`, derived from the uploaded Masnavi
repository files:
  target_tracker/launch/3_cylinder_0.35_world.launch
  target_tracker/params/3_obs_params.yaml

Numerical settings:
  workspace: x in [-4, 4], y in [-3, 3] from 3_obs_params.yaml
  UAV start: (-2, 0) from 3_obs_params.yaml and launch file
  terminal/reference point: (3, 1) from 3_obs_params.yaml
  observed target: (5, 0) from 3_cylinder_0.35_world.launch
  obstacles: (1,-2.5), (-0.5,1.5), (1,1), radius 0.35 from the launch file name/URDF

Important modeling note:
This is not the full ROS/Gazebo implementation. It is a common Python platform
for comparing a discrete soft-occlusion planner against a continuous
Masnavi-style smoothness optimizer. The continuous baseline keeps collision
avoidance hard but moves LOS occlusion into the objective as a soft penalty.
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
from typing import Dict, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.optimize import minimize

Point = Tuple[float, float]


@dataclass
class Scenario:
    name: str = "masnavi_static_3cyl"
    x_min: float = -4.0
    x_max: float = 4.0
    y_min: float = -3.0
    y_max: float = 3.0
    grid_step: float = 0.5
    horizon_points: int = 16
    start_xy_tuple: Point = (-4.0, 0.0)
    goal_xy_tuple: Point = (5.0, 1.0)
    target_xy_tuple: Point = (5.0, 1.0)
    obstacle_centers_list: List[Point] = field(default_factory=lambda: [(1.0, -2.5), (-0.5, 1.5), (1.0, 1.0)])
    obstacle_radius: float = 0.35
    buffer_distance: float = 0.70
    lambda_occ: float = 8.0
    lambda_buffer: float = 0.35

    def __post_init__(self) -> None:
        self.x_values = np.arange(self.x_min, self.x_max + 0.5 * self.grid_step, self.grid_step)
        self.y_values = np.arange(self.y_min, self.y_max + 0.5 * self.grid_step, self.grid_step)
        self.nx = len(self.x_values)
        self.ny = len(self.y_values)

    @property
    def start_xy(self) -> np.ndarray:
        return np.array(self.start_xy_tuple, dtype=float)

    @property
    def goal_xy(self) -> np.ndarray:
        return np.array(self.goal_xy_tuple, dtype=float)

    @property
    def target_xy(self) -> np.ndarray:
        return np.array(self.target_xy_tuple, dtype=float)

    @property
    def obstacle_centers(self) -> np.ndarray:
        return np.array(self.obstacle_centers_list, dtype=float)

    @property
    def num_nodes(self) -> int:
        return self.nx * self.ny

    def node_index(self, iy: int, ix: int) -> int:
        return iy * self.nx + ix

    def index_to_grid(self, v: int) -> Tuple[int, int]:
        iy, ix = divmod(v, self.nx)
        return iy, ix

    def index_to_point(self, v: int) -> np.ndarray:
        iy, ix = self.index_to_grid(v)
        return np.array([self.x_values[ix], self.y_values[iy]], dtype=float)

    def nearest_index(self, p: np.ndarray) -> int:
        ix = int(np.argmin(np.abs(self.x_values - p[0])))
        iy = int(np.argmin(np.abs(self.y_values - p[1])))
        return self.node_index(iy, ix)

    @property
    def start_index(self) -> int:
        return self.nearest_index(self.start_xy)

    @property
    def goal_index(self) -> int:
        return self.nearest_index(self.goal_xy)

    def neighbors(self, v: int, allow_diag: bool = True, allow_hover: bool = True) -> Set[int]:
        iy, ix = self.index_to_grid(v)
        out: Set[int] = {v} if allow_hover else set()
        steps = [(-1,0),(1,0),(0,-1),(0,1)]
        if allow_diag:
            steps += [(-1,-1),(-1,1),(1,-1),(1,1)]
        for dy, dx in steps:
            ny, nx = iy + dy, ix + dx
            if 0 <= ny < self.ny and 0 <= nx < self.nx:
                out.add(self.node_index(ny, nx))
        return out

    def point_obstacle_clearances_all(self, p: np.ndarray) -> np.ndarray:
        return np.linalg.norm(self.obstacle_centers - p[None, :], axis=1) - self.obstacle_radius

    def point_obstacle_clearance(self, p: np.ndarray) -> float:
        return float(np.min(self.point_obstacle_clearances_all(p)))

    def segment_obstacle_clearances_all(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        v = q - p
        l2 = float(v @ v)
        if l2 <= 1e-12:
            return self.point_obstacle_clearances_all(p)
        c = self.obstacle_centers
        tau = ((c - p[None, :]) @ v) / l2
        tau = np.clip(tau, 0.0, 1.0)
        closest = p[None, :] + tau[:, None] * v[None, :]
        return np.linalg.norm(closest - c, axis=1) - self.obstacle_radius

    def segment_obstacle_clearance(self, p: np.ndarray, q: np.ndarray) -> float:
        return float(np.min(self.segment_obstacle_clearances_all(p, q)))

    def los_clearances_all(self, p: np.ndarray) -> np.ndarray:
        return self.segment_obstacle_clearances_all(p, self.target_xy)

    def los_clearance(self, p: np.ndarray) -> float:
        return float(np.min(self.los_clearances_all(p)))

    def soft_occ_cost(self, p: np.ndarray) -> float:
        # Sum overlap/violation magnitudes. Zero if LOS segment clears all obstacles.
        return float(np.sum(np.maximum(0.0, -self.los_clearances_all(p))))

    @property
    def free_indices(self) -> Set[int]:
        out: Set[int] = set()
        for v in range(self.num_nodes):
            if self.point_obstacle_clearance(self.index_to_point(v)) >= -1e-9:
                out.add(v)
        return out

    @property
    def buffer_indices(self) -> Set[int]:
        out: Set[int] = set()
        for v in self.free_indices:
            p = self.index_to_point(v)
            if self.point_obstacle_clearance(p) <= self.buffer_distance:
                out.add(v)
        return out

    def describe(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "workspace": [self.x_min, self.x_max, self.y_min, self.y_max],
            "grid_step": self.grid_step,
            "nx": self.nx,
            "ny": self.ny,
            "num_grid_nodes": self.num_nodes,
            "horizon_points": self.horizon_points,
            "start_xy": self.start_xy_tuple,
            "goal_xy_terminal_reference": self.goal_xy_tuple,
            "observed_target_xy_for_LOS": self.target_xy_tuple,
            "obstacle_centers": self.obstacle_centers_list,
            "obstacle_radius": self.obstacle_radius,
            "num_free_nodes": len(self.free_indices),
            "start_los_clearance": self.los_clearance(self.start_xy),
            "goal_los_clearance": self.los_clearance(self.goal_xy),
            "start_occ_cost": self.soft_occ_cost(self.start_xy),
            "goal_occ_cost": self.soft_occ_cost(self.goal_xy),
            "lambda_occ": self.lambda_occ,
            "lambda_buffer": self.lambda_buffer,
        }


def build_scenario(name: str, grid_step: float, lambda_occ: float) -> Scenario:
    if name != "masnavi_static_3cyl":
        raise ValueError(name)
    return Scenario(grid_step=grid_step, lambda_occ=lambda_occ)


def astar_soft_occ(s: Scenario) -> Optional[List[int]]:
    allowed = s.free_indices
    if s.start_index not in allowed or s.goal_index not in allowed:
        return None

    def h(v: int) -> float:
        p = s.index_to_point(v)
        return float(np.linalg.norm(p - s.goal_xy))

    pq: List[Tuple[float, int]] = [(h(s.start_index), s.start_index)]
    g: Dict[int, float] = {s.start_index: 0.0}
    parent: Dict[int, int] = {}
    closed: Set[int] = set()
    while pq:
        _, u = heapq.heappop(pq)
        if u in closed:
            continue
        closed.add(u)
        if u == s.goal_index:
            path = [u]
            while path[-1] != s.start_index:
                path.append(parent[path[-1]])
            path.reverse()
            if len(path) <= s.horizon_points:
                path = path + [s.goal_index] * (s.horizon_points - len(path))
            elif len(path) > s.horizon_points:
                # Keep the full path for geometric evaluation; SA will be initialized with full path as well.
                pass
            return path
        for v in s.neighbors(u, allow_diag=True, allow_hover=False) & allowed:
            p_u = s.index_to_point(u)
            p_v = s.index_to_point(v)
            if s.segment_obstacle_clearance(p_u, p_v) < -1e-9:
                continue
            step_len = float(np.linalg.norm(p_v - p_u))
            occ = s.soft_occ_cost(p_v)
            buffer_cost = s.lambda_buffer if v in s.buffer_indices else 0.0
            ng = g[u] + step_len + s.lambda_occ * occ + buffer_cost
            if ng < g.get(v, math.inf):
                g[v] = ng
                parent[v] = u
                heapq.heappush(pq, (ng + h(v), v))
    return None


def trajectory_points_from_nodes(s: Scenario, traj: Sequence[int]) -> np.ndarray:
    return np.vstack([s.index_to_point(v) for v in traj])


def resample_points(points: np.ndarray, N: int) -> np.ndarray:
    if len(points) == N:
        return points.copy()
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    if dist[-1] <= 1e-12:
        return np.repeat(points[:1], N, axis=0)
    new_d = np.linspace(0.0, dist[-1], N)
    out = np.zeros((N, 2))
    out[:, 0] = np.interp(new_d, dist, points[:, 0])
    out[:, 1] = np.interp(new_d, dist, points[:, 1])
    return out


def discrete_energy(s: Scenario, traj: Sequence[int]) -> float:
    pts = trajectory_points_from_nodes(s, traj)
    length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    occ = float(np.sum([s.soft_occ_cost(p) for p in pts]))
    buffer_pen = 0.0
    B = s.buffer_indices
    for t in range(1, len(traj) - 1):
        if traj[t - 1] in B and traj[t] in B and traj[t + 1] in B:
            buffer_pen += 1.0
    # Hard obstacle/motion infeasibility is rejected by move generator, not added here.
    return length + s.lambda_occ * occ + 5.0 * buffer_pen


def trajectory_sa_soft_occ(s: Scenario, init_path: Sequence[int], runs: int, sweeps: int, seed: int) -> Dict[str, object]:
    start_t = time.perf_counter()
    rng = np.random.default_rng(seed)
    allowed = s.free_indices
    best_global = list(init_path)
    best_global_e = discrete_energy(s, best_global)
    accept = 0
    evals = 0
    history: List[float] = []

    for _ in range(runs):
        traj = list(init_path)
        e = discrete_energy(s, traj)
        best = list(traj)
        best_e = e
        temp = 5.0
        cooling = (0.01 / temp) ** (1.0 / max(1, sweeps))
        for _sw in range(sweeps):
            if len(traj) <= 2:
                break
            for t in rng.permutation(np.arange(1, len(traj) - 1)):
                prev_v = traj[t - 1]
                next_v = traj[t + 1]
                cands = (s.neighbors(prev_v, allow_diag=True) & s.neighbors(next_v, allow_diag=True) & allowed)
                # Reject candidates that make the displayed motion collide with obstacles.
                cands = {v for v in cands if s.segment_obstacle_clearance(s.index_to_point(prev_v), s.index_to_point(v)) >= -1e-9
                         and s.segment_obstacle_clearance(s.index_to_point(v), s.index_to_point(next_v)) >= -1e-9}
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
                evals += 1
                if delta <= 0 or rng.random() < math.exp(-delta / max(temp, 1e-12)):
                    e = after
                    accept += 1
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
        "energy": float(best_global_e),
        "runtime_s": time.perf_counter() - start_t,
        "acceptance_rate": accept / max(1, evals),
        "runs": runs,
        "sweeps": sweeps,
        "history": history,
    }


def masnavi_style_continuous_soft_occ(s: Scenario, init_points: np.ndarray, occ_weight: float, maxiter: int = 800) -> Dict[str, object]:
    start_t = time.perf_counter()
    N = s.horizon_points
    init_points = resample_points(init_points, N)
    centers = s.obstacle_centers
    r = s.obstacle_radius
    target = s.target_xy
    bounds = [(s.x_min, s.x_max), (s.y_min, s.y_max)] * (N - 2)

    def unpack(z: np.ndarray) -> np.ndarray:
        pts = np.zeros((N, 2), dtype=float)
        pts[0] = s.start_xy
        pts[-1] = s.goal_xy
        pts[1:-1] = z.reshape(N - 2, 2)
        return pts

    def objective(z: np.ndarray) -> float:
        pts = unpack(z)
        acc = pts[2:] - 2.0 * pts[1:-1] + pts[:-2]
        smooth = float(np.sum(acc * acc))
        length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        occ = float(np.sum([s.soft_occ_cost(p) for p in pts]))
        # Masnavi-style smoothness objective plus soft LOS/occlusion penalty.
        return smooth + 0.05 * length + occ_weight * occ

    def segment_margins(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        v = q - p
        l2 = float(v @ v)
        if l2 <= 1e-12:
            d = np.linalg.norm(centers - p[None, :], axis=1) - r
            return d
        taus = ((centers - p[None, :]) @ v) / l2
        taus = np.clip(taus, 0.0, 1.0)
        closest = p[None, :] + taus[:, None] * v[None, :]
        d = np.linalg.norm(closest - centers, axis=1) - r
        return d

    def hard_collision_constraints(z: np.ndarray) -> np.ndarray:
        pts = unpack(z)
        margins: List[float] = []
        # point collision
        for p in pts:
            margins.extend((np.linalg.norm(centers - p[None, :], axis=1) - r).tolist())
        # continuous motion segment collision
        for i in range(len(pts) - 1):
            margins.extend(segment_margins(pts[i], pts[i + 1]).tolist())
        return np.array(margins, dtype=float)

    z0 = init_points[1:-1].reshape(-1).copy()
    for i, (lo, hi) in enumerate(bounds):
        z0[i] = min(max(z0[i], lo + 1e-5), hi - 1e-5)

    res = minimize(
        objective,
        z0,
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "ineq", "fun": hard_collision_constraints}],
        options={"maxiter": maxiter, "ftol": 1e-8, "disp": False},
    )
    pts = unpack(res.x)
    return {
        "points": pts,
        "runtime_s": time.perf_counter() - start_t,
        "success": bool(res.success),
        "message": str(res.message),
        "objective": float(res.fun),
        "nit": int(getattr(res, "nit", -1)),
        "min_collision_margin": float(np.min(hard_collision_constraints(res.x))),
        "occ_weight": occ_weight,
    }


def trajectory_metrics(s: Scenario, points: np.ndarray, runtime_s: float, name: str) -> Dict[str, object]:
    diffs = np.diff(points, axis=0)
    path_length = float(np.sum(np.linalg.norm(diffs, axis=1)))
    acc = points[2:] - 2.0 * points[1:-1] + points[:-2]
    smoothness = float(np.sum(acc * acc)) if len(points) >= 3 else 0.0
    point_clear = np.array([s.point_obstacle_clearance(p) for p in points])
    motion_clear = np.array([s.segment_obstacle_clearance(points[i], points[i + 1]) for i in range(len(points) - 1)]) if len(points) >= 2 else point_clear
    los_clear = np.array([s.los_clearance(p) for p in points])
    occ_cost = np.array([s.soft_occ_cost(p) for p in points])
    start_error = float(np.linalg.norm(points[0] - s.start_xy))
    terminal_error = float(np.linalg.norm(points[-1] - s.goal_xy))
    structural_feasible = bool(point_clear.min() >= -1e-7 and motion_clear.min() >= -1e-7 and start_error <= 1e-6 and terminal_error <= 1e-6)
    return {
        "method": name,
        "runtime_s": float(runtime_s),
        "path_length": path_length,
        "smoothness_sum_squared_acceleration": smoothness,
        "min_point_obstacle_clearance": float(point_clear.min()),
        "min_motion_obstacle_clearance": float(motion_clear.min()),
        "min_los_clearance": float(los_clear.min()),
        "visibility_percent": float(np.mean(los_clear >= -1e-7) * 100.0),
        "total_soft_occlusion_cost": float(np.sum(occ_cost)),
        "start_error": start_error,
        "terminal_error": terminal_error,
        "structural_feasible": structural_feasible,
    }


def plot_comparison(s: Scenario, paths: Dict[str, np.ndarray], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    # Plot background soft occlusion cost on the discrete grid.
    xx, yy = np.meshgrid(s.x_values, s.y_values)
    occ = np.zeros_like(xx, dtype=float)
    for iy in range(s.ny):
        for ix in range(s.nx):
            occ[iy, ix] = s.soft_occ_cost(np.array([xx[iy, ix], yy[iy, ix]]))
    im = ax.imshow(occ, extent=[s.x_min, s.x_max, s.y_max, s.y_min], alpha=0.32, cmap="Reds")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("soft LOS occlusion cost")

    # Plot grid points lightly.
    for x in s.x_values:
        ax.axvline(x, color="0.85", linewidth=0.5, zorder=0)
    for y in s.y_values:
        ax.axhline(y, color="0.85", linewidth=0.5, zorder=0)

    # Obstacles.
    for c in s.obstacle_centers:
        ax.add_patch(patches.Circle((c[0], c[1]), s.obstacle_radius, color="black", alpha=0.85, zorder=5))
        ax.add_patch(patches.Circle((c[0], c[1]), s.buffer_distance, fill=False, edgecolor="gray", linestyle="--", alpha=0.45, zorder=4))

    # Target LOS point and terminal goal.
    ax.plot(s.target_xy[0], s.target_xy[1], "r*", markersize=17, markeredgecolor="black", label="Observed target for LOS", zorder=10)
    ax.plot(s.goal_xy[0], s.goal_xy[1], "D", markersize=9, markeredgecolor="black", label="Terminal/reference point", zorder=10)
    ax.plot(s.start_xy[0], s.start_xy[1], "go", markersize=12, markeredgecolor="black", label="Start", zorder=10)

    for name, pts in paths.items():
        ax.plot(pts[:, 0], pts[:, 1], "o-", linewidth=2.2, markersize=4.2, label=name, zorder=8)
        # show LOS segments at a few points to make soft occlusion interpretation visible
        for idx in np.linspace(0, len(pts) - 1, min(5, len(pts)), dtype=int):
            p = pts[idx]
            ax.plot([p[0], s.target_xy[0]], [p[1], s.target_xy[1]], color="gray", alpha=0.12, linewidth=1.0, zorder=1)

    ax.set_xlim(s.x_min - 0.2, max(s.x_max, s.target_xy[0]) + 0.5)
    ax.set_ylim(s.y_min - 0.2, s.y_max + 0.2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Soft-occlusion comparison on Masnavi repository scenario")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_metrics(metrics: List[Dict[str, object]], out_path: str) -> None:
    names = [str(m["method"]) for m in metrics]
    keys = ["runtime_s", "path_length", "smoothness_sum_squared_acceleration", "total_soft_occlusion_cost", "min_los_clearance", "visibility_percent"]
    titles = ["Runtime (s)", "Path length", "Smoothness: sum ||acc||²", "Soft occlusion cost", "Minimum LOS clearance", "Visibility (%)"]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.0))
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
        writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        for row in metrics:
            writer.writerow(row)
    for name, pts in paths.items():
        safe = name.lower().replace(" ", "_").replace("*", "astar").replace("/", "_")
        with open(os.path.join(out_dir, f"{safe}_trajectory.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "x", "y", "los_clearance", "soft_occ_cost", "point_obstacle_clearance"])
            for t, p in enumerate(pts):
                writer.writerow([t, p[0], p[1], s.los_clearance(p), s.soft_occ_cost(p), s.point_obstacle_clearance(p)])
    with open(os.path.join(out_dir, "comparison_results.json"), "w") as f:
        json.dump({"scenario": s.describe(), "metrics": metrics, "extra": extra}, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["masnavi_static_3cyl"], default="masnavi_static_3cyl")
    parser.add_argument("--grid-step", type=float, default=0.5)
    parser.add_argument("--lambda-occ", type=float, default=8.0)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--sweeps", type=int, default=350)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out-dir", default="outputs_soft_occ")
    args = parser.parse_args()

    s = build_scenario(args.scenario, grid_step=args.grid_step, lambda_occ=args.lambda_occ)
    print("Scenario:")
    print(json.dumps(s.describe(), indent=2))

    t0 = time.perf_counter()
    astar_path = astar_soft_occ(s)
    astar_runtime = time.perf_counter() - t0
    if astar_path is None:
        raise RuntimeError("No collision-free path found in the Masnavi-derived discrete grid.")
    astar_pts_raw = trajectory_points_from_nodes(s, astar_path)
    astar_pts = resample_points(astar_pts_raw, s.horizon_points)

    # For SA, use nearest grid-node version; keep full A* node path, so SA can refine but structural endpoints are fixed.
    sa = trajectory_sa_soft_occ(s, astar_path, runs=args.runs, sweeps=args.sweeps, seed=args.seed)
    sa_pts_raw = trajectory_points_from_nodes(s, sa["trajectory"])
    sa_pts = resample_points(sa_pts_raw, s.horizon_points)

    cont = masnavi_style_continuous_soft_occ(s, astar_pts, occ_weight=s.lambda_occ, maxiter=800)
    cont_pts = cont["points"]

    metrics = [
        trajectory_metrics(s, astar_pts, astar_runtime, "A* soft-LOS"),
        trajectory_metrics(s, sa_pts, float(sa["runtime_s"]), "SA soft-LOS HUBO"),
        trajectory_metrics(s, cont_pts, float(cont["runtime_s"]), "Masnavi-style continuous soft-LOS"),
    ]
    paths = {
        "A* soft-LOS": astar_pts,
        "SA soft-LOS HUBO": sa_pts,
        "Masnavi-style continuous soft-LOS": cont_pts,
    }
    extra = {
        "sa": {k: v for k, v in sa.items() if k not in {"trajectory", "history"}},
        "continuous": {k: v for k, v in cont.items() if k != "points"},
        "modeling_note": "Obstacle/collision avoidance is hard. LOS occlusion is soft and appears in the objective/cost only. Structural feasibility excludes LOS visibility.",
    }
    write_outputs(args.out_dir, s, metrics, paths, extra)
    print("Metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"Outputs written to {args.out_dir}")


if __name__ == "__main__":
    main()
