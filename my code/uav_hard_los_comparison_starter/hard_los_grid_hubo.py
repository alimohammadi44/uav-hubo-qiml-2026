"""
Hard-LOS Grid HUBO starter code for fair comparison with Masnavi-style
occlusion-free MPC.

This script modifies the earlier grid HUBO benchmark in the important way:
LOS visibility is treated as a hard feasibility requirement, not only as a
soft occlusion penalty.

Outputs are written to ./outputs_hard_los/ by default.

Run examples:
  python hard_los_grid_hubo.py --scenario visible_start --runs 10 --sweeps 200
  python hard_los_grid_hubo.py --scenario original --check-only

The original paper scenario start=(0,0), target=(7,7) is infeasible under
hard per-step LOS because the start cell is already occluded by the central
obstacle block. The default `visible_start` scenario keeps the same map and
target, but starts from (3,0), a visible cell with a feasible occlusion-free
path to the target.
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
from typing import Dict, Iterable, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

HUBO = Dict[Tuple[int, ...], float]
Cell = Tuple[int, int]


@dataclass
class Scenario:
    grid_size: int = 8
    horizon: int = 12
    start: Cell = (3, 0)
    target: Cell = (7, 7)
    obstacles: List[Cell] = field(default_factory=lambda: [
        (3, 3), (3, 4), (4, 3), (4, 4),  # central block
        (2, 6), (5, 1),                  # scattered blockers
    ])
    buffer_radius: int = 1

    # Hard penalty weights. These are large because they represent constraints.
    lambda_uniq: float = 1000.0
    lambda_start: float = 1000.0
    lambda_move: float = 1000.0
    lambda_obs: float = 1000.0
    lambda_los: float = 1000.0
    lambda_terminal: float = 1000.0

    # Soft trajectory-quality terms.
    lambda_prox: float = 5.0

    def cell_index(self, r: int, c: int) -> int:
        return r * self.grid_size + c

    def index_to_cell(self, v: int) -> Cell:
        return divmod(v, self.grid_size)

    @property
    def num_cells(self) -> int:
        return self.grid_size * self.grid_size

    @property
    def start_index(self) -> int:
        return self.cell_index(*self.start)

    @property
    def target_index(self) -> int:
        return self.cell_index(*self.target)

    @property
    def obstacle_indices(self) -> Set[int]:
        return {self.cell_index(r, c) for r, c in self.obstacles}

    @property
    def buffer_indices(self) -> Set[int]:
        obs = self.obstacle_indices
        buf: Set[int] = set()
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                v = self.cell_index(r, c)
                if v in obs:
                    continue
                for ro, co in self.obstacles:
                    if max(abs(r - ro), abs(c - co)) <= self.buffer_radius:
                        buf.add(v)
                        break
        return buf

    def neighbors(self, v: int, allow_hover: bool = True) -> Set[int]:
        r, c = self.index_to_cell(v)
        out: Set[int] = {v} if allow_hover else set()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                out.add(self.cell_index(nr, nc))
        return out

    def line_of_sight(self, u: int, v: Optional[int] = None) -> List[int]:
        """Bresenham cells strictly between u and v. Default v=target."""
        if v is None:
            v = self.target_index
        r0, c0 = self.index_to_cell(u)
        r1, c1 = self.index_to_cell(v)
        cells: List[int] = []
        dr, dc = abs(r1 - r0), abs(c1 - c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1
        err = dr - dc
        r, c = r0, c0
        while True:
            cells.append(self.cell_index(r, c))
            if r == r1 and c == c1:
                break
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dr
                c += sc
        return cells[1:-1] if len(cells) > 2 else []

    def los_blockers(self, u: int) -> List[int]:
        obs = self.obstacle_indices
        return [w for w in self.line_of_sight(u) if w in obs]

    def occlusion_count(self, u: int) -> int:
        return len(self.los_blockers(u))

    def is_visible(self, u: int) -> bool:
        return self.occlusion_count(u) == 0

    @property
    def visible_indices(self) -> Set[int]:
        return {v for v in range(self.num_cells) if self.is_visible(v)}

    @property
    def free_visible_indices(self) -> Set[int]:
        return self.visible_indices - self.obstacle_indices

    def describe(self) -> Dict[str, object]:
        return {
            "grid_size": self.grid_size,
            "horizon": self.horizon,
            "start": self.start,
            "target": self.target,
            "obstacles": self.obstacles,
            "num_cells": self.num_cells,
            "num_obstacles": len(self.obstacles),
            "num_visible_cells": len(self.visible_indices),
            "num_free_visible_cells": len(self.free_visible_indices),
            "start_visible": self.is_visible(self.start_index),
            "target_visible": self.is_visible(self.target_index),
        }


class GridHUBO:
    def __init__(self, scenario: Scenario):
        self.s = scenario
        self.T = scenario.horizon
        self.V = scenario.num_cells
        self.num_vars = self.T * self.V

    def var(self, t: int, v: int) -> int:
        return t * self.V + v

    def unvar(self, idx: int) -> Tuple[int, int]:
        return divmod(idx, self.V)

    def H_uniq(self) -> HUBO:
        d: HUBO = {}
        for t in range(self.T):
            for v in range(self.V):
                _add(d, (self.var(t, v),), -1.0)
            for u in range(self.V):
                for v in range(u + 1, self.V):
                    _add(d, (self.var(t, u), self.var(t, v)), 2.0)
        return d

    def H_start(self) -> HUBO:
        d: HUBO = {}
        _add(d, (self.var(0, self.s.start_index),), -1.0)
        for v in range(self.V):
            if v != self.s.start_index:
                _add(d, (self.var(0, v),), 1.0)
        return d

    def H_move(self) -> HUBO:
        d: HUBO = {}
        for t in range(self.T - 1):
            for u in range(self.V):
                nb = self.s.neighbors(u)
                for v in range(self.V):
                    if v not in nb:
                        _add(d, tuple(sorted((self.var(t, u), self.var(t + 1, v)))), 1.0)
        return d

    def H_obs(self) -> HUBO:
        d: HUBO = {}
        for t in range(self.T):
            for v in self.s.obstacle_indices:
                _add(d, (self.var(t, v),), 1.0)
        return d

    def H_los(self) -> HUBO:
        """Hard LOS penalty: zero only for cells with unobstructed LOS to target."""
        d: HUBO = {}
        for v in range(self.V):
            cnt = self.s.occlusion_count(v)
            if cnt > 0:
                for t in range(self.T):
                    _add(d, (self.var(t, v),), float(cnt))
        return d

    def H_terminal(self) -> HUBO:
        """Hard terminal target condition x[T-1, target] = 1."""
        d: HUBO = {}
        tv = self.s.target_index
        _add(d, (self.var(self.T - 1, tv),), -1.0)
        for v in range(self.V):
            if v != tv:
                _add(d, (self.var(self.T - 1, v),), 1.0)
        return d

    def H_prox(self) -> HUBO:
        """Cubic soft penalty for three consecutive buffer-cell states."""
        d: HUBO = {}
        B = self.s.buffer_indices
        for t in range(1, self.T - 1):
            for u in B:
                for v in B & self.s.neighbors(u):
                    for w in B & self.s.neighbors(v):
                        key = tuple(sorted((self.var(t - 1, u), self.var(t, v), self.var(t + 1, w))))
                        _add(d, key, 1.0)
        return d

    def build(self) -> HUBO:
        out: HUBO = {}
        _scale_add(out, self.H_uniq(), self.s.lambda_uniq)
        _scale_add(out, self.H_start(), self.s.lambda_start)
        _scale_add(out, self.H_move(), self.s.lambda_move)
        _scale_add(out, self.H_obs(), self.s.lambda_obs)
        _scale_add(out, self.H_los(), self.s.lambda_los)
        _scale_add(out, self.H_terminal(), self.s.lambda_terminal)
        _scale_add(out, self.H_prox(), self.s.lambda_prox)
        return out

    def term_summary(self, hubo: HUBO) -> Dict[str, int]:
        out = {"linear": 0, "quadratic": 0, "cubic": 0, "higher": 0}
        for key in hubo:
            if len(key) == 1:
                out["linear"] += 1
            elif len(key) == 2:
                out["quadratic"] += 1
            elif len(key) == 3:
                out["cubic"] += 1
            else:
                out["higher"] += 1
        return out

    def trajectory_to_bits(self, traj: List[int]) -> np.ndarray:
        bits = np.zeros(self.num_vars, dtype=np.int8)
        for t, v in enumerate(traj):
            bits[self.var(t, v)] = 1
        return bits

    def eval_bits(self, hubo: HUBO, bits: np.ndarray) -> float:
        total = 0.0
        for key, coeff in hubo.items():
            prod = 1
            for idx in key:
                prod *= int(bits[idx])
                if prod == 0:
                    break
            total += coeff * prod
        return total

    def eval_trajectory(self, hubo: HUBO, traj: List[int]) -> float:
        return self.eval_bits(hubo, self.trajectory_to_bits(traj))

    def feasibility_report(self, traj: List[int]) -> Dict[str, object]:
        rep = {
            "start_violation": traj[0] != self.s.start_index,
            "terminal_violation": traj[-1] != self.s.target_index,
            "move_violations": 0,
            "obstacle_collisions": 0,
            "los_violations": 0,
            "visible_steps": 0,
            "visibility_percent": 0.0,
            "all_hard_satisfied": False,
        }
        for t in range(self.T - 1):
            if traj[t + 1] not in self.s.neighbors(traj[t]):
                rep["move_violations"] += 1
        for v in traj:
            if v in self.s.obstacle_indices:
                rep["obstacle_collisions"] += 1
            if not self.s.is_visible(v):
                rep["los_violations"] += 1
            else:
                rep["visible_steps"] += 1
        rep["visibility_percent"] = 100.0 * rep["visible_steps"] / len(traj)
        rep["all_hard_satisfied"] = (
            not rep["start_violation"] and
            not rep["terminal_violation"] and
            rep["move_violations"] == 0 and
            rep["obstacle_collisions"] == 0 and
            rep["los_violations"] == 0
        )
        return rep


def _add(d: HUBO, key: Tuple[int, ...], value: float) -> None:
    if value != 0.0:
        d[key] = d.get(key, 0.0) + value


def _scale_add(dst: HUBO, src: HUBO, scale: float) -> None:
    for k, v in src.items():
        _add(dst, k, scale * v)


def astar_hard_los(s: Scenario) -> Optional[List[int]]:
    """A* baseline over free-visible cells only; pads with target hovers if needed."""
    allowed = s.free_visible_indices
    if s.start_index not in allowed or s.target_index not in allowed:
        return None

    def h(v: int) -> int:
        r, c = s.index_to_cell(v)
        tr, tc = s.target
        return abs(r - tr) + abs(c - tc)

    pq: List[Tuple[float, int, int]] = []
    heapq.heappush(pq, (h(s.start_index), 0, s.start_index))
    parent: Dict[int, Optional[int]] = {s.start_index: None}
    g: Dict[int, float] = {s.start_index: 0.0}

    while pq:
        _, _, v = heapq.heappop(pq)
        if v == s.target_index:
            break
        for w in s.neighbors(v, allow_hover=False):
            if w not in allowed:
                continue
            # soft cost: one step plus small buffer penalty
            step = 1.0 + (0.1 if w in s.buffer_indices else 0.0)
            ng = g[v] + step
            if ng < g.get(w, float("inf")):
                g[w] = ng
                parent[w] = v
                heapq.heappush(pq, (ng + h(w), len(parent), w))

    if s.target_index not in parent:
        return None

    path: List[int] = []
    v: Optional[int] = s.target_index
    while v is not None:
        path.append(v)
        v = parent[v]
    path.reverse()

    if len(path) > s.horizon:
        return None
    while len(path) < s.horizon:
        path.append(s.target_index)
    return path


def random_feasible_walk(s: Scenario, rng: np.random.Generator) -> Optional[List[int]]:
    """Random feasible walk seeded by A* fallback; returns terminal-feasible path."""
    base = astar_hard_los(s)
    if base is None:
        return None
    # Perturbing a terminal hard path safely is nontrivial; for robust initialization use A*.
    # SA randomization is provided by stochastic move proposals.
    return list(base)


def trajectory_sa(hubo: HUBO, grid: GridHUBO, init_traj: List[int], runs: int, sweeps: int,
                  seed: int = 42) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    s, T = grid.s, grid.T
    allowed = s.free_visible_indices
    best_global = list(init_traj)
    best_global_e = grid.eval_trajectory(hubo, best_global)
    histories: List[float] = []
    total_evals = 0
    total_acc = 0
    start_time = time.perf_counter()

    def local_terms_by_t() -> List[List[Tuple[Tuple[int, ...], float]]]:
        per_t: List[List[Tuple[Tuple[int, ...], float]]] = [[] for _ in range(T)]
        for key, coeff in hubo.items():
            ts = {grid.unvar(idx)[0] for idx in key}
            for tt in ts:
                per_t[tt].append((key, coeff))
        # remove duplicated keys per t
        return [list(dict(items).items()) for items in per_t]

    per_t = local_terms_by_t()

    def eval_terms(traj: List[int], t: int) -> float:
        total = 0.0
        for key, coeff in per_t[t]:
            prod = 1
            for idx in key:
                tt, vv = grid.unvar(idx)
                prod *= int(traj[tt] == vv)
                if prod == 0:
                    break
            total += coeff * prod
        return total

    for run in range(runs):
        traj = list(init_traj)
        energy = grid.eval_trajectory(hubo, traj)
        best = list(traj)
        best_e = energy
        T_start, T_end = 25.0, 0.01
        alpha = (T_end / T_start) ** (1.0 / max(1, sweeps))
        Temp = T_start
        for _ in range(sweeps):
            for t in rng.permutation(T):
                if t == 0 or t == T - 1:
                    continue  # start and terminal hard constraints are fixed
                prev_v = traj[t - 1]
                next_v = traj[t + 1]
                cands = (s.neighbors(prev_v) & s.neighbors(next_v) & allowed)
                if not cands:
                    continue
                new_v = int(rng.choice(list(cands)))
                if new_v == traj[t]:
                    continue
                before = eval_terms(traj, t)
                old_v = traj[t]
                traj[t] = new_v
                after = eval_terms(traj, t)
                delta = after - before
                total_evals += 1
                if delta <= 0 or rng.random() < math.exp(-delta / Temp):
                    energy += delta
                    total_acc += 1
                    if energy < best_e:
                        best_e = energy
                        best = list(traj)
                else:
                    traj[t] = old_v
            histories.append(energy)
            Temp *= alpha
        if best_e < best_global_e:
            best_global_e = best_e
            best_global = best

    return {
        "best_traj": best_global,
        "best_energy": best_global_e,
        "history": histories,
        "runtime_s": time.perf_counter() - start_time,
        "acceptance_rate": total_acc / max(1, total_evals),
        "runs": runs,
        "sweeps": sweeps,
    }


def plot_grid(s: Scenario, traj: Optional[List[int]], out_path: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    for r in range(s.grid_size):
        for c in range(s.grid_size):
            v = s.cell_index(r, c)
            if v in s.obstacle_indices:
                face = "black"
                alpha = 0.85
            elif not s.is_visible(v):
                face = "lightcoral"
                alpha = 0.50
            elif v in s.buffer_indices:
                face = "khaki"
                alpha = 0.60
            else:
                face = "white"
                alpha = 1.00
            ax.add_patch(patches.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                           facecolor=face, edgecolor="gray", alpha=alpha))
            if not s.is_visible(v) and v not in s.obstacle_indices:
                ax.text(c, r, str(s.occlusion_count(v)), ha="center", va="center", fontsize=7)
    if traj:
        cells = [s.index_to_cell(v) for v in traj]
        ys = [r for r, _ in cells]
        xs = [c for _, c in cells]
        ax.plot(xs, ys, "o-", linewidth=2.0, markersize=5, label="UAV path")
        for t, (x, y) in enumerate(zip(xs, ys)):
            if t == 0 or t == len(xs) - 1 or t % 2 == 0:
                ax.annotate(str(t), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.plot(s.start[1], s.start[0], "go", markersize=12, markeredgecolor="black", label="Start")
    ax.plot(s.target[1], s.target[0], "r*", markersize=16, markeredgecolor="black", label="Target")
    ax.set_title(title)
    ax.set_xticks(range(s.grid_size))
    ax.set_yticks(range(s.grid_size))
    ax.set_xlim(-0.5, s.grid_size - 0.5)
    ax.set_ylim(s.grid_size - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    legend_handles = [
        patches.Patch(facecolor="black", alpha=0.85, label="Obstacle"),
        patches.Patch(facecolor="lightcoral", alpha=0.50, label="LOS-forbidden"),
        patches.Patch(facecolor="khaki", alpha=0.60, label="Buffer"),
    ]
    h, l = ax.get_legend_handles_labels()
    ax.legend(handles=h + legend_handles, loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def save_outputs(out_dir: str, s: Scenario, grid: GridHUBO, hubo: HUBO,
                 astar_path: Optional[List[int]], sa_result: Optional[Dict[str, object]]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    metrics: Dict[str, object] = {
        "scenario": s.describe(),
        "hubo": {
            "num_variables": grid.num_vars,
            "num_terms": len(hubo),
            "term_summary": grid.term_summary(hubo),
        },
        "astar": None,
        "sa": None,
    }
    if astar_path:
        metrics["astar"] = {
            "trajectory": [s.index_to_cell(v) for v in astar_path],
            "energy": grid.eval_trajectory(hubo, astar_path),
            "feasibility": grid.feasibility_report(astar_path),
        }
        with open(os.path.join(out_dir, "astar_trajectory.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "row", "col", "visible", "occlusion_count"])
            for t, v in enumerate(astar_path):
                r, c = s.index_to_cell(v)
                writer.writerow([t, r, c, int(s.is_visible(v)), s.occlusion_count(v)])
        plot_grid(s, astar_path, os.path.join(out_dir, "astar_hard_los_trajectory.png"),
                  "A* baseline with hard LOS visibility")
    else:
        plot_grid(s, None, os.path.join(out_dir, "scenario_hard_los_infeasible_or_unplanned.png"),
                  "Hard LOS visibility map")

    if sa_result:
        traj = sa_result["best_traj"]  # type: ignore[index]
        metrics["sa"] = {
            "best_energy": sa_result["best_energy"],
            "runtime_s": sa_result["runtime_s"],
            "acceptance_rate": sa_result["acceptance_rate"],
            "runs": sa_result["runs"],
            "sweeps": sa_result["sweeps"],
            "trajectory": [s.index_to_cell(v) for v in traj],
            "feasibility": grid.feasibility_report(traj),
        }
        plot_grid(s, traj, os.path.join(out_dir, "sa_hard_los_trajectory.png"),
                  "Trajectory-level SA with hard LOS visibility")
        np.save(os.path.join(out_dir, "sa_energy_history.npy"), np.array(sa_result["history"]))
        plt.figure(figsize=(7, 4))
        plt.plot(sa_result["history"])
        plt.xlabel("SA accepted sweep index")
        plt.ylabel("Energy")
        plt.title("SA energy history, hard LOS formulation")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "sa_energy_history.png"), dpi=160)
        plt.close()

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def build_scenario(name: str) -> Scenario:
    if name == "original":
        return Scenario(start=(0, 0), target=(7, 7), horizon=15)
    if name == "visible_start":
        return Scenario(start=(3, 0), target=(7, 7), horizon=12)
    raise ValueError(f"Unknown scenario {name!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["original", "visible_start"], default="visible_start")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--sweeps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="outputs_hard_los")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    s = build_scenario(args.scenario)
    grid = GridHUBO(s)
    hubo = grid.build()

    print("Scenario:")
    print(json.dumps(s.describe(), indent=2))
    print("HUBO:", grid.num_vars, "variables,", len(hubo), "terms", grid.term_summary(hubo))

    astar_path = astar_hard_los(s)
    if astar_path is None:
        print("No hard-LOS A* path exists for this scenario/horizon.")
        print("This usually means the start is occluded, the target is unreachable, or the horizon is too short.")
        save_outputs(args.out_dir, s, grid, hubo, None, None)
        return

    print("A* hard-LOS path:", [s.index_to_cell(v) for v in astar_path])
    print("A* feasibility:", grid.feasibility_report(astar_path))
    print("A* energy:", grid.eval_trajectory(hubo, astar_path))

    if args.check_only:
        save_outputs(args.out_dir, s, grid, hubo, astar_path, None)
        return

    init = random_feasible_walk(s, np.random.default_rng(args.seed))
    if init is None:
        print("Cannot build feasible initialization.")
        save_outputs(args.out_dir, s, grid, hubo, astar_path, None)
        return

    print(f"Running trajectory SA: runs={args.runs}, sweeps={args.sweeps}")
    sa = trajectory_sa(hubo, grid, init, runs=args.runs, sweeps=args.sweeps, seed=args.seed)
    print("SA best energy:", sa["best_energy"])
    print("SA runtime_s:", sa["runtime_s"])
    print("SA feasibility:", grid.feasibility_report(sa["best_traj"]))

    save_outputs(args.out_dir, s, grid, hubo, astar_path, sa)
    print("Outputs saved to", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
