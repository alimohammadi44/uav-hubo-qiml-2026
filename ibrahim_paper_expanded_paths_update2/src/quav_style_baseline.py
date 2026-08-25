"""
QUAV-style baseline for UAV obstacle-avoidance path planning.

Purpose
-------
This script is NOT the official QUAV implementation. It is a transparent
reproduction-style baseline inspired by the published QUAV workflow:

    grid/graph construction -> candidate path generation / segmentation
    -> cost assignment -> QAOA-style quantum-assisted selection
    -> comparison with A* and RRT.

It is designed to run inside the same code_pkg/src folder as Ibrahim's
HUBO code (task2_grid_hubo.py). It uses the same 8x8 scenario, obstacles,
start/target, horizon, and path-validity checks, so the results can be
compared with the HUBO/SA/neal paper experiments.

Default quantum simulation is a small path-candidate QAOA selector. It uses
one qubit per candidate path, with a one-hot penalty forcing the selection of
one path. This keeps the comparison reproducible and avoids claiming that this
is the full QUAV code or the full HUBO formulation.

Outputs
-------
outputs/quav_style_baseline/
    quav_style_results.json
    quav_style_report.md
    quav_style_paths.png
    quav_style_metrics.png
    quav_style_loss.png

Run
---
python quav_style_baseline.py
python quav_style_baseline.py --max-candidates 16 --qaoa-steps 80
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except Exception:  # plotting optional
    plt = None
    mpatches = None

from task2_grid_hubo import Scenario, GridHUBO

Cell = Tuple[int, int]
Path = List[Cell]

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "quav_style_baseline")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Basic path utilities
# ---------------------------------------------------------------------------

def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def neighbors(cell: Cell, s: Scenario, include_self: bool = False) -> List[Cell]:
    r, c = cell
    out = []
    if include_self:
        out.append(cell)
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < s.grid_size and 0 <= nc < s.grid_size:
            out.append((nr, nc))
    return out


def pad_to_horizon(path: Path, s: Scenario) -> Path:
    """Pad a target-reaching path with target self-loops to match horizon."""
    if len(path) < s.horizon:
        path = path + [path[-1]] * (s.horizon - len(path))
    elif len(path) > s.horizon:
        path = path[: s.horizon]
    return path


def path_to_indices(path: Path, s: Scenario) -> List[int]:
    return [s.cell_index(r, c) for r, c in path]


def turn_count(path: Path) -> int:
    dirs = []
    for a, b in zip(path[:-1], path[1:]):
        dr, dc = b[0] - a[0], b[1] - a[1]
        if (dr, dc) != (0, 0):
            dirs.append((dr, dc))
    return sum(1 for i in range(1, len(dirs)) if dirs[i] != dirs[i - 1])


def path_length(path: Path) -> int:
    return sum(1 for a, b in zip(path[:-1], path[1:]) if a != b)


def collision_count(path: Path, s: Scenario) -> int:
    obs = set(s.obstacles)
    return sum(1 for cell in path if cell in obs)


def buffer_count(path: Path, s: Scenario) -> int:
    return sum(1 for cell in path if s.cell_index(*cell) in s.buffer_indices)


def visibility_rate(path: Path, s: Scenario) -> float:
    if not path:
        return 0.0
    visible = 0
    for cell in path:
        v = s.cell_index(*cell)
        if s.occlusion_count(v) == 0:
            visible += 1
    return visible / len(path)


def reaches_target(path: Path, s: Scenario) -> bool:
    return bool(path) and path[-1] == s.target


def valid_motion(path: Path, s: Scenario) -> bool:
    for a, b in zip(path[:-1], path[1:]):
        if s.cell_index(*b) not in s.neighbors(s.cell_index(*a)):
            return False
    return True


def quav_style_cost(path: Path, s: Scenario) -> float:
    """
    QUAV-style obstacle-avoidance path cost.
    This follows the published paper's idea: length + obstacle/buffer penalty
    + smoothness; it intentionally does not include occlusion, because QUAV is
    obstacle-avoidance focused.
    """
    return (
        1.0 * path_length(path)
        + 1000.0 * collision_count(path, s)
        + 5.0 * buffer_count(path, s)
        + 0.5 * turn_count(path)
    )


def hubo_energy(path: Path, s: Scenario, grid: GridHUBO, hubo: Dict[Tuple[int, ...], float]) -> float:
    return float(grid.eval_trajectory(hubo, path_to_indices(path, s)))


# ---------------------------------------------------------------------------
# Baseline planners: A* and RRT-style randomized planner
# ---------------------------------------------------------------------------

def astar_path(s: Scenario, random_seed: Optional[int] = None, noise_scale: float = 0.0) -> Path:
    """A* with obstacle/buffer/occlusion-aware cell cost. Random noise gives alternatives."""
    rng = np.random.default_rng(random_seed)
    start, target = s.start, s.target
    obs = set(s.obstacles)

    # fixed random perturbation per cell for reproducible candidate diversity
    noise: Dict[Cell, float] = {}
    for r in range(s.grid_size):
        for c in range(s.grid_size):
            noise[(r, c)] = float(rng.uniform(0.0, noise_scale)) if random_seed is not None else 0.0

    def step_cost(cell: Cell) -> float:
        if cell in obs:
            return float("inf")
        v = s.cell_index(*cell)
        return (
            1.0
            + s.lambda_occ * s.occlusion_count(v)
            + s.lambda_prox * (1.0 if v in s.buffer_indices else 0.0)
            + noise[cell]
        )

    heap = []
    heapq.heappush(heap, (manhattan(start, target), 0.0, start))
    parent: Dict[Cell, Optional[Cell]] = {start: None}
    gscore: Dict[Cell, float] = {start: 0.0}

    while heap:
        _, g, cur = heapq.heappop(heap)
        if cur == target:
            break
        if g > gscore.get(cur, float("inf")):
            continue
        for nb in neighbors(cur, s, include_self=False):
            cost = step_cost(nb)
            if math.isinf(cost):
                continue
            ng = g + cost
            if ng < gscore.get(nb, float("inf")):
                gscore[nb] = ng
                parent[nb] = cur
                heapq.heappush(heap, (ng + manhattan(nb, target), ng, nb))

    if target not in parent:
        return []
    path: Path = []
    cur: Optional[Cell] = target
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return pad_to_horizon(path, s)


def rrt_grid_path(s: Scenario, seed: int, max_iter: int = 1000, goal_bias: float = 0.08) -> Path:
    """Simple grid RRT-style planner for comparison with QUAV paper baseline."""
    rng = np.random.default_rng(seed)
    start, target = s.start, s.target
    obs = set(s.obstacles)
    nodes: List[Cell] = [start]
    parent: Dict[Cell, Optional[Cell]] = {start: None}

    free_cells = [
        (r, c)
        for r in range(s.grid_size)
        for c in range(s.grid_size)
        if (r, c) not in obs
    ]

    def nearest(sample: Cell) -> Cell:
        return min(nodes, key=lambda x: (x[0] - sample[0]) ** 2 + (x[1] - sample[1]) ** 2)

    def step_toward(a: Cell, b: Cell) -> Cell:
        dr, dc = b[0] - a[0], b[1] - a[1]
        if abs(dr) >= abs(dc) and dr != 0:
            cand = (a[0] + (1 if dr > 0 else -1), a[1])
        elif dc != 0:
            cand = (a[0], a[1] + (1 if dc > 0 else -1))
        else:
            cand = a
        return cand

    for _ in range(max_iter):
        sample = target if rng.random() < goal_bias else free_cells[int(rng.integers(len(free_cells)))]
        near = nearest(sample)
        new = step_toward(near, sample)
        if new in obs or not (0 <= new[0] < s.grid_size and 0 <= new[1] < s.grid_size):
            continue
        if new not in parent:
            parent[new] = near
            nodes.append(new)
        if new == target:
            break

    if target not in parent:
        # fall back to A* so the experiment never crashes
        return astar_path(s, random_seed=seed, noise_scale=0.0)

    path: Path = []
    cur: Optional[Cell] = target
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return pad_to_horizon(path, s)


def generate_candidate_paths(s: Scenario, n_random_astar: int = 60, n_rrt: int = 60) -> List[Path]:
    """Generate a diverse candidate set for QUAV-style QAOA selection."""
    candidates: List[Path] = []
    seen = set()

    def add(path: Path):
        if not path:
            return
        key = tuple(path)
        if key not in seen and valid_motion(path, s) and collision_count(path, s) == 0 and reaches_target(path, s):
            seen.add(key)
            candidates.append(path)

    add(astar_path(s, random_seed=None, noise_scale=0.0))
    for seed in range(n_random_astar):
        add(astar_path(s, random_seed=seed, noise_scale=8.0))
    for seed in range(1000, 1000 + n_rrt):
        add(rrt_grid_path(s, seed=seed, max_iter=1200, goal_bias=0.12))
    return candidates


# ---------------------------------------------------------------------------
# Small QAOA simulator for one-hot candidate-path selection
# ---------------------------------------------------------------------------

def bitcount_array(n: int) -> np.ndarray:
    x = np.arange(1 << n, dtype=np.uint32)
    # compatible popcount without numpy.bit_count assumption
    counts = np.zeros_like(x, dtype=np.float64)
    y = x.copy()
    while np.any(y):
        counts += (y & 1)
        y >>= 1
    return counts


def candidate_qubo_energy(costs: np.ndarray, onehot_penalty: float) -> np.ndarray:
    n = len(costs)
    N = 1 << n
    idx = np.arange(N, dtype=np.uint32)
    counts = bitcount_array(n)
    E = onehot_penalty * (counts - 1.0) ** 2
    for i, c in enumerate(costs):
        E += c * ((idx >> i) & 1)
    return E.astype(np.float64)


def apply_rx_mixer(psi: np.ndarray, beta: float, n: int) -> np.ndarray:
    """Apply product_i exp(-i beta X_i) to statevector."""
    c = math.cos(beta)
    s = -1j * math.sin(beta)
    out = psi.copy()
    for q in range(n):
        step = 1 << q
        block = step << 1
        for start in range(0, len(out), block):
            a = out[start : start + step].copy()
            b = out[start + step : start + block].copy()
            out[start : start + step] = c * a + s * b
            out[start + step : start + block] = s * a + c * b
    return out


def qaoa_expectation(params: np.ndarray, energies: np.ndarray, p_layers: int, n: int) -> float:
    gammas = params[:p_layers]
    betas = params[p_layers:]
    psi = np.ones(1 << n, dtype=np.complex128) / math.sqrt(1 << n)
    for gamma, beta in zip(gammas, betas):
        psi *= np.exp(-1j * gamma * energies)
        psi = apply_rx_mixer(psi, beta, n)
    prob = np.abs(psi) ** 2
    return float(np.dot(prob, energies))


def qaoa_distribution(params: np.ndarray, energies: np.ndarray, p_layers: int, n: int) -> np.ndarray:
    gammas = params[:p_layers]
    betas = params[p_layers:]
    psi = np.ones(1 << n, dtype=np.complex128) / math.sqrt(1 << n)
    for gamma, beta in zip(gammas, betas):
        psi *= np.exp(-1j * gamma * energies)
        psi = apply_rx_mixer(psi, beta, n)
    return np.abs(psi) ** 2


def optimize_qaoa(costs: np.ndarray, p_layers: int = 1, steps: int = 60, seed: int = 42):
    """Optimize small candidate-selection QAOA; returns selected candidate index and trace."""
    rng = np.random.default_rng(seed)
    # Normalize costs to keep QAOA angles numerically stable.
    cmin, cmax = float(np.min(costs)), float(np.max(costs))
    span = max(cmax - cmin, 1e-9)
    norm_costs = (costs - cmin) / span
    onehot_penalty = 2.0 + 2.0 * float(np.max(norm_costs))
    energies = candidate_qubo_energy(norm_costs, onehot_penalty)
    n = len(costs)

    best_params = None
    best_val = float("inf")
    trace = []

    try:
        from scipy.optimize import minimize

        def obj(x):
            return qaoa_expectation(x, energies, p_layers, n)

        # Multi-start Nelder-Mead is robust for this tiny p=1/p=2 setting.
        for _ in range(4):
            x0 = np.concatenate([
                rng.uniform(0.0, 2.0 * math.pi, size=p_layers),
                rng.uniform(0.0, math.pi, size=p_layers),
            ])
            local_trace = []

            def cb(xk):
                local_trace.append(float(obj(xk)))

            res = minimize(obj, x0, method="Nelder-Mead", options={"maxiter": steps, "xatol": 1e-3, "fatol": 1e-3}, callback=cb)
            val = float(res.fun)
            trace.extend(local_trace if local_trace else [val])
            if val < best_val:
                best_val = val
                best_params = np.array(res.x, dtype=float)
    except Exception:
        # Fallback random search if scipy is unavailable.
        for _ in range(steps):
            x = np.concatenate([
                rng.uniform(0.0, 2.0 * math.pi, size=p_layers),
                rng.uniform(0.0, math.pi, size=p_layers),
            ])
            val = qaoa_expectation(x, energies, p_layers, n)
            trace.append(val)
            if val < best_val:
                best_val = val
                best_params = x

    probs = qaoa_distribution(best_params, energies, p_layers, n)
    measured = int(np.argmax(probs))
    # Enforce one-hot decode. If argmax not one-hot, take best one-hot bitstring by probability.
    onehot_states = [1 << i for i in range(n)]
    best_state = max(onehot_states, key=lambda z: probs[z])
    selected = int(math.log2(best_state))
    return {
        "selected_index": selected,
        "best_expectation_normalized": best_val,
        "best_state_probability": float(probs[best_state]),
        "argmax_state": measured,
        "argmax_state_popcount": int(bin(measured).count("1")),
        "trace": trace,
        "energies_normalized": energies.tolist(),
        "onehot_penalty": onehot_penalty,
    }


# ---------------------------------------------------------------------------
# Reporting and plotting
# ---------------------------------------------------------------------------

def evaluate_method(name: str, path: Path, s: Scenario, grid: GridHUBO, hubo) -> dict:
    return {
        "method": name,
        "path": [list(x) for x in path],
        "path_length_steps": path_length(path),
        "turns": turn_count(path),
        "buffer_steps": buffer_count(path, s),
        "collisions": collision_count(path, s),
        "collision_free": collision_count(path, s) == 0,
        "valid_motion": valid_motion(path, s),
        "reaches_target": reaches_target(path, s),
        "full_task_success": valid_motion(path, s) and collision_count(path, s) == 0 and reaches_target(path, s),
        "visibility_rate": visibility_rate(path, s),
        "quav_style_cost": quav_style_cost(path, s),
        "hubo_energy": hubo_energy(path, s, grid, hubo),
    }


def plot_paths(results: List[dict], s: Scenario):
    if plt is None:
        return
    fig, axes = plt.subplots(1, len(results), figsize=(5.0 * len(results), 5.2))
    if len(results) == 1:
        axes = [axes]
    obs = set(s.obstacles)
    buf = {s.index_to_cell(v) for v in s.buffer_indices}
    for ax, res in zip(axes, results):
        ax.set_title(res["method"])
        ax.set_xlim(-0.5, s.grid_size - 0.5)
        ax.set_ylim(s.grid_size - 0.5, -0.5)
        ax.set_xticks(range(s.grid_size))
        ax.set_yticks(range(s.grid_size))
        ax.grid(True, alpha=0.4)
        for r in range(s.grid_size):
            for c in range(s.grid_size):
                if (r, c) in obs:
                    rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="black", alpha=0.85)
                    ax.add_patch(rect)
                elif (r, c) in buf:
                    rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="orange", alpha=0.15)
                    ax.add_patch(rect)
        path = [tuple(x) for x in res["path"]]
        xs = [c for r, c in path]
        ys = [r for r, c in path]
        ax.plot(xs, ys, marker="o", linewidth=2)
        ax.scatter([s.start[1]], [s.start[0]], s=120, marker="s", label="Start")
        ax.scatter([s.target[1]], [s.target[0]], s=160, marker="*", label="Target")
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "quav_style_paths.png"), dpi=200)
    plt.close(fig)


def plot_metrics(results: List[dict]):
    if plt is None:
        return
    methods = [r["method"] for r in results]
    costs = [r["quav_style_cost"] for r in results]
    lengths = [r["path_length_steps"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(methods))
    w = 0.35
    ax.bar(x - w / 2, costs, width=w, label="QUAV-style cost")
    ax.bar(x + w / 2, lengths, width=w, label="Path length")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Value")
    ax.set_title("QUAV-style baseline comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "quav_style_metrics.png"), dpi=200)
    plt.close(fig)


def plot_loss(trace: List[float]):
    if plt is None or not trace:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(trace, marker="o", markersize=3)
    ax.set_xlabel("Optimizer callback/evaluation")
    ax.set_ylabel("Expected QUBO energy")
    ax.set_title("QUAV-style QAOA candidate-selection convergence")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "quav_style_loss.png"), dpi=200)
    plt.close(fig)


def write_report(results: List[dict], qaoa_info: dict, candidates: List[Path]):
    lines = []
    lines.append("# QUAV-style baseline report")
    lines.append("")
    lines.append("This is a reproduction-style baseline inspired by the QUAV paper, not the official QUAV code.")
    lines.append("")
    lines.append(f"Candidate paths generated: {len(candidates)}")
    lines.append(f"QAOA-selected candidate index: {qaoa_info['selected_index']}")
    lines.append(f"Best one-hot state probability: {qaoa_info['best_state_probability']:.4f}")
    lines.append("")
    lines.append("| Method | Full success | Path length | Turns | Buffer steps | Visibility | QUAV-style cost | HUBO energy |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r['method']} | {r['full_task_success']} | {r['path_length_steps']} | {r['turns']} | "
            f"{r['buffer_steps']} | {100*r['visibility_rate']:.1f}% | {r['quav_style_cost']:.2f} | {r['hubo_energy']:.2f} |"
        )
    with open(os.path.join(OUT_DIR, "quav_style_report.md"), "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-candidates", type=int, default=12, help="Number of candidate paths/qubits used by QAOA selector.")
    parser.add_argument("--qaoa-layers", type=int, default=1)
    parser.add_argument("--qaoa-steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    start_time = time.time()
    s = Scenario()
    grid = GridHUBO(s)
    hubo = grid.build()

    print("QUAV-style baseline on Ibrahim UAV HUBO scenario")
    print("=" * 72)
    print(s.describe())
    print()
    print("Generating A*, RRT, and diverse candidate paths...")

    t0 = time.time()
    a_path = astar_path(s)
    astar_runtime = time.time() - t0

    rrt_paths = [rrt_grid_path(s, seed=i, max_iter=1200, goal_bias=0.12) for i in range(200, 240)]
    rrt_best = min(rrt_paths, key=lambda p: quav_style_cost(p, s))

    candidates = generate_candidate_paths(s, n_random_astar=80, n_rrt=80)
    if not candidates:
        raise RuntimeError("No valid candidate paths generated.")
    candidates = sorted(candidates, key=lambda p: quav_style_cost(p, s))
    candidates = candidates[: max(2, min(args.max_candidates, len(candidates)))]

    print(f"Generated valid candidate paths: {len(candidates)} used for QAOA selector")
    print("Candidate costs:", [round(quav_style_cost(p, s), 2) for p in candidates])

    print("Running QUAV-style QAOA candidate selector...")
    t1 = time.time()
    cand_costs = np.array([quav_style_cost(p, s) for p in candidates], dtype=float)
    qaoa_info = optimize_qaoa(cand_costs, p_layers=args.qaoa_layers, steps=args.qaoa_steps, seed=args.seed)
    quav_runtime = time.time() - t1
    q_path = candidates[qaoa_info["selected_index"]]

    results = [
        evaluate_method("A*", a_path, s, grid, hubo),
        evaluate_method("RRT-best", rrt_best, s, grid, hubo),
        evaluate_method("QUAV-style QAOA", q_path, s, grid, hubo),
    ]
    results[0]["runtime_seconds"] = astar_runtime
    results[1]["runtime_seconds"] = None
    results[2]["runtime_seconds"] = quav_runtime

    payload = {
        "note": "This is a QUAV-style reproduction baseline, not official QUAV code.",
        "scenario": {
            "grid_size": s.grid_size,
            "horizon": s.horizon,
            "start": list(s.start),
            "target": list(s.target),
            "obstacles": [list(x) for x in s.obstacles],
        },
        "candidate_count_total": len(candidates),
        "candidate_costs": [float(x) for x in cand_costs],
        "qaoa_info": qaoa_info,
        "results": results,
        "total_runtime_seconds": time.time() - start_time,
    }

    with open(os.path.join(OUT_DIR, "quav_style_results.json"), "w") as f:
        json.dump(payload, f, indent=2)

    plot_paths(results, s)
    plot_metrics(results)
    plot_loss(qaoa_info.get("trace", []))
    write_report(results, qaoa_info, candidates)

    print("\nResults")
    print("-" * 72)
    for r in results:
        print(
            f"{r['method']:16s} success={str(r['full_task_success']):5s} "
            f"length={r['path_length_steps']:2d} turns={r['turns']:2d} "
            f"buffer={r['buffer_steps']:2d} visibility={100*r['visibility_rate']:5.1f}% "
            f"quav_cost={r['quav_style_cost']:7.2f} hubo_energy={r['hubo_energy']:10.2f}"
        )
    print("\nSaved outputs to:", OUT_DIR)


if __name__ == "__main__":
    main()
