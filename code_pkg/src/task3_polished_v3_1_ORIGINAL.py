"""
=============================================================================
Task 3 v3.1 — SA at 500 Sweeps (Targeted Improvement)
=============================================================================
This script fixes the "SA reaches target only 2% of the time" issue
from v3 by re-running SA with 500 sweeps per run (instead of 200).
500 sweeps matches the original Task 2 SA configuration, so this is
SA running at its true capability rather than a handicapped version.

WHAT THIS DOES
--------------
  1. Re-runs Classical SA — 50 runs × 500 sweeps
  2. Re-runs Neal — 50 reads × 3000 sweeps (same as v3, for fresh sample data)
  3. Re-runs penalty sweep (same as v3, for completeness)
  4. Saves all plots and JSON

EXPECTED RUNTIME
----------------
  Neal:    ~25 seconds
  SA:      ~60 minutes  (50 runs × 500 sweeps)
  Sweep:   ~3 minutes
  TOTAL:   ~65 minutes

REQUIREMENTS
------------
  - task2_grid_hubo.py in same folder
  - pip install dwave-neal dimod numpy matplotlib

HOW TO RUN
----------
  cd C:\\Users\\ibrah\\OneDrive\\Desktop\\hubo
  python task3_polished_v3_1.py

OUTPUTS  →  ./outputs/task3_polished_v3_1/
=============================================================================
"""

import os
import json
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, Tuple, List

# Make Task 2 modules importable from same folder
TASK2_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK2_DIR)

from task2_grid_hubo import (
    Scenario, GridHUBO,
    trajectory_sa,
    greedy_init, random_init,
)

import dimod
import neal

OUT_DIR = os.path.join(TASK2_DIR, "outputs", "task3_polished_v3_1")
os.makedirs(OUT_DIR, exist_ok=True)

HUBO = Dict[Tuple[int, ...], float]


# ===========================================================================
# 1. HELPERS — proper feasibility check
# ===========================================================================

def build_scenario_with_weights(uniq=1000.0, start=1000.0, move=1000.0):
    s = Scenario()
    s.lambda_uniq  = uniq
    s.lambda_start = start
    s.lambda_move  = move
    return s


def proper_feasibility_check(bits, scenario, grid) -> dict:
    T, V = grid.T, grid.V

    trajectory = []
    uniqueness_violations = 0
    for t in range(T):
        active = [v for v in range(V) if bits[t*V + v] == 1]
        if len(active) != 1:
            uniqueness_violations += 1
        trajectory.append(active[0] if active else None)

    start_v = scenario.cell_index(*scenario.start)
    start_violation = (trajectory[0] != start_v
                       if trajectory[0] is not None else True)

    move_violations = 0
    for t in range(T - 1):
        if trajectory[t] is None or trajectory[t+1] is None:
            move_violations += 1
            continue
        if trajectory[t+1] not in scenario.neighbors(trajectory[t]):
            move_violations += 1

    collisions = sum(1 for v in trajectory
                     if v is not None and v in scenario.obstacle_indices)

    target_v = scenario.cell_index(*scenario.target)
    reaches_target = (trajectory[-1] == target_v)

    all_satisfied = (
        uniqueness_violations == 0 and
        not start_violation and
        move_violations == 0 and
        collisions == 0
    )

    return {
        "uniqueness_violations": uniqueness_violations,
        "start_violation": bool(start_violation),
        "move_violations": move_violations,
        "obstacle_collisions": collisions,
        "reaches_target": reaches_target,
        "all_hard_satisfied": all_satisfied,
    }


def decompose_energy(scenario, grid, bits) -> dict:
    H_uniq     = grid.H_uniq()
    H_start    = grid.H_start()
    H_move     = grid.H_move()
    H_obs      = grid.H_obs()
    H_occ      = grid.H_occ()
    H_prox     = grid.H_prox()
    H_goal     = grid.H_goal()
    H_terminal = grid.H_terminal()

    def w_eval(terms, weight):
        total = 0.0
        for key, coeff in terms.items():
            prod = 1
            for idx in key:
                prod *= bits[idx]
                if prod == 0: break
            total += weight * coeff * prod
        return total

    hard = (
        w_eval(H_uniq,  scenario.lambda_uniq) +
        w_eval(H_start, scenario.lambda_start) +
        w_eval(H_move,  scenario.lambda_move) +
        w_eval(H_obs,   scenario.lambda_obs)
    )
    soft = (
        w_eval(H_occ,      scenario.lambda_occ) +
        w_eval(H_prox,     scenario.lambda_prox) +
        w_eval(H_goal,     scenario.lambda_goal) +
        w_eval(H_terminal, scenario.lambda_terminal)
    )
    return {"hard_penalty": hard, "soft_cost": soft, "total": hard + soft}


# ===========================================================================
# 2. SOLVERS
# ===========================================================================

def solve_with_neal_proper(hubo, num_vars, scenario, grid,
                            num_reads=50, num_sweeps=3000, seed=42):
    """Run Neal with proper feasibility check on every sample."""
    poly = dimod.BinaryPolynomial(hubo, dimod.BINARY)
    sampler = dimod.HigherOrderComposite(neal.SimulatedAnnealingSampler())
    t0 = time.perf_counter()
    response = sampler.sample_poly(poly, num_reads=num_reads,
                                   num_sweeps=num_sweeps, seed=seed)
    runtime = time.perf_counter() - t0

    samples = []
    for sample, energy, *_ in response.data(["sample", "energy"]):
        bits = np.zeros(num_vars, dtype=np.int8)
        for var, val in sample.items():
            if isinstance(var, int) and 0 <= var < num_vars:
                bits[var] = int(val)
        decomp = decompose_energy(scenario, grid, bits)
        feas   = proper_feasibility_check(bits, scenario, grid)
        samples.append({
            "bits": bits,
            "total_energy": float(energy),
            "hard_penalty": decomp["hard_penalty"],
            "soft_cost": decomp["soft_cost"],
            "feasible": feas["all_hard_satisfied"],
            "reaches_target": feas["reaches_target"],
        })
    samples.sort(key=lambda s: s["total_energy"])
    return {"samples": samples, "runtime": runtime,
            "num_reads": num_reads, "num_sweeps": num_sweeps}


def solve_with_sa_500_sweeps(scenario, grid, hubo,
                              num_runs=50, num_sweeps=500):
    """
    Run SA at 500 sweeps per run (matches original Task 2 configuration).
    This gives SA enough time to converge to the global optimum,
    so most random-init runs will reach the target.
    """
    samples = []
    t0 = time.perf_counter()

    for run in range(num_runs):
        if run == 0:
            init_traj = greedy_init(scenario)
        else:
            init_traj = random_init(scenario, seed=run * 17 + 11)

        result = trajectory_sa(
            hubo=hubo, grid=grid, init_traj=init_traj,
            num_sweeps=num_sweeps, T_start=25.0, T_end=0.01,
            seed=42 + run, verbose=False,
        )
        bits = grid.trajectory_to_bits(result["best_traj"])
        decomp = decompose_energy(scenario, grid, bits)
        feas   = proper_feasibility_check(bits, scenario, grid)
        samples.append({
            "bits": bits,
            "total_energy": result["best_energy"],
            "hard_penalty": decomp["hard_penalty"],
            "soft_cost": decomp["soft_cost"],
            "feasible": feas["all_hard_satisfied"],
            "reaches_target": feas["reaches_target"],
        })

        # Progress every 5 runs
        if (run + 1) % 5 == 0:
            elapsed = time.perf_counter() - t0
            eta = elapsed / (run + 1) * (num_runs - run - 1)
            target_count = sum(1 for s in samples if s["reaches_target"])
            print(f"  SA {run+1}/{num_runs} done, "
                  f"best={min(s['total_energy'] for s in samples):.0f}, "
                  f"reaches_target={target_count}/{run+1}, "
                  f"elapsed={elapsed:.0f}s, ETA={eta:.0f}s",
                  flush=True)

    runtime = time.perf_counter() - t0
    samples.sort(key=lambda s: s["total_energy"])
    return {"samples": samples, "runtime": runtime,
            "num_runs": num_runs, "num_sweeps": num_sweeps}


# ===========================================================================
# 3. PENALTY SWEEP
# ===========================================================================

def run_penalty_sweep_proper(weights_to_try, num_reads=15, num_sweeps=2000):
    print(f"\n{'─' * 70}")
    print("Penalty-Weight Sweep (proper feasibility check)")
    print(f"{'─' * 70}")

    results = []
    for w in weights_to_try:
        scenario = build_scenario_with_weights(uniq=w, start=w, move=w)
        grid = GridHUBO(scenario)
        hubo = grid.build()

        poly = dimod.BinaryPolynomial(hubo, dimod.BINARY)
        sampler = dimod.HigherOrderComposite(neal.SimulatedAnnealingSampler())
        response = sampler.sample_poly(poly, num_reads=num_reads,
                                       num_sweeps=num_sweeps, seed=42)

        feas_count = 0
        best_total_energy = float("inf")
        best_feasible_soft = float("inf")
        for sample, energy, *_ in response.data(["sample", "energy"]):
            bits = np.zeros(grid.num_vars, dtype=np.int8)
            for var, val in sample.items():
                if isinstance(var, int) and 0 <= var < grid.num_vars:
                    bits[var] = int(val)
            feas = proper_feasibility_check(bits, scenario, grid)
            if feas["all_hard_satisfied"]:
                feas_count += 1
                decomp = decompose_energy(scenario, grid, bits)
                if decomp["soft_cost"] < best_feasible_soft:
                    best_feasible_soft = decomp["soft_cost"]
            if energy < best_total_energy:
                best_total_energy = float(energy)

        feas_rate = feas_count / num_reads
        results.append({
            "weight": w,
            "feasibility_rate": feas_rate,
            "feasible_count": feas_count,
            "total_reads": num_reads,
            "best_total_energy": best_total_energy,
            "best_feasible_soft_cost": (best_feasible_soft
                                         if best_feasible_soft != float("inf")
                                         else None),
        })
        soft_str = (f"{best_feasible_soft:.1f}"
                    if best_feasible_soft != float("inf")
                    else "N/A")
        print(f"  λ={w:>6.0f}: feasibility = {feas_rate*100:5.1f}% "
              f"({feas_count}/{num_reads}), "
              f"best feasible soft = {soft_str}",
              flush=True)
    return results


# ===========================================================================
# 4. PLOTTING
# ===========================================================================

def plot_total_energy(neal_samples, sa_samples, path):
    neal_e = [s["total_energy"] for s in neal_samples]
    sa_e   = [s["total_energy"] for s in sa_samples]
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(min(neal_e + sa_e), max(neal_e + sa_e), 30)
    ax.hist(neal_e, bins=bins, alpha=0.65, color="steelblue",
            label=f"D-Wave Neal (n={len(neal_e)})", edgecolor="black")
    ax.hist(sa_e, bins=bins, alpha=0.65, color="darkorange",
            label=f"Classical SA (n={len(sa_e)})", edgecolor="black")
    ax.axvline(min(neal_e), color="steelblue", linestyle="--",
               label=f"Neal best: {min(neal_e):.0f}")
    ax.axvline(min(sa_e), color="darkorange", linestyle="--",
               label=f"SA best: {min(sa_e):.0f}")
    ax.set_xlabel("Total HUBO Energy")
    ax.set_ylabel("Count")
    ax.set_title("Total Energy Distribution  (50 vs 50, SA at 500 sweeps)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_split_energy(neal_samples, sa_samples, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    neal_hard = [s["hard_penalty"] for s in neal_samples]
    sa_hard   = [s["hard_penalty"] for s in sa_samples]
    all_hard = neal_hard + sa_hard
    lo, hi = min(all_hard), max(all_hard)
    if abs(hi - lo) < 1:
        lo -= 100; hi += 100
    bins = np.linspace(lo - 50, hi + 50, 30)

    axes[0].hist(neal_hard, bins=bins, alpha=0.7, color="steelblue",
                 label="Neal", edgecolor="black")
    axes[0].hist(sa_hard, bins=bins, alpha=0.7, color="darkorange",
                 label="Classical SA", edgecolor="black")
    feas_floor = -16000
    axes[0].axvline(feas_floor, color="green", linestyle="--", linewidth=2,
                     label=f"Feasibility floor ({feas_floor})")
    axes[0].set_xlabel("Hard Constraint Penalty"); axes[0].set_ylabel("Count")
    axes[0].set_title(f"Hard-Constraint Penalty\n"
                       f"(closer to {feas_floor} = feasible)",
                       fontsize=11)
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    neal_soft = [s["soft_cost"] for s in neal_samples]
    sa_soft   = [s["soft_cost"] for s in sa_samples]
    bins = np.linspace(min(neal_soft + sa_soft), max(neal_soft + sa_soft), 30)
    axes[1].hist(neal_soft, bins=bins, alpha=0.7, color="steelblue",
                 label="Neal", edgecolor="black")
    axes[1].hist(sa_soft, bins=bins, alpha=0.7, color="darkorange",
                 label="Classical SA", edgecolor="black")
    axes[1].set_xlabel("Soft Cost (occ + prox + goal + terminal)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Soft-Cost Energy\n(actual trajectory quality)",
                       fontsize=11)
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle("Energy decomposed into hard constraint vs soft cost",
                 fontsize=13, y=1.02)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_runtime(neal_rt, sa_rt, neal_n, sa_n, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(["D-Wave Neal", "Classical SA"],
                  [neal_rt, sa_rt],
                  color=["steelblue", "darkorange"],
                  alpha=0.85, edgecolor="black", linewidth=1)
    for b, v, label in zip(bars, [neal_rt, sa_rt],
                            [f"{neal_n} reads", f"{sa_n} runs"]):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                f"{v:.1f}s\n({label})",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Runtime — 50 vs 50 matched samples (SA at 500 sweeps)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_trajectory(scenario, bits, grid, title, path):
    T, V = grid.T, grid.V
    trajectory = []
    for t in range(T):
        active = [v for v in range(V) if bits[t*V + v] == 1]
        trajectory.append(active[0] if len(active) == 1 else None)

    gs = scenario.grid_size
    img = np.zeros((gs, gs))
    for v in scenario.buffer_indices:
        r, c = scenario.index_to_cell(v); img[r,c] = 1.0
    for r, c in scenario.obstacles: img[r,c] = 2.0

    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.cm.Reds
    ax.imshow(img, cmap=cmap, origin="upper", vmin=0, vmax=2, alpha=0.65)

    valid = [(t, scenario.index_to_cell(v))
             for t, v in enumerate(trajectory) if v is not None]
    if valid:
        ts, cells = zip(*valid)
        ys = [c[0] for c in cells]; xs = [c[1] for c in cells]
        ax.plot(xs, ys, "b-o", linewidth=2.5, markersize=8, zorder=5)
        for t, (y, x) in zip(ts, zip(ys, xs)):
            if t % 2 == 0 or t == len(valid) - 1:
                ax.annotate(f"t={t}", (x, y), xytext=(5, 5),
                            textcoords="offset points", fontsize=8,
                            color="navy", fontweight="bold")

    ax.plot(scenario.start[1], scenario.start[0], "go",
            markersize=18, markeredgecolor="black", zorder=6, label="Start")
    ax.plot(scenario.target[1], scenario.target[0], "r*",
            markersize=22, markeredgecolor="black", zorder=6, label="Target")
    p_obs = mpatches.Patch(color=cmap(1.0), alpha=0.7, label="Obstacle")
    p_buf = mpatches.Patch(color=cmap(0.5), alpha=0.7, label="Buffer")
    h, l = ax.get_legend_handles_labels()
    ax.legend(handles=h + [p_obs, p_buf], loc="upper left", fontsize=9)
    ax.set_xticks(range(gs)); ax.set_yticks(range(gs))
    ax.set_xlim(-0.5, gs - 0.5); ax.set_ylim(gs - 0.5, -0.5)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=12)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_penalty_sweep(sweep_results, path):
    weights = [r["weight"] for r in sweep_results]
    feas_rates = [r["feasibility_rate"] * 100 for r in sweep_results]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    color1 = "steelblue"
    ax.set_xlabel("Penalty Weight  λ_uniq = λ_start = λ_move", fontsize=11)
    ax.set_ylabel("Feasibility Rate (%)", color=color1, fontsize=11)
    ax.set_xscale("log")
    ax.plot(weights, feas_rates, "o-", color=color1,
            linewidth=2.5, markersize=10, label="Feasibility rate")
    ax.tick_params(axis="y", labelcolor=color1)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3, axis="y")

    for w, fr in zip(weights, feas_rates):
        ax.annotate(f"{fr:.0f}%", (w, fr), xytext=(0, 10),
                    textcoords="offset points",
                    fontsize=10, ha="center", color=color1,
                    fontweight="bold")

    ax.set_title("Soft-Constraint Exploitation Curve\n"
                 "Neal feasibility rate vs penalty weight",
                 fontsize=12)
    ax.legend(loc="upper left", fontsize=10)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_metrics_table(rows, path):
    fig, ax = plt.subplots(figsize=(11, 0.5 + 0.4 * len(rows)))
    ax.axis("off")
    tbl = ax.table(cellText=rows[1:], colLabels=rows[0],
                   cellLoc="left", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.5)
    for col_idx in range(len(rows[0])):
        c = tbl[(0, col_idx)]
        c.set_facecolor("#0D1B4B")
        c.set_text_props(color="white", fontweight="bold")
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# ===========================================================================
# 5. MAIN
# ===========================================================================

def main():
    print("=" * 70)
    print("  Task 3 v3.1 — SA at 500 Sweeps (Fair-Fight Configuration)")
    print("=" * 70)
    print("\nThis runs SA at 500 sweeps per run (vs 200 in v3).")
    print("Matches the original Task 2 SA configuration, giving SA enough")
    print("time to converge to the global optimum from random initializations.")
    print("\nExpected runtime: ~65 minutes (SA is the long part).")

    # Build HUBO with hardened weights
    scenario = build_scenario_with_weights(uniq=1000, start=1000, move=1000)
    grid = GridHUBO(scenario)
    hubo = grid.build()
    print(f"\nHUBO: {grid.num_vars} variables, {len(hubo)} terms")
    print(f"Hardened weights: λ_uniq=λ_start=λ_move=1000")

    # ---- D-Wave Neal ----
    print(f"\n{'─' * 70}")
    print("D-Wave Neal — 50 reads with proper feasibility checking")
    print(f"{'─' * 70}")
    neal_result = solve_with_neal_proper(hubo, grid.num_vars,
                                          scenario, grid,
                                          num_reads=50, num_sweeps=3000)
    neal_samples = neal_result["samples"]
    neal_feas_count = sum(1 for s in neal_samples if s["feasible"])
    neal_target_count = sum(1 for s in neal_samples if s["reaches_target"])
    print(f"  Best total energy: {neal_samples[0]['total_energy']:.2f}")
    print(f"  Best soft cost   : {neal_samples[0]['soft_cost']:.2f}")
    print(f"  Mean total       : {np.mean([s['total_energy'] for s in neal_samples]):.2f}")
    print(f"  Runtime          : {neal_result['runtime']:.2f}s")
    print(f"  TRUE feasibility : {neal_feas_count}/50 = {neal_feas_count*100/50:.1f}%")
    print(f"  Reaches target   : {neal_target_count}/50 = {neal_target_count*100/50:.1f}%")

    feasible_neal = [s for s in neal_samples if s["feasible"]]
    if feasible_neal:
        neal_best_feasible = feasible_neal[0]
        print(f"  Best FEASIBLE Neal: energy={neal_best_feasible['total_energy']:.2f}, "
              f"soft={neal_best_feasible['soft_cost']:.2f}")
    else:
        neal_best_feasible = neal_samples[0]

    # ---- Classical SA at 500 sweeps (fair-fight configuration) ----
    print(f"\n{'─' * 70}")
    print("Classical SA — 50 runs × 500 sweeps (fair-fight configuration)")
    print(f"{'─' * 70}")
    print("This is the long part — expected ~60 minutes.")
    print("Progress will be reported every 5 runs.")
    sa_result = solve_with_sa_500_sweeps(scenario, grid, hubo,
                                          num_runs=50, num_sweeps=500)
    sa_samples = sa_result["samples"]
    sa_feas_count = sum(1 for s in sa_samples if s["feasible"])
    sa_target_count = sum(1 for s in sa_samples if s["reaches_target"])
    print(f"  Best total energy: {sa_samples[0]['total_energy']:.2f}")
    print(f"  Best soft cost   : {sa_samples[0]['soft_cost']:.2f}")
    print(f"  Mean total       : {np.mean([s['total_energy'] for s in sa_samples]):.2f}")
    print(f"  Runtime          : {sa_result['runtime']:.2f}s")
    print(f"  TRUE feasibility : {sa_feas_count}/50 = {sa_feas_count*100/50:.1f}%")
    print(f"  Reaches target   : {sa_target_count}/50 = {sa_target_count*100/50:.1f}%")

    # ---- Penalty sweep ----
    sweep_weights = [50, 100, 300, 600, 1000, 2000]
    sweep_results = run_penalty_sweep_proper(sweep_weights,
                                              num_reads=15, num_sweeps=2000)

    # ---- Save outputs ----
    print(f"\n{'─' * 70}")
    print("Saving outputs ...")
    print(f"{'─' * 70}")

    plot_total_energy(neal_samples, sa_samples,
                       os.path.join(OUT_DIR, "energy_distribution_total.png"))
    print(f"  Saved: energy_distribution_total.png")

    plot_split_energy(neal_samples, sa_samples,
                       os.path.join(OUT_DIR, "energy_distribution_split.png"))
    print(f"  Saved: energy_distribution_split.png")

    plot_runtime(neal_result["runtime"], sa_result["runtime"],
                  50, 50,
                  os.path.join(OUT_DIR, "runtime_comparison.png"))
    print(f"  Saved: runtime_comparison.png")

    feas_label = ("FEASIBLE ✓" if neal_best_feasible["feasible"]
                  else "INFEASIBLE ✗")
    plot_trajectory(scenario, neal_best_feasible["bits"], grid,
                     f"D-Wave Neal — Best Feasible Trajectory ({feas_label})\n"
                     f"energy={neal_best_feasible['total_energy']:.0f}, "
                     f"soft={neal_best_feasible['soft_cost']:.0f}",
                     os.path.join(OUT_DIR, "neal_trajectory.png"))
    print(f"  Saved: neal_trajectory.png")

    plot_trajectory(scenario, sa_samples[0]["bits"], grid,
                     f"Classical SA — Best Trajectory (500 sweeps)\n"
                     f"energy={sa_samples[0]['total_energy']:.0f}, "
                     f"soft={sa_samples[0]['soft_cost']:.0f}",
                     os.path.join(OUT_DIR, "sa_trajectory.png"))
    print(f"  Saved: sa_trajectory.png")

    plot_penalty_sweep(sweep_results,
                        os.path.join(OUT_DIR, "penalty_sweep.png"))
    print(f"  Saved: penalty_sweep.png")

    # Metrics table
    table_rows = [
        ["Metric", "D-Wave Neal", "Classical SA (500 sweeps)"],
        ["Best total energy",
         f"{neal_samples[0]['total_energy']:.2f}",
         f"{sa_samples[0]['total_energy']:.2f}"],
        ["Best soft cost",
         f"{neal_samples[0]['soft_cost']:.2f}",
         f"{sa_samples[0]['soft_cost']:.2f}"],
        ["Best FEASIBLE soft cost",
         (f"{neal_best_feasible['soft_cost']:.2f}"
          if neal_best_feasible['feasible']
          else "N/A"),
         f"{sa_samples[0]['soft_cost']:.2f}"],
        ["Mean total energy",
         f"{np.mean([s['total_energy'] for s in neal_samples]):.2f}",
         f"{np.mean([s['total_energy'] for s in sa_samples]):.2f}"],
        ["Std dev",
         f"{np.std([s['total_energy'] for s in neal_samples]):.2f}",
         f"{np.std([s['total_energy'] for s in sa_samples]):.2f}"],
        ["Samples (matched)",
         "50 reads",
         "50 runs"],
        ["Sweeps per sample",
         "3000",
         "500"],
        ["Total runtime",
         f"{neal_result['runtime']:.2f}s",
         f"{sa_result['runtime']:.2f}s"],
        ["Time per sample",
         f"{neal_result['runtime']/50:.3f}s",
         f"{sa_result['runtime']/50:.3f}s"],
        ["TRUE feasibility rate",
         f"{neal_feas_count*100/50:.1f}%",
         f"{sa_feas_count*100/50:.1f}%"],
        ["Reaches-target rate",
         f"{neal_target_count*100/50:.1f}%",
         f"{sa_target_count*100/50:.1f}%"],
        ["Best sample feasible",
         "✓" if neal_samples[0]['feasible'] else "✗",
         "✓" if sa_samples[0]['feasible'] else "✗"],
    ]
    plot_metrics_table(table_rows,
                        os.path.join(OUT_DIR, "metrics_table.png"))
    print(f"  Saved: metrics_table.png")

    # JSON
    json_payload = {
        "version": "v3.1 — SA at 500 sweeps (fair-fight)",
        "hubo": {
            "num_variables": grid.num_vars,
            "num_terms": len(hubo),
            "lambda_uniq_start_move": 1000,
        },
        "methodology": (
            "v3.1: matched 50 vs 50 samples, proper feasibility check, "
            "SA at 500 sweeps to match original Task 2 configuration."
        ),
        "d_wave_neal": {
            "num_reads": 50,
            "num_sweeps": 3000,
            "best_total_energy": neal_samples[0]['total_energy'],
            "best_soft_cost": neal_samples[0]['soft_cost'],
            "mean_total_energy": float(np.mean(
                [s['total_energy'] for s in neal_samples])),
            "std_total_energy": float(np.std(
                [s['total_energy'] for s in neal_samples])),
            "runtime_s": neal_result["runtime"],
            "true_feasibility_rate": neal_feas_count / 50,
            "feasible_sample_count": neal_feas_count,
            "reaches_target_rate": neal_target_count / 50,
            "best_sample_feasible": bool(neal_samples[0]['feasible']),
            "best_sample_reaches_target": bool(neal_samples[0]['reaches_target']),
            "best_feasible_sample": {
                "total_energy": neal_best_feasible['total_energy'],
                "soft_cost": neal_best_feasible['soft_cost'],
                "feasible": bool(neal_best_feasible['feasible']),
                "reaches_target": bool(neal_best_feasible['reaches_target']),
            },
        },
        "classical_sa": {
            "num_runs": 50,
            "num_sweeps": 500,
            "best_total_energy": sa_samples[0]['total_energy'],
            "best_soft_cost": sa_samples[0]['soft_cost'],
            "mean_total_energy": float(np.mean(
                [s['total_energy'] for s in sa_samples])),
            "std_total_energy": float(np.std(
                [s['total_energy'] for s in sa_samples])),
            "runtime_s": sa_result["runtime"],
            "true_feasibility_rate": sa_feas_count / 50,
            "reaches_target_rate": sa_target_count / 50,
            "best_sample_feasible": bool(sa_samples[0]['feasible']),
        },
        "penalty_sweep": sweep_results,
    }
    with open(os.path.join(OUT_DIR, "benchmark_results_v3_1.json"), "w") as f:
        json.dump(json_payload, f, indent=2)
    print(f"  Saved: benchmark_results_v3_1.json")

    # ---- Final summary ----
    print(f"\n{'=' * 70}")
    print(f"  TASK 3 v3.1 — FINAL RESULTS (SA at 500 sweeps)")
    print(f"{'=' * 70}")
    print(f"  D-Wave Neal:")
    print(f"    Best total energy : {neal_samples[0]['total_energy']:.2f}")
    print(f"    Best soft cost    : {neal_samples[0]['soft_cost']:.2f}")
    print(f"    Feasibility       : {neal_feas_count*100/50:.1f}% ({neal_feas_count}/50)")
    print(f"    Reaches target    : {neal_target_count*100/50:.1f}% ({neal_target_count}/50)")
    print(f"  Classical SA (500 sweeps):")
    print(f"    Best total energy : {sa_samples[0]['total_energy']:.2f}")
    print(f"    Best soft cost    : {sa_samples[0]['soft_cost']:.2f}")
    print(f"    Feasibility       : {sa_feas_count*100/50:.1f}% ({sa_feas_count}/50)")
    print(f"    Reaches target    : {sa_target_count*100/50:.1f}% ({sa_target_count}/50)")
    print(f"  Outputs : {OUT_DIR}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
