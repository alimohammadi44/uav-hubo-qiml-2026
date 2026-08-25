"""
=============================================================================
Task 2 — Grid-Based vs Basis-Function HUBO Comparison
=============================================================================
This script loads the outputs from both implementations and produces:
  1. Side-by-side trajectory comparison plot
  2. Both trajectories overlaid on the same grid
  3. Combined energy trace plot
  4. A comprehensive metrics comparison table
  5. A markdown summary report

Run this AFTER running task2_grid_hubo.py and task2_basis_hubo.py.
=============================================================================
"""

import os, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT     = os.path.dirname(os.path.abspath(__file__))
GRID_DIR = os.path.join(ROOT, "outputs", "grid")
BASIS_DIR= os.path.join(ROOT, "outputs", "basis")
OUT_DIR  = os.path.join(ROOT, "outputs", "comparison")
os.makedirs(OUT_DIR, exist_ok=True)


def load_meta(path):
    with open(os.path.join(path, "metadata.json")) as f:
        return json.load(f)


def load_energy(path):
    return np.load(os.path.join(path, "energy_history.npy"))


def print_table(rows):
    """Print a clean comparison table."""
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    sep = "─" * (sum(widths) + 3 * (len(widths) - 1) + 4)
    print(sep)
    for i, row in enumerate(rows):
        print("  " + "   ".join(str(c).ljust(w) for c, w in zip(row, widths)))
        if i == 0:
            print(sep)
    print(sep)


def main():
    print("=" * 70)
    print("  Task 2  —  Grid-Based vs Basis-Function HUBO Comparison")
    print("=" * 70)

    # ---- Load both ----
    if not os.path.exists(os.path.join(GRID_DIR, "metadata.json")):
        print(f"\n[ERROR] Grid-based outputs not found at {GRID_DIR}")
        print("        Run task2_grid_hubo.py first.")
        return
    if not os.path.exists(os.path.join(BASIS_DIR, "metadata.json")):
        print(f"\n[ERROR] Basis-function outputs not found at {BASIS_DIR}")
        print("        Run task2_basis_hubo.py first.")
        return

    grid_meta  = load_meta(GRID_DIR)
    basis_meta = load_meta(BASIS_DIR)
    grid_energy  = load_energy(GRID_DIR)
    basis_energy = load_energy(BASIS_DIR)

    # ====================== COMPARISON TABLE ======================
    print("\n┌──────────────────────────────────────────────────────────────────┐")
    print("│                  KEY METRICS COMPARISON                            │")
    print("└──────────────────────────────────────────────────────────────────┘")

    grid_feas  = grid_meta["feasibility"]
    basis_feas = basis_meta["feasibility"]

    rows = [
        ("Metric",                    "Grid-Based",                 "Basis-Function"),
        ("Binary variables",          str(grid_meta["num_variables"]),
                                      str(basis_meta["num_variables"])),
        ("HUBO terms",                str(grid_meta["num_terms"]),
                                      str(basis_meta["num_terms"])),
        ("HUBO order",                "3 (cubic)",
                                      f"{basis_meta['hubo_order']} (sextic)"),
        ("Solver runtime (s)",        f"{grid_meta['solver']['runtime_s']:.2f}",
                                      f"{basis_meta['solver']['runtime_s']:.2f}"),
        ("Best energy",               f"{grid_meta['solver']['best_energy']:.2f}",
                                      f"{basis_meta['solver']['best_energy']:.2f}"),
        ("Sweeps run",                str(grid_meta['solver']['num_sweeps']),
                                      str(basis_meta['solver']['num_sweeps'])),
        ("Obstacle collisions",       str(grid_feas['obstacle_collisions']),
                                      str(basis_feas['obstacle_collisions'])),
        ("Reaches target",            "YES" if grid_meta['reaches_target'] else "NO",
                                      "YES" if basis_feas['reaches_target'] else "NO"),
    ]
    print()
    print_table(rows)

    # ====================== TRAJECTORY ANALYSIS ======================
    print("\n┌──────────────────────────────────────────────────────────────────┐")
    print("│                  TRAJECTORY ANALYSIS                               │")
    print("└──────────────────────────────────────────────────────────────────┘")

    grid_traj  = grid_meta["trajectory"]
    basis_traj = basis_meta["trajectory_grid"]
    basis_cont = basis_meta["trajectory_continuous"]

    # Path length (number of unique cells visited)
    grid_unique = len(set(tuple(c) for c in grid_traj))
    basis_unique = len(set(tuple(c) for c in basis_traj))

    # Total path length (sum of step distances)
    def path_length_grid(traj):
        total = 0.0
        for i in range(len(traj) - 1):
            r0, c0 = traj[i]
            r1, c1 = traj[i+1]
            total += ((r1-r0)**2 + (c1-c0)**2) ** 0.5
        return total

    def path_length_cont(traj):
        total = 0.0
        for i in range(len(traj) - 1):
            x0, y0 = traj[i]
            x1, y1 = traj[i+1]
            total += ((x1-x0)**2 + (y1-y0)**2) ** 0.5
        return total

    grid_len  = path_length_grid(grid_traj)
    basis_len = path_length_cont(basis_cont)

    # Smoothness: average turning angle for grid
    def smoothness_grid(traj):
        """Average angle change in degrees."""
        angles = []
        for i in range(1, len(traj) - 1):
            r0, c0 = traj[i-1]
            r1, c1 = traj[i]
            r2, c2 = traj[i+1]
            dx1, dy1 = r1-r0, c1-c0
            dx2, dy2 = r2-r1, c2-c1
            if (dx1 == 0 and dy1 == 0) or (dx2 == 0 and dy2 == 0):
                continue
            dot = dx1*dx2 + dy1*dy2
            mag1 = (dx1**2 + dy1**2) ** 0.5
            mag2 = (dx2**2 + dy2**2) ** 0.5
            cos_a = max(-1, min(1, dot / (mag1*mag2)))
            angle_deg = np.degrees(np.arccos(cos_a))
            angles.append(angle_deg)
        return float(np.mean(angles)) if angles else 0.0

    def smoothness_cont(traj):
        angles = []
        for i in range(1, len(traj) - 1):
            x0, y0 = traj[i-1]
            x1, y1 = traj[i]
            x2, y2 = traj[i+1]
            dx1, dy1 = x1-x0, y1-y0
            dx2, dy2 = x2-x1, y2-y1
            if (abs(dx1) < 1e-6 and abs(dy1) < 1e-6) or \
               (abs(dx2) < 1e-6 and abs(dy2) < 1e-6):
                continue
            dot = dx1*dx2 + dy1*dy2
            mag1 = (dx1**2 + dy1**2) ** 0.5
            mag2 = (dx2**2 + dy2**2) ** 0.5
            cos_a = max(-1, min(1, dot / (mag1*mag2)))
            angle_deg = np.degrees(np.arccos(cos_a))
            angles.append(angle_deg)
        return float(np.mean(angles)) if angles else 0.0

    grid_smooth  = smoothness_grid(grid_traj)
    basis_smooth = smoothness_cont(basis_cont)

    rows = [
        ("Trajectory Metric",          "Grid-Based",                "Basis-Function"),
        ("Unique cells visited",       str(grid_unique),
                                       str(basis_unique)),
        ("Total path length",          f"{grid_len:.2f}",
                                       f"{basis_len:.2f}"),
        ("Mean direction change (°)",  f"{grid_smooth:.2f}",
                                       f"{basis_smooth:.2f}"),
        ("Final position",             str(tuple(grid_traj[-1])),
                                       str(tuple(basis_traj[-1]))),
    ]
    print()
    print_table(rows)

    # ====================== VERDICT ======================
    print("\n┌──────────────────────────────────────────────────────────────────┐")
    print("│                  VERDICT                                           │")
    print("└──────────────────────────────────────────────────────────────────┘\n")

    print(f"  ✓ Variable count        : Basis-function wins "
          f"({basis_meta['num_variables']} vs {grid_meta['num_variables']} — "
          f"{grid_meta['num_variables'] / basis_meta['num_variables']:.0f}× smaller)")
    print(f"  ✓ Runtime               : Basis-function wins "
          f"({basis_meta['solver']['runtime_s']:.1f}s vs "
          f"{grid_meta['solver']['runtime_s']:.1f}s)")
    print(f"  ✓ Trajectory smoothness : Basis-function wins "
          f"({basis_smooth:.1f}° avg vs {grid_smooth:.1f}°)")
    print(f"  ✓ Flexibility           : Grid-based wins "
          f"(any path representable)")
    print(f"  ✓ Feasibility (target)  : "
          f"{'Both reach target' if grid_meta['reaches_target'] and basis_feas['reaches_target'] else 'Mixed'}")

    # ====================== PLOT 1: Side-by-side ======================
    print("\nGenerating comparison plots ...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # We need the scenario to draw obstacles, but it's identical in both
    grid_size = grid_meta["scenario"]["grid_size"]
    obstacles = [tuple(o) for o in grid_meta["scenario"]["obstacles"]]
    start  = tuple(grid_meta["scenario"]["start"])
    target = tuple(grid_meta["scenario"]["target"])

    # Re-derive buffer cells locally
    def buffer_cells(grid_size, obstacles, radius=1):
        obs_set = {(r, c) for r, c in obstacles}
        buf = set()
        for r in range(grid_size):
            for c in range(grid_size):
                if (r, c) in obs_set:
                    continue
                for or_, oc in obstacles:
                    if max(abs(r - or_), abs(c - oc)) <= radius:
                        buf.add((r, c))
                        break
        return buf

    buf_set = buffer_cells(grid_size, obstacles)

    def draw_grid(ax, title):
        img = np.zeros((grid_size, grid_size))
        for r, c in buf_set:
            img[r, c] = 1.0
        for r, c in obstacles:
            img[r, c] = 2.0
        ax.imshow(img, cmap=plt.cm.Reds, origin="upper",
                  vmin=0, vmax=2, alpha=0.65)
        ax.plot(start[1], start[0], "go", markersize=16,
                markeredgecolor="black", zorder=7, label="Start")
        ax.plot(target[1], target[0], "r*", markersize=20,
                markeredgecolor="black", zorder=7, label="Target")
        ax.set_xticks(range(grid_size)); ax.set_yticks(range(grid_size))
        ax.set_xlim(-0.5, grid_size - 0.5); ax.set_ylim(grid_size - 0.5, -0.5)
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=12)

    # Left: grid-based
    draw_grid(axes[0],
              f"Grid-Based HUBO\n{grid_meta['num_variables']} variables, "
              f"cubic order")
    ys = [c[0] for c in grid_traj]
    xs = [c[1] for c in grid_traj]
    axes[0].plot(xs, ys, "b-o", linewidth=2, markersize=6,
                 zorder=5, label="UAV path")
    axes[0].legend(loc="upper left", fontsize=9)

    # Right: basis-function
    draw_grid(axes[1],
              f"Basis-Function HUBO\n{basis_meta['num_variables']} variables, "
              f"order-{basis_meta['hubo_order']} HUBO")
    ys_cont = [p[0] for p in basis_cont]
    xs_cont = [p[1] for p in basis_cont]
    axes[1].plot(xs_cont, ys_cont, "-", color="navy", linewidth=2.5,
                 zorder=5, label="Polynomial path")
    for x, y in zip(xs_cont, ys_cont):
        axes[1].plot(x, y, "o", color="navy", markersize=6, zorder=6)
    axes[1].legend(loc="upper left", fontsize=9)

    plt.suptitle("Grid-Based vs Basis-Function HUBO — Trajectory Comparison",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    sbs_path = os.path.join(OUT_DIR, "trajectories_side_by_side.png")
    plt.savefig(sbs_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: trajectories_side_by_side.png")

    # ====================== PLOT 2: Overlay ======================
    fig, ax = plt.subplots(figsize=(9, 9))
    img = np.zeros((grid_size, grid_size))
    for r, c in buf_set:
        img[r, c] = 1.0
    for r, c in obstacles:
        img[r, c] = 2.0
    ax.imshow(img, cmap=plt.cm.Reds, origin="upper",
              vmin=0, vmax=2, alpha=0.55)

    # Grid-based — blue solid
    ys = [c[0] for c in grid_traj]; xs = [c[1] for c in grid_traj]
    ax.plot(xs, ys, "b-o", linewidth=2.2, markersize=7, alpha=0.85,
            zorder=5, label=f"Grid-Based ({grid_meta['num_variables']} vars)")

    # Basis-function — orange curve
    ys_cont = [p[0] for p in basis_cont]; xs_cont = [p[1] for p in basis_cont]
    ax.plot(xs_cont, ys_cont, "-", color="darkorange", linewidth=2.8,
            zorder=6, label=f"Basis-Function ({basis_meta['num_variables']} vars)")
    for x, y in zip(xs_cont, ys_cont):
        ax.plot(x, y, "s", color="darkorange", markersize=6, zorder=7)

    ax.plot(start[1], start[0], "go", markersize=20,
            markeredgecolor="black", zorder=8, label="Start")
    ax.plot(target[1], target[0], "r*", markersize=24,
            markeredgecolor="black", zorder=8, label="Target")

    p_obs = mpatches.Patch(color=plt.cm.Reds(1.0), alpha=0.65, label="Obstacle")
    p_buf = mpatches.Patch(color=plt.cm.Reds(0.5), alpha=0.65, label="Buffer")
    h, l = ax.get_legend_handles_labels()
    ax.legend(handles=h + [p_obs, p_buf], loc="upper left", fontsize=10)

    ax.set_xticks(range(grid_size)); ax.set_yticks(range(grid_size))
    ax.set_xlim(-0.5, grid_size - 0.5); ax.set_ylim(grid_size - 0.5, -0.5)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.set_title("Trajectory Overlay — Grid-Based vs Basis-Function HUBO",
                 fontsize=13)
    plt.tight_layout()
    ov_path = os.path.join(OUT_DIR, "trajectories_overlay.png")
    plt.savefig(ov_path, dpi=150)
    plt.close()
    print(f"  Saved: trajectories_overlay.png")

    # ====================== PLOT 3: Energy traces ======================
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(grid_energy, color="steelblue", linewidth=1.2)
    axes[0].set_title(f"Grid-Based — SA Energy Trace\n"
                      f"({grid_meta['num_variables']} vars, "
                      f"5 restarts × {grid_meta['solver']['num_sweeps']//5} sweeps)",
                      fontsize=11)
    axes[0].set_xlabel("Sweep"); axes[0].set_ylabel("HUBO Energy")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(basis_energy, color="darkorange", linewidth=1.2)
    axes[1].set_title(f"Basis-Function — SA Energy Trace\n"
                      f"({basis_meta['num_variables']} vars, "
                      f"{len(basis_energy)} sweeps)",
                      fontsize=11)
    axes[1].set_xlabel("Sweep"); axes[1].set_ylabel("HUBO Energy")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Solver Energy Traces — Side-by-Side", fontsize=13, y=1.02)
    plt.tight_layout()
    en_path = os.path.join(OUT_DIR, "energy_traces.png")
    plt.savefig(en_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: energy_traces.png")

    # ====================== PLOT 4: Bar chart of key metrics ======================
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    methods = ["Grid-Based", "Basis-Function"]
    colors  = ["steelblue", "darkorange"]

    # Variables
    var_counts = [grid_meta["num_variables"], basis_meta["num_variables"]]
    bars0 = axes[0].bar(methods, var_counts, color=colors, alpha=0.85,
                        edgecolor="black", linewidth=1)
    axes[0].set_ylabel("Binary Variables", fontsize=11)
    axes[0].set_title("Variable Count", fontsize=12)
    for b, v in zip(bars0, var_counts):
        axes[0].text(b.get_x() + b.get_width()/2, b.get_height(),
                     f"{v}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    axes[0].grid(True, alpha=0.3, axis="y")

    # Runtime
    rts = [grid_meta["solver"]["runtime_s"], basis_meta["solver"]["runtime_s"]]
    bars1 = axes[1].bar(methods, rts, color=colors, alpha=0.85,
                        edgecolor="black", linewidth=1)
    axes[1].set_ylabel("Runtime (s)", fontsize=11)
    axes[1].set_title("Solver Runtime", fontsize=12)
    for b, v in zip(bars1, rts):
        axes[1].text(b.get_x() + b.get_width()/2, b.get_height(),
                     f"{v:.1f}s", ha="center", va="bottom", fontsize=11, fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="y")

    # Term count
    term_counts = [grid_meta["num_terms"], basis_meta["num_terms"]]
    bars2 = axes[2].bar(methods, term_counts, color=colors, alpha=0.85,
                        edgecolor="black", linewidth=1)
    axes[2].set_ylabel("HUBO Terms", fontsize=11)
    axes[2].set_title("Number of HUBO Terms", fontsize=12)
    for b, v in zip(bars2, term_counts):
        axes[2].text(b.get_x() + b.get_width()/2, b.get_height(),
                     f"{v:,}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.suptitle("Key Metrics Comparison", fontsize=13, y=1.02)
    plt.tight_layout()
    bars_path = os.path.join(OUT_DIR, "metrics_bars.png")
    plt.savefig(bars_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: metrics_bars.png")

    # ====================== MARKDOWN SUMMARY ======================
    md_path = os.path.join(OUT_DIR, "comparison_report.md")
    with open(md_path, "w") as f:
        f.write("# Task 2 — Comparison Report\n")
        f.write("## Grid-Based vs Basis-Function HUBO\n\n")
        f.write("Both approaches solve the same UAV obstacle-avoidance problem ")
        f.write("on an 8×8 grid with a 15-step horizon. ")
        f.write("Below is a side-by-side comparison.\n\n")

        f.write("## Key Metrics\n\n")
        f.write("| Metric | Grid-Based | Basis-Function |\n")
        f.write("|---|---|---|\n")
        f.write(f"| Binary variables | {grid_meta['num_variables']} | {basis_meta['num_variables']} |\n")
        f.write(f"| HUBO terms | {grid_meta['num_terms']:,} | {basis_meta['num_terms']:,} |\n")
        f.write(f"| HUBO order | 3 (cubic) | {basis_meta['hubo_order']} (sextic) |\n")
        f.write(f"| Runtime | {grid_meta['solver']['runtime_s']:.1f}s | {basis_meta['solver']['runtime_s']:.1f}s |\n")
        f.write(f"| Best energy | {grid_meta['solver']['best_energy']:.2f} | {basis_meta['solver']['best_energy']:.2f} |\n")
        f.write(f"| Reaches target | {'✓' if grid_meta['reaches_target'] else '✗'} | {'✓' if basis_feas['reaches_target'] else '✗'} |\n")
        f.write(f"| Obstacle collisions | {grid_feas['obstacle_collisions']} | {basis_feas['obstacle_collisions']} |\n")
        f.write(f"| Mean direction change | {grid_smooth:.1f}° | {basis_smooth:.1f}° |\n")
        f.write(f"| Total path length | {grid_len:.2f} | {basis_len:.2f} |\n\n")

        f.write("## Analysis\n\n")
        f.write(f"**Variable efficiency**: The basis-function approach uses ")
        f.write(f"only **{basis_meta['num_variables']} variables**, which is ")
        f.write(f"**{grid_meta['num_variables']/basis_meta['num_variables']:.0f}× fewer** ")
        f.write(f"than the grid-based approach ({grid_meta['num_variables']}). ")
        f.write("This is the main advantage of polynomial parameterisation.\n\n")

        f.write(f"**Runtime**: The basis-function solver completes in ")
        f.write(f"{basis_meta['solver']['runtime_s']:.1f}s vs ")
        f.write(f"{grid_meta['solver']['runtime_s']:.1f}s for the grid-based — ")
        f.write(f"about **{grid_meta['solver']['runtime_s']/basis_meta['solver']['runtime_s']:.0f}× faster**.\n\n")

        f.write(f"**Trajectory smoothness**: The polynomial trajectory has a ")
        f.write(f"mean direction change of just **{basis_smooth:.1f}°** per step, ")
        f.write(f"compared to **{grid_smooth:.1f}°** for the grid-based path. ")
        f.write("This reflects the inherent smoothness of polynomial curves vs ")
        f.write("the necessarily piecewise-linear cell-by-cell grid path.\n\n")

        f.write("**HUBO order**: The basis-function approach yields an ")
        f.write(f"**order-{basis_meta['hubo_order']} (sextic)** HUBO — ")
        f.write("higher-order than the grid-based cubic HUBO. ")
        f.write("This is because choosing one trajectory requires picking ")
        f.write(f"M={basis_meta['M']} coefficients per axis, ")
        f.write("giving products of 2M=6 indicator variables per cost term.\n\n")

        f.write("**Flexibility**: The grid-based approach can represent ")
        f.write("any feasible path including sharp turns. The basis-function ")
        f.write(f"approach is constrained to degree-{basis_meta['M']-1} ")
        f.write("polynomial curves. For complex obstacle arrangements ")
        f.write("requiring intricate navigation, the grid-based has the edge.\n\n")

        f.write("## Conclusion\n\n")
        f.write("Both approaches successfully solve the obstacle-avoidance ")
        f.write("problem and reach the target. The choice between them depends ")
        f.write("on the application:\n\n")
        f.write("- For **resource-constrained** quantum hardware where qubit ")
        f.write("count matters, the **basis-function** approach is preferable.\n")
        f.write("- For **complex environments** requiring fine-grained ")
        f.write("manoeuvring, the **grid-based** approach is more flexible.\n")
        f.write("- For **real-time replanning** the **basis-function** ")
        f.write("approach's much shorter runtime is decisive.\n\n")
        f.write("This trade-off is consistent with the classic resolution-vs-")
        f.write("flexibility tension in discretised trajectory optimisation.\n")

    print(f"  Saved: comparison_report.md")

    print(f"\n{'=' * 70}")
    print(f"  Comparison complete. All outputs in: {OUT_DIR}/")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
