"""
=============================================================================
Task 2 — Basis-Function HUBO for UAV Obstacle Avoidance
=============================================================================
Based on:
  Masnavi et al., "Real-Time Multi-Convex Model Predictive Control for
  Occlusion-Free Target Tracking With Quadrotors," IEEE Access, 2022.

FORMULATION OVERVIEW
--------------------
Unlike the grid-based approach (which encodes UAV position cell-by-cell),
this implementation follows the paper's polynomial-basis parameterisation:

    x(t) = c_x[0] + c_x[1]·t + c_x[2]·t²
    y(t) = c_y[0] + c_y[1]·t + c_y[2]·t²

To make this a HUBO, each coefficient c_x[m] / c_y[m] is discretised into
one of K candidate values via ONE-HOT encoding:

    b_x[m, k] ∈ {0, 1}      with     Σ_k b_x[m, k] = 1
    c_x[m]   = Σ_k b_x[m, k] · candidate_value[m, k]

VARIABLE LAYOUT
---------------
  M = 3 basis functions    (constant, linear, quadratic)
  K = 5 candidate values per coefficient
  2 axes (x, y)
  Start coefficients (m=0) are HARD-PINNED → 1 candidate only
  Higher-order coefficients (m=1, 2) → K=5 candidates each

  Total binary variables = 2·(1 + K + K) = 2·11 = 22

HUBO ORDER
----------
Each trajectory is uniquely determined by ONE choice per (axis, m).
A trajectory-cost term is therefore a product of 2M = 6 one-hot
indicator variables — making this an order-6 HUBO. Higher than the
grid-based approach's cubic order, this is a genuinely
"higher-order Ising" problem.

CONTRAST WITH GRID-BASED APPROACH
----------------------------------
  Grid-based   : 960 binary variables, cubic order, 87,684 terms
  Basis-fn     : ~22 binary variables, sextic order, ~3,125 terms

The trade-off is resolution vs flexibility:
  - Grid-based: any path representable, but bulky (1 var per cell per step)
  - Basis-fn  : compact (only 6 numbers per trajectory), but constrained
                to smooth polynomial curves

OUTPUTS  →  ./outputs/basis/
  bitstring.npy / bitstring.txt    best solution bitstring
  trajectory.csv                   decoded UAV path (continuous + rounded)
  energy_history.npy               SA energy per sweep
  metadata.json                    statistics and result summary
  trajectory.png                   UAV polynomial curve on the grid
  cost_landscape.png               per-cell obstacle + LOS occlusion cost

Requirements: numpy, matplotlib
=============================================================================
"""

import os, json, time, itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "outputs", "basis")
os.makedirs(OUT_DIR, exist_ok=True)

HUBO = Dict[Tuple[int, ...], float]


# ===========================================================================
# 1.  SCENARIO  (same as grid-based — shared problem definition)
# ===========================================================================

@dataclass
class Scenario:
    """
    8×8 grid UAV navigation problem — identical to grid-based version
    so the two methods solve the exact same physical problem.
    """
    grid_size : int = 8
    horizon   : int = 15

    start  : Tuple[int,int] = (0, 0)
    target : Tuple[int,int] = (7, 7)

    obstacles : List[Tuple[int,int]] = field(default_factory=lambda: [
        (3,3),(3,4),(4,3),(4,4),
        (2,6),(5,1),
    ])
    buffer_radius : int = 1

    # Cost weights (kept proportional to grid-based)
    lambda_obs      : float = 80.0
    lambda_occ      : float = 10.0
    lambda_prox     : float = 5.0
    lambda_move     : float = 50.0   # smoothness penalty (per-step jump)
    lambda_terminal : float = 50.0
    lambda_goal     : float = 8.0
    lambda_uniq     : float = 50.0   # one-hot enforcement weight

    def cell_index(self, r:int, c:int) -> int: return r*self.grid_size + c
    def index_to_cell(self, v:int) -> Tuple[int,int]: return divmod(v, self.grid_size)

    @property
    def num_cells(self) -> int: return self.grid_size**2

    @property
    def obstacle_indices(self) -> Set[int]:
        return {self.cell_index(r,c) for r,c in self.obstacles}

    @property
    def buffer_indices(self) -> Set[int]:
        obs = self.obstacle_indices
        buf: Set[int] = set()
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                v = self.cell_index(r,c)
                if v in obs: continue
                for or_,oc in self.obstacles:
                    if max(abs(r-or_),abs(c-oc)) <= self.buffer_radius:
                        buf.add(v); break
        return buf

    def line_of_sight(self, u:int, v:int) -> List[int]:
        r0,c0 = self.index_to_cell(u)
        r1,c1 = self.index_to_cell(v)
        cells=[]
        dr,dc = abs(r1-r0),abs(c1-c0)
        sr = 1 if r0<r1 else -1
        sc = 1 if c0<c1 else -1
        err=dr-dc; r,c=r0,c0
        while True:
            cells.append(self.cell_index(r,c))
            if r==r1 and c==c1: break
            e2=2*err
            if e2>-dc: err-=dc; r+=sr
            if e2< dr: err+=dr; c+=sc
        return cells[1:-1] if len(cells)>2 else []

    def occlusion_count(self, u:int) -> int:
        tv = self.cell_index(*self.target)
        return sum(1 for w in self.line_of_sight(u,tv)
                   if w in self.obstacle_indices)


# ===========================================================================
# 2.  BASIS-FUNCTION HUBO BUILDER
# ===========================================================================

class BasisHUBO:
    """
    Polynomial-basis HUBO for UAV obstacle avoidance.

    Variables
    ---------
        For each axis (x, y) and each basis index m (0..M-1):
            One-hot binary vector b[axis, m, k] for k in 0..K(m)-1
        Start coefficients (m=0) are hard-pinned with K(0)=1
        Higher-order coefficients (m>0) use K candidates

    Indexing
    --------
        var(axis, m, k) returns the global binary variable index

    Cost function
    -------------
        H = Σ_traj  trajectory_cost(traj) · Π_indicators
            + λ_uniq · one-hot enforcement penalties
    """

    def __init__(
        self,
        scenario : Scenario,
        M : int = 3,    # number of basis functions: 1, t, t²
        K : int = 5,    # candidate values for m>0 (m=0 is pinned)
    ):
        self.s = scenario
        self.M = M
        self.K = K
        self.T = scenario.horizon

        # Per-coefficient candidate counts: m=0 is hard-pinned (start = 0)
        # so only one candidate. m≥1 uses K candidates.
        # K_per[axis][m] = number of candidates
        self.K_per = []
        for axis in range(2):
            row = [1]                  # m=0 → pinned
            for m in range(1, M):
                row.append(K)
            self.K_per.append(row)

        # Candidate values for each (axis, m).
        # axis 0 = x (row index), axis 1 = y (column index)
        # We choose ranges that allow paths from (0,0) to (grid-1, grid-1).
        max_grid = scenario.grid_size - 1
        T_horizon = self.T - 1     # last time index
        self.candidates = []
        for axis in range(2):
            row = []
            # m=0 : pinned at start (row 0 for axis=0, col 0 for axis=1)
            start_value = scenario.start[axis]
            row.append(np.array([float(start_value)]))
            # m=1 : linear slope. Range allows reaching the opposite corner
            slope_max = max_grid / T_horizon * 1.5
            row.append(np.linspace(-slope_max, slope_max, K))
            # m≥2 : higher-order, smaller range
            for m in range(2, M):
                cap = max_grid / (T_horizon ** m) * 3.0
                row.append(np.linspace(-cap, cap, K))
            self.candidates.append(row)

        # Build variable index lookup
        self._build_var_index()

    def _build_var_index(self):
        """Assign global binary variable indices."""
        self._var = {}
        idx = 0
        for axis in range(2):
            for m in range(self.M):
                for k in range(self.K_per[axis][m]):
                    self._var[(axis, m, k)] = idx
                    idx += 1
        self.num_vars = idx

    def var(self, axis: int, m: int, k: int) -> int:
        return self._var[(axis, m, k)]

    # ---------------------- trajectory evaluation ----------------------

    def trajectory_from_choices(
        self,
        x_choices: List[int],
        y_choices: List[int],
    ) -> List[Tuple[float, float]]:
        """
        Given M chosen candidate indices per axis, return the continuous
        polynomial trajectory at each time step.
        """
        cx = np.array([self.candidates[0][m][x_choices[m]] for m in range(self.M)])
        cy = np.array([self.candidates[1][m][y_choices[m]] for m in range(self.M)])
        traj = []
        for t in range(self.T):
            x = sum(cx[m] * (t ** m) for m in range(self.M))
            y = sum(cy[m] * (t ** m) for m in range(self.M))
            traj.append((float(x), float(y)))
        return traj

    def trajectory_to_grid(
        self,
        traj_cont: List[Tuple[float, float]],
    ) -> List[Tuple[int, int]]:
        """Round continuous trajectory to nearest grid cells (clipped)."""
        gs = self.s.grid_size
        grid_traj = []
        for x, y in traj_cont:
            r = max(0, min(gs - 1, int(round(x))))
            c = max(0, min(gs - 1, int(round(y))))
            grid_traj.append((r, c))
        return grid_traj

    # ---------------------- physical cost per trajectory ----------------

    def physical_cost(self, traj_cont: List[Tuple[float, float]]) -> float:
        """
        Cost of a single complete polynomial trajectory.
        Combines all penalty terms (obs, occ, prox, smoothness, goal,
        terminal) into one scalar. This scalar becomes the coefficient
        on the corresponding order-2M HUBO term.
        """
        traj_grid = self.trajectory_to_grid(traj_cont)
        cost = 0.0

        # Obstacle collisions
        for cell in traj_grid:
            v = self.s.cell_index(*cell)
            if v in self.s.obstacle_indices:
                cost += self.s.lambda_obs

        # LOS occlusion
        for cell in traj_grid:
            v = self.s.cell_index(*cell)
            occ = self.s.occlusion_count(v)
            if occ > 0:
                cost += self.s.lambda_occ * occ

        # Proximity — sustained buffer cell occupancy (3 in a row)
        B = self.s.buffer_indices
        for t in range(1, len(traj_grid) - 1):
            v_prev = self.s.cell_index(*traj_grid[t - 1])
            v_curr = self.s.cell_index(*traj_grid[t])
            v_next = self.s.cell_index(*traj_grid[t + 1])
            if v_prev in B and v_curr in B and v_next in B:
                cost += self.s.lambda_prox * 3.0

        # Smoothness penalty (replaces H_move) — large per-step jumps cost extra
        # This corresponds to bounded velocity in the paper.
        for t in range(len(traj_cont) - 1):
            x0, y0 = traj_cont[t]
            x1, y1 = traj_cont[t + 1]
            jump = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
            if jump > 1.5:           # any step > 1.5 cells is a kinematic violation
                cost += self.s.lambda_move * (jump - 1.5)

        # Goal attraction
        tr, tc = self.s.target
        for t, (x, y) in enumerate(traj_cont):
            dist = ((x - tr) ** 2 + (y - tc) ** 2) ** 0.5
            w = (t + 1) / len(traj_cont)
            cost += self.s.lambda_goal * dist * w

        # Terminal bonus
        x_final, y_final = traj_cont[-1]
        terminal_err = ((x_final - tr) ** 2 + (y_final - tc) ** 2) ** 0.5
        cost += self.s.lambda_terminal * terminal_err

        return cost

    # ---------------------- HUBO construction ----------------------

    def build(self) -> HUBO:
        """
        Build the complete order-2M HUBO.

        Two parts:
          1. One-hot enforcement penalty for each (axis, m) where K>1
          2. Physical cost over all enumerated trajectories
        """
        terms: HUBO = {}

        # ---- one-hot enforcement ----
        # For each (axis, m) with multiple candidates:
        #   λ_uniq · ( Σ_k b[axis,m,k]  -  1 )²
        for axis in range(2):
            for m in range(self.M):
                Km = self.K_per[axis][m]
                if Km <= 1:
                    continue
                vars_mk = [self.var(axis, m, k) for k in range(Km)]
                # Linear: −λ · b[k]
                for v in vars_mk:
                    _add(terms, (v,), -1.0 * self.s.lambda_uniq)
                # Quadratic: +2λ · b[i]·b[j] for i<j
                for i in range(Km):
                    for j in range(i + 1, Km):
                        key = tuple(sorted([vars_mk[i], vars_mk[j]]))
                        _add(terms, key, 2.0 * self.s.lambda_uniq)

        # ---- physical cost over all trajectories ----
        # For each combination of choices (x_choices, y_choices):
        #   coefficient = physical_cost(corresponding trajectory)
        #   term = product of 2M one-hot indicators
        K_choices_x = [list(range(self.K_per[0][m])) for m in range(self.M)]
        K_choices_y = [list(range(self.K_per[1][m])) for m in range(self.M)]

        for x_choices in itertools.product(*K_choices_x):
            for y_choices in itertools.product(*K_choices_y):
                traj = self.trajectory_from_choices(list(x_choices), list(y_choices))
                cost = self.physical_cost(traj)
                if cost == 0.0:
                    continue
                # Build the order-2M product term
                key_vars = []
                for m in range(self.M):
                    key_vars.append(self.var(0, m, x_choices[m]))
                for m in range(self.M):
                    key_vars.append(self.var(1, m, y_choices[m]))
                key = tuple(sorted(key_vars))
                _add(terms, key, cost)

        return terms

    # ---------------------- evaluation / decode ----------------------

    def evaluate(self, hubo: HUBO, bits: np.ndarray) -> float:
        total = 0.0
        for key, coeff in hubo.items():
            prod = 1
            for idx in key:
                prod *= bits[idx]
                if prod == 0: break
            total += coeff * prod
        return total

    def decode(self, bits: np.ndarray) -> Tuple[List[int], List[int]]:
        """Read x_choices and y_choices from a bitstring."""
        x_choices = []
        y_choices = []
        for m in range(self.M):
            # x axis
            picks = []
            for k in range(self.K_per[0][m]):
                if bits[self.var(0, m, k)] == 1:
                    picks.append(k)
            x_choices.append(picks[0] if picks else 0)
            # y axis
            picks = []
            for k in range(self.K_per[1][m]):
                if bits[self.var(1, m, k)] == 1:
                    picks.append(k)
            y_choices.append(picks[0] if picks else 0)
        return x_choices, y_choices

    def term_summary(self, hubo: HUBO) -> Dict[str, int]:
        c = {}
        for key in hubo:
            n = len(key)
            label = {1: "linear", 2: "quadratic", 3: "cubic",
                     4: "quartic", 5: "quintic", 6: "sextic"}.get(n, f"order-{n}")
            c[label] = c.get(label, 0) + 1
        return c

    def feasibility_report(self, x_choices, y_choices) -> dict:
        """Check feasibility of decoded trajectory."""
        traj_cont = self.trajectory_from_choices(x_choices, y_choices)
        traj_grid = self.trajectory_to_grid(traj_cont)

        report = {
            "start_violation": False,
            "obstacle_collisions": 0,
            "max_step_jump": 0.0,
            "kinematic_violations": 0,
            "reaches_target": False,
            "final_position": tuple(traj_grid[-1]),
            "target_distance": 0.0,
        }

        if traj_grid[0] != self.s.start:
            report["start_violation"] = True

        for cell in traj_grid:
            v = self.s.cell_index(*cell)
            if v in self.s.obstacle_indices:
                report["obstacle_collisions"] += 1

        for t in range(len(traj_cont) - 1):
            x0, y0 = traj_cont[t]
            x1, y1 = traj_cont[t + 1]
            jump = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
            if jump > report["max_step_jump"]:
                report["max_step_jump"] = jump
            if jump > 1.5:
                report["kinematic_violations"] += 1

        tr, tc = self.s.target
        final_dist = ((traj_grid[-1][0] - tr) ** 2 +
                      (traj_grid[-1][1] - tc) ** 2) ** 0.5
        report["target_distance"] = final_dist
        report["reaches_target"] = (traj_grid[-1] == self.s.target)

        return report


# ===========================================================================
# 3.  SIMULATED ANNEALING — one-hot preserving
# ===========================================================================

def simulated_annealing_basis(
    builder    : BasisHUBO,
    hubo       : HUBO,
    num_sweeps : int = 1000,
    T_start    : float = 5.0,
    T_end      : float = 0.01,
    seed       : int = 42,
    verbose    : bool = True,
) -> dict:
    """
    SA over one-hot blocks.

    State: for each (axis, m), which candidate index k is selected.
    Move : pick a random (axis, m) with K>1 and change k to a new value.

    This preserves the one-hot constraint automatically. Energy is
    evaluated by mapping the choices to a bitstring, then evaluating
    the HUBO.
    """
    rng = np.random.default_rng(seed)
    M = builder.M

    # Initialize: middle candidate for each non-pinned coefficient
    x_state = []
    y_state = []
    for m in range(M):
        Km_x = builder.K_per[0][m]
        Km_y = builder.K_per[1][m]
        x_state.append(Km_x // 2)
        y_state.append(Km_y // 2)

    def make_bits(xs, ys):
        bits = np.zeros(builder.num_vars, dtype=np.int8)
        for m in range(M):
            bits[builder.var(0, m, xs[m])] = 1
            bits[builder.var(1, m, ys[m])] = 1
        return bits

    bits = make_bits(x_state, y_state)
    energy = builder.evaluate(hubo, bits)

    best_x, best_y = list(x_state), list(y_state)
    best_energy = energy

    history = []
    alpha = (T_end / T_start) ** (1.0 / num_sweeps)
    T = T_start
    n_eval = n_acc = 0
    t0 = time.perf_counter()

    # List of mutable (axis, m) pairs (those with Km > 1)
    mutables = []
    for axis in range(2):
        for m in range(M):
            if builder.K_per[axis][m] > 1:
                mutables.append((axis, m))

    for sweep in range(num_sweeps):
        # Each sweep: one mutation per mutable coefficient
        for axis, m in mutables:
            Km = builder.K_per[axis][m]
            # Propose a new k different from current
            old_k = x_state[m] if axis == 0 else y_state[m]
            new_k = rng.integers(0, Km)
            if new_k == old_k:
                new_k = (new_k + 1) % Km

            # Apply mutation
            if axis == 0:
                x_state[m] = new_k
            else:
                y_state[m] = new_k

            new_bits = make_bits(x_state, y_state)
            new_energy = builder.evaluate(hubo, new_bits)

            delta = new_energy - energy
            n_eval += 1
            if delta <= 0 or rng.random() < np.exp(-delta / T):
                bits = new_bits
                energy = new_energy
                n_acc += 1
                if energy < best_energy:
                    best_energy = energy
                    best_x = list(x_state)
                    best_y = list(y_state)
            else:
                # Revert
                if axis == 0:
                    x_state[m] = old_k
                else:
                    y_state[m] = old_k

        history.append(energy)
        T *= alpha

        if verbose and sweep % max(1, num_sweeps // 10) == 0:
            print(f"  sweep {sweep:4d}/{num_sweeps}  T={T:7.4f}  "
                  f"energy={energy:10.2f}  best={best_energy:10.2f}")

    return dict(
        best_x_choices = best_x,
        best_y_choices = best_y,
        best_energy    = float(best_energy),
        history        = history,
        runtime        = time.perf_counter() - t0,
        n_eval         = n_eval,
        n_acc          = n_acc,
    )


# ===========================================================================
# 4.  HELPERS
# ===========================================================================

def _add(d: HUBO, key: tuple, c: float):
    if c: d[key] = d.get(key, 0.0) + c


# ===========================================================================
# 5.  VISUALISATION
# ===========================================================================

def plot_trajectory(scenario, builder, x_choices, y_choices, path):
    """Plot the continuous polynomial trajectory on the grid."""
    gs = scenario.grid_size

    # Background: obstacles and buffer
    img = np.zeros((gs, gs))
    for v in scenario.buffer_indices:
        r, c = scenario.index_to_cell(v)
        img[r, c] = 1.0
    for r, c in scenario.obstacles:
        img[r, c] = 2.0

    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.cm.Reds
    ax.imshow(img, cmap=cmap, origin="upper", vmin=0, vmax=2, alpha=0.65)

    # Continuous trajectory (high-resolution sampling)
    cx = np.array([builder.candidates[0][m][x_choices[m]] for m in range(builder.M)])
    cy = np.array([builder.candidates[1][m][y_choices[m]] for m in range(builder.M)])
    ts = np.linspace(0, builder.T - 1, 100)
    xs_cont = sum(cx[m] * (ts ** m) for m in range(builder.M))
    ys_cont = sum(cy[m] * (ts ** m) for m in range(builder.M))

    # Plot smooth curve
    ax.plot(ys_cont, xs_cont, "-", color="navy", linewidth=2.5,
            zorder=5, label="Polynomial trajectory")

    # Plot the discrete-time-step samples
    traj_cont = builder.trajectory_from_choices(x_choices, y_choices)
    for t, (x, y) in enumerate(traj_cont):
        ax.plot(y, x, "o", color="navy", markersize=7, zorder=6)
        if t % 2 == 0 or t == len(traj_cont) - 1:
            ax.annotate(f"t={t}", (y, x), xytext=(6, 6),
                        textcoords="offset points",
                        fontsize=8, color="navy", fontweight="bold")

    # Start and target
    ax.plot(scenario.start[1], scenario.start[0],
            "go", markersize=18, markeredgecolor="black",
            zorder=7, label="Start (0,0)")
    ax.plot(scenario.target[1], scenario.target[0],
            "r*", markersize=22, markeredgecolor="black",
            zorder=7, label="Target (7,7)")

    p_obs = mpatches.Patch(color=cmap(1.0), alpha=0.7, label="Obstacle")
    p_buf = mpatches.Patch(color=cmap(0.5), alpha=0.7, label="Buffer zone")
    h, l = ax.get_legend_handles_labels()
    ax.legend(handles=h + [p_obs, p_buf], loc="upper left", fontsize=9)

    ax.set_xticks(range(gs)); ax.set_yticks(range(gs))
    ax.set_xlim(-0.5, gs - 0.5); ax.set_ylim(gs - 0.5, -0.5)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.set_title(f"Basis-Function HUBO — Polynomial Trajectory\n"
                 f"M={builder.M}, K={builder.K}, "
                 f"{builder.num_vars} binary variables", fontsize=12)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {os.path.basename(path)}")


def plot_energy_trace(history, path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history, color="steelblue", linewidth=1.2)
    ax.set_xlabel("Sweep", fontsize=12)
    ax.set_ylabel("HUBO Energy", fontsize=12)
    ax.set_title("Basis-Function HUBO — SA Energy Trace", fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {os.path.basename(path)}")


def plot_cost_landscape(scenario, path):
    """Same cost landscape as grid-based for direct comparison."""
    gs = scenario.grid_size
    cost = np.zeros((gs, gs))
    for v in range(scenario.num_cells):
        r, c = scenario.index_to_cell(v)
        if v in scenario.obstacle_indices:
            cost[r, c] = scenario.lambda_obs
        else:
            occ = scenario.occlusion_count(v)
            cost[r, c] = scenario.lambda_occ * occ

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(cost, cmap="hot_r", origin="upper")
    plt.colorbar(im, ax=ax, label="Cell cost (λ_obs·[obs] + λ_occ·|LOS∩O|)")
    ax.plot(scenario.start[1], scenario.start[0], "go",
            markersize=15, markeredgecolor="black", label="Start")
    ax.plot(scenario.target[1], scenario.target[0], "b*",
            markersize=19, markeredgecolor="black", label="Target")
    ax.set_xticks(range(gs)); ax.set_yticks(range(gs))
    ax.set_aspect("equal"); ax.grid(True, alpha=0.2, color="white")
    ax.set_title("Basis-Function HUBO — Cost Landscape\n"
                 "(Obstacle + LOS-Occlusion Penalty per Cell)", fontsize=13)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {os.path.basename(path)}")


# ===========================================================================
# 6.  SAVE OUTPUTS
# ===========================================================================

def save_all(scenario, builder, hubo, result):
    x_choices = result["best_x_choices"]
    y_choices = result["best_y_choices"]
    traj_cont = builder.trajectory_from_choices(x_choices, y_choices)
    traj_grid = builder.trajectory_to_grid(traj_cont)

    # Bitstring
    bits = np.zeros(builder.num_vars, dtype=np.int8)
    for m in range(builder.M):
        bits[builder.var(0, m, x_choices[m])] = 1
        bits[builder.var(1, m, y_choices[m])] = 1
    np.save(os.path.join(OUT_DIR, "bitstring.npy"), bits)
    with open(os.path.join(OUT_DIR, "bitstring.txt"), "w") as f:
        f.write("".join(str(int(b)) for b in bits) + "\n")
    print("  Saved: bitstring.npy / bitstring.txt")

    # Trajectory CSV (both continuous and grid)
    with open(os.path.join(OUT_DIR, "trajectory.csv"), "w") as f:
        f.write("t,x_cont,y_cont,row_grid,col_grid,obstacle,occluded\n")
        for t, ((x, y), (r, c)) in enumerate(zip(traj_cont, traj_grid)):
            v = scenario.cell_index(r, c)
            obs = int(v in scenario.obstacle_indices)
            occ = scenario.occlusion_count(v)
            f.write(f"{t},{x:.3f},{y:.3f},{r},{c},{obs},{occ}\n")
    print("  Saved: trajectory.csv")

    # Energy history
    np.save(os.path.join(OUT_DIR, "energy_history.npy"),
            np.array(result["history"]))
    print("  Saved: energy_history.npy")

    # Metadata
    summary = builder.term_summary(hubo)
    feas = builder.feasibility_report(x_choices, y_choices)

    # Extract chosen polynomial coefficients
    cx = [float(builder.candidates[0][m][x_choices[m]])
          for m in range(builder.M)]
    cy = [float(builder.candidates[1][m][y_choices[m]])
          for m in range(builder.M)]

    meta = {
        "formulation"    : "basis-function (polynomial) HUBO",
        "num_variables"  : builder.num_vars,
        "num_terms"      : len(hubo),
        "term_breakdown" : summary,
        "hubo_order"     : 2 * builder.M,
        "M"              : builder.M,
        "K"              : builder.K,
        "K_per_axis"     : builder.K_per,
        "scenario": {
            "grid_size" : scenario.grid_size,
            "horizon"   : scenario.horizon,
            "start"     : list(scenario.start),
            "target"    : list(scenario.target),
            "obstacles" : scenario.obstacles,
        },
        "polynomial_coefficients": {
            "c_x": cx,
            "c_y": cy,
        },
        "solver": {
            "method"         : "block one-hot simulated annealing",
            "num_sweeps"     : len(result["history"]),
            "best_energy"    : result["best_energy"],
            "runtime_s"      : round(result["runtime"], 3),
            "acceptance_rate": round(result["n_acc"] / max(1, result["n_eval"]), 4),
        },
        "feasibility": {
            "start_violation"        : feas["start_violation"],
            "obstacle_collisions"    : feas["obstacle_collisions"],
            "max_step_jump"          : round(feas["max_step_jump"], 3),
            "kinematic_violations"   : feas["kinematic_violations"],
            "reaches_target"         : feas["reaches_target"],
            "final_position"         : list(feas["final_position"]),
            "target_distance"        : round(feas["target_distance"], 3),
        },
        "trajectory_continuous": [
            [round(x, 3), round(y, 3)] for x, y in traj_cont],
        "trajectory_grid": [list(c) for c in traj_grid],
    }
    with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("  Saved: metadata.json")


# ===========================================================================
# 7.  MAIN
# ===========================================================================

def main():
    print("=" * 68)
    print("  Task 2  —  Basis-Function HUBO for UAV Obstacle Avoidance")
    print("=" * 68)

    scenario = Scenario()
    md = abs(scenario.target[0] - scenario.start[0]) + \
         abs(scenario.target[1] - scenario.start[1])
    print(f"\nScenario  : {scenario.grid_size}×{scenario.grid_size} grid, "
          f"horizon = {scenario.horizon} steps")
    print(f"  Start   : {scenario.start}")
    print(f"  Target  : {scenario.target}  (Manhattan dist = {md})")
    print(f"  Obstacles ({len(scenario.obstacles)}): {scenario.obstacles}")

    # ---- Build HUBO ----
    print("\nBuilding basis-function HUBO ...")
    builder = BasisHUBO(scenario, M=3, K=5)
    print(f"  M = {builder.M}, K = {builder.K}")
    print(f"  Variables : {builder.num_vars}  "
          f"(2 axes × (1 pinned + {builder.M-1}×{builder.K} candidates))")

    # Show candidate values
    axis_names = ["x (row)", "y (col)"]
    for axis in range(2):
        print(f"  Candidates for {axis_names[axis]}:")
        for m in range(builder.M):
            vals = builder.candidates[axis][m]
            print(f"    m={m}: {[round(v, 3) for v in vals]}")

    hubo = builder.build()
    summary = builder.term_summary(hubo)
    print(f"  HUBO terms     : {len(hubo)}")
    print(f"  Term breakdown : {summary}")
    print(f"  HUBO order     : {2 * builder.M}")

    # ---- Solve ----
    print("\nRunning block-one-hot simulated annealing ...")
    result = simulated_annealing_basis(
        builder    = builder,
        hubo       = hubo,
        num_sweeps = 1000,
        T_start    = 5.0,
        T_end      = 0.01,
        seed       = 42,
        verbose    = True,
    )

    # ---- Report ----
    x_choices = result["best_x_choices"]
    y_choices = result["best_y_choices"]
    feas = builder.feasibility_report(x_choices, y_choices)

    print(f"\n{'─' * 50}")
    print(f"  Best energy    : {result['best_energy']:.4f}")
    print(f"  Runtime        : {result['runtime']:.2f}s")
    print(f"  Acceptance rate: {result['n_acc'] / max(1, result['n_eval']):.4f}")
    print(f"{'─' * 50}")

    print("\nDecoded polynomial coefficients:")
    for m in range(builder.M):
        cx = builder.candidates[0][m][x_choices[m]]
        cy = builder.candidates[1][m][y_choices[m]]
        print(f"  c_x[{m}] = {cx:7.3f}  c_y[{m}] = {cy:7.3f}")

    print("\nFeasibility:")
    for k, v in feas.items():
        if isinstance(v, bool):
            status = "✓ OK" if (k == "reaches_target" and v) or \
                              (k != "reaches_target" and not v) else "✗"
            print(f"  {k:25s}: {v}  {status}")
        else:
            print(f"  {k:25s}: {v}")

    print("\nDecoded trajectory (grid-rounded):")
    traj_cont = builder.trajectory_from_choices(x_choices, y_choices)
    traj_grid = builder.trajectory_to_grid(traj_cont)
    for t, ((x, y), (r, c)) in enumerate(zip(traj_cont, traj_grid)):
        v = scenario.cell_index(r, c)
        tags = []
        if v in scenario.obstacle_indices: tags.append("OBSTACLE")
        if v in scenario.buffer_indices:   tags.append("buffer")
        occ = scenario.occlusion_count(v)
        if occ > 0: tags.append(f"occluded({occ})")
        if (r, c) == scenario.target: tags.append("TARGET ★")
        tag_str = "  ← " + ", ".join(tags) if tags else ""
        print(f"  t={t:2d}: cont=({x:6.2f}, {y:6.2f})  "
              f"grid={(r, c)}{tag_str}")

    # ---- Save outputs ----
    print("\nSaving outputs ...")
    save_all(scenario, builder, hubo, result)
    plot_trajectory(scenario, builder, x_choices, y_choices,
                    os.path.join(OUT_DIR, "trajectory.png"))
    plot_energy_trace(result["history"],
                      os.path.join(OUT_DIR, "energy_trace.png"))
    plot_cost_landscape(scenario,
                        os.path.join(OUT_DIR, "cost_landscape.png"))

    # ---- Final summary ----
    print(f"\n{'=' * 68}")
    print(f"  FINAL SUMMARY")
    print(f"{'=' * 68}")
    print(f"  Formulation    : Basis-function (polynomial) HUBO")
    print(f"  Variables      : {builder.num_vars}  "
          f"(vs 960 in grid-based)")
    print(f"  HUBO order     : {2 * builder.M}")
    print(f"  Terms          : {len(hubo)}")
    print(f"  Best energy    : {result['best_energy']:.4f}")
    print(f"  Runtime        : {result['runtime']:.2f}s")
    print(f"  Obstacle hits  : {feas['obstacle_collisions']}")
    print(f"  Reaches target : {'YES ✓' if feas['reaches_target'] else 'NO ✗'}")
    print(f"  Final position : {feas['final_position']}")
    print(f"  Outputs        : {OUT_DIR}/")
    print(f"{'=' * 68}")


if __name__ == "__main__":
    main()
