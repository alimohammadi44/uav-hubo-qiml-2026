"""
=============================================================================
Task 2 — Grid-Based HUBO for UAV Obstacle Avoidance
=============================================================================
Based on:
  Masnavi et al., "Real-Time Multi-Convex Model Predictive Control for
  Occlusion-Free Target Tracking With Quadrotors," IEEE Access, 2022.

FORMULATION
-----------
Binary variable:  x[t, v] ∈ {0, 1}
  x[t, v] = 1  →  UAV occupies grid cell v at time t
  Total variables: horizon × |V|; upgraded conference run uses 20 × 64 = 1280

Cost terms (matching Task 1 mathematical formulation exactly):
  H_uniq    (quadratic)  one active cell per time step
  H_start   (linear)     UAV starts at known cell
  H_move    (quadratic)  only adjacent transitions
  H_obs     (linear)     hard collision avoidance
  H_occ     (linear)     LOS occlusion penalty — static target form
  H_prox    (CUBIC)      persistent near-obstacle buffer penalty ← HUBO term
  H_goal    (linear)     distance-to-target incentive
  H_terminal(linear)     strong bonus for reaching target at final step

NOTE ON H_goal AND H_terminal
------------------------------
The task statement says "tracking has already been completed separately."
H_goal and H_terminal are not the full tracking term from Task 1.
They are lightweight stand-ins so the optimizer has a directional incentive.
Without them the UAV has no reason to move from the start cell.

SOLVER DESIGN — Why Trajectory-Level SA
-----------------------------------------
Previous versions used bit-level swap moves (flip one cell OFF, adjacent
cell ON at the same time step). These preserve per-step uniqueness but
do NOT preserve inter-step feasibility: independently moving the cell at
t=5 and the cell at t=6 can create a gap between them.

This version uses TRAJECTORY-LEVEL SA:
- The state is a list of cell indices [v_0, v_1, ..., v_{T-1}]
- Each move proposes a new cell for ONE time step t, choosing only from
  cells that are adjacent to BOTH v_{t-1} AND v_{t+1}
- This guarantees H_uniq, H_start, and H_move are NEVER violated
- The bitstring is computed from the trajectory only at the end

This is the correct approach for trajectory optimization problems with
one-hot and adjacency constraints.

FIXES APPLIED vs. FIRST VERSION
---------------------------------
Fix 1: Horizon 6 → 14  (UAV can now physically reach the target)
Fix 2: Weight ratio 100:1 → 10:1  (solver can explore)
Fix 3: T_start 8 → 25, sweeps 300 → 500  (wider exploration)
Fix 4: Multiple random restarts  (escapes local optima)
Fix 5: Trajectory-level SA  (100% feasibility guaranteed)
Fix 6: Terminal bonus H_terminal  (UAV reaches exact target)

OUTPUTS  →  ./outputs/grid/
  bitstring.npy / bitstring.txt    best solution bitstring
  trajectory.csv                   decoded UAV path
  energy_history.npy               SA energy per sweep
  metadata.json                    statistics and result summary
  energy_trace.png                 energy vs. sweep
  trajectory.png                   UAV path on the grid
  cost_landscape.png               per-cell obstacle+occlusion heatmap

Requirements: numpy, matplotlib
=============================================================================
"""

import os, json, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "outputs", "grid")
os.makedirs(OUT_DIR, exist_ok=True)

HUBO = Dict[Tuple[int, ...], float]


# ===========================================================================
# 1.  SCENARIO
# ===========================================================================

@dataclass
class Scenario:
    """
    8×8 grid UAV navigation problem.
    Start (0,0) → Target (7,7), 6 obstacles, 20-layer horizon.
    """
    grid_size : int = 8
    horizon   : int = 20       # Conference run: longer than Manhattan distance=14
                               # allows detours / hovering for visibility trade-offs

    start  : Tuple[int,int] = (0, 0)
    target : Tuple[int,int] = (7, 7)

    obstacles : List[Tuple[int,int]] = field(default_factory=lambda: [
        (3,3),(3,4),(4,3),(4,4),   # central block
        (2,6),(5,1),               # scattered
    ])
    buffer_radius : int = 1

    # Hard constraints — large but not so large the solver freezes
    # Fix 2: was 1000, now 100 (10:1 ratio with soft terms)
    lambda_uniq     : float = 1000.0
    lambda_start    : float = 1000.0
    lambda_move     : float = 1000.0
    lambda_obs      : float = 1000.0
    # Soft safety
    lambda_occ      : float = 10.0
    lambda_prox     : float = 5.0
    # Goal terms (stand-ins for the separate tracking implementation)
    lambda_goal     : float = 8.0
    lambda_terminal : float = 50.0   # Fix 6: strong bonus for exact arrival

    # ---- helpers ----
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

    def neighbors(self, v:int) -> Set[int]:
        """4-connected + self (hover allowed)."""
        r,c = self.index_to_cell(v)
        nb = {v}
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr,nc = r+dr, c+dc
            if 0<=nr<self.grid_size and 0<=nc<self.grid_size:
                nb.add(self.cell_index(nr,nc))
        return nb

    def line_of_sight(self, u:int, v:int) -> List[int]:
        """Intermediate cells on Bresenham line from u to v (endpoints excluded)."""
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
        """|LOS(u, target) ∩ O|  — number of obstacle blockers."""
        tv = self.cell_index(*self.target)
        return sum(1 for w in self.line_of_sight(u,tv)
                   if w in self.obstacle_indices)

    def describe(self) -> str:
        md = abs(self.target[0]-self.start[0]) + abs(self.target[1]-self.start[1])
        return (f"Scenario  : {self.grid_size}×{self.grid_size} grid, "
                f"horizon = {self.horizon} steps\n"
                f"  Start   : {self.start}\n"
                f"  Target  : {self.target}  (Manhattan dist = {md})\n"
                f"  Obstacles ({len(self.obstacles)}): {self.obstacles}\n"
                f"  Buffer cells : {len(self.buffer_indices)}\n"
                f"  |V|          : {self.num_cells}\n"
                f"  Variables    : {self.horizon} × {self.num_cells}"
                f" = {self.horizon * self.num_cells}")


# ===========================================================================
# 2.  GRID-BASED HUBO BUILDER
# ===========================================================================

class GridHUBO:
    """
    Builds each penalty term of the HUBO as a dict of
    variable-index tuples → coefficients.
    """

    def __init__(self, s:Scenario):
        self.s=s; self.T=s.horizon; self.V=s.num_cells
        self.num_vars = self.T * self.V

    def var(self, t:int, v:int) -> int: return t*self.V + v
    def unvar(self, idx:int) -> Tuple[int,int]: return divmod(idx, self.V)

    # ---- individual terms ----

    def H_uniq(self) -> HUBO:
        """(Σ_v x[t,v] − 1)² per time step — exactly one active cell."""
        d: HUBO = {}
        for t in range(self.T):
            for v in range(self.V): _add(d,(self.var(t,v),),-1.0)
            for u in range(self.V):
                for v in range(u+1,self.V):
                    _add(d,(self.var(t,u),self.var(t,v)),2.0)
        return d

    def H_start(self) -> HUBO:
        """Force x[0,s]=1; penalise all other cells at t=0."""
        d: HUBO = {}
        s = self.s.cell_index(*self.s.start)
        _add(d,(self.var(0,s),),-1.0)
        for v in range(self.V):
            if v!=s: _add(d,(self.var(0,v),),1.0)
        return d

    def H_move(self) -> HUBO:
        """Penalise non-adjacent transitions between consecutive steps."""
        d: HUBO = {}
        for t in range(self.T-1):
            for u in range(self.V):
                nb = self.s.neighbors(u)
                for v in range(self.V):
                    if v not in nb:
                        key = tuple(sorted([self.var(t,u),self.var(t+1,v)]))
                        _add(d,key,1.0)
        return d

    def H_obs(self) -> HUBO:
        """Hard linear penalty for occupying obstacle cells."""
        d: HUBO = {}
        for t in range(self.T):
            for v in self.s.obstacle_indices:
                _add(d,(self.var(t,v),),1.0)
        return d

    def H_occ(self) -> HUBO:
        """
        Linear LOS occlusion penalty.
        ω(u) = |LOS(u, target) ∩ O| — obstacle blockers on the LOS.
        Corresponds to d_oi(t, u_j) ≥ 1 from the paper (static target).
        """
        d: HUBO = {}
        for u in range(self.V):
            cnt = self.s.occlusion_count(u)
            if cnt>0:
                for t in range(self.T):
                    _add(d,(self.var(t,u),),float(cnt))
        return d

    def H_prox(self) -> HUBO:
        """
        CUBIC proximity term.
        Activated when the UAV spends THREE consecutive steps in buffer cells.
        η_{uvw}·x[t-1,u]·x[t,v]·x[t+1,w]  for u,v,w ∈ B, v∈N(u), w∈N(v).
        This is the term that makes the model a genuine HUBO (not QUBO).
        """
        d: HUBO = {}
        B = self.s.buffer_indices
        for t in range(1,self.T-1):
            for u in B:
                for v in B & self.s.neighbors(u):
                    for w in B & self.s.neighbors(v):
                        key = tuple(sorted([
                            self.var(t-1,u),self.var(t,v),self.var(t+1,w)]))
                        _add(d,key,3.0)   # η = ρ_u+ρ_v+ρ_w = 3
        return d

    def H_goal(self) -> HUBO:
        """
        Distance-to-target cost, weighted by time step.
        Stand-in for the tracking term (implemented separately).
        lambda_goal=8.0 gives meaningful pull without overriding safety.
        """
        d: HUBO = {}
        tr,tc = self.s.target
        for v in range(self.V):
            r,c = self.s.index_to_cell(v)
            dist = ((r-tr)**2+(c-tc)**2)**0.5
            for t in range(self.T):
                _add(d,(self.var(t,v),), dist*(t+1)/self.T)
        return d

    def H_terminal(self) -> HUBO:
        """
        Strong bonus for being at the TARGET cell at the FINAL time step.
        Penalises every non-target cell at t=T-1 by lambda_terminal.
        This ensures the UAV actually arrives at (7,7).
        """
        d: HUBO = {}
        tv = self.s.cell_index(*self.s.target)
        for v in range(self.V):
            if v!=tv: _add(d,(self.var(self.T-1,v),),1.0)
        return d

    # ---- assembly ----

    def build(self) -> HUBO:
        master: HUBO = {}
        _scale_add(master, self.H_uniq(),     self.s.lambda_uniq)
        _scale_add(master, self.H_start(),    self.s.lambda_start)
        _scale_add(master, self.H_move(),     self.s.lambda_move)
        _scale_add(master, self.H_obs(),      self.s.lambda_obs)
        _scale_add(master, self.H_occ(),      self.s.lambda_occ)
        _scale_add(master, self.H_prox(),     self.s.lambda_prox)
        _scale_add(master, self.H_goal(),     self.s.lambda_goal)
        _scale_add(master, self.H_terminal(), self.s.lambda_terminal)
        return master

    # ---- utilities ----

    def trajectory_to_bits(self, traj:List[int]) -> np.ndarray:
        """Convert a list of cell indices → binary array."""
        bits = np.zeros(self.num_vars, dtype=np.int8)
        for t,v in enumerate(traj): bits[self.var(t,v)] = 1
        return bits

    def bits_to_trajectory(self, bits:np.ndarray) -> List[Optional[int]]:
        traj=[]
        for t in range(self.T):
            active=[v for v in range(self.V) if bits[self.var(t,v)]==1]
            traj.append(active[0] if len(active)==1 else None)
        return traj

    def eval_trajectory(self, hubo:HUBO, traj:List[int]) -> float:
        bits = self.trajectory_to_bits(traj)
        total = 0.0
        for key,coeff in hubo.items():
            prod=1
            for idx in key:
                prod*=bits[idx]
                if prod==0: break
            total+=coeff*prod
        return total

    def term_summary(self, hubo:HUBO) -> dict:
        c={"linear":0,"quadratic":0,"cubic":0,"higher":0}
        for key in hubo:
            n=len(key)
            if n==1: c["linear"]+=1
            elif n==2: c["quadratic"]+=1
            elif n==3: c["cubic"]+=1
            else: c["higher"]+=1
        return c

    def feasibility_report(self, traj:List[int]) -> dict:
        """Check all hard constraints on a trajectory."""
        report={"uniqueness_violations":0,"start_violation":False,
                "move_violations":0,"obstacle_collisions":0}
        if traj[0] != self.s.cell_index(*self.s.start):
            report["start_violation"]=True
        for t in range(self.T-1):
            if traj[t+1] not in self.s.neighbors(traj[t]):
                report["move_violations"]+=1
        for v in traj:
            if v in self.s.obstacle_indices:
                report["obstacle_collisions"]+=1
        return report


# ===========================================================================
# 3.  HELPERS
# ===========================================================================

def _add(d:HUBO, key:tuple, c:float):
    if c: d[key]=d.get(key,0.0)+c

def _scale_add(dst:HUBO, src:HUBO, s:float):
    for k,v in src.items(): _add(dst,k,s*v)


def greedy_init(scenario:Scenario) -> List[int]:
    """
    Greedy path toward target, one step at a time.
    Always feasible: satisfies H_start, H_move, H_uniq, avoids obstacles.
    """
    obs=scenario.obstacle_indices
    tr,tc=scenario.target
    cur=scenario.cell_index(*scenario.start)
    traj=[cur]
    for _ in range(scenario.horizon-1):
        r0,c0=scenario.index_to_cell(cur)
        best=cur; best_d=(r0-tr)**2+(c0-tc)**2
        for nv in scenario.neighbors(cur):
            if nv in obs: continue
            nr,nc=scenario.index_to_cell(nv)
            d=(nr-tr)**2+(nc-tc)**2
            if d<best_d: best_d,best=d,nv
        cur=best; traj.append(cur)
    return traj


def random_init(scenario:Scenario, seed:int=0) -> List[int]:
    """
    Random walk from start — always feasible, used for random restarts.
    """
    rng=np.random.default_rng(seed)
    obs=scenario.obstacle_indices
    cur=scenario.cell_index(*scenario.start)
    traj=[cur]
    for _ in range(scenario.horizon-1):
        cands=[v for v in scenario.neighbors(cur) if v not in obs]
        if not cands: cands=list(scenario.neighbors(cur))
        cur=int(rng.choice(cands)); traj.append(cur)
    return traj


# ===========================================================================
# 4.  TRAJECTORY-LEVEL SIMULATED ANNEALING
# ===========================================================================

def trajectory_sa(
    hubo       : HUBO,
    grid       : GridHUBO,
    init_traj  : List[int],
    num_sweeps : int   = 500,   # Fix 3: was 300
    T_start    : float = 25.0,  # Fix 3: was 8
    T_end      : float = 0.01,
    seed       : int   = 42,
    verbose    : bool  = True,
) -> dict:
    """
    Trajectory-level simulated annealing.

    State: a list of cell indices [v_0, v_1, ..., v_{T-1}]
    Move:  pick a random time step t (1 ≤ t ≤ T-2) and propose a new
           cell v' that is adjacent to BOTH v_{t-1} AND v_{t+1}.
           For t=0: fixed (start constraint).
           For t=T-1: propose any cell adjacent to v_{T-2}.

    WHY THIS GUARANTEES FEASIBILITY
    --------------------------------
    H_uniq  : each time step has exactly one cell (enforced by list repr.)
    H_start : traj[0] = start, never changed.
    H_move  : v' is chosen from neighbors(v_{t-1}) ∩ neighbors(v_{t+1}),
              so the move at t never creates a violation at t-1→t or t→t+1.
    H_obs   : obstacle cells are excluded from candidates.

    The key difference from the bit-level approach: moves at step t
    KNOW about steps t-1 and t+1 and explicitly enforce adjacency.

    ENERGY EVALUATION
    -----------------
    Only terms that contain variables for time step t change when we move
    the UAV at t. We precompute per-timestep term lists for O(1) delta
    computation per move.
    """

    rng   = np.random.default_rng(seed)
    s     = grid.s
    T, V  = grid.T, grid.V

    # Precompute: for each (t, v), the HUBO terms that contain var(t, v)
    # We only need terms that depend on the CELL at each time step.
    # Group terms by the set of time steps they involve.
    # For delta-E computation: when we change cell at time t,
    # only terms involving var(t, *) can change.
    per_t: List[List[Tuple[tuple,float]]] = [[] for _ in range(T)]
    for key,coeff in hubo.items():
        involved_t = set()
        for idx in key:
            tt,_ = grid.unvar(idx)
            involved_t.add(tt)
        for tt in involved_t:
            per_t[tt].append((key,coeff))

    # Remove duplicates
    for t in range(T):
        per_t[t] = list({k:c for k,c in per_t[t]}.items())

    def eval_terms_at_t(traj:List[int], t:int) -> float:
        bits_t = {}   # local bit lookup for speed
        # Fill in bits for time steps involved in terms at t
        for key,_ in per_t[t]:
            for idx in key:
                tt,vv = grid.unvar(idx)
                bits_t[idx] = int(vv == traj[tt])
        total=0.0
        for key,coeff in per_t[t]:
            prod=1
            for idx in key:
                prod*=bits_t[idx]
                if prod==0: break
            total+=coeff*prod
        return total

    # Initial state
    traj = list(init_traj)
    energy = grid.eval_trajectory(hubo, traj)

    alpha  = (T_end/T_start)**(1.0/num_sweeps)
    Temp   = T_start
    best_traj   = list(traj)
    best_energy = energy
    history = []
    n_eval = n_acc = 0
    t0 = time.perf_counter()

    for sweep in range(num_sweeps):
        # One sweep = T proposed moves (one per time step, random order)
        for t in rng.permutation(T):
            if t == 0:
                continue   # start is fixed

            prev = traj[t-1]
            nxt  = traj[t+1] if t < T-1 else None

            # Candidate cells: adjacent to prev AND (if exists) to nxt
            if nxt is not None:
                cands = list(s.neighbors(prev) & s.neighbors(nxt))
            else:
                cands = list(s.neighbors(prev))

            # Publication fix: constrained trajectory SA should not propose obstacle cells.
            # Otherwise obstacle avoidance is only penalized, not guaranteed by construction.
            cands = [v for v in cands if v not in s.obstacle_indices]

            if not cands:
                continue

            new_v = int(rng.choice(cands))
            if new_v == traj[t]:
                continue   # no change

            # Compute delta energy
            e_before = eval_terms_at_t(traj, t)
            old_v    = traj[t]
            traj[t]  = new_v
            e_after  = eval_terms_at_t(traj, t)
            delta    = e_after - e_before

            n_eval += 1
            if delta<=0 or rng.random()<np.exp(-delta/Temp):
                energy  += delta
                n_acc   += 1
                if energy < best_energy:
                    best_energy = energy
                    best_traj   = list(traj)
            else:
                traj[t] = old_v   # revert

        history.append(energy)
        Temp *= alpha

        if verbose and sweep % max(1,num_sweeps//10) == 0:
            print(f"  sweep {sweep:4d}/{num_sweeps}  "
                  f"T={Temp:8.4f}  energy={energy:12.2f}  "
                  f"best={best_energy:12.2f}")

    return dict(
        best_traj    = best_traj,
        best_energy  = float(best_energy),
        history      = history,
        runtime      = time.perf_counter()-t0,
        n_eval       = n_eval,
        n_acc        = n_acc,
    )


# ===========================================================================
# 5.  VISUALISATION
# ===========================================================================

def plot_energy_trace(history:List[float], path:str) -> None:
    fig,ax = plt.subplots(figsize=(10,4))
    ax.plot(history, color="steelblue", linewidth=1.2)
    ax.set_xlabel("Sweep",fontsize=12); ax.set_ylabel("HUBO Energy",fontsize=12)
    ax.set_title("Grid-based HUBO — SA Energy Trace (trajectory-level)",fontsize=13)
    ax.grid(True,alpha=0.3)
    # Shade exploration vs convergence
    half = len(history)//2
    ax.axvspan(0,half,alpha=0.06,color="orange",label="Exploration phase")
    ax.axvspan(half,len(history),alpha=0.06,color="green",label="Convergence phase")
    ax.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(path,dpi=150); plt.close()
    print(f"  Saved: {os.path.basename(path)}")


def plot_trajectory(scenario:Scenario, traj:List[int],
                    grid:GridHUBO, path:str) -> None:
    gs = scenario.grid_size
    img = np.zeros((gs,gs))
    for v in scenario.buffer_indices:
        r,c=scenario.index_to_cell(v); img[r,c]=1.0
    for r,c in scenario.obstacles: img[r,c]=2.0

    fig,ax=plt.subplots(figsize=(8,8))
    cmap=plt.cm.Reds
    ax.imshow(img,cmap=cmap,origin="upper",vmin=0,vmax=2,alpha=0.65)

    # Draw path
    ys=[scenario.index_to_cell(v)[0] for v in traj]
    xs=[scenario.index_to_cell(v)[1] for v in traj]
    ax.plot(xs,ys,"b-o",linewidth=2.5,markersize=8,zorder=5,label="UAV path")
    for t,(y,x) in enumerate(zip(ys,xs)):
        if t%2==0 or t==len(traj)-1:
            ax.annotate(f"t={t}",(x,y),xytext=(5,5),
                        textcoords="offset points",
                        fontsize=8,color="navy",fontweight="bold")

    ax.plot(scenario.start[1],scenario.start[0],"go",markersize=18,
            markeredgecolor="black",zorder=6,label="Start (0,0)")
    ax.plot(scenario.target[1],scenario.target[0],"r*",markersize=22,
            markeredgecolor="black",zorder=6,label="Target (7,7)")

    p_obs=mpatches.Patch(color=cmap(1.0),alpha=0.7,label="Obstacle")
    p_buf=mpatches.Patch(color=cmap(0.5),alpha=0.7,label="Buffer zone")
    h,l=ax.get_legend_handles_labels()
    ax.legend(handles=h+[p_obs,p_buf],loc="upper left",fontsize=9)

    ax.set_xticks(range(gs)); ax.set_yticks(range(gs))
    ax.set_xlim(-0.5,gs-0.5); ax.set_ylim(gs-0.5,-0.5)
    ax.set_aspect("equal"); ax.grid(True,alpha=0.3)
    ax.set_title(f"Grid-based HUBO — UAV Trajectory  "
                 f"(horizon={scenario.horizon}, 100% feasible)",fontsize=13)
    plt.tight_layout(); plt.savefig(path,dpi=150); plt.close()
    print(f"  Saved: {os.path.basename(path)}")


def plot_cost_landscape(scenario:Scenario, path:str) -> None:
    gs=scenario.grid_size
    cost=np.zeros((gs,gs))
    for v in range(scenario.num_cells):
        r,c=scenario.index_to_cell(v)
        if v in scenario.obstacle_indices: cost[r,c]=scenario.lambda_obs
        else:
            occ=scenario.occlusion_count(v)
            cost[r,c]=scenario.lambda_occ*occ

    fig,ax=plt.subplots(figsize=(7,7))
    im=ax.imshow(cost,cmap="hot_r",origin="upper")
    plt.colorbar(im,ax=ax,label="Cell cost (λ_obs·[obs] + λ_occ·|LOS∩O|)")
    ax.plot(scenario.start[1],scenario.start[0],"go",markersize=15,
            markeredgecolor="black",label="Start")
    ax.plot(scenario.target[1],scenario.target[0],"b*",markersize=19,
            markeredgecolor="black",label="Target")
    ax.set_xticks(range(gs)); ax.set_yticks(range(gs))
    ax.set_aspect("equal"); ax.grid(True,alpha=0.2,color="white")
    ax.set_title("Grid-based HUBO — Cost Landscape\n"
                 "(Obstacle + LOS-Occlusion Penalty per Cell)",fontsize=13)
    ax.legend(loc="upper left",fontsize=9)
    plt.tight_layout(); plt.savefig(path,dpi=150); plt.close()
    print(f"  Saved: {os.path.basename(path)}")


def index_to_cell(scenario, v): return scenario.index_to_cell(v)


# ===========================================================================
# 6.  SAVE STRUCTURED OUTPUTS
# ===========================================================================

def save_all(scenario, grid, best_traj, best_energy, history,
             runtime, n_eval, n_acc, hubo) -> None:

    bits = grid.trajectory_to_bits(best_traj)

    # bitstring
    np.save(os.path.join(OUT_DIR,"bitstring.npy"), bits)
    with open(os.path.join(OUT_DIR,"bitstring.txt"),"w") as f:
        f.write("".join(str(int(b)) for b in bits)+"\n")
    print("  Saved: bitstring.npy / bitstring.txt")

    # trajectory CSV
    tv = scenario.cell_index(*scenario.target)
    with open(os.path.join(OUT_DIR,"trajectory.csv"),"w") as f:
        f.write("t,row,col,occluded,in_buffer,obstacle,at_target\n")
        for t,v in enumerate(best_traj):
            r,c=scenario.index_to_cell(v)
            occ=scenario.occlusion_count(v)
            buf=int(v in scenario.buffer_indices)
            obs=int(v in scenario.obstacle_indices)
            tgt=int(v==tv)
            f.write(f"{t},{r},{c},{occ},{buf},{obs},{tgt}\n")
    print("  Saved: trajectory.csv")

    # energy history
    np.save(os.path.join(OUT_DIR,"energy_history.npy"), np.array(history))
    print("  Saved: energy_history.npy")

    # metadata
    summary   = grid.term_summary(hubo)
    feas      = grid.feasibility_report(best_traj)
    meta={
        "formulation"    : "grid-based HUBO",
        "num_variables"  : grid.num_vars,
        "num_terms"      : len(hubo),
        "term_breakdown" : summary,
        "is_genuine_HUBO": summary["cubic"]>0,
        "scenario":{
            "grid_size"  : scenario.grid_size,
            "horizon"    : scenario.horizon,
            "start"      : list(scenario.start),
            "target"     : list(scenario.target),
            "obstacles"  : scenario.obstacles,
            "manhattan_dist": (abs(scenario.target[0]-scenario.start[0])+
                               abs(scenario.target[1]-scenario.start[1])),
        },
        "penalty_weights":{
            "lambda_uniq"    :scenario.lambda_uniq,
            "lambda_start"   :scenario.lambda_start,
            "lambda_move"    :scenario.lambda_move,
            "lambda_obs"     :scenario.lambda_obs,
            "lambda_occ"     :scenario.lambda_occ,
            "lambda_prox"    :scenario.lambda_prox,
            "lambda_goal"    :scenario.lambda_goal,
            "lambda_terminal":scenario.lambda_terminal,
        },
        "solver":{
            "method"         :"trajectory-level simulated annealing",
            "num_sweeps"     :len(history),
            "T_start"        :25.0, "T_end":0.01,
            "num_restarts"   :5,
            "best_energy"    :best_energy,
            "runtime_s"      :round(runtime,3),
            "acceptance_rate":round(n_acc/max(1,n_eval),4),
        },
        "feasibility"   :feas,
        "reaches_target": best_traj[-1]==scenario.cell_index(*scenario.target),
        "trajectory"    :[[*scenario.index_to_cell(v)] for v in best_traj],
    }
    with open(os.path.join(OUT_DIR,"metadata.json"),"w") as f:
        json.dump(meta,f,indent=2)
    print("  Saved: metadata.json")


# ===========================================================================
# 7.  MAIN
# ===========================================================================

def main():
    print("="*68)
    print("  Task 2  —  Grid-Based HUBO for UAV Obstacle Avoidance")
    print("="*68)

    scenario = Scenario()
    print(f"\n{scenario.describe()}\n")

    # ---- Build HUBO ----
    print("Building HUBO ...")
    grid    = GridHUBO(scenario)
    hubo    = grid.build()
    summary = grid.term_summary(hubo)
    print(f"  Variables      : {grid.num_vars}")
    print(f"  Total terms    : {len(hubo)}")
    print(f"  Term breakdown : {summary}")
    print(f"  Cubic terms    : {summary['cubic']}  → genuine HUBO (not QUBO)")

    # ---- Greedy init ----
    print("\nBuilding greedy initial trajectory ...")
    g_traj = greedy_init(scenario)
    g_energy = grid.eval_trajectory(hubo, g_traj)
    g_feas   = grid.feasibility_report(g_traj)
    print(f"  Path   : {[scenario.index_to_cell(v) for v in g_traj]}")
    print(f"  Energy : {g_energy:.2f}")
    print(f"  Feasibility: {g_feas}")

    # ---- Multiple restarts (Fix 4) ----
    print("\nRunning SA with multiple restarts ...")
    NUM_RESTARTS = 5
    best_result  = None
    best_energy  = float("inf")
    all_history  = []
    total_runtime = 0.0
    total_n_eval  = 0
    total_n_acc   = 0

    for restart in range(NUM_RESTARTS):
        print(f"\n  --- Restart {restart+1}/{NUM_RESTARTS} ---")
        if restart == 0:
            init = g_traj
            print(f"  Init: greedy path  (energy {g_energy:.2f})")
        else:
            init = random_init(scenario, seed=restart*17)
            e0   = grid.eval_trajectory(hubo, init)
            print(f"  Init: random walk  (energy {e0:.2f})")

        res = trajectory_sa(
            hubo=hubo, grid=grid, init_traj=init,
            num_sweeps=500, T_start=25.0, T_end=0.01,
            seed=42+restart, verbose=True,
        )
        all_history.extend(res["history"])
        total_runtime += res["runtime"]
        total_n_eval  += res["n_eval"]
        total_n_acc   += res["n_acc"]
        print(f"  Best energy this restart: {res['best_energy']:.4f}")
        if res["best_energy"] < best_energy:
            best_energy = res["best_energy"]
            best_result = res
            print(f"  ✓ New overall best!")

    best_traj = best_result["best_traj"]

    # ---- Report ----
    feas = grid.feasibility_report(best_traj)
    ok   = all(v==0 or v is False for v in feas.values())
    at_target = best_traj[-1] == scenario.cell_index(*scenario.target)

    print(f"\n{'─'*52}")
    print(f"  Best energy across all restarts : {best_energy:.4f}")
    print(f"  Total runtime                   : {total_runtime:.2f}s")
    print(f"  Acceptance rate                 : "
          f"{total_n_acc/max(1,total_n_eval):.4f}")
    print(f"{'─'*52}")

    print("\nFeasibility check:")
    for k,v in feas.items():
        status = "✓ OK" if (v==0 or v is False) else "✗ VIOLATED"
        print(f"  {k:35s}: {v}  {status}")
    print(f"  Overall: {'✓ FEASIBLE' if ok else '✗ INFEASIBLE'}")
    print(f"  Reaches target at final step: {'YES ✓' if at_target else 'NO ✗'}")

    print("\nDecoded UAV trajectory:")
    tv = scenario.cell_index(*scenario.target)
    for t,v in enumerate(best_traj):
        cell = scenario.index_to_cell(v)
        tags=[]
        if v in scenario.obstacle_indices: tags.append("OBSTACLE")
        if v in scenario.buffer_indices:   tags.append("buffer")
        occ = scenario.occlusion_count(v)
        if occ>0: tags.append(f"occluded({occ})")
        if v==tv: tags.append("TARGET ★")
        tag_str = "  ← "+", ".join(tags) if tags else ""
        print(f"  t={t:2d}: {str(cell):12s}{tag_str}")

    # ---- Save outputs ----
    print("\nSaving outputs ...")
    save_all(scenario, grid, best_traj, best_energy, all_history,
             total_runtime, total_n_eval, total_n_acc, hubo)
    plot_energy_trace(all_history, os.path.join(OUT_DIR,"energy_trace.png"))
    plot_trajectory(scenario, best_traj, grid,
                    os.path.join(OUT_DIR,"trajectory.png"))
    plot_cost_landscape(scenario, os.path.join(OUT_DIR,"cost_landscape.png"))

    # ---- Final summary ----
    print(f"\n{'='*68}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*68}")
    print(f"  Formulation   : Grid-based HUBO")
    print(f"  Variables     : {grid.num_vars}  "
          f"(horizon {scenario.horizon} × {scenario.num_cells} cells)")
    print(f"  Terms         : {len(hubo)}  "
          f"({summary['cubic']} cubic → genuine HUBO)")
    print(f"  Best energy   : {best_energy:.4f}")
    print(f"  Runtime       : {total_runtime:.2f}s  "
          f"({NUM_RESTARTS} restarts × 500 sweeps)")
    print(f"  Feasible      : {'YES ✓' if ok else 'NO ✗'}")
    print(f"  Reaches target: {'YES ✓' if at_target else 'NO ✗'}")
    print(f"  Outputs       : {OUT_DIR}/")
    print(f"{'='*68}")


if __name__ == "__main__":
    main()
