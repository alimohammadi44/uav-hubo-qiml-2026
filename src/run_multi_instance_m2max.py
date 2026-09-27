"""Run a validated JSON experiment plan on the paper's full method suite."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
DEFAULT_PLAN = PROJECT_DIR / "configs" / "multi_instance_plan.json"

sys.path.insert(0, str(THIS_DIR))
import cp_milp_exact_baselines as exact_module
from task2_grid_hubo import GridHUBO, Scenario
from scenario_config import load_run_plan, scenario_from_mapping, selected_runs, shortest_path_distance
from task3_polished_v3_1 import (
    solve_with_neal_proper,
    solve_with_sa_500_sweeps,
)
from quav_style_baseline import (
    astar_path,
    buffer_count,
    evaluate_method,
    generate_candidate_paths,
    optimize_qaoa,
    path_length,
    quav_style_cost,
    rrt_grid_path,
    turn_count,
    visibility_rate,
)

Cell = Tuple[int, int]


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> List[float]:
    if total <= 0:
        return [0.0, 0.0]
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return [max(0.0, center - radius), min(1.0, center + radius)]


def stochastic_summary(samples: Sequence[Dict[str, Any]], runtime: float) -> Dict[str, Any]:
    total = len(samples)
    counts = {
        "feasible": sum(bool(x["feasible"]) for x in samples),
        "goal": sum(bool(x["reaches_target"]) for x in samples),
        "full": sum(bool(x["full_task_success"]) for x in samples),
    }
    full_pool = [x for x in samples if x["full_task_success"]]
    feasible_pool = [x for x in samples if x["feasible"]]
    representative = (full_pool or feasible_pool or list(samples))[0]
    summary: Dict[str, Any] = {
        "sample_count": total,
        "runtime_total_s": float(runtime),
        "runtime_per_sample_s": float(runtime) / max(1, total),
        "counts": counts,
        "rates": {key: value / total for key, value in counts.items()},
        "wilson_95": {key: wilson_interval(value, total) for key, value in counts.items()},
        "representative_selection": "best_full_success" if full_pool else ("best_feasible" if feasible_pool else "best_energy"),
        "representative": {key: value for key, value in representative.items() if key != "bits"},
        "best_energy": float(samples[0]["total_energy"]),
        "best_soft_cost": float(samples[0]["soft_cost"]),
        "mean_energy": float(np.mean([x["total_energy"] for x in samples])),
        "std_energy": float(np.std([x["total_energy"] for x in samples], ddof=1)),
    }
    return summary


def representative_metrics(summary: Dict[str, Any], scenario: Scenario) -> Dict[str, Any]:
    trajectory = summary["representative"]["trajectory"]
    if any(cell is None for cell in trajectory):
        return {"path_length": None, "turns": None, "buffer_steps": None, "visibility_rate": None}
    path = [tuple(cell) for cell in trajectory]
    return {
        "path_length": path_length(path),
        "turns": turn_count(path),
        "buffer_steps": buffer_count(path, scenario),
        "visibility_rate": visibility_rate(path, scenario),
    }


def deterministic_row(result: Dict[str, Any], runtime_divisor: int = 1) -> Dict[str, Any]:
    return {
        "feasible": bool(result["valid_motion"] and result["collision_free"]),
        "goal": bool(result["reaches_target"]),
        "full": bool(result["full_task_success"]),
        "path_length": result["path_length_steps"],
        "turns": result["turns"],
        "buffer_steps": result["buffer_steps"],
        "visibility_rate": result["visibility_rate"],
        "hubo_energy": result["hubo_energy"],
        "runtime_total_s": result["runtime_seconds"],
        "runtime_per_run_s": result["runtime_seconds"] / runtime_divisor,
    }


def merged_solver_settings(
    defaults: Mapping[str, Any] | None,
    overrides: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    settings: Dict[str, Any] = {
        "solver_seed": 42,
        "rrt_first_seed": 200,
        "rrt_trials": 40,
        "rrt_max_iterations": 1200,
        "rrt_goal_bias": 0.12,
        "candidate_random_astar": 80,
        "candidate_rrt": 80,
        "qaoa_max_candidates": 12,
        "qaoa_layers": 1,
        "qaoa_steps": 60,
        "cpsat_time_limit_s": 300.0,
        "cpsat_workers": 8,
        "neal_reads": 50,
        "neal_sweeps": 3000,
        "sa_runs": 50,
        "sa_sweeps": 500,
    }
    settings.update(defaults or {})
    settings.update(overrides or {})
    integer_keys = {
        "solver_seed", "rrt_first_seed", "rrt_trials", "rrt_max_iterations",
        "candidate_random_astar", "candidate_rrt", "qaoa_max_candidates",
        "qaoa_layers", "qaoa_steps", "cpsat_workers", "neal_reads",
        "neal_sweeps", "sa_runs", "sa_sweeps",
    }
    for key in integer_keys:
        settings[key] = int(settings[key])
        if settings[key] < 1 and key not in {"rrt_first_seed", "solver_seed"}:
            raise ValueError(f"Solver setting {key} must be positive.")
    settings["rrt_goal_bias"] = float(settings["rrt_goal_bias"])
    if not 0.0 <= settings["rrt_goal_bias"] <= 1.0:
        raise ValueError("Solver setting rrt_goal_bias must be between 0 and 1.")
    settings["cpsat_time_limit_s"] = float(settings["cpsat_time_limit_s"])
    return settings


def run_case(
    case_name: str,
    scenario: Scenario,
    settings: Mapping[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    obstacles = list(scenario.obstacles)
    print(
        f"\n{'=' * 78}\nCASE {case_name}: "
        f"grid={scenario.grid_size}x{scenario.grid_size}, start={scenario.start}, "
        f"target={scenario.target}, obstacles={obstacles}\n{'=' * 78}",
        flush=True,
    )
    grid = GridHUBO(scenario)
    hubo = grid.build()
    term_summary = grid.term_summary(hubo)

    t0 = time.perf_counter()
    a_path = astar_path(scenario)
    astar_runtime = time.perf_counter() - t0
    astar = evaluate_method("A*", a_path, scenario, grid, hubo)
    astar["runtime_seconds"] = astar_runtime

    t0 = time.perf_counter()
    rrt_seeds = list(
        range(
            settings["rrt_first_seed"],
            settings["rrt_first_seed"] + settings["rrt_trials"],
        )
    )
    rrt_paths = [
        rrt_grid_path(
            scenario,
            seed=seed,
            max_iter=settings["rrt_max_iterations"],
            goal_bias=settings["rrt_goal_bias"],
        )
        for seed in rrt_seeds
    ]
    rrt_path = min(rrt_paths, key=lambda path: quav_style_cost(path, scenario))
    rrt_runtime = time.perf_counter() - t0
    rrt = evaluate_method("RRT-best", rrt_path, scenario, grid, hubo)
    rrt["runtime_seconds"] = rrt_runtime

    candidates = generate_candidate_paths(
        scenario,
        n_random_astar=settings["candidate_random_astar"],
        n_rrt=settings["candidate_rrt"],
    )
    candidates = sorted(candidates, key=lambda path: quav_style_cost(path, scenario))[
        : settings["qaoa_max_candidates"]
    ]
    t0 = time.perf_counter()
    qaoa_info = optimize_qaoa(
        np.asarray([quav_style_cost(path, scenario) for path in candidates]),
        p_layers=settings["qaoa_layers"],
        steps=settings["qaoa_steps"],
        seed=settings["solver_seed"],
    )
    qaoa_runtime = time.perf_counter() - t0
    qaoa_path = candidates[qaoa_info["selected_index"]]
    qaoa = evaluate_method("QUAV-style QAOA", qaoa_path, scenario, grid, hubo)
    qaoa["runtime_seconds"] = qaoa_runtime

    print("Running CP-SAT...", flush=True)
    cpsat = exact_module.solve_exact_cpsat(
        time_limit_s=settings["cpsat_time_limit_s"],
        workers=settings["cpsat_workers"],
        scenario=scenario,
    )
    print(f"CP-SAT status={cpsat['status']} runtime={cpsat['runtime_s']:.3f}s", flush=True)

    print(
        f"Running neal ({settings['neal_reads']} reads x "
        f"{settings['neal_sweeps']} sweeps)...",
        flush=True,
    )
    neal_result = solve_with_neal_proper(
        hubo, grid.num_vars, scenario, grid,
        num_reads=settings["neal_reads"],
        num_sweeps=settings["neal_sweeps"],
        seed=settings["solver_seed"],
    )
    neal_summary = stochastic_summary(neal_result["samples"], neal_result["runtime"])
    print(f"neal rates={neal_summary['rates']} runtime={neal_result['runtime']:.3f}s", flush=True)

    print(
        f"Running trajectory-SA ({settings['sa_runs']} runs x "
        f"{settings['sa_sweeps']} sweeps)...",
        flush=True,
    )
    sa_result = solve_with_sa_500_sweeps(
        scenario,
        grid,
        hubo,
        num_runs=settings["sa_runs"],
        num_sweeps=settings["sa_sweeps"],
    )
    sa_summary = stochastic_summary(sa_result["samples"], sa_result["runtime"])
    print(f"SA rates={sa_summary['rates']} runtime={sa_result['runtime']:.3f}s", flush=True)

    methods: Dict[str, Any] = {
        "A*": deterministic_row(astar),
        "RRT-best": deterministic_row(rrt, len(rrt_seeds)),
        "QUAV-style QAOA": deterministic_row(qaoa),
    }
    cp_metrics = cpsat["metrics"]
    methods["CP-SAT exact"] = {
        "feasible": cp_metrics["feasible"],
        "goal": cp_metrics["goal_arrival"],
        "full": cp_metrics["full_success"],
        "path_length": cp_metrics["path_length"],
        "turns": cp_metrics["turns"],
        "buffer_steps": cp_metrics["buffer_steps"],
        "visibility_rate": cp_metrics["visibility_rate"],
        "hubo_energy": cp_metrics["native_hubo_energy"],
        "soft_cost": cp_metrics["soft_cost_if_feasible"],
        "runtime_total_s": cpsat["runtime_s"],
        "runtime_per_run_s": cpsat["runtime_s"],
        "status": cpsat["status"],
        "is_optimal": cpsat["is_optimal"],
    }
    for name, summary in (("HUBO trajectory-SA", sa_summary), ("HUBO->QUBO/neal", neal_summary)):
        rep = representative_metrics(summary, scenario)
        methods[name] = {
            "feasible_rate": summary["rates"]["feasible"],
            "goal_rate": summary["rates"]["goal"],
            "full_rate": summary["rates"]["full"],
            "wilson_95": summary["wilson_95"],
            **rep,
            "hubo_energy": summary["representative"]["total_energy"],
            "soft_cost": summary["representative"]["soft_cost"],
            "runtime_total_s": summary["runtime_total_s"],
            "runtime_per_run_s": summary["runtime_per_sample_s"],
            "representative_selection": summary["representative_selection"],
        }

    payload = {
        "case": case_name,
        "scenario": {
            "grid_size": scenario.grid_size,
            "horizon": scenario.horizon,
            "start": list(scenario.start),
            "target": list(scenario.target),
            "obstacles": [list(cell) for cell in obstacles],
            "buffer_radius": scenario.buffer_radius,
            "shortest_path_distance": shortest_path_distance(scenario),
            "buffer_cell_count": len(scenario.buffer_indices),
            "weights": {
                "lambda_uniq": scenario.lambda_uniq,
                "lambda_start": scenario.lambda_start,
                "lambda_move": scenario.lambda_move,
                "lambda_obs": scenario.lambda_obs,
                "lambda_occ": scenario.lambda_occ,
                "lambda_prox": scenario.lambda_prox,
                "lambda_goal": scenario.lambda_goal,
                "lambda_terminal": scenario.lambda_terminal,
            },
        },
        "hubo": {
            "original_binary_variables": grid.num_vars,
            "polynomial_terms": len(hubo),
            **term_summary,
        },
        "seeds_and_budgets": {**settings, "rrt_seeds": rrt_seeds},
        "methods": methods,
        "details": {
            "qaoa_candidate_count": len(candidates),
            "qaoa": qaoa_info,
            "cpsat": cpsat,
            "neal": neal_summary,
            "trajectory_sa": sa_summary,
        },
    }
    with (output_dir / f"{case_name}.json").open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    return payload


def command_text(args: Sequence[str]) -> str | None:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip() or None
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def package_versions() -> Dict[str, str | None]:
    """Record the installed solver stack used for an experiment."""
    packages = ("numpy", "scipy", "dimod", "dwave-neal", "ortools", "matplotlib")
    versions: Dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def project_relative_path(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        return str(resolved.relative_to(PROJECT_DIR))
    except ValueError:
        return str(resolved)


def hardware_metadata() -> Dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "chip": command_text(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "model": command_text(["sysctl", "-n", "hw.model"]),
        "memory_bytes": int(command_text(["sysctl", "-n", "hw.memsize"]) or 0),
        "physical_cpu_cores": int(command_text(["sysctl", "-n", "hw.physicalcpu"]) or 0),
        "logical_cpu_cores": int(command_text(["sysctl", "-n", "hw.logicalcpu"]) or 0),
    }


def save_summary(
    cases: Iterable[Dict[str, Any]],
    metadata: Dict[str, Any],
    output_dir: Path,
) -> None:
    cases = list(cases)
    rows = []
    for case in cases:
        for method, metrics in case["methods"].items():
            rows.append({
                "case": case["case"],
                "grid_size": case["scenario"]["grid_size"],
                "horizon": case["scenario"]["horizon"],
                "start": str(case["scenario"]["start"]),
                "target": str(case["scenario"]["target"]),
                "obstacles": str(case["scenario"]["obstacles"]),
                "method": method,
                "feasible": metrics.get("feasible", metrics.get("feasible_rate")),
                "goal": metrics.get("goal", metrics.get("goal_rate")),
                "full": metrics.get("full", metrics.get("full_rate")),
                "path_length": metrics.get("path_length"),
                "turns": metrics.get("turns"),
                "buffer_steps": metrics.get("buffer_steps"),
                "visibility_rate": metrics.get("visibility_rate"),
                "hubo_energy": metrics.get("hubo_energy"),
                "soft_cost": metrics.get("soft_cost"),
                "runtime_total_s": metrics.get("runtime_total_s"),
                "runtime_per_run_s": metrics.get("runtime_per_run_s"),
            })
    with (output_dir / "multi_instance_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        **metadata,
        "cases": [
            {
                "case": case["case"],
                "obstacles": case["scenario"]["obstacles"],
                "output": f"{case['case']}.json",
            }
            for case in cases
        ],
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")


def plot_cases(cases: Sequence[Dict[str, Any]], output_dir: Path) -> None:
    fig, axes = plt.subplots(
        1,
        len(cases),
        figsize=(max(5.0, 4.8 * len(cases)), 4.6),
        constrained_layout=True,
    )
    if len(cases) == 1:
        axes = [axes]
    for ax, case in zip(axes, cases):
        obstacles = {tuple(cell) for cell in case["scenario"]["obstacles"]}
        grid_size = int(case["scenario"]["grid_size"])
        start = tuple(case["scenario"]["start"])
        target = tuple(case["scenario"]["target"])
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(grid_size - 0.5, -0.5)
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))
        ax.grid(True, alpha=0.4)
        for r, c in obstacles:
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="black"))
        ax.scatter([start[1]], [start[0]], s=100, color="green", marker="o", edgecolor="black", label="Start")
        ax.scatter([target[1]], [target[0]], s=160, color="red", marker="*", edgecolor="black", label="Goal")
        ax.set_title(
            f"{case['case']}: {grid_size}x{grid_size}, "
            f"{len(obstacles)} obstacles"
        )
        ax.set_aspect("equal")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.savefig(output_dir / "configured_maps.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one or more validated grid maps from a JSON plan."
    )
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument(
        "--run",
        action="append",
        dest="run_names",
        help="Run only this named case; repeat the option to select several.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and print the configured maps without running solvers.",
    )
    parser.add_argument(
        "--allow-hardware-mismatch",
        action="store_true",
        help="Run even when required_chip does not match the current machine.",
    )
    parser.add_argument(
        "--allow-dirty-source",
        action="store_true",
        help="Run even when the Git working tree has uncommitted changes.",
    )
    args = parser.parse_args()

    plan = load_run_plan(args.plan)
    run_specs = selected_runs(plan, args.run_names)
    scenario_defaults = plan.get("scenario_defaults", {})
    solver_defaults = plan.get("solver_defaults", {})
    configured = [
        (
            spec["name"],
            scenario_from_mapping(spec, scenario_defaults),
            merged_solver_settings(solver_defaults, spec.get("solver")),
        )
        for spec in run_specs
    ]
    print(f"Validated plan: {plan['_path']}", flush=True)
    for name, scenario, settings in configured:
        print(
            f"  {name}: grid={scenario.grid_size}x{scenario.grid_size}, "
            f"horizon={scenario.horizon}, start={scenario.start}, "
            f"target={scenario.target}, obstacles={scenario.obstacles}, "
            f"neal={settings['neal_reads']}x{settings['neal_sweeps']}, "
            f"SA={settings['sa_runs']}x{settings['sa_sweeps']}",
            flush=True,
        )
    if args.validate_only:
        return

    hardware = hardware_metadata()
    required_chip = str(plan.get("required_chip", "")).strip()
    if (
        required_chip
        and required_chip.lower() not in str(hardware.get("chip", "")).lower()
        and not args.allow_hardware_mismatch
    ):
        raise RuntimeError(
            f"Plan requires chip {required_chip!r}, but detected {hardware.get('chip')!r}. "
            "Use --allow-hardware-mismatch only for non-comparable diagnostic runs."
        )
    print(f"Hardware verified: {hardware}", flush=True)

    source_revision = command_text(["git", "-C", str(PROJECT_DIR), "rev-parse", "HEAD"])
    source_status = command_text(["git", "-C", str(PROJECT_DIR), "status", "--porcelain"]) or ""
    if source_status and not args.allow_dirty_source:
        raise RuntimeError(
            "The Git working tree is not clean. Commit or stash changes before a "
            "publication run, or use --allow-dirty-source for a diagnostic run."
        )

    experiment_name = str(plan.get("experiment_name", "configurable_experiment"))
    safe_name = "".join(character if character.isalnum() or character in "-_" else "_" for character in experiment_name)
    safe_name = safe_name or "configurable_experiment"
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else PROJECT_DIR / "results" / safe_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_snapshot = {key: value for key, value in plan.items() if key != "_path"}
    with (output_dir / "plan_used.json").open("w", encoding="utf-8") as stream:
        json.dump(plan_snapshot, stream, indent=2)
        stream.write("\n")
    started = datetime.now(timezone.utc)
    outputs = [
        run_case(name, scenario, settings, output_dir)
        for name, scenario, settings in configured
    ]
    metadata = {
        "schema_version": 2,
        "experiment_name": experiment_name,
        "plan_path": project_relative_path(plan["_path"]),
        "command_argv": [project_relative_path(sys.argv[0]), *sys.argv[1:]],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "started_at_utc": started.isoformat(),
        "git_revision": source_revision,
        "source_clean_at_start": not bool(source_status),
        "python_version": platform.python_version(),
        "package_versions": package_versions(),
        "hardware": hardware,
        "required_chip": required_chip or None,
        "selected_runs": [name for name, _scenario, _settings in configured],
    }
    save_summary(outputs, metadata, output_dir)
    plot_cases(outputs, output_dir)
    print(f"\nComplete. Results saved under {output_dir}", flush=True)


if __name__ == "__main__":
    main()
