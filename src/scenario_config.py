"""Validated JSON configuration for UAV grid-planning experiments."""

from __future__ import annotations

import json
import re
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from task2_grid_hubo import Scenario

Cell = Tuple[int, int]

SCENARIO_FIELDS = {
    "grid_size",
    "horizon",
    "start",
    "target",
    "obstacles",
    "buffer_radius",
}

WEIGHT_FIELDS = {
    "lambda_uniq",
    "lambda_start",
    "lambda_move",
    "lambda_obs",
    "lambda_occ",
    "lambda_prox",
    "lambda_goal",
    "lambda_terminal",
}


def _cell(value: Any, label: str) -> Cell:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must be a two-element [row, column] coordinate.")
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        raise ValueError(f"{label} coordinates must be integers: {value!r}")
    return int(value[0]), int(value[1])


def shortest_path_distance(scenario: Scenario) -> int | None:
    """Four-connected start-to-target distance with obstacle cells removed."""
    blocked = set(scenario.obstacles)
    queue = deque([(scenario.start, 0)])
    seen = {scenario.start}
    while queue:
        cell, distance = queue.popleft()
        if cell == scenario.target:
            return distance
        row, col = cell
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (row + dr, col + dc)
            if not (0 <= nxt[0] < scenario.grid_size and 0 <= nxt[1] < scenario.grid_size):
                continue
            if nxt in blocked or nxt in seen:
                continue
            seen.add(nxt)
            queue.append((nxt, distance + 1))
    return None


def validate_scenario(scenario: Scenario, require_reachable: bool = True) -> None:
    if not isinstance(scenario.grid_size, int) or isinstance(scenario.grid_size, bool):
        raise ValueError("grid_size must be the integer 8 for publication experiments.")
    if scenario.grid_size != 8:
        raise ValueError("grid_size must be exactly 8 for publication experiments.")
    if not isinstance(scenario.horizon, int) or scenario.horizon < 2:
        raise ValueError("horizon must be an integer of at least 2.")
    if not isinstance(scenario.buffer_radius, int) or scenario.buffer_radius < 0:
        raise ValueError("buffer_radius must be a nonnegative integer.")

    def in_bounds(cell: Cell) -> bool:
        return all(0 <= coordinate < scenario.grid_size for coordinate in cell)

    if not in_bounds(scenario.start):
        raise ValueError(f"start {scenario.start} is outside the grid.")
    if not in_bounds(scenario.target):
        raise ValueError(f"target {scenario.target} is outside the grid.")
    if scenario.start == scenario.target:
        raise ValueError("start and target must be different cells.")
    if len(set(scenario.obstacles)) != len(scenario.obstacles):
        raise ValueError("obstacle coordinates must be unique.")
    for obstacle in scenario.obstacles:
        if not in_bounds(obstacle):
            raise ValueError(f"obstacle {obstacle} is outside the grid.")
    if scenario.start in scenario.obstacles:
        raise ValueError("the start cell cannot be an obstacle.")
    if scenario.target in scenario.obstacles:
        raise ValueError("the target cell cannot be an obstacle.")

    if require_reachable:
        distance = shortest_path_distance(scenario)
        if distance is None:
            raise ValueError("target is unreachable from the start.")
        if distance > scenario.horizon - 1:
            raise ValueError(
                f"shortest path requires {distance} moves, but horizon={scenario.horizon} "
                f"allows only {scenario.horizon - 1}."
            )


def scenario_from_mapping(
    mapping: Mapping[str, Any],
    defaults: Mapping[str, Any] | None = None,
) -> Scenario:
    """Create and validate a Scenario from a run mapping plus optional defaults."""
    merged: Dict[str, Any] = dict(defaults or {})
    merged.update(mapping)
    unknown = set(merged) - SCENARIO_FIELDS - WEIGHT_FIELDS - {
        "name",
        "solver",
        "notes",
        "require_reachable",
    }
    if unknown:
        raise ValueError(f"Unknown scenario fields: {sorted(unknown)}")

    missing = [key for key in ("grid_size", "horizon", "start", "target", "obstacles") if key not in merged]
    if missing:
        raise ValueError(f"Missing required scenario fields: {missing}")

    kwargs: Dict[str, Any] = {
        "grid_size": merged["grid_size"],
        "horizon": merged["horizon"],
        "start": _cell(merged["start"], "start"),
        "target": _cell(merged["target"], "target"),
        "obstacles": [_cell(value, f"obstacles[{index}]") for index, value in enumerate(merged["obstacles"])],
        "buffer_radius": merged.get("buffer_radius", 1),
    }
    for key in WEIGHT_FIELDS:
        if key in merged:
            kwargs[key] = float(merged[key])
    scenario = Scenario(**kwargs)
    validate_scenario(scenario, require_reachable=bool(merged.get("require_reachable", True)))
    return scenario


def load_run_plan(path: str | Path) -> Dict[str, Any]:
    plan_path = Path(path).expanduser().resolve()
    with plan_path.open("r", encoding="utf-8") as stream:
        plan = json.load(stream)
    if not isinstance(plan, dict):
        raise ValueError("Run plan must be a JSON object.")
    runs = plan.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("Run plan must contain a nonempty 'runs' array.")
    defaults = plan.get("scenario_defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("scenario_defaults must be a JSON object.")
    names = []
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise ValueError(f"runs[{index}] must be a JSON object.")
        name = run.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"runs[{index}].name must be a nonempty string.")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) is None:
            raise ValueError(
                f"runs[{index}].name must contain only letters, digits, '.', '_', or '-' "
                "and must begin with a letter or digit."
            )
        names.append(name)
        scenario_from_mapping(run, defaults)
    if len(set(names)) != len(names):
        raise ValueError("Run names must be unique.")
    plan["_path"] = str(plan_path)
    return plan


def selected_runs(plan: Mapping[str, Any], names: Iterable[str] | None = None) -> List[Dict[str, Any]]:
    runs = [dict(run) for run in plan["runs"]]
    if not names:
        return runs
    requested = set(names)
    available = {run["name"] for run in runs}
    missing = requested - available
    if missing:
        raise ValueError(f"Unknown run names {sorted(missing)}; available runs are {sorted(available)}")
    return [run for run in runs if run["name"] in requested]
