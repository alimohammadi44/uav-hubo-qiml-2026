"""Collect one-machine timing outputs for the QIML UAV benchmark.

This script does not run solvers. It validates and consolidates the JSON files
created by ``run_m2max_benchmark.command`` into publication-review artifacts.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent

QUAV_JSON = THIS_DIR / "outputs" / "quav_style_baseline" / "quav_style_results.json"
CPSAT_JSON = THIS_DIR / "outputs" / "cp_milp_exact_baselines" / "exact_baseline_results.json"
TASK3_JSON = THIS_DIR / "outputs" / "task3_polished_v3_1" / "benchmark_results_v3_1.json"
BASE_TABLE = PROJECT_DIR / "results" / "unified_comparison_with_cpsat.csv"


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required benchmark output is missing: {path}\n"
            "Run ./run_m2max_benchmark.command from the repository root."
        )
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def command_output(args: List[str]) -> Optional[str]:
    try:
        value = subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    value = value.strip()
    return value or None


def integer_command(args: List[str]) -> Optional[int]:
    value = command_output(args)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def mac_hardware() -> Dict[str, Any]:
    hardware: Dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
    }
    if platform.system() != "Darwin":
        return hardware

    hardware.update(
        {
            "model": command_output(["sysctl", "-n", "hw.model"]),
            "chip": command_output(["sysctl", "-n", "machdep.cpu.brand_string"]),
            "physical_cpu_cores": integer_command(["sysctl", "-n", "hw.physicalcpu"]),
            "logical_cpu_cores": integer_command(["sysctl", "-n", "hw.logicalcpu"]),
            "memory_bytes": integer_command(["sysctl", "-n", "hw.memsize"]),
            "macos_version": command_output(["sw_vers", "-productVersion"]),
            "macos_build": command_output(["sw_vers", "-buildVersion"]),
        }
    )

    profiler = command_output(["system_profiler", "SPHardwareDataType", "-json"])
    if profiler:
        try:
            records = json.loads(profiler).get("SPHardwareDataType", [])
            if records:
                record = records[0]
                hardware["chip"] = record.get("chip_type") or hardware.get("chip")
                hardware["model_name"] = record.get("machine_name")
                hardware["model_identifier"] = record.get("machine_model") or hardware.get("model")
                hardware["memory"] = record.get("physical_memory")
        except json.JSONDecodeError:
            pass
    return hardware


def package_versions(names: Iterable[str]) -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def git_revision() -> Optional[str]:
    return command_output(["git", "-C", str(PROJECT_DIR), "rev-parse", "HEAD"])


def quav_result_map(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(item["method"]): item for item in payload.get("results", [])}


def runtime_row(method: str, total_s: float, units: int, unit_name: str) -> Dict[str, Any]:
    return {
        "Method": method,
        "Runtime total s": float(total_s),
        "Work units": int(units),
        "Work unit": unit_name,
        "Runtime per unit s": float(total_s) / max(1, int(units)),
    }


def collect_rows(
    quav: Dict[str, Any], cpsat: Dict[str, Any], task3: Dict[str, Any]
) -> List[Dict[str, Any]]:
    qmap = quav_result_map(quav)
    required_quav = ["A*", "RRT-best", "QUAV-style QAOA"]
    missing = [name for name in required_quav if name not in qmap]
    if missing:
        raise KeyError(f"QUAV output is missing methods: {missing}")

    cpsat_result = cpsat.get("cpsat")
    if not isinstance(cpsat_result, dict) or "runtime_s" not in cpsat_result:
        raise KeyError("CP-SAT output does not contain cpsat.runtime_s")

    neal = task3.get("d_wave_neal")
    sa = task3.get("classical_sa")
    if not isinstance(neal, dict) or not isinstance(sa, dict):
        raise KeyError("Task 3 output is missing d_wave_neal or classical_sa")

    return [
        runtime_row("A*", qmap["A*"]["runtime_seconds"], 1, "path"),
        runtime_row("RRT-best", qmap["RRT-best"]["runtime_seconds"], 40, "fixed-seed trial"),
        runtime_row(
            "QUAV-style QAOA",
            qmap["QUAV-style QAOA"]["runtime_seconds"],
            1,
            "optimizer run",
        ),
        runtime_row("CP-SAT exact", cpsat_result["runtime_s"], 1, "solve"),
        runtime_row(
            "HUBO trajectory-SA",
            sa["runtime_s"],
            int(sa.get("num_runs", 50)),
            "run",
        ),
        runtime_row(
            "HUBO->QUBO/neal",
            neal["runtime_s"],
            int(neal.get("num_reads", 50)),
            "read",
        ),
    ]


def write_runtime_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            formatted["Runtime total s"] = f"{row['Runtime total s']:.9f}"
            formatted["Runtime per unit s"] = f"{row['Runtime per unit s']:.9f}"
            writer.writerow(formatted)


def write_proposed_table(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not BASE_TABLE.is_file():
        raise FileNotFoundError(f"Base comparison table is missing: {BASE_TABLE}")
    runtime_map = {row["Method"]: row["Runtime total s"] for row in rows}
    with BASE_TABLE.open("r", newline="", encoding="utf-8") as stream:
        table_rows = list(csv.DictReader(stream))
        fieldnames = list(table_rows[0].keys())

    for row in table_rows:
        method = row["Method"]
        if method not in runtime_map:
            raise KeyError(f"No M2 Max runtime was collected for table method: {method}")
        row["Runtime s"] = f"{runtime_map[method]:.9f}"

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="Apple M2 Max")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "results")
    args = parser.parse_args()

    quav = read_json(QUAV_JSON)
    cpsat = read_json(CPSAT_JSON)
    task3 = read_json(TASK3_JSON)
    rows = collect_rows(quav, cpsat, task3)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    hardware = mac_hardware()
    chip_text = " ".join(
        str(hardware.get(key) or "") for key in ("chip", "processor", "model_name")
    )
    chip_verified = "m2 max" in chip_text.lower()

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_label": args.label,
        "apple_m2_max_detected": chip_verified,
        "git_revision": git_revision(),
        "hardware": hardware,
        "software": {
            "python": sys.version,
            "python_executable": sys.executable,
            "packages": package_versions(
                ["numpy", "matplotlib", "pandas", "scipy", "dimod", "dwave-neal", "ortools"]
            ),
        },
        "timing_protocol": {
            "scope": "solver-level wall-clock time from the existing implementations",
            "excluded": ["dependency installation", "plotting", "file serialization"],
            "cpsat_workers": 8,
            "rrt_trials": 40,
            "neal_reads": 50,
            "trajectory_sa_runs": 50,
        },
        "methods": rows,
        "source_outputs": {
            "quav": str(QUAV_JSON.relative_to(PROJECT_DIR)),
            "cpsat": str(CPSAT_JSON.relative_to(PROJECT_DIR)),
            "neal_and_sa": str(TASK3_JSON.relative_to(PROJECT_DIR)),
        },
    }

    json_path = output_dir / "m2max_runtime_benchmark.json"
    csv_path = output_dir / "m2max_runtime_benchmark.csv"
    table_path = output_dir / "unified_comparison_m2max.csv"

    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    write_runtime_csv(csv_path, rows)
    write_proposed_table(table_path, rows)

    if not chip_verified:
        print("WARNING: Apple M2 Max was not detected in the recorded hardware metadata.")
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {table_path}")


if __name__ == "__main__":
    main()

