"""Cross-check camera-ready claims against committed experiment artifacts."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "results" / "multi_instance_m2max"
sys.path.insert(0, str(ROOT / "src"))
from task2_grid_hubo import GridHUBO, Scenario

CASE_NAMES = (
    "primary_6_obstacles",
    "random_2_obstacles",
    "random_3_obstacles",
)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def publication_table_module():
    path = ROOT / "src" / "generate_publication_tables.py"
    spec = importlib.util.spec_from_file_location("publication_tables", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PublicationConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = load_json(ROOT / "configs" / "multi_instance_plan.json")
        cls.manifest = load_json(RESULT_DIR / "manifest.json")
        cls.cases = {
            name: load_json(RESULT_DIR / f"{name}.json")
            for name in CASE_NAMES
        }

    def test_plan_snapshot_and_provenance(self) -> None:
        self.assertEqual(load_json(RESULT_DIR / "plan_used.json"), self.plan)
        self.assertEqual(self.manifest["schema_version"], 2)
        self.assertEqual(self.manifest["selected_runs"], list(CASE_NAMES))
        self.assertTrue(self.manifest["source_clean_at_start"])
        self.assertRegex(self.manifest["git_revision"], r"^[0-9a-f]{40}$")
        self.assertIn("Apple M2 Max", self.manifest["hardware"]["chip"])
        self.assertEqual(self.manifest["hardware"]["memory_bytes"], 32 * 1024**3)
        for package in ("numpy", "scipy", "dimod", "dwave-neal", "ortools"):
            self.assertIsNotNone(self.manifest["package_versions"][package])

    def test_case_geometry_budgets_and_exact_status(self) -> None:
        plan_runs = {run["name"]: run for run in self.plan["runs"]}
        for name, case in self.cases.items():
            scenario = case["scenario"]
            self.assertEqual(scenario["grid_size"], 8)
            self.assertEqual(scenario["horizon"], 20)
            self.assertEqual(scenario["start"], [0, 0])
            self.assertEqual(scenario["target"], [7, 7])
            self.assertEqual(scenario["obstacles"], plan_runs[name]["obstacles"])
            self.assertEqual(case["details"]["neal"]["sample_count"], 50)
            self.assertEqual(case["details"]["trajectory_sa"]["sample_count"], 50)
            cpsat = case["methods"]["CP-SAT exact"]
            self.assertEqual(cpsat["status"], "OPTIMAL")
            self.assertTrue(cpsat["is_optimal"])

    def test_primary_model_size_and_weights_match_paper(self) -> None:
        scenario = Scenario()
        grid = GridHUBO(scenario)
        hubo = grid.build()
        self.assertEqual(grid.num_vars, 1280)
        self.assertEqual(len(hubo), 118344)
        self.assertEqual(
            grid.term_summary(hubo),
            {"linear": 1280, "quadratic": 112672, "cubic": 4392, "higher": 0},
        )
        primary_plan = self.plan["runs"][0]
        self.assertEqual([list(cell) for cell in scenario.obstacles], primary_plan["obstacles"])
        self.assertEqual(
            (
                scenario.lambda_uniq,
                scenario.lambda_start,
                scenario.lambda_move,
                scenario.lambda_obs,
                scenario.lambda_occ,
                scenario.lambda_prox,
                scenario.lambda_goal,
                scenario.lambda_terminal,
            ),
            (1000.0, 1000.0, 1000.0, 1000.0, 10.0, 5.0, 8.0, 50.0),
        )

    def test_stochastic_rates_and_wilson_intervals(self) -> None:
        z = 1.959963984540054
        for case in self.cases.values():
            for detail_name in ("trajectory_sa", "neal"):
                summary = case["details"][detail_name]
                total = summary["sample_count"]
                for metric in ("feasible", "goal", "full"):
                    successes = summary["counts"][metric]
                    rate = successes / total
                    self.assertAlmostEqual(summary["rates"][metric], rate)
                    denominator = 1 + z * z / total
                    center = (rate + z * z / (2 * total)) / denominator
                    radius = z * math.sqrt(
                        rate * (1 - rate) / total + z * z / (4 * total * total)
                    ) / denominator
                    expected = (max(0.0, center - radius), min(1.0, center + radius))
                    stored = summary["wilson_95"][metric]
                    self.assertAlmostEqual(stored[0], expected[0])
                    self.assertAlmostEqual(stored[1], expected[1])

    def test_primary_rates_and_energies_match_table_2_source(self) -> None:
        with (ROOT / "results" / "unified_comparison_with_cpsat.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            csv_rows = {row["Method"]: row for row in csv.DictReader(stream)}
        methods = self.cases["primary_6_obstacles"]["methods"]
        for name, row in csv_rows.items():
            method = methods[name]
            self.assertAlmostEqual(
                float(method["hubo_energy"]),
                float(row["Native HUBO energy"]),
                places=2,
            )
        for name in ("HUBO trajectory-SA", "HUBO->QUBO/neal"):
            method = methods[name]
            row = csv_rows[name]
            for key, column in (
                ("feasible_rate", "Feasible"),
                ("goal_rate", "Goal arrival"),
                ("full_rate", "Full success"),
            ):
                expected = float(row[column].rstrip("%")) / 100.0
                self.assertAlmostEqual(method[key], expected)

    def test_generated_tables_are_current(self) -> None:
        module = publication_table_module()
        self.assertEqual(
            (ROOT / "paper" / "generated_primary_table.tex").read_text(encoding="utf-8"),
            module.render_primary(),
        )
        self.assertEqual(
            (ROOT / "paper" / "generated_sensitivity_table.tex").read_text(encoding="utf-8"),
            module.render_sensitivity(),
        )

    def test_abstract_and_scope_claims_match_results(self) -> None:
        abstract = (ROOT / "paper" / "abstract_body.tex").read_text(encoding="utf-8")
        main = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
        standalone = (ROOT / "paper" / "abstract.tex").read_text(encoding="utf-8")
        self.assertIn(r"\input{abstract_body}", main)
        self.assertIn(r"\input{abstract_body}", standalone)
        sa_rates = [
            round(100 * self.cases[name]["methods"]["HUBO trajectory-SA"]["full_rate"])
            for name in CASE_NAMES
        ]
        self.assertIn(
            f"SA full-task success is {sa_rates[0]}\\%, {sa_rates[1]}\\%, and {sa_rates[2]}\\%",
            abstract,
        )
        neal_full = [
            self.cases[name]["methods"]["HUBO->QUBO/neal"]["full_rate"]
            for name in CASE_NAMES
        ]
        self.assertEqual(neal_full, [0.0, 0.0, 0.0])
        combined = "\n".join(
            (
                main,
                abstract,
                (ROOT / "README.md").read_text(encoding="utf-8"),
                (ROOT / "paper" / "README.md").read_text(encoding="utf-8"),
            )
        )
        for stale_claim in (
            "Evidence covers one map",
            "Multi-instance evaluation is future work",
            "on one fully specified instance",
        ):
            self.assertNotIn(stale_claim, combined)


if __name__ == "__main__":
    unittest.main()
