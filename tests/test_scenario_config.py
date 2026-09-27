import sys
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scenario_config import load_run_plan, scenario_from_mapping, shortest_path_distance


class ScenarioConfigTests(unittest.TestCase):
    def test_variable_grid_start_target_and_obstacles(self):
        scenario = scenario_from_mapping({
            "grid_size": 6,
            "horizon": 16,
            "start": [0, 1],
            "target": [5, 4],
            "obstacles": [[2, 2], [3, 3], [4, 1]],
        })
        self.assertEqual(scenario.grid_size, 6)
        self.assertEqual(scenario.start, (0, 1))
        self.assertEqual(scenario.target, (5, 4))
        self.assertEqual(shortest_path_distance(scenario), 8)

    def test_rejects_out_of_bounds_obstacle(self):
        with self.assertRaisesRegex(ValueError, "outside the grid"):
            scenario_from_mapping({
                "grid_size": 5,
                "horizon": 12,
                "start": [0, 0],
                "target": [4, 4],
                "obstacles": [[5, 1]],
            })

    def test_rejects_duplicate_obstacles(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            scenario_from_mapping({
                "grid_size": 5,
                "horizon": 12,
                "start": [0, 0],
                "target": [4, 4],
                "obstacles": [[2, 2], [2, 2]],
            })

    def test_rejects_target_obstacle(self):
        with self.assertRaisesRegex(ValueError, "target cell"):
            scenario_from_mapping({
                "grid_size": 5,
                "horizon": 12,
                "start": [0, 0],
                "target": [4, 4],
                "obstacles": [[4, 4]],
            })

    def test_rejects_horizon_shorter_than_shortest_path(self):
        with self.assertRaisesRegex(ValueError, "horizon"):
            scenario_from_mapping({
                "grid_size": 5,
                "horizon": 5,
                "start": [0, 0],
                "target": [4, 4],
                "obstacles": [],
            })

    def test_rejects_unsafe_run_name(self):
        plan = {
            "runs": [{
                "name": "../outside",
                "grid_size": 5,
                "horizon": 10,
                "start": [0, 0],
                "target": [4, 4],
                "obstacles": [],
            }]
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "plan.json"
            path.write_text(json.dumps(plan), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must contain only"):
                load_run_plan(path)


if __name__ == "__main__":
    unittest.main()
