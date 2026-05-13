from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from experiments.mcdp.routing_bird import (BUS_TYPES, GRID_AVG_SPEEDS,
                                           GRID_LAMBDAS, GRID_METHODS,
                                           GRID_PARTIAL,
                                           bird_config_for_grid_point,
                                           config_posets, fleet_catalogue,
                                           iter_grid,
                                           read_bus_inventory_counts,
                                           routing_service_catalogue,
                                           routing_service_entry,
                                           write_policy_module, write_poset)


class RoutingBirdMcdpTests(unittest.TestCase):
    def test_bus_inventory_counts_match_current_csv(self) -> None:
        self.assertEqual(
            read_bus_inventory_counts(),
            {"C": 52, "B": 21, "BWC": 11, "WC": 2},
        )

    def test_routing_service_entry_uses_expected_yaml_shape(self) -> None:
        summary = {
            "students_served": 3,
            "sped_students_served": 2,
            "wheelchair_students_served": 1,
            "students_unserved": 4,
            "sped_students_unserved": 1,
            "wheelchair_students_unserved": 0,
            "stops_used": 5,
            "monitor_buses": 2,
            "by_type": {
                "C": {"buses_used": 1, "distance_km": 12.5, "runtime_s": 100.0},
                "BWC": {"buses_used": 1, "distance_km": 7.5, "runtime_s": 50.0},
            },
        }
        entry = routing_service_entry(
            summary,
            {
                "bird_method": "scenario",
                "bird_lambda": "lambda_1e4",
                "bird_partial": "partial_true",
                "bird_dwell": "dwell_10",
                "bird_arrival_window": "arrival_default",
                "bird_avg_speed": "speed_30mph",
            },
        )
        catalogue = routing_service_catalogue({"tiny": entry})

        self.assertEqual(catalogue["F"], ["Nat", "Nat", "Nat"])
        self.assertEqual(catalogue["R"][-6:], [
            "`bird_method",
            "`bird_lambda",
            "`bird_partial",
            "`bird_dwell",
            "`bird_arrival_window",
            "`bird_avg_speed",
        ])
        self.assertEqual(entry["f_max"], ["3", "2", "1"])
        self.assertEqual(entry["r_min"][:9], ["4", "1", "0", "5", "2", "1", "0", "1", "0"])
        self.assertEqual(entry["r_min"][9:13], ["12.5 km", "0 km", "7.5 km", "0 km"])
        self.assertEqual(entry["r_min"][13:17], ["100 s", "0 s", "50 s", "0 s"])
        self.assertEqual(entry["r_min"][-6:], [
            "`bird_method: scenario",
            "`bird_lambda: lambda_1e4",
            "`bird_partial: partial_true",
            "`bird_dwell: dwell_10",
            "`bird_arrival_window: arrival_default",
            "`bird_avg_speed: speed_30mph",
        ])

    def test_config_values_are_finite_unordered_posets(self) -> None:
        posets = config_posets()

        self.assertEqual(posets["bird_method"], GRID_METHODS)
        self.assertEqual(
            posets["bird_lambda"],
            tuple(label for _value, label in GRID_LAMBDAS),
        )
        self.assertEqual(
            posets["bird_partial"],
            tuple(label for _value, label in GRID_PARTIAL),
        )
        self.assertEqual(
            posets["bird_avg_speed"],
            tuple(f"speed_{speed}mph" for speed in GRID_AVG_SPEEDS),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bird_lambda.mcdp_poset"
            write_poset(path, posets["bird_lambda"])
            contents = path.read_text(encoding="utf-8")

        self.assertIn("poset {", contents)
        self.assertIn("lambda_1e3 lambda_1e4 lambda_1e5 lambda_1e6", contents)
        self.assertNotIn(">=", contents)

    def test_grid_average_speeds_reach_bird_config_and_labels(self) -> None:
        grid_points = list(iter_grid())
        speed_labels = {grid_point.average_speed_label for grid_point in grid_points}

        self.assertEqual(
            {grid_point.average_speed_mph for grid_point in grid_points},
            {float(speed) for speed in GRID_AVG_SPEEDS},
        )
        self.assertEqual(speed_labels, set(config_posets()["bird_avg_speed"]))
        for grid_point in grid_points:
            self.assertIn(grid_point.average_speed_label, grid_point.label)
            self.assertEqual(
                grid_point.config_labels["bird_avg_speed"],
                grid_point.average_speed_label,
            )
            self.assertEqual(
                bird_config_for_grid_point(grid_point).bus_mph,
                grid_point.average_speed_mph,
            )

    def test_fleet_catalogue_uses_type_count_interface_and_costs(self) -> None:
        costs = {
            "capital": {"C": 10, "B": 20, "BWC": 30, "WC": 40},
            "capital_annualization_factor": 2.0,
        }
        catalogue = fleet_catalogue({"C": 1, "B": 1, "BWC": 0, "WC": 0}, costs)

        self.assertEqual(catalogue["F"], ["Nat", "Nat", "Nat", "Nat"])
        self.assertEqual(catalogue["R"], ["USD"])
        self.assertEqual(set(BUS_TYPES), {"C", "B", "BWC", "WC"})
        self.assertEqual(
            catalogue["implementations"]["fleet_1_1_0_0"],
            {"f_max": ["1", "1", "0", "0"], "r_min": ["60 USD"]},
        )

    def test_checked_in_routing_service_catalogue_schema_matches_plan(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "routing.mcdplib"
            / "yaml_catalogues"
            / "routing_service.dpc.yaml"
        )
        data = yaml.safe_load(path.read_text(encoding="utf-8"))

        self.assertEqual(data["F"], ["Nat", "Nat", "Nat"])
        self.assertEqual(data["R"], routing_service_catalogue({})["R"])

    def test_checked_in_routing_service_catalogue_values_match_posets(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "routing.mcdplib"
            / "yaml_catalogues"
            / "routing_service.dpc.yaml"
        )
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        posets = config_posets()
        config_indices = {
            "bird_method": 17,
            "bird_lambda": 18,
            "bird_partial": 19,
            "bird_dwell": 20,
            "bird_arrival_window": 21,
            "bird_avg_speed": 22,
        }

        def parse_poset_value(value: str) -> str:
            text = str(value).strip().strip('"').strip("'").strip("`")
            return text.split(":", 1)[1].strip()

        for implementation in data["implementations"].values():
            self.assertEqual(len(implementation["r_min"]), len(data["R"]))
            for poset_name, index in config_indices.items():
                self.assertIn(
                    parse_poset_value(implementation["r_min"][index]),
                    posets[poset_name],
                )

    def test_generator_writes_routing_policy_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = Path(tmpdir) / "routing.mcdplib"
            lib.mkdir()
            write_policy_module(lib)
            contents = (lib / "routing_policy.mcdp").read_text(encoding="utf-8")

        self.assertIn("sub g = instance `guidelines", contents)
        self.assertIn("requires policy_cost [USD]", contents)
        self.assertIn("required policy_cost <= roi_budget provided by g", contents)


if __name__ == "__main__":
    unittest.main()
