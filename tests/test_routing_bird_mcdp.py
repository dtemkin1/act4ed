from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from experiments.mcdp.routing_bird import (BUS_TYPES, GRID_AVG_SPEEDS,
                                           GRID_LAMBDAS, GRID_METHODS,
                                           GRID_PARTIAL,
                                           GRID_STUDENT_POLICIES,
                                           bird_config_for_grid_point,
                                           config_posets,
                                           fleet_catalogue,
                                           guidelines_catalogue,
                                           iter_grid,
                                           read_bus_inventory_counts,
                                           routing_service_catalogue,
                                           routing_service_entry,
                                           write_guidelines_module,
                                           write_poset,
                                           write_static_catalogues)


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
                "student_policy": "all_students",
            },
        )
        catalogue = routing_service_catalogue({"tiny": entry})

        self.assertEqual(catalogue["F"], ["Nat", "Nat", "Nat"])
        self.assertEqual(catalogue["R"][-7:], [
            "`bird_method",
            "`bird_lambda",
            "`bird_partial",
            "`bird_dwell",
            "`bird_arrival_window",
            "`bird_avg_speed",
            "`student_policy",
        ])
        self.assertEqual(entry["f_max"], ["3", "2", "1"])
        self.assertEqual(entry["r_min"][:9], ["4", "1", "0", "5", "2", "1", "0", "1", "0"])
        self.assertEqual(entry["r_min"][9:13], ["12.5 km", "0 km", "7.5 km", "0 km"])
        self.assertEqual(entry["r_min"][13:17], ["100 s", "0 s", "50 s", "0 s"])
        self.assertEqual(entry["r_min"][-7:], [
            "`bird_method: scenario",
            "`bird_lambda: lambda_1e4",
            "`bird_partial: partial_true",
            "`bird_dwell: dwell_10",
            "`bird_arrival_window: arrival_default",
            "`bird_avg_speed: speed_30mph",
            "`student_policy: all_students",
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
        self.assertEqual(posets["student_policy"], GRID_STUDENT_POLICIES)

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
        policy_labels = {grid_point.student_policy for grid_point in grid_points}

        self.assertEqual(
            {grid_point.average_speed_mph for grid_point in grid_points},
            {float(speed) for speed in GRID_AVG_SPEEDS},
        )
        self.assertEqual(speed_labels, set(config_posets()["bird_avg_speed"]))
        self.assertEqual(policy_labels, set(GRID_STUDENT_POLICIES))
        for grid_point in grid_points:
            self.assertIn(grid_point.average_speed_label, grid_point.label)
            self.assertIn(grid_point.student_policy, grid_point.label)
            self.assertEqual(
                grid_point.config_labels["bird_avg_speed"],
                grid_point.average_speed_label,
            )
            self.assertEqual(
                grid_point.config_labels["student_policy"],
                grid_point.student_policy,
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

    def test_static_catalogues_bound_fleet_by_observed_route_usage(self) -> None:
        costs = {
            "capital": {"C": 10, "B": 20, "BWC": 30, "WC": 40},
            "capital_annualization_factor": 1.0,
            "school_days": 180,
            "driver_yearly_pay": 1,
            "monitor_yearly_pay": 1,
            "diesel_cost_per_gallon": 1,
            "diesel_co2_kg_per_gallon": 1,
            "fuel_cost_per_km": 1,
            "emissions_kg_per_km": 1,
            "maintenance_distance_factor": {"C": 0, "B": 0, "BWC": 0, "WC": 0},
            "maintenance_runtime_factor": {"C": 0, "B": 0, "BWC": 0, "WC": 0},
        }
        route = routing_service_entry(
            {
                "students_served": 3,
                "sped_students_served": 0,
                "wheelchair_students_served": 0,
                "students_unserved": 0,
                "sped_students_unserved": 0,
                "wheelchair_students_unserved": 0,
                "stops_used": 1,
                "monitor_buses": 0,
                "by_type": {
                    "C": {"buses_used": 2},
                    "BWC": {"buses_used": 1},
                },
            },
            {
                "bird_method": "scenario",
                "bird_lambda": "lambda_1e3",
                "bird_partial": "partial_true",
                "bird_dwell": "dwell_0",
                "bird_arrival_window": "arrival_default",
                "bird_avg_speed": "speed_30mph",
                "student_policy": "all_students",
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            routing_lib = Path(tmpdir) / "routing.mcdplib"
            write_static_catalogues(
                routing_lib=routing_lib,
                costs=costs,
                routing_implementations={"route": route},
            )
            fleet_path = routing_lib / "yaml_catalogues" / "fleet_bus_count.dpc.yaml"
            catalogue = yaml.safe_load(fleet_path.read_text(encoding="utf-8"))

        self.assertEqual(len(catalogue["implementations"]), 6)
        self.assertIn("fleet_2_0_1_0", catalogue["implementations"])
        self.assertNotIn("fleet_3_0_1_0", catalogue["implementations"])

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
            "student_policy": 23,
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

    def test_generator_writes_guidelines_module(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = Path(tmpdir) / "routing.mcdplib"
            lib.mkdir()
            write_guidelines_module(lib)
            guidelines = (lib / "guidelines.mcdp").read_text(encoding="utf-8")

        self.assertIn("provides students_unserved [Nat]", guidelines)
        self.assertIn("requires student_policy [`student_policy]", guidelines)
        self.assertFalse((lib / "routing_policy.mcdp").exists())

    def test_policy_catalogue_schema(self) -> None:
        entry = {
            "f_max": ["4500000 USD", "0", "0", "0"],
            "r_min": ["10", "2", "1", "`student_policy: all_students"],
        }
        catalogue = guidelines_catalogue({"guideline_all_students": entry})

        self.assertEqual(catalogue["F"], ["USD", "Nat", "Nat", "Nat"])
        self.assertEqual(catalogue["R"], ["Nat", "Nat", "Nat", "`student_policy"])
        self.assertEqual(
            catalogue["implementations"]["guideline_all_students"],
            entry,
        )


if __name__ == "__main__":
    unittest.main()
