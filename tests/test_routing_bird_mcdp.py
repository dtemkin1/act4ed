from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from experiments.mcdp.routing_bird import (BUS_TYPES, config_posets,
                                           fleet_catalogue,
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
            },
        )
        catalogue = routing_service_catalogue({"tiny": entry})

        self.assertEqual(catalogue["F"], ["Nat", "Nat", "Nat"])
        self.assertEqual(catalogue["R"][-5:], [
            "`bird_method",
            "`bird_lambda",
            "`bird_partial",
            "`bird_dwell",
            "`bird_arrival_window",
        ])
        self.assertEqual(entry["f_max"], ["3", "2", "1"])
        self.assertEqual(entry["r_min"][:9], ["4", "1", "0", "5", "2", "1", "0", "1", "0"])
        self.assertEqual(entry["r_min"][9:13], ["12.5 km", "0 km", "7.5 km", "0 km"])
        self.assertEqual(entry["r_min"][13:17], ["100 s", "0 s", "50 s", "0 s"])
        self.assertEqual(entry["r_min"][-5:], [
            "`bird_method: scenario",
            "`bird_lambda: lambda_1e4",
            "`bird_partial: partial_true",
            "`bird_dwell: dwell_10",
            "`bird_arrival_window: arrival_default",
        ])

    def test_config_values_are_finite_unordered_posets(self) -> None:
        posets = config_posets()

        self.assertEqual(posets["bird_method"], ("lbh", "scenario"))
        self.assertEqual(posets["bird_lambda"], ("lambda_1e3", "lambda_1e4", "lambda_1e5"))
        self.assertEqual(posets["bird_partial"], ("partial_false", "partial_true"))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bird_lambda.mcdp_poset"
            write_poset(path, posets["bird_lambda"])
            contents = path.read_text(encoding="utf-8")

        self.assertIn("poset {", contents)
        self.assertIn("lambda_1e3 lambda_1e4 lambda_1e5", contents)
        self.assertNotIn(">=", contents)

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
        path = Path(__file__).resolve().parents[1] / "routing.mcdplib" / "yaml_catalogues" / "routing_service.dpc.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))

        self.assertEqual(data["F"], ["Nat", "Nat", "Nat"])
        self.assertEqual(data["R"], routing_service_catalogue({})["R"])

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
