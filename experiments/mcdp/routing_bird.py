from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import yaml

from experiments.existing_data.utils import get_assigned_students
from experiments.helpers import setup
from formulation.bird_adapter import (
    BirdAdapterConfig,
    BirdBackendSolution,
    BirdExportInstance,
    export_bird_instance,
    summarize_bird_solution_for_mcdp,
)
from formulation.common import Bus
from formulation.common.problems import FilteredProblemData, ProblemData

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROUTING_LIB = PROJECT_ROOT / "routing.mcdplib"
BUS_CSV = PROJECT_ROOT / "experiments" / "data" / "buses.csv"
COST_CONFIG = PROJECT_ROOT / "experiments" / "mcdp" / "routing_costs.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "outputs" / "routing_bird_grid"
DEFAULT_PLACE_NAME = "Framingham, Massachusetts, USA"
DEFAULT_PROBLEM_NAME = "framingham"

BUS_TYPES = ("C", "B", "BWC", "WC")

ROUTING_F = ["Nat", "Nat", "Nat"]
ROUTING_R = [
    "Nat",                   # students_unserved
    "Nat",                   # sped_students_unserved
    "Nat",                   # wheelchair_students_unserved
    "Nat",                   # stops_used
    "Nat",                   # monitor_buses (total buses requiring a monitor)
    "Nat",                   # buses_used C
    "Nat",                   # buses_used B
    "Nat",                   # buses_used BWC
    "Nat",                   # buses_used WC
    "km",                    # distance_km C
    "km",                    # distance_km B
    "km",                    # distance_km BWC
    "km",                    # distance_km WC
    "s",                     # runtime_s C
    "s",                     # runtime_s B
    "s",                     # runtime_s BWC
    "s",                     # runtime_s WC
    "`bird_method",          # routing algorithm (scenario | lbh)
    "`bird_lambda",          # distance/ride-time trade-off weight
    "`bird_partial",         # whether partial assignment is allowed
    "`bird_dwell",           # school dwell time setting
    "`bird_arrival_window",  # earliest/latest arrival buffer setting
]

GRID_FLEET = {
    "C": (0, 15, 30, 45),
    "B": (0, 8, 17),
    "BWC": (0, 4, 9),
    "WC": (0, 1),
}
GRID_METHODS = ("scenario", "lbh")
GRID_LAMBDAS = ((1.0e3, "lambda_1e3"), (1.0e4, "lambda_1e4"), (1.0e5, "lambda_1e5"))
GRID_PARTIAL = ((False, "partial_false"), (True, "partial_true"))
GRID_SPILLOVER = ((False, "spillover_false"), (True, "spillover_true"))
GRID_DWELL = ((0.0, "dwell_0"), (10.0, "dwell_10"))
GRID_ARRIVAL_WINDOWS = (
    (None, None, "arrival_default"),
    (30.0, 10.0, "arrival_early30_late10"),  # Minutes before school bell time
)

DEFAULT_COSTS: dict[str, Any] = {
    "school_days": 180,
    "capital_annualization_factor": 1.0,
    "capital": {"C": 50000, "B": 70000, "BWC": 80000, "WC": 90000},
    "driver_yearly_pay": 50000,
    "monitor_yearly_pay": 20000,
    "fuel_cost_per_km": 0.60,
    "emissions_kg_per_km": 1.20,
    "maintenance_distance_factor": {"C": 0.15, "B": 0.15, "BWC": 0.15, "WC": 0.15},
    "maintenance_runtime_factor": {"C": 0.0, "B": 0.0, "BWC": 0.0, "WC": 0.0},
}


@dataclass(frozen=True)
class GridPoint:
    counts: dict[str, int]
    method: str
    lambda_value: float
    lambda_label: str
    conventional_spillover: bool
    spillover_label: str
    allow_partial: bool
    partial_label: str
    school_dwell_time: float
    dwell_label: str
    earliest_arrival_buffer: float | None
    latest_arrival_buffer: float | None
    arrival_label: str

    @property
    def label(self) -> str:
        count_label = "_".join(f"{bus_type}{self.counts[bus_type]}" for bus_type in BUS_TYPES)
        return (
            f"bird_{count_label}_{self.method}_{self.lambda_label}_"
            f"{self.partial_label}_{self.dwell_label}_{self.arrival_label}"
        )

    @property
    def config_labels(self) -> dict[str, str]:
        return {
            "bird_method": self.method,
            "bird_lambda": self.lambda_label,
            "bird_partial": self.partial_label,
            "bird_dwell": self.dwell_label,
            "bird_arrival_window": self.arrival_label,
        }


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def load_routing_costs(path: Path = COST_CONFIG) -> dict[str, Any]:
    if not path.exists():
        return dict(DEFAULT_COSTS)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"expected mapping in routing cost config {path}")
    return _deep_merge(DEFAULT_COSTS, data)


def read_bus_inventory_counts(path: Path = BUS_CSV) -> dict[str, int]:
    counts = {bus_type: 0 for bus_type in BUS_TYPES}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            bus_type = row["type"]
            if bus_type not in counts:
                raise ValueError(f"unknown bus type {bus_type!r} in {path}")
            counts[bus_type] += 1
    return counts


def read_bus_inventory_order(path: Path = BUS_CSV) -> dict[str, list[str]]:
    order = {bus_type: [] for bus_type in BUS_TYPES}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            bus_type = row["type"]
            if bus_type not in order:
                raise ValueError(f"unknown bus type {bus_type!r} in {path}")
            order[bus_type].append(row["num"])
    return order


def select_buses_by_type_counts(
    buses: Iterable[Bus],
    counts: Mapping[str, int],
    *,
    bus_order: Mapping[str, list[str]] | None = None,
) -> tuple[Bus, ...]:
    # TODO check if this should be by name or by id
    by_name = {bus.name: bus for bus in buses}
    if bus_order is None:
        ordered_names: dict[str, list[str]] = {bus_type: [] for bus_type in BUS_TYPES}
        for bus in buses:
            if bus.type is not None and bus.type.name in ordered_names:
                ordered_names[bus.type.name].append(bus.name)
    else:
        ordered_names = {bus_type: list(bus_order[bus_type]) for bus_type in BUS_TYPES}

    selected: list[Bus] = []
    for bus_type in BUS_TYPES:
        requested = int(counts.get(bus_type, 0))
        available = [name for name in ordered_names[bus_type] if name in by_name]
        # TODO make this dependent on whether we consider the current fleet requirements or
        # if we want to allow to go over it.
        if requested > len(available):
            raise ValueError(
                f"requested {requested} {bus_type} buses but only {len(available)} are available"
            )
        selected.extend(by_name[name] for name in available[:requested])
    return tuple(selected)


def _format_number(value: float | int) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.12g}"


def _with_unit(value: float | int, unit: str) -> str:
    return f"{_format_number(value)} {unit}"


def _poset_value(poset_name: str, value: str) -> str:
    return f"`{poset_name}: {value}"


def _type_summary(summary: Mapping[str, Any], bus_type: str) -> Mapping[str, Any]:
    by_type = summary.get("by_type", {})
    if not isinstance(by_type, Mapping):
        by_type = {}
    value = by_type.get(bus_type, {})
    return value if isinstance(value, Mapping) else {}


def routing_service_entry(
    summary: Mapping[str, Any],
    config_labels: Mapping[str, str],
) -> dict[str, list[str]]:
    return {
        "f_max": [
            str(int(summary["students_served"])),
            str(int(summary["sped_students_served"])),
            str(int(summary["wheelchair_students_served"])),
        ],
        "r_min": [
            str(int(summary["students_unserved"])),
            str(int(summary["sped_students_unserved"])),
            str(int(summary["wheelchair_students_unserved"])),
            str(int(summary["stops_used"])),
            str(int(summary["monitor_buses"])),
            *[
                str(int(_type_summary(summary, bus_type).get("buses_used", 0)))
                for bus_type in BUS_TYPES
            ],
            *[
                _with_unit(
                    float(_type_summary(summary, bus_type).get("distance_km", 0.0)),
                    "km",
                )
                for bus_type in BUS_TYPES
            ],
            *[
                _with_unit(
                    float(_type_summary(summary, bus_type).get("runtime_s", 0.0)),
                    "s",
                )
                for bus_type in BUS_TYPES
            ],
            _poset_value("bird_method", config_labels["bird_method"]),
            _poset_value("bird_lambda", config_labels["bird_lambda"]),
            _poset_value("bird_partial", config_labels["bird_partial"]),
            _poset_value("bird_dwell", config_labels["bird_dwell"]),
            _poset_value("bird_arrival_window", config_labels["bird_arrival_window"]),
        ],
    }


def routing_service_catalogue(
    implementations: Mapping[str, dict[str, list[str]]],
) -> dict[str, Any]:
    return {"F": ROUTING_F, "R": ROUTING_R, "implementations": dict(implementations)}


def fleet_catalogue(
    max_counts: Mapping[str, int],
    costs: Mapping[str, Any],
) -> dict[str, Any]:
    capital = costs["capital"]
    annualization = float(costs["capital_annualization_factor"])
    implementations: dict[str, dict[str, list[str]]] = {}
    ranges = [range(int(max_counts[bus_type]) + 1) for bus_type in BUS_TYPES]

    for values in product(*ranges):
        counts = dict(zip(BUS_TYPES, values, strict=True))
        name = "fleet_" + "_".join(str(counts[bus_type]) for bus_type in BUS_TYPES)
        cost = sum(float(capital[bus_type]) * counts[bus_type] for bus_type in BUS_TYPES)
        implementations[name] = {
            "f_max": [str(counts[bus_type]) for bus_type in BUS_TYPES],
            "r_min": [_with_unit(cost * annualization, "USD")],
        }

    return {"F": ["Nat", "Nat", "Nat", "Nat"], "R": ["USD"], "implementations": implementations}


def config_posets() -> dict[str, tuple[str, ...]]:
    return {
        "bird_method": GRID_METHODS,
        "bird_lambda": tuple(label for _value, label in GRID_LAMBDAS),
        "bird_partial": tuple(label for _value, label in GRID_PARTIAL),
        "bird_dwell": tuple(label for _value, label in GRID_DWELL),
        "bird_arrival_window": tuple(label for _earliest, _latest, label in GRID_ARRIVAL_WINDOWS),
    }


def write_poset(path: Path, elements: Iterable[str]) -> None:
    path.write_text("poset {\n    " + " ".join(elements) + "\n}\n", encoding="utf-8")


def write_catalogue(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(data), sort_keys=False), encoding="utf-8")


def write_cost_modules(routing_lib: Path, costs: Mapping[str, Any]) -> None:
    school_days = _format_number(costs["school_days"])
    driver_pay = _format_number(costs["driver_yearly_pay"])
    monitor_pay = _format_number(costs["monitor_yearly_pay"])
    fuel_cost_per_km = _format_number(costs["fuel_cost_per_km"])
    emissions_kg_per_km = _format_number(costs["emissions_kg_per_km"])
    distance_factor = costs["maintenance_distance_factor"]
    runtime_factor = costs["maintenance_runtime_factor"]

    (routing_lib / "driver.mcdp").write_text(
        f"""mcdp {{
  provides used_C [Nat]
  provides used_B [Nat]
  provides used_BWC [Nat]
  provides used_WC [Nat]

  requires driver_cost [USD]

  constant yearly_pay_per_bus = {driver_pay} USD

  required driver_cost >= (
    provided used_C +
    provided used_B +
    provided used_BWC +
    provided used_WC
  ) * yearly_pay_per_bus
}}
""",
        encoding="utf-8",
    )
    (routing_lib / "monitor.mcdp").write_text(
        f"""mcdp {{
  # buses with monitors on them
  provides monitor_buses [Nat]

  requires monitor_cost [USD]

  constant yearly_pay_per_monitor_bus = {monitor_pay} USD

  required monitor_cost >= provided monitor_buses * yearly_pay_per_monitor_bus
}}
""",
        encoding="utf-8",
    )
    (routing_lib / "fuel.mcdp").write_text(
        f"""mcdp {{
    provides distance_C [km]
    provides distance_B [km]
    provides distance_BWC [km]
    provides distance_WC [km]

    requires fuel_cost [USD]
    requires emissions [kg]

    constant school_days = {school_days}
    constant fuel_cost_per_km = {fuel_cost_per_km} USD / km
    constant emissions_per_km = {emissions_kg_per_km} kg / km

    required fuel_cost >= (
        provided distance_C +
        provided distance_B +
        provided distance_BWC +
        provided distance_WC
    ) * school_days * fuel_cost_per_km

    required emissions >= (
        provided distance_C +
        provided distance_B +
        provided distance_BWC +
        provided distance_WC
    ) * school_days * emissions_per_km
}}
""",
        encoding="utf-8",
    )
    (routing_lib / "maintenance.mcdp").write_text(
        f"""mcdp {{
    provides distance_C [km]
    provides distance_B [km]
    provides distance_BWC [km]
    provides distance_WC [km]

    provides runtime_C [s]
    provides runtime_B [s]
    provides runtime_BWC [s]
    provides runtime_WC [s]

    requires maintenance_cost [USD]

    constant school_days = {school_days}

    constant distance_factor_C = {_format_number(distance_factor["C"])} USD / km
    constant distance_factor_B = {_format_number(distance_factor["B"])} USD / km
    constant distance_factor_BWC = {_format_number(distance_factor["BWC"])} USD / km
    constant distance_factor_WC = {_format_number(distance_factor["WC"])} USD / km

    constant runtime_factor_C = {_format_number(runtime_factor["C"])} USD / s
    constant runtime_factor_B = {_format_number(runtime_factor["B"])} USD / s
    constant runtime_factor_BWC = {_format_number(runtime_factor["BWC"])} USD / s
    constant runtime_factor_WC = {_format_number(runtime_factor["WC"])} USD / s

    required maintenance_cost >= school_days * (
        provided distance_C * distance_factor_C +
        provided distance_B * distance_factor_B +
        provided distance_BWC * distance_factor_BWC +
        provided distance_WC * distance_factor_WC +
        provided runtime_C * runtime_factor_C +
        provided runtime_B * runtime_factor_B +
        provided runtime_BWC * runtime_factor_BWC +
        provided runtime_WC * runtime_factor_WC
    )
}}
""",
        encoding="utf-8",
    )


def write_policy_module(routing_lib: Path) -> None:
    (routing_lib / "routing_policy.mcdp").write_text(
        """mcdp {
    provides students_served [Nat]
    provides sped_students_served [Nat]
    provides wheelchair_students_served [Nat]

    requires policy_cost [USD]

    sub g = instance `guidelines

    provided students_served >= students_served required by g
    provided sped_students_served >= sped_students_served required by g
    provided wheelchair_students_served >= wheelchair_students_served required by g

    required policy_cost <= roi_budget provided by g
}
""",
        encoding="utf-8",
    )


def iter_grid() -> Iterable[GridPoint]:
    count_values = [GRID_FLEET[bus_type] for bus_type in BUS_TYPES]
    for counts_tuple in product(*count_values):
        counts = dict(zip(BUS_TYPES, counts_tuple, strict=True))
        for method, lambda_pair, partial_pair, spillover_pair, dwell_pair, window in product(
            GRID_METHODS,
            GRID_LAMBDAS,
            GRID_PARTIAL,
            GRID_SPILLOVER,
            GRID_DWELL,
            GRID_ARRIVAL_WINDOWS,
        ):
            lambda_value, lambda_label = lambda_pair
            allow_partial, partial_label = partial_pair
            conventional_spillover, spillover_label = spillover_pair
            dwell_value, dwell_label = dwell_pair
            earliest, latest, arrival_label = window
            yield GridPoint(
                counts=counts,
                method=method,
                lambda_value=lambda_value,
                lambda_label=lambda_label,
                allow_partial=allow_partial,
                partial_label=partial_label,
                conventional_spillover=conventional_spillover,
                spillover_label=spillover_label,
                school_dwell_time=dwell_value,
                dwell_label=dwell_label,
                earliest_arrival_buffer=earliest,
                latest_arrival_buffer=latest,
                arrival_label=arrival_label,
            )


def _load_assigned_framingham_problem(
        problem_name: str = DEFAULT_PROBLEM_NAME, 
        place_name: str = DEFAULT_PLACE_NAME    
    ) -> ProblemData:
    problem_data = setup(DEFAULT_PROBLEM_NAME, DEFAULT_PLACE_NAME, None)
    assigned_students = get_assigned_students(problem_data.schools, problem_data.stops)
    return FilteredProblemData(
        name=f"{problem_data.name}_assigned",
        base_problem_data=problem_data,
        _students=assigned_students,
    )


def _load_full_framingham_problem(
        problem_name: str = DEFAULT_PROBLEM_NAME, 
        place_name: str = DEFAULT_PLACE_NAME
    ) -> ProblemData:
    return setup(problem_name, place_name, None)


def _problem_with_buses(problem_data: ProblemData, buses: tuple[Bus, ...]) -> ProblemData:
    return FilteredProblemData(
        name=f"{problem_data.name}_fleet_{len(buses)}",
        base_problem_data=problem_data,
        _buses=buses,
    )


def solve_grid_point(
    problem_data: ProblemData,
    grid_point: GridPoint,
    *,
    output_dir: Path,
    bus_order: Mapping[str, list[str]],
) -> tuple[dict[str, Any], BirdExportInstance, BirdBackendSolution]:
    selected_buses = select_buses_by_type_counts(
        problem_data.buses,
        grid_point.counts,
        bus_order=bus_order,
    )
    if not selected_buses:
        raise ValueError("BiRD fleet-aware export requires at least one selected bus")

    run_problem = _problem_with_buses(problem_data, selected_buses)
    config = BirdAdapterConfig(
        cohort="all",
        fleet_aware=True,
        conventional_spillover=grid_point.conventional_spillover,
        allow_partial=grid_point.allow_partial,
        lambda_value=grid_point.lambda_value,
        school_dwell_time=grid_point.school_dwell_time,
        earliest_arrival_buffer=grid_point.earliest_arrival_buffer,
        latest_arrival_buffer=grid_point.latest_arrival_buffer,
        method=grid_point.method
    )

    instance_path = export_bird_instance(
        run_problem,
        output_dir / f"{grid_point.label}_instance.npz",
        config,
    )
    solution_path = output_dir / f"{grid_point.label}_solution.npz"
    log_file = output_dir / f"{grid_point.label}.log"

    julia_cmd = os.environ.get("JULIA_CMD", "julia")
    julia_project = os.environ.get("JULIA_PROJECT", "julia")
    command = [
        julia_cmd,
        f"--project={julia_project}",
        str(PROJECT_ROOT / "experiments" / "solve_bird_backend_julia.jl"),
        "--instance",
        str(instance_path),
        "--solution",
        str(solution_path),
        "--log-file",
        str(log_file),
    ]
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)

    instance = BirdExportInstance.load(instance_path)
    solution = BirdBackendSolution.load(solution_path)
    if solution.status not in {"OPTIMAL", "PARTIAL"}:
        raise RuntimeError(f"BiRD returned unsupported status {solution.status!r}")
    return summarize_bird_solution_for_mcdp(instance, solution), instance, solution


def write_static_catalogues(
    *,
    routing_lib: Path = ROUTING_LIB,
    costs: Mapping[str, Any] | None = None,
    routing_implementations: Mapping[str, dict[str, list[str]]] | None = None,
) -> None:
    selected_costs = dict(costs or load_routing_costs())
    yaml_dir = routing_lib / "yaml_catalogues"
    write_catalogue(
        yaml_dir / "fleet_bus_count.dpc.yaml",
        fleet_catalogue(read_bus_inventory_counts(), selected_costs),
    )
    if routing_implementations is not None:
        write_catalogue(
            yaml_dir / "routing_service.dpc.yaml",
            routing_service_catalogue(routing_implementations),
        )
    for name, elements in config_posets().items():
        write_poset(routing_lib / f"{name}.mcdp_poset", elements)
    write_cost_modules(routing_lib, selected_costs)
    write_policy_module(routing_lib)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--routing-lib", type=Path, default=ROUTING_LIB)
    parser.add_argument("--cost-config", type=Path, default=COST_CONFIG)
    parser.add_argument("--current-routes", type=bool, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    costs = load_routing_costs(args.cost_config)
    if args.current_routes:
        problem_data = _load_assigned_framingham_problem()
    else:
        problem_data = _load_full_framingham_problem()

    bus_order = read_bus_inventory_order()

    implementations: dict[str, dict[str, list[str]]] = {}
    partial_result: dict[str, Any] = {}
    errors: list[dict[str, Any]] = []

    attempted = 0
    for grid_point in iter_grid():
        if sum(grid_point.counts.values()) == 0:
            continue
        if args.limit is not None and attempted >= args.limit:
            break
        attempted += 1
        try:
            summary, _instance, solution = solve_grid_point(
                problem_data,
                grid_point,
                output_dir=args.output_dir,
                bus_order=bus_order,
            )
        except Exception as exc:
            errors.append({"label": grid_point.label, "error": str(exc)})
            continue

        implementations[grid_point.label] = routing_service_entry(
            summary,
            grid_point.config_labels,
        )
        partial_result[grid_point.label] = {
            "counts": grid_point.counts,
            "method": grid_point.method,
            "lambda_value": grid_point.lambda_value,
            "allow_partial": grid_point.allow_partial,
            "school_dwell_time": grid_point.school_dwell_time,
            "earliest_arrival_buffer": grid_point.earliest_arrival_buffer,
            "latest_arrival_buffer": grid_point.latest_arrival_buffer,
            "status": solution.status,
            "summary": summary,
        }

    write_static_catalogues(
        routing_lib=args.routing_lib,
        costs=costs,
        routing_implementations=implementations or None,
    )
    (args.output_dir / "routing_bird_catalogue_partial_result.json").write_text(
        json.dumps(partial_result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.output_dir / "routing_bird_errors.json").write_text(
        json.dumps(errors, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
