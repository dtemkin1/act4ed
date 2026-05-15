from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from collections.abc import Iterable, Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Literal

import yaml
from loguru import logger
from tqdm import tqdm

from experiments.existing_data.utils import get_assigned_students
from experiments.helpers import setup
from formulation.bird_adapter import (BirdAdapterConfig, BirdBackendSolution,
                                      BirdExportInstance,
                                      bird_export_instance_from_template,
                                      build_bird_export_instance,
                                      export_bird_instance,
                                      summarize_bird_solution_for_mcdp)
from formulation.common import Bus, KM_PER_MILE, Student
from formulation.common.problems import FilteredProblemData, ProblemData

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROUTING_LIB = PROJECT_ROOT / "routing.mcdplib"
BUS_CSV = PROJECT_ROOT / "experiments" / "data" / "buses.csv"
COST_CONFIG = PROJECT_ROOT / "experiments" / "mcdp" / "routing_costs.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "outputs" / "routing_bird_grid"
DEFAULT_PLACE_NAME = "Framingham, Massachusetts, USA"
DEFAULT_PROBLEM_NAME = "framingham"
DEFAULT_CPUS_PER_SOLVE = 4
PROGRESS_HEARTBEAT_SECONDS = 60.0

BUS_TYPES = ("C", "B", "BWC", "WC")

ROUTING_F = ["Nat", "Nat", "Nat"]
ROUTING_R = [
    "Nat",  # students_unserved
    "Nat",  # sped_students_unserved
    "Nat",  # wheelchair_students_unserved
    "Nat",  # stops_used
    "Nat",  # monitor_buses (total buses requiring a monitor)
    "Nat",  # buses_used C
    "Nat",  # buses_used B
    "Nat",  # buses_used BWC
    "Nat",  # buses_used WC
    "km",  # distance_km C
    "km",  # distance_km B
    "km",  # distance_km BWC
    "km",  # distance_km WC
    "s",  # runtime_s C
    "s",  # runtime_s B
    "s",  # runtime_s BWC
    "s",  # runtime_s WC
    "`bird_method",  # routing algorithm (scenario | lbh)
    "`bird_lambda",  # distance/ride-time trade-off weight
    "`bird_partial",  # whether partial assignment is allowed
    "`bird_dwell",  # school dwell time setting
    "`bird_arrival_window",  # earliest/latest arrival buffer setting
    "`bird_avg_speed",  # average bus speed setting
    "`student_policy",  # student body/policy cohort selected for routing
]

GRID_FLEET = {
    "C": (52,),
    "B": (21,),
    "BWC": (11,),
    "WC": (2,),
}
GRID_METHODS = ("lbh", "scenario")
GRID_LAMBDAS = (
    (1.0e2, "lambda_1e3"),
    (1.0e4, "lambda_1e4"),
    (1.0e5, "lambda_1e5"),
    (1.0e6, "lambda_1e6"),
)
GRID_PARTIAL = ((True, "partial_true"),)
GRID_SPILLOVER = ((False, "spillover_false"), (True, "spillover_true"))
GRID_DWELL = ((10.0, "dwell_0"), (15.0, "dwell_10"))
# Minutes before school bell time
GRID_ARRIVAL_WINDOWS = (
    (None, None, "arrival_default"),
    (20.0, 10.0, "arrival_early20_late10"),
    (30.0, 10.0, "arrival_early30_late10"),
    (40.0, 10.0, "arrival_early40_late10"),
)
GRID_AVG_SPEEDS = (10, 20, 30)
GRID_STUDENT_POLICIES = (
    "current_assignment",
    "distance_gt_2mi",
    "distance_gt_1p5mi",
    "distance_gt_1mi",
    "distance_gt_0p5mi",
    "all_students",
)
DEFAULT_GUIDELINE_BUDGET_USD = 4_500_000
DEFAULT_ALLOWABLE_UNSERVED = 2_000

# GUIDELINE_BUDGETS = (
#     10_000_000,
#     5_000_000,
#     2_500_000,
#     1_000_000,
# )

# ALLOWABLE_UNSERVED = (
#     2_000,
#     1_000,
#     100,
#     0
# )

DEFAULT_COSTS: dict[str, Any] = {
    "school_days": 180,
    "capital_annualization_factor": 1.0,
    "capital": {
        "C": 128780,
        "B": 110060,
        "BWC": 126660,
        "WC": 136780
    },
    "driver_yearly_pay": 46571,
    "monitor_yearly_pay": 26609,
    "diesel_cost_per_gallon": 3.09,
    "diesel_co2_kg_per_gallon": 10.21,
    "fuel_cost_per_km": None,
    "emissions_kg_per_km": 1.20,
    "maintenance_distance_factor": {
        "C": 0.15,
        "B": 0.15,
        "BWC": 0.15,
        "WC": 0.15
    },
    "maintenance_runtime_factor": {
        "C": 0.0,
        "B": 0.0,
        "BWC": 0.0,
        "WC": 0.0
    },
}


def _available_cpus() -> int:
    return os.process_cpu_count() or os.cpu_count() or 1


def _default_worker_count(cpus_per_solve: int = DEFAULT_CPUS_PER_SOLVE) -> int:
    return max(1, _available_cpus() // cpus_per_solve)


def _problem_size_summary(problem_data: ProblemData) -> dict[str, int]:
    return {
        "schools": len(problem_data.schools),
        "stops": len(problem_data.stops),
        "students": len(problem_data.students),
        "buses": len(problem_data.buses),
    }


@dataclass(frozen=True)
class StudentPolicySpec:
    label: str
    description: str
    min_distance_miles: float | None = None
    current_assignment: bool = False


STUDENT_POLICY_SPECS = {
    "current_assignment": StudentPolicySpec(
        label="current_assignment",
        description="current assignment",
        current_assignment=True,
    ),
    "distance_gt_2mi": StudentPolicySpec(
        label="distance_gt_2mi",
        description="students living farther than 2 miles",
        min_distance_miles=2.0,
    ),
    "distance_gt_1p5mi": StudentPolicySpec(
        label="distance_gt_1p5mi",
        description="students living farther than 1.5 miles",
        min_distance_miles=1.5,
    ),
    "distance_gt_1mi": StudentPolicySpec(
        label="distance_gt_1mi",
        description="students living farther than 1 mile",
        min_distance_miles=1.0,
    ),
    "distance_gt_0p5mi": StudentPolicySpec(
        label="distance_gt_0p5mi",
        description="students living farther than 0.5 miles",
        min_distance_miles=0.5,
    ),
    "all_students": StudentPolicySpec(
        label="all_students",
        description="all students",
    ),
}


@dataclass(frozen=True)
class GridPoint:
    counts: dict[str, int]
    student_policy: str
    method: Literal["lbh", "scenario"]
    lambda_value: float
    lambda_label: str
    average_speed_mph: float
    average_speed_label: str
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
        count_label = "_".join(
            f"{bus_type}{self.counts[bus_type]}" for bus_type in BUS_TYPES
        )
        return (
            f"bird_{self.student_policy}_{count_label}_{self.method}_{self.lambda_label}_"
            f"{self.average_speed_label}_{self.partial_label}_{self.spillover_label}_"
            f"{self.dwell_label}_{self.arrival_label}"
        )

    @property
    def config_labels(self) -> dict[str, str]:
        return {
            "bird_method": self.method,
            "bird_lambda": self.lambda_label,
            "bird_partial": self.partial_label,
            "bird_dwell": self.dwell_label,
            "bird_arrival_window": self.arrival_label,
            "bird_avg_speed": self.average_speed_label,
            "student_policy": self.student_policy,
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


def _speed_label(speed_mph: float | int) -> str:
    return f"speed_{_format_number(speed_mph).replace('.', 'p')}mph"


def _grid_speed_pair(value: float | int | tuple[float | int, str]) -> tuple[float, str]:
    if isinstance(value, tuple):
        speed_mph, label = value
        return float(speed_mph), label
    return float(value), _speed_label(value)


def _fuel_cost_per_km(costs: Mapping[str, Any]) -> float:
    configured = costs.get("fuel_cost_per_km")
    if configured is not None:
        return float(configured)
    return (
        float(costs["diesel_cost_per_gallon"])
        * float(costs["emissions_kg_per_km"])
        / float(costs["diesel_co2_kg_per_gallon"])
    )


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
            _poset_value("bird_avg_speed", config_labels["bird_avg_speed"]),
            _poset_value("student_policy", config_labels["student_policy"]),
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
        cost = sum(
            float(capital[bus_type]) * counts[bus_type] for bus_type in BUS_TYPES
        )
        implementations[name] = {
            "f_max": [str(counts[bus_type]) for bus_type in BUS_TYPES],
            "r_min": [_with_unit(cost * annualization, "USD")],
        }

    return {
        "F": ["Nat", "Nat", "Nat", "Nat"],
        "R": ["USD"],
        "implementations": implementations,
    }


def fleet_bounds_for_routing_implementations(
    implementations: Mapping[str, dict[str, list[str]]],
) -> dict[str, int]:
    bounds = {bus_type: 0 for bus_type in BUS_TYPES}
    for implementation in implementations.values():
        r_min = implementation["r_min"]
        for index, bus_type in enumerate(BUS_TYPES):
            bounds[bus_type] = max(bounds[bus_type], int(r_min[5 + index]))
    return bounds


def _student_counts(students: Iterable[Student]) -> dict[str, int]:
    total = 0
    sped = 0
    wheelchair = 0
    for student in students:
        total += 1
        sped += int(bool(student.attributes.special_ed))
        wheelchair += int(bool(student.attributes.wheelchair_user))
    return {
        "students": total,
        "sped_students": sped,
        "wheelchair_students": wheelchair,
    }


def guideline_entry_for_policy(
    policy_problem_data: ProblemData,
    policy_label: str,
    *,
    roi_budget_usd: float = DEFAULT_GUIDELINE_BUDGET_USD,
) -> dict[str, list[str]]:
    counts = _student_counts(policy_problem_data.students)
    return {
        "f_max": [
            _with_unit(roi_budget_usd, "USD"),
            "0",
            "0",
            "0",
        ],
        "r_min": [
            str(counts["students"]),
            str(counts["sped_students"]),
            str(counts["wheelchair_students"]),
            _poset_value("student_policy", policy_label),
        ],
    }


def default_guideline_entry(policy_label: str) -> dict[str, list[str]]:
    return {
        "f_max": [
            _with_unit(DEFAULT_GUIDELINE_BUDGET_USD, "USD"),
            str(DEFAULT_ALLOWABLE_UNSERVED),
            str(DEFAULT_ALLOWABLE_UNSERVED),
            str(DEFAULT_ALLOWABLE_UNSERVED),
        ],
        "r_min": [
            "0",
            "0",
            "0",
            _poset_value("student_policy", policy_label),
        ],
    }


def guidelines_catalogue(
    implementations: Mapping[str, dict[str, list[str]]],
) -> dict[str, Any]:
    return {
        "F": ["USD", "Nat", "Nat", "Nat"],
        "R": ["Nat", "Nat", "Nat", "`student_policy"],
        "implementations": dict(implementations),
    }


def config_posets() -> dict[str, tuple[str, ...]]:
    return {
        "bird_method": GRID_METHODS,
        "bird_lambda": tuple(label for _value, label in GRID_LAMBDAS),
        "bird_partial": tuple(label for _value, label in GRID_PARTIAL),
        "bird_dwell": tuple(label for _value, label in GRID_DWELL),
        "bird_arrival_window": tuple(
            label for _earliest, _latest, label in GRID_ARRIVAL_WINDOWS
        ),
        "bird_avg_speed": tuple(
            label for _speed_mph, label in map(_grid_speed_pair, GRID_AVG_SPEEDS)
        ),
        "student_policy": GRID_STUDENT_POLICIES,
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
    fuel_cost_per_km = _format_number(_fuel_cost_per_km(costs))
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


def write_guidelines_module(routing_lib: Path) -> None:
    (routing_lib / "guidelines.mcdp").write_text(
        """dp {
  # Policy/planning budget cap and allowable unserved counts.
  provides roi_budget [USD]
  provides students_unserved [Nat]
  provides sped_students_unserved [Nat]
  provides wheelchair_students_unserved [Nat]

  # Minimum service level and student body selected by policy/planning.
  requires students_served [Nat]
  requires sped_students_served [Nat]
  requires wheelchair_students_served [Nat]
  requires student_policy [`student_policy]

  implemented-by yaml resource("guidelines.dpc.yaml")
}
""",
        encoding="utf-8",
    )


def write_policy_module(routing_lib: Path) -> None:
    write_guidelines_module(routing_lib)


def iter_grid(policy_labels: Iterable[str] | None = None) -> Iterable[GridPoint]:
    selected_policy_labels = tuple(policy_labels or GRID_STUDENT_POLICIES)
    count_values = [GRID_FLEET[bus_type] for bus_type in BUS_TYPES]
    for counts_tuple in product(*count_values):
        counts: dict[str, int] = dict(zip(BUS_TYPES, counts_tuple, strict=True))
        for (
            student_policy,
            method,
            lambda_pair,
            partial_pair,
            spillover_pair,
            dwell_pair,
            window,
            speed_pair,
        ) in product(
            selected_policy_labels,
            GRID_METHODS,
            GRID_LAMBDAS,
            GRID_PARTIAL,
            GRID_SPILLOVER,
            GRID_DWELL,
            GRID_ARRIVAL_WINDOWS,
            tuple(map(_grid_speed_pair, GRID_AVG_SPEEDS)),
        ):
            lambda_value, lambda_label = lambda_pair
            allow_partial, partial_label = partial_pair
            conventional_spillover, spillover_label = spillover_pair
            dwell_value, dwell_label = dwell_pair
            earliest, latest, arrival_label = window
            average_speed_mph, average_speed_label = speed_pair
            yield GridPoint(
                counts=counts,
                student_policy=student_policy,
                method=method,
                lambda_value=lambda_value,
                lambda_label=lambda_label,
                average_speed_mph=average_speed_mph,
                average_speed_label=average_speed_label,
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


def bird_config_for_grid_point(grid_point: GridPoint) -> BirdAdapterConfig:
    return BirdAdapterConfig(
        cohort="all",
        fleet_aware=True,
        conventional_spillover=grid_point.conventional_spillover,
        allow_partial=grid_point.allow_partial,
        lambda_value=grid_point.lambda_value,
        school_dwell_time=grid_point.school_dwell_time,
        earliest_arrival_buffer=grid_point.earliest_arrival_buffer,
        latest_arrival_buffer=grid_point.latest_arrival_buffer,
        bus_mph=grid_point.average_speed_mph,
        method=grid_point.method,
    )


def _student_distance_to_school_km(
    problem_data: ProblemData, student: Student
) -> float:
    if student.stop.node_id == student.school.node_id:
        return 0.0
    edge_data = problem_data.service_graph.get_edge_data(
        student.stop.node_id,
        student.school.node_id,
        key=0,
    )
    if edge_data is not None:
        return float(edge_data["length"])
    length_m, _path = problem_data.get_shortest_path_base(
        student.stop.node_id,
        student.school.node_id,
    )
    return length_m / 1000.0


def students_for_policy(
    problem_data: ProblemData,
    policy_label: str,
) -> tuple[Student, ...]:
    try:
        policy = STUDENT_POLICY_SPECS[policy_label]
    except KeyError as exc:
        raise ValueError(f"unknown student policy {policy_label!r}") from exc

    if policy.current_assignment:
        return get_assigned_students(problem_data.schools, problem_data.stops)
    if policy.min_distance_miles is None:
        return tuple(problem_data.students)

    threshold_km = policy.min_distance_miles * KM_PER_MILE
    return tuple(
        student
        for student in problem_data.students
        if _student_distance_to_school_km(problem_data, student) > threshold_km
    )


def problem_for_student_policy(
    problem_data: ProblemData,
    policy_label: str,
) -> ProblemData:
    return FilteredProblemData(
        name=f"{problem_data.name}_{policy_label}",
        base_problem_data=problem_data,
        _students=students_for_policy(problem_data, policy_label),
    )


def _load_assigned_framingham_problem(
    problem_name: str = DEFAULT_PROBLEM_NAME, place_name: str = DEFAULT_PLACE_NAME
) -> ProblemData:
    problem_data = setup(problem_name, place_name, None)
    assigned_students = get_assigned_students(problem_data.schools, problem_data.stops)
    return FilteredProblemData(
        name=f"{problem_data.name}_assigned",
        base_problem_data=problem_data,
        _students=assigned_students,
    )


def _load_full_framingham_problem(
    problem_name: str = DEFAULT_PROBLEM_NAME, place_name: str = DEFAULT_PLACE_NAME
) -> ProblemData:
    return setup(problem_name, place_name, None)


def _problem_with_buses(
    problem_data: ProblemData, buses: tuple[Bus, ...]
) -> ProblemData:
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
    template: BirdExportInstance | None = None,
    julia_timing_log: bool = True,
    gurobi_verbose: bool = False,
    cpus_per_solve: int = DEFAULT_CPUS_PER_SOLVE,
) -> tuple[dict[str, Any], BirdExportInstance, BirdBackendSolution]:
    if cpus_per_solve < 1:
        raise ValueError("cpus_per_solve must be at least 1")

    selected_buses = select_buses_by_type_counts(
        problem_data.buses,
        grid_point.counts,
        bus_order=bus_order,
    )
    if not selected_buses:
        raise ValueError("BiRD fleet-aware export requires at least one selected bus")

    config = bird_config_for_grid_point(grid_point)

    instance_path = output_dir / f"{grid_point.label}_instance.npz"
    if template is None:
        run_problem = _problem_with_buses(problem_data, selected_buses)
        export_bird_instance(run_problem, instance_path, config)
    else:
        bird_export_instance_from_template(template, selected_buses, config).save(
            instance_path
        )
    solution_path = output_dir / f"{grid_point.label}_solution.npz"
    log_file = output_dir / f"{grid_point.label}.log"

    julia_cmd = os.environ.get("JULIA_CMD", "julia")
    julia_project = os.environ.get("JULIA_PROJECT", "julia")
    command = [
        julia_cmd,
        f"--threads={cpus_per_solve}",
        f"--project={julia_project}",
        str(PROJECT_ROOT / "experiments" / "solve_bird_backend_julia.jl"),
        "--instance",
        str(instance_path),
        "--solution",
        str(solution_path),
        "--log-file",
        str(log_file),
        "--gurobi-threads",
        str(cpus_per_solve),
    ]
    if julia_timing_log:
        command.append("--timing-log")
    if gurobi_verbose:
        command.append("--gurobi-verbose")
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)

    instance = BirdExportInstance.load(instance_path)
    solution = BirdBackendSolution.load(solution_path)
    if solution.status not in {"OPTIMAL", "PARTIAL"}:
        raise RuntimeError(f"BiRD returned unsupported status {solution.status!r}")
    return summarize_bird_solution_for_mcdp(instance, solution), instance, solution


@dataclass(frozen=True)
class GridPointSolveResult:
    grid_point: GridPoint
    summary: dict[str, Any] | None = None
    status: str | None = None
    error: str | None = None


def solve_grid_point_result(
    problem_data: ProblemData,
    grid_point: GridPoint,
    *,
    output_dir: Path,
    bus_order: Mapping[str, list[str]],
    template: BirdExportInstance | None,
    julia_timing_log: bool,
    gurobi_verbose: bool,
    cpus_per_solve: int,
) -> GridPointSolveResult:
    try:
        summary, _instance, solution = solve_grid_point(
            problem_data,
            grid_point,
            output_dir=output_dir,
            bus_order=bus_order,
            template=template,
            julia_timing_log=julia_timing_log,
            gurobi_verbose=gurobi_verbose,
            cpus_per_solve=cpus_per_solve,
        )
    except Exception as exc:
        return GridPointSolveResult(grid_point=grid_point, error=str(exc))
    return GridPointSolveResult(
        grid_point=grid_point,
        summary=summary,
        status=solution.status,
    )


def write_static_catalogues(
    *,
    routing_lib: Path = ROUTING_LIB,
    costs: Mapping[str, Any] | None = None,
    routing_implementations: Mapping[str, dict[str, list[str]]] | None = None,
    guideline_implementations: Mapping[str, dict[str, list[str]]] | None = None,
) -> None:
    selected_costs = dict(costs or load_routing_costs())
    yaml_dir = routing_lib / "yaml_catalogues"
    fleet_bounds = (
        fleet_bounds_for_routing_implementations(routing_implementations)
        if routing_implementations is not None
        else read_bus_inventory_counts()
    )
    write_catalogue(
        yaml_dir / "fleet_bus_count.dpc.yaml",
        fleet_catalogue(fleet_bounds, selected_costs),
    )
    write_catalogue(
        yaml_dir / "guidelines.dpc.yaml",
        guidelines_catalogue(
            guideline_implementations
            if guideline_implementations is not None
            else {
                f"guideline_{policy_label}": default_guideline_entry(policy_label)
                for policy_label in GRID_STUDENT_POLICIES
            }
        ),
    )
    if routing_implementations is not None:
        write_catalogue(
            yaml_dir / "routing_service.dpc.yaml",
            routing_service_catalogue(routing_implementations),
        )
    for name, elements in config_posets().items():
        write_poset(routing_lib / f"{name}.mcdp_poset", elements)
    write_cost_modules(routing_lib, selected_costs)
    write_guidelines_module(routing_lib)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--routing-lib", type=Path, default=ROUTING_LIB)
    parser.add_argument("--cost-config", type=Path, default=COST_CONFIG)
    parser.add_argument("--current-routes", action="store_true", default=False)
    parser.add_argument("--julia-timing-log", action="store_true", default=False)
    parser.add_argument("--gurobi-verbose", action="store_true", default=False)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Grid points to solve concurrently; defaults to available CPUs divided "
            "by --cpus-per-solve."
        ),
    )
    parser.add_argument(
        "--cpus-per-solve",
        type=int,
        default=DEFAULT_CPUS_PER_SOLVE,
        help="CPU/thread budget for each grid point solve and Julia/Gurobi process.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    available_cpus = _available_cpus()

    output_dir = (
        args.output_dir.with_name(f"{args.output_dir.name}_current_routes")
        if args.current_routes
        else args.output_dir
    )
    logger.info(
        "routing_bird starting: output_dir={}, routing_lib={}, cost_config={}, "
        "current_routes={}, available_cpus={}, requested_workers={}, cpus_per_solve={}",
        output_dir,
        args.routing_lib,
        args.cost_config,
        args.current_routes,
        available_cpus,
        args.workers,
        args.cpus_per_solve,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory ready: {}", output_dir)

    logger.info("Loading routing costs from {}", args.cost_config)
    costs = load_routing_costs(args.cost_config)
    logger.info("Routing costs loaded")

    policy_labels = (
        ("current_assignment",) if args.current_routes else GRID_STUDENT_POLICIES
    )
    logger.info("Loading full Framingham problem data")
    base_problem_data = _load_full_framingham_problem()
    logger.info("Problem data loaded: {}", _problem_size_summary(base_problem_data))

    policy_problems: dict[str, ProblemData] = {}
    bird_templates: dict[str, BirdExportInstance] = {}
    guideline_implementations: dict[str, dict[str, list[str]]] = {}
    for policy_label in policy_labels:
        policy_problem = problem_for_student_policy(base_problem_data, policy_label)
        policy_problems[policy_label] = policy_problem
        guideline_implementations[f"guideline_{policy_label}"] = (
            guideline_entry_for_policy(policy_problem, policy_label)
        )
        logger.info(
            "Building Bird export template for policy {}: {}",
            policy_label,
            _problem_size_summary(policy_problem),
        )
        bird_template = build_bird_export_instance(
            policy_problem,
            BirdAdapterConfig(cohort="all", fleet_aware=True),
        )
        bird_templates[policy_label] = bird_template
        logger.info(
            "Bird export template built for {}: demand_rows={}, schools={}, "
            "fleet_size={}",
            policy_label,
            len(bird_template.demand_rows),
            len(bird_template.schools),
            bird_template.fleet_size,
        )

    logger.info("Reading bus inventory order from {}", BUS_CSV)
    bus_order = read_bus_inventory_order()
    logger.info(
        "Bus inventory order loaded: {}",
        {bus_type: len(names) for bus_type, names in bus_order.items()},
    )

    implementations: dict[str, dict[str, list[str]]] = {}
    partial_result: dict[str, Any] = {}
    errors: list[dict[str, Any]] = []

    if args.cpus_per_solve < 1:
        raise ValueError("--cpus-per-solve must be at least 1")
    if args.workers is not None and args.workers < 1:
        raise ValueError("--workers must be at least 1")

    grid_points = [
        grid_point
        for grid_point in iter_grid(policy_labels)
        if sum(grid_point.counts.values()) != 0
    ]
    worker_count = min(
        args.workers or _default_worker_count(args.cpus_per_solve),
        len(grid_points) or 1,
    )
    success_count = 0
    error_count = 0

    logger.info(
        "Grid prepared: grid_points={}, worker_count={}, cpus_per_solve={}, "
        "julia_timing_log={}, gurobi_verbose={}, policies={}",
        len(grid_points),
        worker_count,
        args.cpus_per_solve,
        args.julia_timing_log,
        args.gurobi_verbose,
        policy_labels,
    )

    def record_result(result: GridPointSolveResult) -> None:
        nonlocal success_count, error_count
        grid_point = result.grid_point
        if result.error is not None:
            errors.append({"label": grid_point.label, "error": result.error})
            error_count += 1
            return
        if result.summary is None or result.status is None:
            errors.append(
                {
                    "label": grid_point.label,
                    "error": "grid point solve returned an incomplete result",
                }
            )
            error_count += 1
            return

        implementations[grid_point.label] = routing_service_entry(
            result.summary,
            grid_point.config_labels,
        )
        partial_result[grid_point.label] = {
            "counts": grid_point.counts,
            "method": grid_point.method,
            "lambda_value": grid_point.lambda_value,
            "student_policy": grid_point.student_policy,
            "conventional_spillover": grid_point.conventional_spillover,
            "allow_partial": grid_point.allow_partial,
            "school_dwell_time": grid_point.school_dwell_time,
            "earliest_arrival_buffer": grid_point.earliest_arrival_buffer,
            "latest_arrival_buffer": grid_point.latest_arrival_buffer,
            "average_speed_mph": grid_point.average_speed_mph,
            "status": result.status,
            "summary": result.summary,
        }
        success_count += 1

    def update_progress(progress: tqdm, *, pending: int | None = None) -> None:
        postfix: dict[str, int] = {"ok": success_count, "errors": error_count}
        if pending is not None:
            postfix["pending"] = pending
        progress.set_postfix(postfix, refresh=False)

    if worker_count == 1:
        logger.info("Starting routing grid solve in serial mode")
        progress = tqdm(grid_points, desc="routing grid", unit="solve")
        for grid_point in progress:
            record_result(
                solve_grid_point_result(
                    policy_problems[grid_point.student_policy],
                    grid_point,
                    output_dir=output_dir,
                    bus_order=bus_order,
                    template=bird_templates[grid_point.student_policy],
                    julia_timing_log=args.julia_timing_log,
                    gurobi_verbose=args.gurobi_verbose,
                    cpus_per_solve=args.cpus_per_solve,
                )
            )
            update_progress(progress)
    else:
        logger.info("Starting routing grid solve with {} worker threads", worker_count)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            pending = {
                executor.submit(
                    solve_grid_point_result,
                    policy_problems[grid_point.student_policy],
                    grid_point,
                    output_dir=output_dir,
                    bus_order=bus_order,
                    template=bird_templates[grid_point.student_policy],
                    julia_timing_log=args.julia_timing_log,
                    gurobi_verbose=args.gurobi_verbose,
                    cpus_per_solve=args.cpus_per_solve,
                )
                for grid_point in grid_points
            }
            logger.info("Submitted {} grid point solves", len(pending))
            with tqdm(
                total=len(pending),
                desc=f"routing grid ({worker_count} workers)",
                unit="solve",
            ) as progress:
                while pending:
                    done, pending = wait(
                        pending,
                        timeout=PROGRESS_HEARTBEAT_SECONDS,
                        return_when=FIRST_COMPLETED,
                    )
                    if not done:
                        update_progress(progress, pending=len(pending))
                        logger.info(
                            "Routing grid progress: {}/{} complete, {} ok, "
                            "{} errors, {} pending",
                            progress.n,
                            progress.total,
                            success_count,
                            error_count,
                            len(pending),
                        )
                        continue
                    for future in done:
                        record_result(future.result())
                        progress.update()
                    update_progress(progress, pending=len(pending))

    logger.info(
        "Routing grid solve finished: attempted={}, ok={}, errors={}",
        len(grid_points),
        success_count,
        error_count,
    )

    logger.info("Writing static routing catalogues to {}", args.routing_lib)
    write_static_catalogues(
        routing_lib=args.routing_lib,
        costs=costs,
        routing_implementations=implementations or None,
        guideline_implementations=guideline_implementations,
    )
    partial_result_path = output_dir / "routing_bird_catalogue_partial_result.json"
    errors_path = output_dir / "routing_bird_errors.json"
    partial_result_path.write_text(
        json.dumps(partial_result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    errors_path.write_text(
        json.dumps(errors, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info("Wrote partial results to {}", partial_result_path)
    if errors:
        logger.warning("Wrote {} routing errors to {}", len(errors), errors_path)
    else:
        logger.info("Wrote empty routing error log to {}", errors_path)


if __name__ == "__main__":
    main()
