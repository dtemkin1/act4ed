from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Literal

import numpy as np

from formulation.common import (
    Bus,
    BusType,
    Depot,
    ProblemData,
    School,
    Stop,
    Student,
    l_s,
)
from formulation.formulation_3.problem3_definition import MPH_TO_KILOMETERS_PER_MINUTE
from formulation.normalized_result import (
    NormalizedBusItinerary,
    NormalizedRoute,
    NormalizedRoutingResult,
    RoutingSolutionJson,
    RoutingSolutionMetadata,
    RoutingSolutionRow,
)


_BIRD_INSTANCE_SCHEMA_VERSION = 4
_BIRD_SOLUTION_SCHEMA_VERSION = 1
_DEFAULT_BUS_SPEED_KM_PER_MINUTE = 40 * MPH_TO_KILOMETERS_PER_MINUTE
_DEFAULT_BIRD_LAMBDA_VALUE = 1.0e4
_DEFAULT_STOP_ASSIGNMENT_LAMBDA = 1.0e4


BirdCohort = Literal["conventional", "sped_no_wheelchair"]


@dataclass(frozen=True, slots=True)
class BirdAdapterConfig:
    cohort: BirdCohort = "conventional"
    bus_type: BusType | str | int | None = None
    max_time_on_bus: float = 120.0
    constant_stop_time: float = 0.0
    stop_time_per_student: float = 0.3
    school_dwell_time: float = 0.0
    speed_km_per_minute: float = _DEFAULT_BUS_SPEED_KM_PER_MINUTE
    lambda_value: float = _DEFAULT_BIRD_LAMBDA_VALUE
    reassign_stops: bool = False
    stop_assignment_lambda: float = _DEFAULT_STOP_ASSIGNMENT_LAMBDA
    max_walking_distance_km: float | None = None


@dataclass(frozen=True, slots=True)
class BirdDemandRow:
    external_stop_id: str
    source_stop_id: str
    stop_name: str
    school_id: str
    school_name: str
    students: int
    student_names: list[str]
    stop_node_id: int


@dataclass(frozen=True, slots=True)
class BirdSchoolView:
    id: str
    name: str
    node_id: int
    start_time: int


@dataclass(frozen=True, slots=True)
class BirdDepotView:
    name: str
    node_id: int


@dataclass(frozen=True, slots=True)
class BirdExportInstance:
    max_time_on_bus: float
    constant_stop_time: float
    stop_time_per_student: float
    school_dwell_time: float
    speed_km_per_minute: float
    lambda_value: float
    stop_assignment_enabled: bool
    stop_assignment_lambda: float
    max_walking_distance_km: float | None
    bus_capacity: int
    fleet_size: int
    cohort: str
    bus_type: str
    schools: list[School | BirdSchoolView]
    depots: list[Depot | BirdDepotView]
    demand_rows: list[BirdDemandRow]
    school_start_times: np.ndarray
    school_dwell_times: np.ndarray
    travel_distance_km: np.ndarray
    travel_time_min: np.ndarray
    demand_school_indices: np.ndarray

    def to_payload(self) -> dict[str, np.ndarray]:
        demand_student_names_ptr, demand_student_names_values = (
            _encode_ragged_bytes_array([row.student_names for row in self.demand_rows])
        )
        return {
            "schema_version": np.asarray(_BIRD_INSTANCE_SCHEMA_VERSION, dtype=np.int64),
            "max_time_on_bus": np.asarray(self.max_time_on_bus, dtype=np.float64),
            "constant_stop_time": np.asarray(self.constant_stop_time, dtype=np.float64),
            "stop_time_per_student": np.asarray(
                self.stop_time_per_student,
                dtype=np.float64,
            ),
            "school_dwell_time": np.asarray(self.school_dwell_time, dtype=np.float64),
            "speed_km_per_minute": np.asarray(
                self.speed_km_per_minute, dtype=np.float64
            ),
            "lambda_value": np.asarray(self.lambda_value, dtype=np.float64),
            "stop_assignment_enabled": np.asarray(
                1 if self.stop_assignment_enabled else 0,
                dtype=np.int64,
            ),
            "stop_assignment_lambda": np.asarray(
                self.stop_assignment_lambda,
                dtype=np.float64,
            ),
            "max_walking_distance_km": np.asarray(
                np.nan
                if self.max_walking_distance_km is None
                else self.max_walking_distance_km,
                dtype=np.float64,
            ),
            "bus_capacity": np.asarray(self.bus_capacity, dtype=np.int64),
            "fleet_size": np.asarray(self.fleet_size, dtype=np.int64),
            "cohort": _encode_bytes_array([self.cohort]),
            "bus_type": _encode_bytes_array([self.bus_type]),
            "school_ids": _encode_bytes_array(
                [str(school.id) for school in self.schools]
            ),
            "school_names": _encode_bytes_array(
                [school.name for school in self.schools]
            ),
            "school_start_times": self.school_start_times,
            "school_dwell_times": self.school_dwell_times,
            "school_node_ids": np.asarray(
                [school.node_id for school in self.schools],
                dtype=np.int64,
            ),
            "depot_ids": _encode_bytes_array(
                [str(depot.name) for depot in self.depots]
            ),
            "depot_names": _encode_bytes_array([depot.name for depot in self.depots]),
            "depot_node_ids": np.asarray(
                [depot.node_id for depot in self.depots],
                dtype=np.int64,
            ),
            "demand_external_stop_ids": _encode_bytes_array(
                [row.external_stop_id for row in self.demand_rows]
            ),
            "demand_source_stop_ids": _encode_bytes_array(
                [row.source_stop_id for row in self.demand_rows]
            ),
            "demand_stop_names": _encode_bytes_array(
                [row.stop_name for row in self.demand_rows]
            ),
            "demand_school_ids": _encode_bytes_array(
                [row.school_id for row in self.demand_rows]
            ),
            "demand_school_indices": self.demand_school_indices,
            "demand_students": np.asarray(
                [row.students for row in self.demand_rows],
                dtype=np.int64,
            ),
            "demand_student_names_ptr": demand_student_names_ptr,
            "demand_student_names_values": demand_student_names_values,
            "demand_stop_node_ids": np.asarray(
                [row.stop_node_id for row in self.demand_rows],
                dtype=np.int64,
            ),
            "travel_distance_km": self.travel_distance_km,
            "travel_time_min": self.travel_time_min,
        }

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        if output_path.suffix != ".npz":
            output_path = output_path.with_suffix(".npz")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **self.to_payload())
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "BirdExportInstance":
        with np.load(Path(path), allow_pickle=False) as payload:
            schema_version = int(np.asarray(payload["schema_version"]).item())
            if schema_version not in (1, 2, 3, _BIRD_INSTANCE_SCHEMA_VERSION):
                raise ValueError("unsupported Bird instance schema")

            schools = [
                BirdSchoolView(
                    id=school_id,
                    name=name,
                    node_id=int(node_id),
                    start_time=int(round(float(start_time))),
                )
                for school_id, name, node_id, start_time in zip(
                    _decode_bytes_array(payload["school_ids"]),
                    _decode_bytes_array(payload["school_names"]),
                    np.asarray(payload["school_node_ids"], dtype=np.int64),
                    np.asarray(payload["school_start_times"], dtype=np.float64),
                    strict=True,
                )
            ]
            depots = [
                BirdDepotView(name=depot_name, node_id=int(node_id))
                for depot_name, node_id in zip(
                    _decode_bytes_array(payload["depot_names"]),
                    np.asarray(payload["depot_node_ids"], dtype=np.int64),
                    strict=True,
                )
            ]
            if schema_version >= 3 and "demand_student_names_ptr" in payload.files:
                demand_student_names = _decode_ragged_bytes_array(
                    np.asarray(payload["demand_student_names_ptr"], dtype=np.int64),
                    np.asarray(payload["demand_student_names_values"]),
                )
            else:
                demand_student_names = [
                    []
                    for _ in range(
                        len(np.asarray(payload["demand_students"], dtype=np.int64))
                    )
                ]
            demand_rows = [
                BirdDemandRow(
                    external_stop_id=external_stop_id,
                    source_stop_id=source_stop_id,
                    stop_name=stop_name,
                    school_id=school_id,
                    school_name=next(
                        school.name for school in schools if str(school.id) == school_id
                    ),
                    students=int(students),
                    student_names=student_names,
                    stop_node_id=int(stop_node_id),
                )
                for external_stop_id, source_stop_id, stop_name, school_id, students, student_names, stop_node_id in zip(
                    _decode_bytes_array(payload["demand_external_stop_ids"]),
                    _decode_bytes_array(payload["demand_source_stop_ids"]),
                    _decode_bytes_array(payload["demand_stop_names"]),
                    _decode_bytes_array(payload["demand_school_ids"]),
                    np.asarray(payload["demand_students"], dtype=np.int64),
                    demand_student_names,
                    np.asarray(payload["demand_stop_node_ids"], dtype=np.int64),
                    strict=True,
                )
            ]
            return cls(
                max_time_on_bus=float(np.asarray(payload["max_time_on_bus"]).item()),
                constant_stop_time=float(
                    np.asarray(payload["constant_stop_time"]).item()
                ),
                stop_time_per_student=float(
                    np.asarray(payload["stop_time_per_student"]).item()
                ),
                school_dwell_time=float(
                    np.asarray(payload["school_dwell_time"]).item()
                ),
                speed_km_per_minute=float(
                    np.asarray(payload["speed_km_per_minute"]).item()
                ),
                lambda_value=(
                    float(np.asarray(payload["lambda_value"]).item())
                    if schema_version >= 2 and "lambda_value" in payload.files
                    else _DEFAULT_BIRD_LAMBDA_VALUE
                ),
                stop_assignment_enabled=(
                    int(np.asarray(payload["stop_assignment_enabled"]).item()) == 1
                    if schema_version >= 4
                    and "stop_assignment_enabled" in payload.files
                    else False
                ),
                stop_assignment_lambda=(
                    float(np.asarray(payload["stop_assignment_lambda"]).item())
                    if schema_version >= 4 and "stop_assignment_lambda" in payload.files
                    else _DEFAULT_STOP_ASSIGNMENT_LAMBDA
                ),
                max_walking_distance_km=(
                    None
                    if schema_version < 4
                    or "max_walking_distance_km" not in payload.files
                    or np.isnan(
                        float(np.asarray(payload["max_walking_distance_km"]).item())
                    )
                    else float(np.asarray(payload["max_walking_distance_km"]).item())
                ),
                bus_capacity=int(np.asarray(payload["bus_capacity"]).item()),
                fleet_size=int(np.asarray(payload["fleet_size"]).item()),
                cohort=_decode_bytes_array(payload["cohort"])[0],
                bus_type=_decode_bytes_array(payload["bus_type"])[0],
                schools=schools,
                depots=depots,
                demand_rows=demand_rows,
                school_start_times=np.asarray(
                    payload["school_start_times"], dtype=np.float64
                ),
                school_dwell_times=np.asarray(
                    payload["school_dwell_times"], dtype=np.float64
                ),
                travel_distance_km=np.asarray(
                    payload["travel_distance_km"], dtype=np.float64
                ),
                travel_time_min=np.asarray(
                    payload["travel_time_min"], dtype=np.float64
                ),
                demand_school_indices=np.asarray(
                    payload["demand_school_indices"],
                    dtype=np.int64,
                ),
            )


@dataclass(frozen=True, slots=True)
class BirdBackendSolution:
    status: str
    objective_value: float | None
    runtime_seconds: float
    buses_used: int
    total_distance_km: float
    total_service_time_min: float
    assignment_bus_ids: np.ndarray
    assignment_orders: np.ndarray
    assignment_school_indices: np.ndarray
    assignment_arrival_times: np.ndarray
    assignment_distance_km: np.ndarray
    assignment_service_time_min: np.ndarray
    assignment_stop_ptr: np.ndarray
    assignment_stop_values: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "BirdBackendSolution":
        with np.load(Path(path), allow_pickle=False) as payload:
            if (
                int(np.asarray(payload["schema_version"]).item())
                != _BIRD_SOLUTION_SCHEMA_VERSION
            ):
                raise ValueError("unsupported Bird solution schema")
            has_objective = int(np.asarray(payload["has_objective_value"]).item()) == 1
            return cls(
                status=_decode_utf8_array(payload["status_name_utf8"]),
                objective_value=(
                    float(np.asarray(payload["objective_value"]).item())
                    if has_objective
                    else None
                ),
                runtime_seconds=float(np.asarray(payload["runtime_seconds"]).item()),
                buses_used=int(np.asarray(payload["buses_used"]).item()),
                total_distance_km=float(
                    np.asarray(payload["total_distance_km"]).item()
                ),
                total_service_time_min=float(
                    np.asarray(payload["total_service_time_min"]).item()
                ),
                assignment_bus_ids=np.asarray(
                    payload["assignment_bus_ids"], dtype=np.int64
                ),
                assignment_orders=np.asarray(
                    payload["assignment_orders"], dtype=np.int64
                ),
                assignment_school_indices=np.asarray(
                    payload["assignment_school_indices"],
                    dtype=np.int64,
                ),
                assignment_arrival_times=np.asarray(
                    payload["assignment_arrival_times"],
                    dtype=np.float64,
                ),
                assignment_distance_km=np.asarray(
                    payload["assignment_distance_km"],
                    dtype=np.float64,
                ),
                assignment_service_time_min=np.asarray(
                    payload["assignment_service_time_min"],
                    dtype=np.float64,
                ),
                assignment_stop_ptr=np.asarray(
                    payload["assignment_stop_ptr"], dtype=np.int64
                ),
                assignment_stop_values=np.asarray(
                    payload["assignment_stop_values"],
                    dtype=np.int64,
                ),
            )


def _encode_bytes_array(values: list[str]) -> np.ndarray:
    return np.asarray([value.encode("utf-8") for value in values], dtype=np.bytes_)


def _encode_ragged_bytes_array(
    values: list[list[str]],
) -> tuple[np.ndarray, np.ndarray]:
    ptr = [0]
    flat: list[str] = []
    for group in values:
        flat.extend(group)
        ptr.append(len(flat))
    return np.asarray(ptr, dtype=np.int64), _encode_bytes_array(flat)


def _decode_bytes_array(values: np.ndarray) -> list[str]:
    return [bytes(value).decode("utf-8") for value in values.tolist()]


def _decode_ragged_bytes_array(ptr: np.ndarray, values: np.ndarray) -> list[list[str]]:
    decoded = _decode_bytes_array(values)
    return [decoded[int(ptr[idx]) : int(ptr[idx + 1])] for idx in range(len(ptr) - 1)]


def _decode_utf8_array(values: np.ndarray) -> str:
    return bytes(np.asarray(values, dtype=np.uint8).tolist()).decode("utf-8")


def _coerce_bus_type(value: BusType | str | int | None, buses: list[Bus]) -> BusType:
    if value is None:
        bus_types = {bus.type for bus in buses}
        if len(bus_types) != 1 or None in bus_types:
            raise ValueError("bus_type must be provided for mixed or untyped fleets")
        return next(iter(bus_types))
    if isinstance(value, BusType):
        return value
    if isinstance(value, str):
        if value.isdigit():
            return BusType(int(value))
        return BusType[value.upper()]
    return BusType(value)


def _filter_students_for_cohort(
    students: list[Student],
    cohort: BirdCohort,
) -> list[Student]:
    if cohort == "conventional":
        return [
            student
            for student in students
            if not student.requires_monitor and not student.requires_wheelchair
        ]
    return [
        student
        for student in students
        if student.requires_monitor and not student.requires_wheelchair
    ]


def _distance_km(
    problem_data: ProblemData, start_node_id: int, end_node_id: int
) -> float:
    if start_node_id == end_node_id:
        return 0.0
    edge_data = problem_data.service_graph.get_edge_data(
        start_node_id, end_node_id, key=0
    )
    if edge_data is None:
        return float("inf")
    return float(edge_data["length"])


def _walking_distance_meters(student: Student, stop: Stop) -> float:
    student_lon = math.radians(student.geographic_location.x)
    student_lat = math.radians(student.geographic_location.y)
    stop_lon = math.radians(stop.geographic_location.x)
    stop_lat = math.radians(stop.geographic_location.y)
    delta_lon = stop_lon - student_lon
    delta_lat = stop_lat - student_lat
    hav = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(student_lat) * math.cos(stop_lat) * math.sin(delta_lon / 2.0) ** 2
    )
    return 2.0 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(hav)))


def _assign_students_to_existing_stops(
    students: list[Student],
    stops: list[Stop],
    *,
    lambda_value: float,
    max_walking_distance_km: float | None,
) -> list[Stop]:
    try:
        import gurobipy as gp
    except ImportError as exc:
        raise RuntimeError(
            "optional Bird stop reassignment requires gurobipy to be installed"
        ) from exc

    if not stops:
        raise ValueError("stop reassignment requires at least one candidate stop")

    assigned_stops = [student.stop for student in students]
    max_walking_distance_m = (
        None if max_walking_distance_km is None else max_walking_distance_km * 1000.0
    )
    student_indices_by_school: dict[str | int, list[int]] = defaultdict(list)
    for student_idx, student in enumerate(students):
        student_indices_by_school[student.school.id].append(student_idx)

    for school_student_indices in student_indices_by_school.values():
        feasible_stops_by_student: dict[int, list[tuple[int, float]]] = {}
        active_stop_indices: set[int] = set()
        for student_idx in school_student_indices:
            student = students[student_idx]
            feasible_stops: list[tuple[int, float]] = []
            for stop_idx, stop in enumerate(stops):
                distance_m = _walking_distance_meters(student, stop)
                if (
                    max_walking_distance_m is None
                    or distance_m <= max_walking_distance_m
                    or stop == student.stop
                ):
                    feasible_stops.append((stop_idx, distance_m))
            if not feasible_stops:
                raise ValueError(
                    f"student {student.name} has no feasible stops for Bird reassignment"
                )
            feasible_stops_by_student[student_idx] = feasible_stops
            active_stop_indices.update(
                stop_idx for stop_idx, _distance_m in feasible_stops
            )

        model = gp.Model("bird_stop_assignment")
        model.Params.OutputFlag = 0
        z: dict[tuple[int, int], gp.Var] = {
            (student_idx, stop_idx): model.addVar(vtype=gp.GRB.BINARY)
            for student_idx in school_student_indices
            for stop_idx, _distance_m in feasible_stops_by_student[student_idx]
        }
        used_stop = {
            stop_idx: model.addVar(vtype=gp.GRB.BINARY)
            for stop_idx in sorted(active_stop_indices)
        }
        model.addConstrs(
            (
                gp.quicksum(
                    z[student_idx, stop_idx]
                    for stop_idx, _distance_m in feasible_stops_by_student[student_idx]
                )
                == 1
                for student_idx in school_student_indices
            )
        )
        model.addConstrs(
            (
                z[student_idx, stop_idx] <= used_stop[stop_idx]
                for student_idx in school_student_indices
                for stop_idx, _distance_m in feasible_stops_by_student[student_idx]
            )
        )
        model.setObjective(
            gp.quicksum(used_stop.values())
            + lambda_value
            * gp.quicksum(
                distance_m * z[student_idx, stop_idx]
                for student_idx in school_student_indices
                for stop_idx, distance_m in feasible_stops_by_student[student_idx]
            ),
            gp.GRB.MINIMIZE,
        )
        model.optimize()
        if model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError(
                f"Bird stop reassignment failed with Gurobi status {model.Status}"
            )

        for student_idx in school_student_indices:
            assigned_stop_idx = next(
                stop_idx
                for stop_idx, _distance_m in feasible_stops_by_student[student_idx]
                if z[student_idx, stop_idx].X > 0.5
            )
            assigned_stops[student_idx] = stops[assigned_stop_idx]

    return assigned_stops


def build_bird_export_instance(
    problem_data: ProblemData,
    config: BirdAdapterConfig,
) -> BirdExportInstance:
    bird_bus_type = _coerce_bus_type(config.bus_type, list(problem_data.buses))
    buses = [bus for bus in problem_data.buses if bus.type == bird_bus_type]
    if not buses:
        raise ValueError(f"no buses available for bus type {bird_bus_type.name}")

    capacities = {bus.capacity for bus in buses}
    if len(capacities) != 1:
        raise ValueError("Bird export requires a homogeneous-capacity fleet slice")
    bus_capacity = next(iter(capacities))

    selected_students = _filter_students_for_cohort(
        list(problem_data.students), config.cohort
    )
    if not selected_students:
        raise ValueError(f"no students available for cohort {config.cohort}")

    selected_school_ids = {student.school.id for student in selected_students}
    schools = [
        school for school in problem_data.schools if school.id in selected_school_ids
    ]
    if not schools:
        raise ValueError("selected cohort does not cover any schools")
    school_index = {school.id: idx + 1 for idx, school in enumerate(schools)}

    assigned_stops = (
        _assign_students_to_existing_stops(
            selected_students,
            list(problem_data.stops),
            lambda_value=config.stop_assignment_lambda,
            max_walking_distance_km=config.max_walking_distance_km,
        )
        if config.reassign_stops
        else [student.stop for student in selected_students]
    )

    student_names_by_pair: dict[tuple[Stop, School], list[str]] = defaultdict(list)
    for student, assigned_stop in zip(selected_students, assigned_stops, strict=True):
        student_names_by_pair[(assigned_stop, student.school)].append(student.name)

    demand_rows: list[BirdDemandRow] = []
    demand_school_indices: list[int] = []
    for school in schools:
        for stop in problem_data.stops:
            pair = (stop, school)
            student_names = student_names_by_pair.get(pair, [])
            if not student_names:
                continue
            demand_rows.append(
                BirdDemandRow(
                    external_stop_id=f"{school.id}:{stop.name}",
                    source_stop_id=stop.name,
                    stop_name=stop.name,
                    school_id=str(school.id),
                    school_name=school.name,
                    students=len(student_names),
                    student_names=list(student_names),
                    stop_node_id=stop.node_id,
                )
            )
            demand_school_indices.append(school_index[school.id])

    if not demand_rows:
        raise ValueError("selected cohort does not produce any Bird demand rows")

    depots = list(problem_data.depots)
    nodes: list[tuple[str, int]] = []
    for row in demand_rows:
        nodes.append(("stop", row.stop_node_id))
    for school in schools:
        nodes.append(("school", school.node_id))
    for depot in depots:
        nodes.append(("depot", depot.node_id))

    n_nodes = len(nodes)
    travel_distance_km = np.full((n_nodes, n_nodes), np.inf, dtype=np.float64)
    travel_time_min = np.full((n_nodes, n_nodes), np.inf, dtype=np.float64)
    for i, (_, start_node_id) in enumerate(nodes):
        for j, (_, end_node_id) in enumerate(nodes):
            distance = _distance_km(problem_data, start_node_id, end_node_id)
            if np.isinf(distance):
                continue
            travel_distance_km[i, j] = distance
            travel_time_min[i, j] = distance / config.speed_km_per_minute

    return BirdExportInstance(
        max_time_on_bus=config.max_time_on_bus,
        constant_stop_time=config.constant_stop_time,
        stop_time_per_student=config.stop_time_per_student,
        school_dwell_time=config.school_dwell_time,
        speed_km_per_minute=config.speed_km_per_minute,
        lambda_value=config.lambda_value,
        stop_assignment_enabled=config.reassign_stops,
        stop_assignment_lambda=config.stop_assignment_lambda,
        max_walking_distance_km=config.max_walking_distance_km,
        bus_capacity=bus_capacity,
        fleet_size=len(buses),
        cohort=config.cohort,
        bus_type=bird_bus_type.name,
        schools=schools,
        depots=depots,
        demand_rows=demand_rows,
        school_start_times=np.asarray(
            [l_s(school) for school in schools], dtype=np.float64
        ),
        school_dwell_times=np.asarray(
            [config.school_dwell_time for _ in schools],
            dtype=np.float64,
        ),
        travel_distance_km=travel_distance_km,
        travel_time_min=travel_time_min,
        demand_school_indices=np.asarray(demand_school_indices, dtype=np.int64),
    )


def export_bird_instance(
    problem_data: ProblemData,
    path: str | Path,
    config: BirdAdapterConfig,
) -> Path:
    return build_bird_export_instance(problem_data, config).save(path)


def _bird_demand_rows_by_school(
    instance: BirdExportInstance,
) -> dict[int, list[tuple[int, BirdDemandRow]]]:
    rows_by_school: dict[int, list[tuple[int, BirdDemandRow]]] = defaultdict(list)
    for demand_idx, (row, school_idx) in enumerate(
        zip(instance.demand_rows, instance.demand_school_indices.tolist(), strict=True),
        start=1,
    ):
        rows_by_school[int(school_idx)].append((demand_idx, row))
    return rows_by_school


def normalized_result_from_bird_solution(
    instance: BirdExportInstance,
    solution: BirdBackendSolution,
) -> NormalizedRoutingResult:
    routes: list[NormalizedRoute] = []
    itineraries_by_bus: dict[int, list[NormalizedRoute]] = defaultdict(list)
    school_arrivals: dict[str, list[float]] = defaultdict(list)
    demand_rows_by_school = _bird_demand_rows_by_school(instance)

    for row_idx, bus_id in enumerate(solution.assignment_bus_ids.tolist()):
        start = int(solution.assignment_stop_ptr[row_idx])
        end = int(solution.assignment_stop_ptr[row_idx + 1])
        school_index = int(solution.assignment_school_indices[row_idx]) - 1
        school = instance.schools[school_index]
        school_demand_rows = demand_rows_by_school[school_index + 1]
        stop_ids = [
            school_demand_rows[int(stop_idx) - 1][1].source_stop_id
            for stop_idx in solution.assignment_stop_values[start:end]
        ]
        route = NormalizedRoute(
            bus_id=f"bird_bus_{bus_id}",
            order=int(solution.assignment_orders[row_idx]),
            school_id=str(school.id),
            stop_ids=stop_ids,
            distance_km=float(solution.assignment_distance_km[row_idx]),
            arrival_time_min=float(solution.assignment_arrival_times[row_idx]),
        )
        routes.append(route)
        itineraries_by_bus[bus_id].append(route)
        school_arrivals[str(school.id)].append(
            float(solution.assignment_arrival_times[row_idx])
        )

    itineraries = [
        NormalizedBusItinerary(
            bus_id=f"bird_bus_{bus_id}",
            route_orders=[
                route.order
                for route in sorted(bus_routes, key=lambda route: route.order)
            ],
            school_ids=[
                route.school_id
                for route in sorted(bus_routes, key=lambda route: route.order)
            ],
            distance_km=float(sum(route.distance_km for route in bus_routes)),
        )
        for bus_id, bus_routes in sorted(itineraries_by_bus.items())
    ]

    return NormalizedRoutingResult(
        backend="bird",
        status=solution.status,
        objective_value=solution.objective_value,
        runtime_seconds=solution.runtime_seconds,
        buses_used=solution.buses_used,
        total_distance_km=solution.total_distance_km,
        routes=sorted(routes, key=lambda route: (route.bus_id, route.order)),
        itineraries=itineraries,
        school_arrivals=dict(school_arrivals),
        metadata={
            "cohort": instance.cohort,
            "bus_type": instance.bus_type,
            "fleet_size": instance.fleet_size,
            "bus_capacity": instance.bus_capacity,
            "stop_assignment_enabled": instance.stop_assignment_enabled,
            "stop_assignment_lambda": instance.stop_assignment_lambda,
            "max_walking_distance_km": instance.max_walking_distance_km,
            "wheelchair_supported": False,
        },
    )


def routing_solution_json_from_bird_solution(
    instance: BirdExportInstance,
    solution: BirdBackendSolution,
) -> RoutingSolutionJson:
    demand_rows_by_school = _bird_demand_rows_by_school(instance)
    demand_count = len(instance.demand_rows)
    school_count = len(instance.schools)

    rows_by_bus: dict[int, list[int]] = defaultdict(list)
    for row_idx, bus_id in enumerate(solution.assignment_bus_ids.tolist()):
        rows_by_bus[int(bus_id)].append(row_idx)

    rows: list[RoutingSolutionRow] = []
    served_student_names: set[str] = set()
    initial_origin_node = (
        int(instance.depots[0].node_id) if len(instance.depots) == 1 else None
    )
    initial_origin_matrix_idx = (
        demand_count + school_count + 1 if len(instance.depots) == 1 else None
    )

    for bus_id, bus_rows in sorted(rows_by_bus.items()):
        current_origin_node = initial_origin_node
        current_origin_matrix_idx = initial_origin_matrix_idx
        for row_idx in sorted(
            bus_rows, key=lambda idx: int(solution.assignment_orders[idx])
        ):
            school_index = int(solution.assignment_school_indices[row_idx])
            school = instance.schools[school_index - 1]
            school_demand_rows = demand_rows_by_school[school_index]

            start = int(solution.assignment_stop_ptr[row_idx])
            end = int(solution.assignment_stop_ptr[row_idx + 1])
            local_stop_ids = solution.assignment_stop_values[start:end].tolist()
            route_demand_rows = [
                school_demand_rows[int(stop_id) - 1] for stop_id in local_stop_ids
            ]

            student_names = sorted(
                student_name
                for _global_demand_idx, demand_row in route_demand_rows
                for student_name in demand_row.student_names
            )
            served_student_names.update(student_names)

            end_time = float(solution.assignment_arrival_times[row_idx])
            service_time = float(solution.assignment_service_time_min[row_idx])
            if route_demand_rows and current_origin_matrix_idx is not None:
                first_demand_idx = route_demand_rows[0][0]
                deadhead_time = float(
                    instance.travel_time_min[
                        current_origin_matrix_idx - 1, first_demand_idx - 1
                    ]
                )
                time_spent = deadhead_time + service_time
                start_time = float(end_time - time_spent)
            else:
                time_spent = None
                start_time = None

            rows.append(
                RoutingSolutionRow(
                    bus_name=f"bird_bus_{bus_id}",
                    round=int(solution.assignment_orders[row_idx]),
                    students_served=sum(
                        demand_row.students for _idx, demand_row in route_demand_rows
                    ),
                    distance_km=float(solution.assignment_distance_km[row_idx]),
                    origin_node_id=current_origin_node,
                    destination_node_id=int(school.node_id),
                    school_name=school.name,
                    stop_node_ids=[
                        int(demand_row.stop_node_id)
                        for _idx, demand_row in route_demand_rows
                    ],
                    start_time=start_time,
                    end_time=end_time,
                    time_spent=time_spent,
                    student_names=student_names,
                )
            )

            current_origin_node = int(school.node_id)
            current_origin_matrix_idx = demand_count + school_index

    rows = sorted(
        rows, key=lambda row: (row.bus_name, -1 if row.round is None else row.round)
    )
    return RoutingSolutionJson(
        metadata=RoutingSolutionMetadata(
            backend="bird",
            status=solution.status,
            objective_value=solution.objective_value,
            runtime_seconds=solution.runtime_seconds,
            buses_used=solution.buses_used,
            total_distance_km=solution.total_distance_km,
            total_students_served=len(served_student_names),
        ),
        solution=rows,
    )
