from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
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
from formulation.common.constants import MPH_TO_KM_PER_MIN
from formulation.normalized_result import (
    NormalizedBusItinerary,
    NormalizedRoute,
    NormalizedRoutingResult,
    RoutingSolutionJson,
    RoutingSolutionMetadata,
    RoutingSolutionRow,
)

_BIRD_INSTANCE_SCHEMA_VERSION = 14
_BIRD_SOLUTION_SCHEMA_VERSION = 1
_DEFAULT_BUS_MPH = 40.0
_DEFAULT_BUS_SPEED_KM_PER_MINUTE = _DEFAULT_BUS_MPH / MPH_TO_KM_PER_MIN
_DEFAULT_BIRD_LAMBDA_VALUE = 1.0e4
_DEFAULT_STOP_ASSIGNMENT_LAMBDA = 1.0e4
_SERVICE_GROUP_WHEELCHAIR = 1
_SERVICE_GROUP_SPED = 2
_SERVICE_GROUP_CONVENTIONAL = 3
_UNKNOWN_GRADE = "unknown"


BirdCohort = Literal[
    "all",
    "conventional",
    "sped_no_wheelchair",
    "sped_and_wheelchair",
    "wheelchair_no_sped",
]

# TODO currently this is not correctly respected in BiRD. I.e. if you set
# the policy to "route_assigned", it will just flag all buses as having
# monitors rather than trying to minimize the amount of monitors.
MonitorPolicy = Literal["fleet", "route_assigned"]
OptimizationMethod = Literal["lbh", "scenario"]


@dataclass(frozen=True, slots=True)
class BirdAdapterConfig:
    cohort: BirdCohort = "conventional"
    bus_type: BusType | str | int | None = None
    max_time_on_bus: float = 120.0
    constant_stop_time: float = 0.0
    stop_time_per_student: float = 0.3
    stop_time_per_wheelchair_student: float = 0.0
    stop_time_per_sped: float = 0.0
    school_dwell_time: float = 0.0
    earliest_arrival_buffer: float | None = None
    latest_arrival_buffer: float | None = None
    bus_mph: float = _DEFAULT_BUS_MPH
    speed_km_per_minute: float | None = field(default=None, repr=False)
    lambda_value: float = _DEFAULT_BIRD_LAMBDA_VALUE
    reassign_stops: bool = False
    stop_assignment_lambda: float = _DEFAULT_STOP_ASSIGNMENT_LAMBDA
    max_walking_distance_km: float | None = None
    fleet_aware: bool = False
    conventional_spillover: bool = False
    allow_partial: bool = False
    monitor_policy: MonitorPolicy = "fleet"
    method: OptimizationMethod = "lbh"

    def __post_init__(self) -> None:
        if self.speed_km_per_minute is None:
            if self.bus_mph <= 0:
                raise ValueError("bus_mph must be positive")
            object.__setattr__(
                self,
                "speed_km_per_minute",
                self.bus_mph / MPH_TO_KM_PER_MIN,
            )
            return

        if self.speed_km_per_minute <= 0:
            raise ValueError("speed_km_per_minute must be positive")
        object.__setattr__(
            self,
            "bus_mph",
            self.speed_km_per_minute * MPH_TO_KM_PER_MIN,
        )


@dataclass(frozen=True, slots=True)
class BirdDemandRow:
    external_stop_id: str
    source_stop_id: str
    stop_name: str
    school_id: str
    school_name: str
    service_group: str
    grade: str
    students: int
    special_ed_students: int
    wheelchair_students: int
    student_names: list[str]
    student_ids: list[str]
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
    stop_time_per_wheelchair_student: float
    stop_time_per_sped: float
    school_dwell_time: float
    earliest_arrival_buffer: float
    latest_arrival_buffer: float
    speed_km_per_minute: float
    lambda_value: float
    stop_assignment_enabled: bool
    stop_assignment_lambda: float
    max_walking_distance_km: float | None
    bus_capacity: int
    fleet_size: int
    cohort: str
    bus_type: str
    fleet_aware: bool
    conventional_spillover: bool
    allow_partial: bool
    monitor_policy: str
    method: str
    bus_names: list[str]
    bus_capacities: np.ndarray
    bus_depot_indices: np.ndarray
    bus_has_monitor: np.ndarray
    bus_wheelchair_capacities: np.ndarray
    bus_type_names: list[str]
    schools: list[School | BirdSchoolView]
    depots: list[Depot | BirdDepotView]
    demand_rows: list[BirdDemandRow]
    school_start_times: np.ndarray
    school_dwell_times: np.ndarray
    school_earliest_arrival_buffers: np.ndarray
    school_latest_arrival_buffers: np.ndarray
    travel_distance_km: np.ndarray
    travel_time_min: np.ndarray
    demand_school_indices: np.ndarray

    @property
    def bus_mph(self) -> float:
        return self.speed_km_per_minute * MPH_TO_KM_PER_MIN

    def to_payload(self) -> dict[str, np.ndarray]:
        demand_student_names_ptr, demand_student_names_values = (
            _encode_ragged_bytes_array([row.student_names for row in self.demand_rows])
        )
        demand_student_ids_ptr, demand_student_ids_values = _encode_ragged_bytes_array(
            [row.student_ids for row in self.demand_rows]
        )
        return {
            "schema_version": np.asarray(_BIRD_INSTANCE_SCHEMA_VERSION, dtype=np.int64),
            "max_time_on_bus": np.asarray(self.max_time_on_bus, dtype=np.float64),
            "constant_stop_time": np.asarray(self.constant_stop_time, dtype=np.float64),
            "stop_time_per_student": np.asarray(
                self.stop_time_per_student,
                dtype=np.float64,
            ),
            "stop_time_per_wheelchair_student": np.asarray(
                self.stop_time_per_wheelchair_student,
                dtype=np.float64,
            ),
            "stop_time_per_sped": np.asarray(
                self.stop_time_per_sped,
                dtype=np.float64,
            ),
            "school_dwell_time": np.asarray(self.school_dwell_time, dtype=np.float64),
            "earliest_arrival_buffer": np.asarray(
                self.earliest_arrival_buffer,
                dtype=np.float64,
            ),
            "latest_arrival_buffer": np.asarray(
                self.latest_arrival_buffer,
                dtype=np.float64,
            ),
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
                (
                    np.nan
                    if self.max_walking_distance_km is None
                    else self.max_walking_distance_km
                ),
                dtype=np.float64,
            ),
            "bus_capacity": np.asarray(self.bus_capacity, dtype=np.int64),
            "fleet_size": np.asarray(self.fleet_size, dtype=np.int64),
            "cohort": _encode_bytes_array([self.cohort]),
            "bus_type": _encode_bytes_array([self.bus_type]),
            "fleet_aware": np.asarray(
                1 if self.fleet_aware else 0,
                dtype=np.int64,
            ),
            "conventional_spillover": np.asarray(
                1 if self.conventional_spillover else 0,
                dtype=np.int64,
            ),
            "allow_partial": np.asarray(
                1 if self.allow_partial else 0,
                dtype=np.int64,
            ),
            "monitor_policy": _encode_bytes_array([self.monitor_policy]),
            "method_id": np.asarray(_method_id(self.method), dtype=np.int64),
            "bus_names": _encode_bytes_array(self.bus_names),
            "bus_capacities": self.bus_capacities,
            "bus_depot_indices": self.bus_depot_indices,
            "bus_has_monitor": self.bus_has_monitor,
            "bus_wheelchair_capacities": self.bus_wheelchair_capacities,
            "bus_type_names": _encode_bytes_array(self.bus_type_names),
            "school_ids": _encode_bytes_array(
                [str(school.id) for school in self.schools]
            ),
            "school_names": _encode_bytes_array(
                [school.name for school in self.schools]
            ),
            "school_start_times": self.school_start_times,
            "school_dwell_times": self.school_dwell_times,
            "school_earliest_arrival_buffers": (self.school_earliest_arrival_buffers),
            "school_latest_arrival_buffers": self.school_latest_arrival_buffers,
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
            "demand_service_groups": _encode_bytes_array(
                [row.service_group for row in self.demand_rows]
            ),
            "demand_group_ids": np.asarray(
                [_service_group_id(row.service_group) for row in self.demand_rows],
                dtype=np.int64,
            ),
            "demand_grades": _encode_bytes_array(
                [row.grade for row in self.demand_rows]
            ),
            "demand_grade_ids": np.asarray(
                _demand_grade_ids(self.demand_rows),
                dtype=np.int64,
            ),
            "demand_students": np.asarray(
                [row.students for row in self.demand_rows],
                dtype=np.int64,
            ),
            "demand_special_ed_students": np.asarray(
                [row.special_ed_students for row in self.demand_rows],
                dtype=np.int64,
            ),
            "demand_wheelchair_students": np.asarray(
                [row.wheelchair_students for row in self.demand_rows],
                dtype=np.int64,
            ),
            "demand_student_names_ptr": demand_student_names_ptr,
            "demand_student_names_values": demand_student_names_values,
            "demand_student_ids_ptr": demand_student_ids_ptr,
            "demand_student_ids_values": demand_student_ids_values,
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
            if schema_version not in range(1, _BIRD_INSTANCE_SCHEMA_VERSION + 1):
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
            if schema_version >= 13 and "demand_student_ids_ptr" in payload.files:
                demand_student_ids = _decode_ragged_bytes_array(
                    np.asarray(payload["demand_student_ids_ptr"], dtype=np.int64),
                    np.asarray(payload["demand_student_ids_values"]),
                )
            else:
                demand_student_ids = demand_student_names
            demand_students = np.asarray(payload["demand_students"], dtype=np.int64)
            if schema_version >= 5 and "demand_service_groups" in payload.files:
                demand_service_groups = _decode_bytes_array(
                    payload["demand_service_groups"]
                )
            else:
                demand_service_groups = [
                    "conventional" for _ in range(len(demand_students))
                ]
            if schema_version >= 9 and "demand_grades" in payload.files:
                demand_grades = _decode_bytes_array(payload["demand_grades"])
            else:
                demand_grades = [_UNKNOWN_GRADE for _ in range(len(demand_students))]
            demand_wheelchair_students = (
                np.asarray(payload["demand_wheelchair_students"], dtype=np.int64)
                if schema_version >= 5 and "demand_wheelchair_students" in payload.files
                else np.zeros(len(demand_students), dtype=np.int64)
            )
            demand_special_ed_students = (
                np.asarray(payload["demand_special_ed_students"], dtype=np.int64)
                if schema_version >= 10
                and "demand_special_ed_students" in payload.files
                else np.asarray(
                    [
                        students if service_group == "sped" else 0
                        for students, service_group in zip(
                            demand_students,
                            demand_service_groups,
                            strict=True,
                        )
                    ],
                    dtype=np.int64,
                )
            )
            demand_rows = [
                BirdDemandRow(
                    external_stop_id=external_stop_id,
                    source_stop_id=source_stop_id,
                    stop_name=stop_name,
                    school_id=school_id,
                    school_name=next(
                        school.name for school in schools if str(school.id) == school_id
                    ),
                    service_group=service_group,
                    grade=grade,
                    students=int(students),
                    special_ed_students=int(special_ed_students),
                    wheelchair_students=int(wheelchair_students),
                    student_names=student_names,
                    student_ids=student_ids,
                    stop_node_id=int(stop_node_id),
                )
                for external_stop_id, source_stop_id, stop_name, school_id, service_group, grade, students, special_ed_students, wheelchair_students, student_names, student_ids, stop_node_id in zip(
                    _decode_bytes_array(payload["demand_external_stop_ids"]),
                    _decode_bytes_array(payload["demand_source_stop_ids"]),
                    _decode_bytes_array(payload["demand_stop_names"]),
                    _decode_bytes_array(payload["demand_school_ids"]),
                    demand_service_groups,
                    demand_grades,
                    demand_students,
                    demand_special_ed_students,
                    demand_wheelchair_students,
                    demand_student_names,
                    demand_student_ids,
                    np.asarray(payload["demand_stop_node_ids"], dtype=np.int64),
                    strict=True,
                )
            ]
            if schema_version >= 5 and "bus_names" in payload.files:
                bus_names = _decode_bytes_array(payload["bus_names"])
                bus_capacities = np.asarray(payload["bus_capacities"], dtype=np.int64)
                bus_depot_indices = np.asarray(
                    payload["bus_depot_indices"],
                    dtype=np.int64,
                )
                bus_has_monitor = np.asarray(payload["bus_has_monitor"], dtype=np.int64)
                bus_wheelchair_capacities = np.asarray(
                    payload["bus_wheelchair_capacities"],
                    dtype=np.int64,
                )
                bus_type_names = _decode_bytes_array(payload["bus_type_names"])
            else:
                fleet_size = int(np.asarray(payload["fleet_size"]).item())
                bus_capacity = int(np.asarray(payload["bus_capacity"]).item())
                bus_names = [f"bird_bus_{idx}" for idx in range(1, fleet_size + 1)]
                bus_capacities = np.full(fleet_size, bus_capacity, dtype=np.int64)
                bus_depot_indices = np.ones(fleet_size, dtype=np.int64)
                bus_has_monitor = np.zeros(fleet_size, dtype=np.int64)
                bus_wheelchair_capacities = np.zeros(fleet_size, dtype=np.int64)
                bus_type_names = [
                    _decode_bytes_array(payload["bus_type"])[0]
                    for _ in range(fleet_size)
                ]
            school_dwell_time = float(np.asarray(payload["school_dwell_time"]).item())
            school_dwell_times = np.asarray(
                payload["school_dwell_times"], dtype=np.float64
            )
            if schema_version >= 7 and "school_latest_arrival_buffers" in payload.files:
                school_latest_arrival_buffers = np.asarray(
                    payload["school_latest_arrival_buffers"],
                    dtype=np.float64,
                )
            else:
                school_latest_arrival_buffers = school_dwell_times.copy()
            if (
                schema_version >= 7
                and "school_earliest_arrival_buffers" in payload.files
            ):
                school_earliest_arrival_buffers = np.asarray(
                    payload["school_earliest_arrival_buffers"],
                    dtype=np.float64,
                )
            else:
                school_earliest_arrival_buffers = school_latest_arrival_buffers.copy()
            latest_arrival_buffer = (
                float(np.asarray(payload["latest_arrival_buffer"]).item())
                if schema_version >= 7 and "latest_arrival_buffer" in payload.files
                else school_dwell_time
            )
            earliest_arrival_buffer = (
                float(np.asarray(payload["earliest_arrival_buffer"]).item())
                if schema_version >= 7 and "earliest_arrival_buffer" in payload.files
                else latest_arrival_buffer
            )
            return cls(
                max_time_on_bus=float(np.asarray(payload["max_time_on_bus"]).item()),
                constant_stop_time=float(
                    np.asarray(payload["constant_stop_time"]).item()
                ),
                stop_time_per_student=float(
                    np.asarray(payload["stop_time_per_student"]).item()
                ),
                stop_time_per_wheelchair_student=(
                    float(
                        np.asarray(payload["stop_time_per_wheelchair_student"]).item()
                    )
                    if schema_version >= 8
                    and "stop_time_per_wheelchair_student" in payload.files
                    else 0.0
                ),
                stop_time_per_sped=(
                    float(np.asarray(payload["stop_time_per_sped"]).item())
                    if schema_version >= 14 and "stop_time_per_sped" in payload.files
                    else 0.0
                ),
                school_dwell_time=school_dwell_time,
                earliest_arrival_buffer=earliest_arrival_buffer,
                latest_arrival_buffer=latest_arrival_buffer,
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
                fleet_aware=(
                    int(np.asarray(payload["fleet_aware"]).item()) == 1
                    if schema_version >= 5 and "fleet_aware" in payload.files
                    else False
                ),
                conventional_spillover=(
                    int(np.asarray(payload["conventional_spillover"]).item()) == 1
                    if schema_version >= 5 and "conventional_spillover" in payload.files
                    else False
                ),
                allow_partial=(
                    int(np.asarray(payload["allow_partial"]).item()) == 1
                    if schema_version >= 6 and "allow_partial" in payload.files
                    else False
                ),
                monitor_policy=(
                    _decode_bytes_array(payload["monitor_policy"])[0]
                    if schema_version >= 10 and "monitor_policy" in payload.files
                    else "fleet"
                ),
                method=(
                    _method_from_id(int(np.asarray(payload["method_id"]).item()))
                    if schema_version >= 12 and "method_id" in payload.files
                    else "lbh"
                ),
                bus_names=bus_names,
                bus_capacities=bus_capacities,
                bus_depot_indices=bus_depot_indices,
                bus_has_monitor=bus_has_monitor,
                bus_wheelchair_capacities=bus_wheelchair_capacities,
                bus_type_names=bus_type_names,
                schools=schools,
                depots=depots,
                demand_rows=demand_rows,
                school_start_times=np.asarray(
                    payload["school_start_times"], dtype=np.float64
                ),
                school_dwell_times=school_dwell_times,
                school_earliest_arrival_buffers=school_earliest_arrival_buffers,
                school_latest_arrival_buffers=school_latest_arrival_buffers,
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
    unassigned_school_indices: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=np.int64),
    )
    unassigned_stop_indices: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=np.int64),
    )

    def student_ride_times(
        self,
        instance: BirdExportInstance,
        *,
        validate_service_time: bool = True,
        tolerance_min: float = 1.0e-6,
    ) -> list[dict[str, object]]:
        """Return one timing record per served student using this solution."""
        return bird_student_ride_times(
            instance,
            self,
            validate_service_time=validate_service_time,
            tolerance_min=tolerance_min,
        )

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
                unassigned_school_indices=np.asarray(
                    (
                        payload["unassigned_school_indices"]
                        if "unassigned_school_indices" in payload.files
                        else []
                    ),
                    dtype=np.int64,
                ),
                unassigned_stop_indices=np.asarray(
                    (
                        payload["unassigned_stop_indices"]
                        if "unassigned_stop_indices" in payload.files
                        else []
                    ),
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


def _method_id(method: str) -> int:
    if method == "lbh":
        return 1
    if method == "scenario":
        return 2
    raise ValueError(f"unknown Bird optimization method {method!r}")


def _method_from_id(method_id: int) -> str:
    if method_id == 1:
        return "lbh"
    if method_id == 2:
        return "scenario"
    raise ValueError(f"unknown Bird optimization method id {method_id}")


def _student_attributes(student: Student):
    attributes = getattr(student, "attributes", None)
    if attributes is None:
        attributes = getattr(student, "demographics")
    return attributes


def _student_is_special_ed(student: Student) -> bool:
    return bool(student.attributes.special_ed)


def _student_is_wheelchair_user(student: Student) -> bool:
    return bool(student.attributes.wheelchair_user)


def _student_needs_monitor(student: Student) -> bool:
    return _student_is_special_ed(student) or _student_is_wheelchair_user(student)


def _student_service_group(student: Student) -> str:
    if _student_is_wheelchair_user(student):
        return "wheelchair"
    if _student_is_special_ed(student):
        return "sped"
    return "conventional"


def _student_grade(student: Student) -> str:
    grade = getattr(student, "grade", None)
    if grade is None:
        return _UNKNOWN_GRADE
    grade_text = str(grade).strip()
    return grade_text if grade_text else _UNKNOWN_GRADE


def _service_group_id(group: str) -> int:
    if group == "wheelchair":
        return _SERVICE_GROUP_WHEELCHAIR
    if group == "sped":
        return _SERVICE_GROUP_SPED
    if group == "conventional":
        return _SERVICE_GROUP_CONVENTIONAL
    raise ValueError(f"unknown Bird service group {group!r}")


def _demand_grade_ids(demand_rows: list[BirdDemandRow]) -> list[int]:
    grade_to_id: dict[str, int] = {}
    grade_ids: list[int] = []
    for row in demand_rows:
        if row.grade not in grade_to_id:
            grade_to_id[row.grade] = len(grade_to_id) + 1
        grade_ids.append(grade_to_id[row.grade])
    return grade_ids


def _demand_stop_external_id(
    service_group: str,
    grade: str,
    school_id: str,
    stop_name: str,
    *,
    fleet_aware: bool,
) -> str:
    if grade == _UNKNOWN_GRADE:
        return (
            f"{service_group}:{school_id}:{stop_name}"
            if fleet_aware
            else f"{school_id}:{stop_name}"
        )
    return (
        f"{service_group}:{grade}:{school_id}:{stop_name}"
        if fleet_aware
        else f"{grade}:{school_id}:{stop_name}"
    )


def _wheelchair_capacity_for_bus(bus: Bus) -> int:
    return bus.wheelchair_capacity


def _bus_type_name(bus: Bus) -> str:
    return bus.type.name if bus.type is not None else "untyped"


def _depot_index(depots: list[Depot], depot: Depot) -> int:
    for idx, candidate in enumerate(depots, start=1):
        if candidate == depot:
            return idx
    raise ValueError(f"bus depot {depot.name} is not present in Bird depots")


def _template_depot_index(
    depots: list[Depot | BirdDepotView],
    depot: Depot,
) -> int:
    for idx, candidate in enumerate(depots, start=1):
        if candidate == depot or candidate.node_id == depot.node_id:
            return idx
    raise ValueError(f"bus depot {depot.name} is not present in Bird template depots")


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


def _resolve_arrival_buffers(config: BirdAdapterConfig) -> tuple[float, float]:
    latest_arrival_buffer = (
        config.school_dwell_time
        if config.latest_arrival_buffer is None
        else config.latest_arrival_buffer
    )
    earliest_arrival_buffer = (
        latest_arrival_buffer
        if config.earliest_arrival_buffer is None
        else config.earliest_arrival_buffer
    )
    if latest_arrival_buffer < 0 or earliest_arrival_buffer < 0:
        raise ValueError("Bird arrival buffers must be non-negative")
    if earliest_arrival_buffer < latest_arrival_buffer:
        raise ValueError(
            "Bird earliest_arrival_buffer must be greater than or equal to "
            "latest_arrival_buffer"
        )
    return float(earliest_arrival_buffer), float(latest_arrival_buffer)


def _filter_students_for_cohort(
    students: list[Student],
    cohort: BirdCohort,
) -> list[Student]:
    if cohort == "all":
        return students
    if cohort == "conventional":
        return [
            student
            for student in students
            if not _student_is_special_ed(student)
            and not _student_is_wheelchair_user(student)
        ]
    if cohort == "sped_no_wheelchair":
        return [
            student
            for student in students
            if _student_is_special_ed(student)
            and not _student_is_wheelchair_user(student)
        ]
    if cohort == "sped_and_wheelchair":
        return [
            student
            for student in students
            if _student_is_special_ed(student) or _student_is_wheelchair_user(student)
        ]
    if cohort == "wheelchair_no_sped":
        return [
            student
            for student in students
            if _student_is_wheelchair_user(student)
            and not _student_is_special_ed(student)
        ]
    raise ValueError(f"unknown Bird cohort {cohort!r}")


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


def assign_students_to_existing_stops(
    students: list[Student],
    stops: list[Stop],
    *,
    lambda_value: float,
    max_walking_distance_km: float | None,
) -> list[Stop]:
    """Assign students to existing candidate stops with BiRD's stop MIP."""
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


_assign_students_to_existing_stops = assign_students_to_existing_stops


def build_bird_export_instance(
    problem_data: ProblemData,
    config: BirdAdapterConfig,
) -> BirdExportInstance:
    earliest_arrival_buffer, latest_arrival_buffer = _resolve_arrival_buffers(config)
    all_buses = list(problem_data.buses)
    if config.fleet_aware:
        if config.bus_type is None:
            buses = all_buses
        else:
            bird_bus_type = _coerce_bus_type(config.bus_type, all_buses)
            buses = [bus for bus in all_buses if bus.type == bird_bus_type]
    else:
        bird_bus_type = _coerce_bus_type(config.bus_type, all_buses)
        buses = [bus for bus in all_buses if bus.type == bird_bus_type]
    if not buses:
        raise ValueError("no buses available for Bird export")

    if config.fleet_aware:
        bus_capacity = max(bus.capacity for bus in buses)
        bus_type_names = {_bus_type_name(bus) for bus in buses}
        bus_type = "mixed" if len(bus_type_names) > 1 else next(iter(bus_type_names))
    else:
        capacities = {bus.capacity for bus in buses}
        if len(capacities) != 1:
            raise ValueError("Bird export requires a homogeneous-capacity fleet slice")
        bus_capacity = next(iter(capacities))
        bus_type = bird_bus_type.name

    selected_students = _filter_students_for_cohort(
        list(problem_data.students), config.cohort
    )
    if not selected_students:
        raise ValueError(f"no students available for cohort {config.cohort}")
    if config.monitor_policy not in ("fleet", "route_assigned"):
        raise ValueError(f"unknown Bird monitor policy {config.monitor_policy!r}")

    def bus_has_monitor(bus: Bus) -> bool:
        return config.monitor_policy == "route_assigned" or bus.has_monitor

    if config.fleet_aware:
        has_wheelchair_students = any(
            _student_is_wheelchair_user(student) for student in selected_students
        )
        if has_wheelchair_students and not any(
            bus_has_monitor(bus) and _wheelchair_capacity_for_bus(bus) > 0
            for bus in buses
        ):
            raise ValueError(
                "fleet-aware Bird export has wheelchair students but no monitor bus "
                "with wheelchair capacity"
            )
        has_sped_students = any(
            _student_is_special_ed(student) and not _student_is_wheelchair_user(student)
            for student in selected_students
        )
        if has_sped_students and not any(bus_has_monitor(bus) for bus in buses):
            raise ValueError(
                "fleet-aware Bird export has SPED students but no monitor bus"
            )

    selected_school_ids = {student.school.id for student in selected_students}
    schools = [
        school for school in problem_data.schools if school.id in selected_school_ids
    ]
    if not schools:
        raise ValueError("selected cohort does not cover any schools")
    school_index = {school.id: idx + 1 for idx, school in enumerate(schools)}

    assigned_stops = (
        assign_students_to_existing_stops(
            selected_students,
            list(problem_data.stops),
            lambda_value=config.stop_assignment_lambda,
            max_walking_distance_km=config.max_walking_distance_km,
        )
        if config.reassign_stops
        else [student.stop for student in selected_students]
    )

    students_by_key: dict[tuple[Stop, School, str, str], list[Student]] = defaultdict(
        list
    )
    grade_order: list[str] = []
    for student, assigned_stop in zip(selected_students, assigned_stops, strict=True):
        grade = _student_grade(student)
        if grade not in grade_order:
            grade_order.append(grade)
        students_by_key[
            (assigned_stop, student.school, _student_service_group(student), grade)
        ].append(student)

    demand_rows: list[BirdDemandRow] = []
    demand_school_indices: list[int] = []
    service_groups = ("wheelchair", "sped", "conventional")
    for school in schools:
        for stop in problem_data.stops:
            for service_group in service_groups:
                for grade in grade_order:
                    key = (stop, school, service_group, grade)
                    students_at_stop = students_by_key.get(key, [])
                    if not students_at_stop:
                        continue
                    demand_rows.append(
                        BirdDemandRow(
                            external_stop_id=_demand_stop_external_id(
                                service_group,
                                grade,
                                str(school.id),
                                stop.name,
                                fleet_aware=config.fleet_aware,
                            ),
                            source_stop_id=stop.name,
                            stop_name=stop.name,
                            school_id=str(school.id),
                            school_name=school.name,
                            service_group=service_group,
                            grade=grade,
                            students=len(students_at_stop),
                            special_ed_students=sum(
                                1
                                for student in students_at_stop
                                if _student_is_special_ed(student)
                            ),
                            wheelchair_students=sum(
                                1
                                for student in students_at_stop
                                if _student_is_wheelchair_user(student)
                            ),
                            student_names=[
                                student.name for student in students_at_stop
                            ],
                            student_ids=[
                                str(student.id) for student in students_at_stop
                            ],
                            stop_node_id=stop.node_id,
                        ),
                    )
                    demand_school_indices.append(school_index[school.id])

    if not demand_rows:
        raise ValueError("selected cohort does not produce any Bird demand rows")

    depots = list(problem_data.depots)
    for bus in buses:
        if bus.depot not in depots:
            depots.append(bus.depot)
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
        stop_time_per_wheelchair_student=config.stop_time_per_wheelchair_student,
        stop_time_per_sped=config.stop_time_per_sped,
        school_dwell_time=config.school_dwell_time,
        earliest_arrival_buffer=earliest_arrival_buffer,
        latest_arrival_buffer=latest_arrival_buffer,
        speed_km_per_minute=config.speed_km_per_minute,
        lambda_value=config.lambda_value,
        stop_assignment_enabled=config.reassign_stops,
        stop_assignment_lambda=config.stop_assignment_lambda,
        max_walking_distance_km=config.max_walking_distance_km,
        bus_capacity=bus_capacity,
        fleet_size=len(buses),
        cohort=config.cohort,
        bus_type=bus_type,
        fleet_aware=config.fleet_aware,
        conventional_spillover=config.conventional_spillover,
        allow_partial=config.allow_partial,
        monitor_policy=config.monitor_policy,
        method=config.method,
        bus_names=[bus.name for bus in buses],
        bus_capacities=np.asarray([bus.capacity for bus in buses], dtype=np.int64),
        bus_depot_indices=np.asarray(
            [_depot_index(depots, bus.depot) for bus in buses],
            dtype=np.int64,
        ),
        bus_has_monitor=np.asarray(
            [1 if bus_has_monitor(bus) else 0 for bus in buses],
            dtype=np.int64,
        ),
        bus_wheelchair_capacities=np.asarray(
            [_wheelchair_capacity_for_bus(bus) for bus in buses],
            dtype=np.int64,
        ),
        bus_type_names=[_bus_type_name(bus) for bus in buses],
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
        school_earliest_arrival_buffers=np.asarray(
            [earliest_arrival_buffer for _ in schools],
            dtype=np.float64,
        ),
        school_latest_arrival_buffers=np.asarray(
            [latest_arrival_buffer for _ in schools],
            dtype=np.float64,
        ),
        travel_distance_km=travel_distance_km,
        travel_time_min=travel_time_min,
        demand_school_indices=np.asarray(demand_school_indices, dtype=np.int64),
    )


def bird_export_instance_from_template(
    template: BirdExportInstance,
    buses: Iterable[Bus],
    config: BirdAdapterConfig,
) -> BirdExportInstance:
    selected_buses = list(buses)
    if not selected_buses:
        raise ValueError("no buses available for Bird export")
    if not template.fleet_aware or not config.fleet_aware:
        raise ValueError("Bird template reuse currently requires fleet_aware=True")
    if template.cohort != config.cohort:
        raise ValueError("Bird template cohort does not match requested config")
    if template.stop_assignment_enabled != config.reassign_stops:
        raise ValueError(
            "Bird template stop reassignment setting does not match requested config"
        )
    if config.reassign_stops:
        if (
            template.stop_assignment_lambda != config.stop_assignment_lambda
            or template.max_walking_distance_km != config.max_walking_distance_km
        ):
            raise ValueError(
                "Bird template stop reassignment parameters do not match requested config"
            )
    if config.monitor_policy not in ("fleet", "route_assigned"):
        raise ValueError(f"unknown Bird monitor policy {config.monitor_policy!r}")

    def bus_has_monitor(bus: Bus) -> bool:
        return config.monitor_policy == "route_assigned" or bus.has_monitor

    bus_monitor_flags = [bus_has_monitor(bus) for bus in selected_buses]
    bus_wheelchair_capacities = [
        _wheelchair_capacity_for_bus(bus) for bus in selected_buses
    ]
    if any(row.wheelchair_students > 0 for row in template.demand_rows) and not any(
        has_monitor and wheelchair_capacity > 0
        for has_monitor, wheelchair_capacity in zip(
            bus_monitor_flags,
            bus_wheelchair_capacities,
            strict=True,
        )
    ):
        raise ValueError(
            "fleet-aware Bird export has wheelchair students but no monitor bus "
            "with wheelchair capacity"
        )
    if any(
        row.service_group == "sped" and row.students > 0 for row in template.demand_rows
    ) and not any(bus_monitor_flags):
        raise ValueError("fleet-aware Bird export has SPED students but no monitor bus")

    earliest_arrival_buffer, latest_arrival_buffer = _resolve_arrival_buffers(config)
    bus_type_names = [_bus_type_name(bus) for bus in selected_buses]
    bus_type_set = set(bus_type_names)
    travel_time_min = template.travel_distance_km / config.speed_km_per_minute

    return replace(
        template,
        max_time_on_bus=config.max_time_on_bus,
        constant_stop_time=config.constant_stop_time,
        stop_time_per_student=config.stop_time_per_student,
        stop_time_per_wheelchair_student=config.stop_time_per_wheelchair_student,
        stop_time_per_sped=config.stop_time_per_sped,
        school_dwell_time=config.school_dwell_time,
        earliest_arrival_buffer=earliest_arrival_buffer,
        latest_arrival_buffer=latest_arrival_buffer,
        speed_km_per_minute=config.speed_km_per_minute,
        lambda_value=config.lambda_value,
        stop_assignment_enabled=config.reassign_stops,
        stop_assignment_lambda=config.stop_assignment_lambda,
        max_walking_distance_km=config.max_walking_distance_km,
        bus_capacity=max(bus.capacity for bus in selected_buses),
        fleet_size=len(selected_buses),
        bus_type="mixed" if len(bus_type_set) > 1 else bus_type_names[0],
        conventional_spillover=config.conventional_spillover,
        allow_partial=config.allow_partial,
        monitor_policy=config.monitor_policy,
        method=config.method,
        bus_names=[bus.name for bus in selected_buses],
        bus_capacities=np.asarray(
            [bus.capacity for bus in selected_buses],
            dtype=np.int64,
        ),
        bus_depot_indices=np.asarray(
            [
                _template_depot_index(list(template.depots), bus.depot)
                for bus in selected_buses
            ],
            dtype=np.int64,
        ),
        bus_has_monitor=np.asarray(
            [1 if has_monitor else 0 for has_monitor in bus_monitor_flags],
            dtype=np.int64,
        ),
        bus_wheelchair_capacities=np.asarray(
            bus_wheelchair_capacities,
            dtype=np.int64,
        ),
        bus_type_names=bus_type_names,
        school_dwell_times=np.asarray(
            [config.school_dwell_time for _ in template.schools],
            dtype=np.float64,
        ),
        school_earliest_arrival_buffers=np.asarray(
            [earliest_arrival_buffer for _ in template.schools],
            dtype=np.float64,
        ),
        school_latest_arrival_buffers=np.asarray(
            [latest_arrival_buffer for _ in template.schools],
            dtype=np.float64,
        ),
        travel_time_min=travel_time_min,
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


def _bird_bus_display_name(instance: BirdExportInstance, bus_id: int) -> str:
    if instance.fleet_aware and 1 <= bus_id <= len(instance.bus_names):
        return instance.bus_names[bus_id - 1]
    return f"bird_bus_{bus_id}"


def _bird_school_matrix_index(instance: BirdExportInstance, school_idx: int) -> int:
    return len(instance.demand_rows) + school_idx - 1


def _bird_depot_matrix_index(instance: BirdExportInstance, depot_idx: int) -> int:
    return len(instance.demand_rows) + len(instance.schools) + depot_idx - 1


def _bird_travel_time_min(
    instance: BirdExportInstance,
    from_matrix_idx: int,
    to_matrix_idx: int,
) -> float:
    value = float(instance.travel_time_min[from_matrix_idx, to_matrix_idx])
    if not math.isfinite(value):
        raise ValueError(
            "Bird instance has no finite travel time between matrix indices "
            f"{from_matrix_idx} and {to_matrix_idx}"
        )
    return value


def _bird_stop_dwell_time_min(
    instance: BirdExportInstance,
    demand_row: BirdDemandRow,
) -> float:
    wheelchair_students = int(demand_row.wheelchair_students)
    special_ed_students = int(demand_row.special_ed_students)
    overlap = min(wheelchair_students, special_ed_students)
    wheelchair_only = wheelchair_students - overlap
    sped_only = special_ed_students - overlap
    wheelchair_rate = float(instance.stop_time_per_wheelchair_student)
    sped_rate = float(instance.stop_time_per_sped)
    return (
        float(instance.constant_stop_time)
        + float(instance.stop_time_per_student) * int(demand_row.students)
        + wheelchair_rate * wheelchair_only
        + sped_rate * sped_only
        + max(wheelchair_rate, sped_rate) * overlap
    )


def _bird_route_service_time_min(
    instance: BirdExportInstance,
    route_rows: list[tuple[int, BirdDemandRow]],
    school_idx: int,
) -> float:
    if not route_rows:
        return 0.0

    service_time = 0.0
    for idx, (global_demand_idx, demand_row) in enumerate(route_rows):
        current_matrix_idx = global_demand_idx - 1
        if idx + 1 == len(route_rows):
            next_matrix_idx = _bird_school_matrix_index(instance, school_idx)
        else:
            next_matrix_idx = route_rows[idx + 1][0] - 1
        service_time += _bird_stop_dwell_time_min(instance, demand_row)
        service_time += _bird_travel_time_min(
            instance,
            current_matrix_idx,
            next_matrix_idx,
        )
    return service_time


def _bird_unassigned_demand_rows(
    instance: BirdExportInstance,
    solution: BirdBackendSolution,
) -> list[BirdDemandRow]:
    demand_rows_by_school = _bird_demand_rows_by_school(instance)
    rows: list[BirdDemandRow] = []
    for school_idx, stop_idx in zip(
        solution.unassigned_school_indices.tolist(),
        solution.unassigned_stop_indices.tolist(),
        strict=True,
    ):
        school_rows = demand_rows_by_school[int(school_idx)]
        rows.append(school_rows[int(stop_idx) - 1][1])
    return rows


def bird_student_assignments(
    instance: BirdExportInstance,
    solution: BirdBackendSolution,
) -> list[dict[str, object]]:
    """Return one assignment record per served student in a BiRD solution."""
    demand_rows_by_school = _bird_demand_rows_by_school(instance)
    assignments: list[dict[str, object]] = []

    for route_idx, bus_id_value in enumerate(solution.assignment_bus_ids.tolist()):
        bus_id = int(bus_id_value)
        school_idx = int(solution.assignment_school_indices[route_idx])
        school = instance.schools[school_idx - 1]
        school_rows = demand_rows_by_school[school_idx]
        start = int(solution.assignment_stop_ptr[route_idx])
        end = int(solution.assignment_stop_ptr[route_idx + 1])
        local_stop_ids = solution.assignment_stop_values[start:end].tolist()

        for route_stop_order, local_stop_id in enumerate(local_stop_ids):
            _global_demand_idx, row = school_rows[int(local_stop_id) - 1]
            for student_name in row.student_names:
                assignments.append(
                    {
                        "student_name": student_name,
                        "bus_id": bus_id,
                        "bus_name": _bird_bus_display_name(instance, bus_id),
                        "route_order": int(solution.assignment_orders[route_idx]),
                        "route_index": route_idx,
                        "route_stop_order": route_stop_order,
                        "school_id": str(school.id),
                        "school_name": school.name,
                        "assigned_stop_id": row.source_stop_id,
                        "assigned_stop_name": row.stop_name,
                        "assigned_stop_node_id": int(row.stop_node_id),
                        "service_group": row.service_group,
                        "grade": row.grade,
                    }
                )

    return assignments


def bird_student_ride_times(
    instance: BirdExportInstance,
    solution: BirdBackendSolution,
    *,
    validate_service_time: bool = True,
    tolerance_min: float = 1.0e-6,
) -> list[dict[str, object]]:
    """Return one timing record per served student in a BiRD solution.

    The solution stores school arrival times and route stop order, but not
    per-student times. This reconstructs latest feasible stop service times
    from the exported travel-time matrix and route-level school arrivals.
    """
    demand_rows_by_school = _bird_demand_rows_by_school(instance)
    rows_by_bus: dict[int, list[int]] = defaultdict(list)
    for row_idx, bus_id in enumerate(solution.assignment_bus_ids.tolist()):
        rows_by_bus[int(bus_id)].append(row_idx)

    records: list[dict[str, object]] = []
    for bus_id, bus_rows in sorted(rows_by_bus.items()):
        bus_name = _bird_bus_display_name(instance, bus_id)
        if instance.fleet_aware and 1 <= bus_id <= len(instance.bus_depot_indices):
            depot_idx = int(instance.bus_depot_indices[bus_id - 1])
        else:
            depot_idx = 1
        current_matrix_idx = _bird_depot_matrix_index(instance, depot_idx)

        for route_idx in sorted(
            bus_rows,
            key=lambda idx: int(solution.assignment_orders[idx]),
        ):
            school_idx = int(solution.assignment_school_indices[route_idx])
            school = instance.schools[school_idx - 1]
            school_matrix_idx = _bird_school_matrix_index(instance, school_idx)
            school_rows = demand_rows_by_school[school_idx]
            start = int(solution.assignment_stop_ptr[route_idx])
            end = int(solution.assignment_stop_ptr[route_idx + 1])
            local_stop_ids = solution.assignment_stop_values[start:end].tolist()
            route_rows = [school_rows[int(stop_id) - 1] for stop_id in local_stop_ids]

            route_service_time = _bird_route_service_time_min(
                instance,
                route_rows,
                school_idx,
            )
            stored_service_time = float(
                solution.assignment_service_time_min[route_idx]
            )
            if (
                validate_service_time
                and abs(route_service_time - stored_service_time) > tolerance_min
            ):
                raise ValueError(
                    "reconstructed Bird route service time does not match solution "
                    f"for route index {route_idx}: reconstructed "
                    f"{route_service_time}, solution {stored_service_time}"
                )

            school_arrival_time = float(solution.assignment_arrival_times[route_idx])
            deadhead_time = (
                _bird_travel_time_min(
                    instance,
                    current_matrix_idx,
                    route_rows[0][0] - 1,
                )
                if route_rows
                else 0.0
            )
            route_start_time = school_arrival_time - deadhead_time - route_service_time
            current_route_time = route_start_time
            previous_matrix_idx = current_matrix_idx

            for route_stop_order, (global_demand_idx, demand_row) in enumerate(
                route_rows
            ):
                demand_matrix_idx = global_demand_idx - 1
                current_route_time += _bird_travel_time_min(
                    instance,
                    previous_matrix_idx,
                    demand_matrix_idx,
                )
                boarding_time = current_route_time
                dwell_time = _bird_stop_dwell_time_min(instance, demand_row)
                departure_time = boarding_time + dwell_time
                ride_time = school_arrival_time - boarding_time
                in_vehicle_time = school_arrival_time - departure_time

                for student_id, student_name in zip(
                    demand_row.student_ids,
                    demand_row.student_names,
                    strict=True,
                ):
                    records.append(
                        {
                            "student_id": student_id,
                            "student_name": student_name,
                            "bus_id": bus_id,
                            "bus_name": bus_name,
                            "route_order": int(
                                solution.assignment_orders[route_idx]
                            ),
                            "route_index": route_idx,
                            "route_stop_order": route_stop_order,
                            "school_id": str(school.id),
                            "school_name": school.name,
                            "school_arrival_time_min": school_arrival_time,
                            "assigned_stop_id": demand_row.source_stop_id,
                            "assigned_stop_name": demand_row.stop_name,
                            "assigned_stop_node_id": int(demand_row.stop_node_id),
                            "service_group": demand_row.service_group,
                            "grade": demand_row.grade,
                            "boarding_time_min": boarding_time,
                            "boarding_departure_time_min": departure_time,
                            "ride_time_min": ride_time,
                            "in_vehicle_time_min": in_vehicle_time,
                            "stop_dwell_time_min": dwell_time,
                            "route_start_time_min": route_start_time,
                            "route_service_time_min": route_service_time,
                        }
                    )

                current_route_time = departure_time
                previous_matrix_idx = demand_matrix_idx

            current_matrix_idx = school_matrix_idx

    return records


def bird_stop_assignments(
    instance: BirdExportInstance,
    solution: BirdBackendSolution,
) -> list[dict[str, object]]:
    """Return one assignment record per routed BiRD demand stop."""
    demand_rows_by_school = _bird_demand_rows_by_school(instance)
    assignments: list[dict[str, object]] = []

    for route_idx, bus_id_value in enumerate(solution.assignment_bus_ids.tolist()):
        bus_id = int(bus_id_value)
        school_idx = int(solution.assignment_school_indices[route_idx])
        school = instance.schools[school_idx - 1]
        school_rows = demand_rows_by_school[school_idx]
        start = int(solution.assignment_stop_ptr[route_idx])
        end = int(solution.assignment_stop_ptr[route_idx + 1])
        local_stop_ids = solution.assignment_stop_values[start:end].tolist()

        for route_stop_order, local_stop_id in enumerate(local_stop_ids):
            global_demand_idx, row = school_rows[int(local_stop_id) - 1]
            assignments.append(
                {
                    "bus_id": bus_id,
                    "bus_name": _bird_bus_display_name(instance, bus_id),
                    "route_order": int(solution.assignment_orders[route_idx]),
                    "route_index": route_idx,
                    "route_stop_order": route_stop_order,
                    "school_id": str(school.id),
                    "school_name": school.name,
                    "assigned_stop_id": row.source_stop_id,
                    "assigned_stop_name": row.stop_name,
                    "assigned_stop_node_id": int(row.stop_node_id),
                    "service_group": row.service_group,
                    "grade": row.grade,
                    "student_names": list(row.student_names),
                    "students": int(row.students),
                    "special_ed_students": int(row.special_ed_students),
                    "wheelchair_students": int(row.wheelchair_students),
                    "demand_row_index": int(global_demand_idx),
                    "local_stop_id": int(local_stop_id),
                }
            )

    return assignments


def summarize_bird_solution_for_mcdp(
    instance: BirdExportInstance,
    solution: BirdBackendSolution,
) -> dict[str, object]:
    """Summarize a BiRD solution into scalar values for routing MCDP catalogues."""
    demand_rows_by_school = _bird_demand_rows_by_school(instance)
    demand_count = len(instance.demand_rows)
    school_count = len(instance.schools)

    rows_by_bus: dict[int, list[int]] = defaultdict(list)
    for row_idx, bus_id in enumerate(solution.assignment_bus_ids.tolist()):
        rows_by_bus[int(bus_id)].append(row_idx)

    by_bus: dict[str, dict[str, object]] = {}
    by_type: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "buses_used": 0,
            "monitor_buses": 0,
            "rounds_used": 0,
            "distance_km": 0.0,
            "runtime_s": 0.0,
            "students_served": 0,
            "sped_students_served": 0,
            "wheelchair_students_served": 0,
        }
    )
    totals = {
        "students_served": 0,
        "sped_students_served": 0,
        "wheelchair_students_served": 0,
        "stops_used": 0,
        "monitor_buses": 0,
    }

    for bus_id, bus_rows in sorted(rows_by_bus.items()):
        bus_index = bus_id - 1
        bus_name = _bird_bus_display_name(instance, bus_id)
        if 0 <= bus_index < len(instance.bus_type_names):
            bus_type = instance.bus_type_names[bus_index]
        else:
            bus_type = instance.bus_type
        if 0 <= bus_index < len(instance.bus_depot_indices):
            depot_index = int(instance.bus_depot_indices[bus_index])
        else:
            depot_index = 1

        current_matrix_idx = demand_count + school_count + depot_index
        distance_km = 0.0
        runtime_min = 0.0
        bus_students = 0
        bus_sped = 0
        bus_wheelchair = 0
        bus_stops = 0

        for row_idx in sorted(
            bus_rows, key=lambda idx: int(solution.assignment_orders[idx])
        ):
            school_idx = int(solution.assignment_school_indices[row_idx])
            school_rows = demand_rows_by_school[school_idx]
            start = int(solution.assignment_stop_ptr[row_idx])
            end = int(solution.assignment_stop_ptr[row_idx + 1])
            local_stop_ids = solution.assignment_stop_values[start:end].tolist()
            route_rows = [school_rows[int(stop_id) - 1] for stop_id in local_stop_ids]

            if route_rows:
                first_demand_idx = int(route_rows[0][0])
                deadhead_time_min = float(
                    instance.travel_time_min[
                        current_matrix_idx - 1,
                        first_demand_idx - 1,
                    ]
                )
            else:
                deadhead_time_min = 0.0

            distance_km += float(solution.assignment_distance_km[row_idx])
            runtime_min += deadhead_time_min + float(
                solution.assignment_service_time_min[row_idx]
            )
            bus_students += sum(row.students for _idx, row in route_rows)
            bus_sped += sum(row.special_ed_students for _idx, row in route_rows)
            bus_wheelchair += sum(row.wheelchair_students for _idx, row in route_rows)
            bus_stops += len(route_rows)
            current_matrix_idx = demand_count + school_idx

        depot_matrix_idx = demand_count + school_count + depot_index
        if bus_rows:
            distance_km += float(
                instance.travel_distance_km[
                    current_matrix_idx - 1,
                    depot_matrix_idx - 1,
                ]
            )
            runtime_min += float(
                instance.travel_time_min[
                    current_matrix_idx - 1,
                    depot_matrix_idx - 1,
                ]
            )

        needs_monitor = bus_sped > 0 or bus_wheelchair > 0
        bus_summary = {
            "bus_name": bus_name,
            "bus_type": bus_type,
            "rounds_used": len(bus_rows),
            "distance_km": distance_km,
            "runtime_s": runtime_min * 60.0,
            "students_served": bus_students,
            "sped_students_served": bus_sped,
            "wheelchair_students_served": bus_wheelchair,
            "stops_used": bus_stops,
            "needs_monitor": needs_monitor,
        }
        by_bus[bus_name] = bus_summary

        type_summary = by_type[bus_type]
        type_summary["buses_used"] = int(type_summary["buses_used"]) + 1
        type_summary["monitor_buses"] = int(type_summary["monitor_buses"]) + int(
            needs_monitor
        )
        type_summary["rounds_used"] = int(type_summary["rounds_used"]) + len(bus_rows)
        type_summary["distance_km"] = float(type_summary["distance_km"]) + distance_km
        type_summary["runtime_s"] = (
            float(type_summary["runtime_s"]) + runtime_min * 60.0
        )
        type_summary["students_served"] = (
            int(type_summary["students_served"]) + bus_students
        )
        type_summary["sped_students_served"] = (
            int(type_summary["sped_students_served"]) + bus_sped
        )
        type_summary["wheelchair_students_served"] = (
            int(type_summary["wheelchair_students_served"]) + bus_wheelchair
        )

        totals["students_served"] += bus_students
        totals["sped_students_served"] += bus_sped
        totals["wheelchair_students_served"] += bus_wheelchair
        totals["stops_used"] += bus_stops
        totals["monitor_buses"] += int(needs_monitor)

    unassigned_rows = _bird_unassigned_demand_rows(instance, solution)
    students_unserved = sum(row.students for row in unassigned_rows)
    sped_students_unserved = sum(row.special_ed_students for row in unassigned_rows)
    wheelchair_students_unserved = sum(
        row.wheelchair_students for row in unassigned_rows
    )

    return {
        **totals,
        "students_unserved": students_unserved,
        "sped_students_unserved": sped_students_unserved,
        "wheelchair_students_unserved": wheelchair_students_unserved,
        "total_distance_km": sum(
            float(summary["distance_km"]) for summary in by_bus.values()
        ),
        "total_runtime_s": sum(
            float(summary["runtime_s"]) for summary in by_bus.values()
        ),
        "by_bus": by_bus,
        "by_type": {key: dict(value) for key, value in by_type.items()},
    }


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
        bus_name = _bird_bus_display_name(instance, int(bus_id))
        route = NormalizedRoute(
            bus_id=bus_name,
            order=int(solution.assignment_orders[row_idx]),
            school_id=str(school.id),
            stop_ids=stop_ids,
            distance_km=float(solution.assignment_distance_km[row_idx]),
            arrival_time_min=float(solution.assignment_arrival_times[row_idx]),
        )
        routes.append(route)
        itineraries_by_bus[int(bus_id)].append(route)
        school_arrivals[str(school.id)].append(
            float(solution.assignment_arrival_times[row_idx])
        )

    itineraries = [
        NormalizedBusItinerary(
            bus_id=_bird_bus_display_name(instance, int(bus_id)),
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
    unassigned_rows = _bird_unassigned_demand_rows(instance, solution)
    unassigned_student_names = sorted(
        student_name for row in unassigned_rows for student_name in row.student_names
    )
    unassigned_rows = _bird_unassigned_demand_rows(instance, solution)
    unassigned_student_names = sorted(
        student_name for row in unassigned_rows for student_name in row.student_names
    )

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
            "earliest_arrival_buffer": instance.earliest_arrival_buffer,
            "latest_arrival_buffer": instance.latest_arrival_buffer,
            "bus_mph": instance.bus_mph,
            "speed_km_per_minute": instance.speed_km_per_minute,
            "fleet_aware": instance.fleet_aware,
            "conventional_spillover": instance.conventional_spillover,
            "allow_partial": instance.allow_partial,
            "monitor_policy": instance.monitor_policy,
            "method": instance.method,
            "bus_names": instance.bus_names,
            "bus_capacities": instance.bus_capacities.tolist(),
            "bus_has_monitor": instance.bus_has_monitor.tolist(),
            "bus_wheelchair_capacities": (instance.bus_wheelchair_capacities.tolist()),
            "stop_assignment_enabled": instance.stop_assignment_enabled,
            "stop_assignment_lambda": instance.stop_assignment_lambda,
            "max_walking_distance_km": instance.max_walking_distance_km,
            "wheelchair_supported": bool(
                np.any(instance.bus_wheelchair_capacities > 0)
            ),
            "unassigned_demand_count": len(unassigned_rows),
            "unassigned_student_count": sum(row.students for row in unassigned_rows),
            "unassigned_students": unassigned_student_names,
            "unassigned_demand_rows": [
                {
                    "external_stop_id": row.external_stop_id,
                    "source_stop_id": row.source_stop_id,
                    "stop_name": row.stop_name,
                    "school_id": row.school_id,
                    "school_name": row.school_name,
                    "service_group": row.service_group,
                    "grade": row.grade,
                    "students": row.students,
                    "special_ed_students": row.special_ed_students,
                    "wheelchair_students": row.wheelchair_students,
                    "student_names": row.student_names,
                    "stop_node_id": row.stop_node_id,
                }
                for row in unassigned_rows
            ],
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
    served_student_ids: set[str] = set()
    for bus_id, bus_rows in sorted(rows_by_bus.items()):
        if instance.fleet_aware and 1 <= int(bus_id) <= len(instance.bus_depot_indices):
            depot_index = int(instance.bus_depot_indices[int(bus_id) - 1])
            current_origin_node = int(instance.depots[depot_index - 1].node_id)
            current_origin_matrix_idx = demand_count + school_count + depot_index
        else:
            current_origin_node = (
                int(instance.depots[0].node_id) if len(instance.depots) == 1 else None
            )
            current_origin_matrix_idx = (
                demand_count + school_count + 1 if len(instance.depots) == 1 else None
            )
        bus_name = _bird_bus_display_name(instance, int(bus_id))
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

            student_ids = sorted(
                student_id
                for _global_demand_idx, demand_row in route_demand_rows
                for student_id in demand_row.student_ids
            )
            served_student_ids.update(student_ids)
            has_sped = any(
                demand_row.special_ed_students > 0
                for _global_demand_idx, demand_row in route_demand_rows
            )

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
                    bus_name=bus_name,
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
                    student_ids=student_ids,
                    has_sped=has_sped,
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
            total_students_served=len(served_student_ids),
        ),
        solution=rows,
    )
