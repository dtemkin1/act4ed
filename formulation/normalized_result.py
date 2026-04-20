from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from formulation.common import Place, School, l_s
from formulation.formulation_3.problem3_definition import Formulation3
from formulation.formulation_3.solution import Formulation3Solution


@dataclass(frozen=True, slots=True)
class NormalizedRoute:
    bus_id: str
    order: int
    school_id: str
    stop_ids: list[str]
    distance_km: float
    arrival_time_min: float | None


@dataclass(frozen=True, slots=True)
class NormalizedBusItinerary:
    bus_id: str
    route_orders: list[int]
    school_ids: list[str]
    distance_km: float


@dataclass(frozen=True, slots=True)
class NormalizedRoutingResult:
    backend: str
    status: str
    objective_value: float | None
    runtime_seconds: float
    buses_used: int
    total_distance_km: float
    routes: list[NormalizedRoute]
    itineraries: list[NormalizedBusItinerary]
    school_arrivals: dict[str, list[float]]
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        output_path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "NormalizedRoutingResult":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["routes"] = [NormalizedRoute(**route) for route in payload["routes"]]
        payload["itineraries"] = [
            NormalizedBusItinerary(**itinerary) for itinerary in payload["itineraries"]
        ]
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class RoutingSolutionMetadata:
    backend: str | None
    status: str | None
    objective_value: float | None
    runtime_seconds: float | None
    buses_used: int | None
    total_distance_km: float | None
    total_students_served: int | None


@dataclass(frozen=True, slots=True)
class RoutingSolutionRow:
    bus_name: str
    round: int | None
    students_served: int | None
    distance_km: float | None
    origin_node_id: int | None
    destination_node_id: int | None
    school_name: str | None
    stop_node_ids: list[int] | None
    start_time: float | None
    end_time: float | None
    time_spent: float | None
    student_names: list[str] | None


@dataclass(frozen=True, slots=True)
class RoutingSolutionJson:
    metadata: RoutingSolutionMetadata
    solution: list[RoutingSolutionRow]

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        output_path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "RoutingSolutionJson":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["metadata"] = RoutingSolutionMetadata(**payload["metadata"])
        payload["solution"] = [RoutingSolutionRow(**row) for row in payload["solution"]]
        return cls(**payload)


def _node_index_map(formulation: Formulation3) -> dict[Place, int]:
    return {node: idx for idx, node in enumerate(formulation.N)}


def _time_spent(start_time: float | None, end_time: float | None) -> float | None:
    if start_time is None or end_time is None:
        return None
    return float(end_time - start_time)


def _route_school_from_ordered_paths(
    formulation: Formulation3,
    ordered_paths: list[tuple[Place, Place]],
) -> School | None:
    served_schools = [path[1] for path in ordered_paths if path[1] in formulation.S]
    if not served_schools:
        return None
    school = served_schools[-1]
    if not isinstance(school, School):
        return None
    return school


def _ordered_paths_for_round(
    formulation: Formulation3,
    solution: Formulation3Solution,
    bus_idx: int,
    round_idx: int,
) -> list[tuple[Any, Any]]:
    path_values = solution.variables["x_bqij"]
    selected_paths = [
        path
        for arc_idx, path in enumerate(formulation.A.keys())
        if path_values.get((bus_idx, round_idx, arc_idx), 0.0) > 0.5
    ]
    if not selected_paths:
        return []

    route_by_start = {path[0]: path for path in selected_paths}
    route_destinations = {path[1] for path in selected_paths}
    route_start = next(
        (path[0] for path in selected_paths if path[0] not in route_destinations),
        selected_paths[0][0],
    )
    ordered_paths: list[tuple[Any, Any]] = []
    current_node = route_start
    seen: set[tuple[Any, Any]] = set()
    while current_node in route_by_start:
        next_path = route_by_start[current_node]
        if next_path in seen:
            break
        seen.add(next_path)
        ordered_paths.append(next_path)
        current_node = next_path[1]
    return ordered_paths


def normalized_result_from_formulation3_solution(
    formulation: Formulation3,
    solution: Formulation3Solution,
) -> NormalizedRoutingResult:
    z_bq = solution.variables["z_bq"]
    T_bqi = solution.variables.get("T_bqi", {})
    backend = str(solution.meta.get("backend", "python"))

    routes: list[NormalizedRoute] = []
    itineraries: list[NormalizedBusItinerary] = []
    school_arrivals: dict[str, list[float]] = {
        str(school.id): [] for school in formulation.problem_data.schools
    }

    for bus_idx, bus in enumerate(formulation.B):
        route_orders: list[int] = []
        school_ids: list[str] = []
        bus_distance = 0.0
        for round_idx, _round in enumerate(formulation.Q):
            if z_bq.get((bus_idx, round_idx), 0.0) <= 0.5:
                continue
            ordered_paths = _ordered_paths_for_round(formulation, solution, bus_idx, round_idx)
            if not ordered_paths:
                continue
            bus_distance += sum(formulation.d_ij(*path) for path in ordered_paths)

            served_schools = [
                path[1]
                for path in ordered_paths
                if path[1] in formulation.S
            ]
            school = served_schools[-1] if served_schools else None
            if school is None:
                continue
            stop_ids = [
                str(path[0].name)
                for path in ordered_paths
                if path[0] in formulation.P
            ]
            arrival_time = T_bqi.get((bus_idx, round_idx, formulation.N.index(school)))
            if arrival_time is not None:
                school_arrivals[str(school.id)].append(float(arrival_time))
            else:
                school_arrivals[str(school.id)].append(float(l_s(school)))
            routes.append(
                NormalizedRoute(
                    bus_id=str(bus.name),
                    order=round_idx,
                    school_id=str(school.id),
                    stop_ids=stop_ids,
                    distance_km=float(sum(formulation.d_ij(*path) for path in ordered_paths)),
                    arrival_time_min=(
                        None if arrival_time is None else float(arrival_time)
                    ),
                )
            )
            route_orders.append(round_idx)
            school_ids.append(str(school.id))
        if route_orders:
            itineraries.append(
                NormalizedBusItinerary(
                    bus_id=str(bus.name),
                    route_orders=route_orders,
                    school_ids=school_ids,
                    distance_km=float(bus_distance),
                )
            )

    total_distance = float(sum(route.distance_km for route in routes))
    return NormalizedRoutingResult(
        backend=backend,
        status=solution.status_name,
        objective_value=solution.objective_value,
        runtime_seconds=solution.runtime_seconds,
        buses_used=len(itineraries),
        total_distance_km=total_distance,
        routes=routes,
        itineraries=itineraries,
        school_arrivals={key: values for key, values in school_arrivals.items() if values},
        metadata=dict(solution.meta),
    )


def routing_solution_json_from_formulation3_solution(
    formulation: Formulation3,
    solution: Formulation3Solution,
) -> RoutingSolutionJson:
    z_bq = solution.variables["z_bq"]
    a_mbq = solution.variables.get("a_mbq", {})
    T_bqi = solution.variables.get("T_bqi", {})
    backend = str(solution.meta.get("backend", "python"))
    node_to_idx = _node_index_map(formulation)

    rows: list[RoutingSolutionRow] = []
    served_student_names: set[str] = set()

    for bus_idx, bus in enumerate(formulation.B):
        for round_idx, _round in enumerate(formulation.Q):
            if z_bq.get((bus_idx, round_idx), 0.0) <= 0.5:
                continue

            ordered_paths = _ordered_paths_for_round(formulation, solution, bus_idx, round_idx)
            if not ordered_paths:
                continue

            school = _route_school_from_ordered_paths(formulation, ordered_paths)
            if school is None:
                continue

            student_names = sorted(
                student.name
                for student_idx, student in enumerate(formulation.M)
                if a_mbq.get((student_idx, bus_idx, round_idx), 0.0) > 0.5
            )
            served_student_names.update(student_names)

            origin_node = ordered_paths[0][0]
            start_time = T_bqi.get((bus_idx, round_idx, node_to_idx[origin_node]))
            end_time = T_bqi.get((bus_idx, round_idx, node_to_idx[school]))

            rows.append(
                RoutingSolutionRow(
                    bus_name=str(bus.name),
                    round=round_idx,
                    students_served=len(student_names),
                    distance_km=float(sum(formulation.d_ij(*path) for path in ordered_paths)),
                    origin_node_id=int(origin_node.node_id),
                    destination_node_id=int(school.node_id),
                    school_name=school.name,
                    stop_node_ids=[
                        int(path[0].node_id) for path in ordered_paths if path[0] in formulation.P
                    ],
                    start_time=None if start_time is None else float(start_time),
                    end_time=None if end_time is None else float(end_time),
                    time_spent=_time_spent(
                        None if start_time is None else float(start_time),
                        None if end_time is None else float(end_time),
                    ),
                    student_names=student_names,
                )
            )

    rows = sorted(rows, key=lambda row: (row.bus_name, -1 if row.round is None else row.round))
    return RoutingSolutionJson(
        metadata=RoutingSolutionMetadata(
            backend=backend,
            status=solution.status_name,
            objective_value=solution.objective_value,
            runtime_seconds=solution.runtime_seconds,
            buses_used=len({row.bus_name for row in rows}),
            total_distance_km=float(sum((row.distance_km or 0.0) for row in rows)),
            total_students_served=len(served_student_names),
        ),
        solution=rows,
    )
