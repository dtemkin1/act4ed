from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
from shapely import Point

from formulation.bird_adapter import (
    BirdAdapterConfig,
    BirdBackendSolution,
    build_bird_export_instance,
    routing_solution_json_from_bird_solution,
)
from formulation.common import (
    Bus,
    BusType,
    Depot,
    ProblemData,
    School,
    SchoolType,
    Stop,
    Student,
)
from formulation.formulation_3.problem3_definition import Formulation3
from formulation.formulation_3.solution import Formulation3Solution
from formulation.normalized_result import (
    RoutingSolutionJson,
    routing_solution_json_from_formulation3_solution,
)


@dataclass(frozen=True)
class TinyProblemData(ProblemData):
    _service_graph: nx.MultiDiGraph
    _stops: list[Stop]
    _schools: list[School]
    _depots: list[Depot]
    _students: list[Student]
    _buses: list[Bus]

    @property
    def service_graph(self) -> nx.MultiDiGraph:
        return self._service_graph

    @property
    def stops(self) -> list[Stop]:
        return self._stops

    @property
    def schools(self) -> list[School]:
        return self._schools

    @property
    def depots(self) -> list[Depot]:
        return self._depots

    @property
    def students(self) -> list[Student]:
        return self._students

    @property
    def buses(self) -> list[Bus]:
        return self._buses


def _build_bird_problem() -> TinyProblemData:
    depot = Depot(name="Depot A", geographic_location=Point(0, 0), node_id=100)
    stop_a = Stop(name="Stop A", geographic_location=Point(1, 0), node_id=101)
    stop_b = Stop(name="Stop B", geographic_location=Point(2, 0), node_id=102)
    school_a = School(
        name="School A",
        geographic_location=Point(3, 0),
        node_id=201,
        id="school-a",
        type=SchoolType.E,
        start_time=8 * 60,
    )
    school_b = School(
        name="School B",
        geographic_location=Point(4, 0),
        node_id=202,
        id="school-b",
        type=SchoolType.MS,
        start_time=9 * 60,
    )
    students = [
        Student(
            name="bird-student-a",
            geographic_location=Point(1, 0),
            school=school_a,
            stop=stop_a,
            requires_monitor=False,
            requires_wheelchair=False,
        ),
        Student(
            name="bird-student-b",
            geographic_location=Point(2, 0),
            school=school_b,
            stop=stop_b,
            requires_monitor=False,
            requires_wheelchair=False,
        ),
    ]
    buses = [
        Bus(
            name="bird-bus-1",
            capacity=40,
            range=25,
            has_wheelchair_access=False,
            depot=depot,
            type=BusType.C,
        )
    ]
    graph = nx.MultiDiGraph()
    distances = {
        (100, 101): 5.0,
        (100, 102): 8.0,
        (101, 201): 4.0,
        (102, 202): 6.0,
        (201, 102): 3.0,
        (201, 202): 2.0,
        (101, 202): 7.0,
        (102, 201): 3.0,
        (201, 100): 5.0,
        (202, 100): 8.0,
    }
    for (src, dst), length in distances.items():
        graph.add_edge(src, dst, key=0, length=length, path=[src, dst])
    return TinyProblemData(
        name="bird-json-tiny",
        _service_graph=graph,
        _stops=[stop_a, stop_b],
        _schools=[school_a, school_b],
        _depots=[depot],
        _students=students,
        _buses=buses,
    )


def _build_formulation_problem() -> TinyProblemData:
    depot = Depot(name="Depot A", geographic_location=Point(0, 0), node_id=100)
    stop = Stop(name="Stop A", geographic_location=Point(1, 0), node_id=101)
    school = School(
        name="School A",
        geographic_location=Point(2, 0),
        node_id=201,
        id="school-a",
        type=SchoolType.E,
        start_time=8 * 60,
    )
    student = Student(
        name="formulation-student-a",
        geographic_location=Point(1, 0),
        school=school,
        stop=stop,
        requires_monitor=False,
        requires_wheelchair=False,
    )
    bus = Bus(
        name="formulation-bus-1",
        capacity=40,
        range=25,
        has_wheelchair_access=False,
        depot=depot,
        type=BusType.C,
    )
    graph = nx.MultiDiGraph()
    for (src, dst), length in {
        (100, 101): 5.0,
        (101, 201): 4.0,
    }.items():
        graph.add_edge(src, dst, key=0, length=length, path=[src, dst])
    return TinyProblemData(
        name="formulation-json-tiny",
        _service_graph=graph,
        _stops=[stop],
        _schools=[school],
        _depots=[depot],
        _students=[student],
        _buses=[bus],
    )


class RoutingSolutionJsonTests(unittest.TestCase):
    def test_bird_solution_json_uses_school_local_stop_ids(self) -> None:
        instance = build_bird_export_instance(
            _build_bird_problem(),
            BirdAdapterConfig(cohort="conventional", bus_type="C", speed_km_per_minute=1.0),
        )
        solution = BirdBackendSolution(
            status="OPTIMAL",
            objective_value=2.0,
            runtime_seconds=1.5,
            buses_used=1,
            total_distance_km=28.0,
            total_service_time_min=10.0,
            assignment_bus_ids=np.asarray([1, 1], dtype=np.int64),
            assignment_orders=np.asarray([0, 1], dtype=np.int64),
            assignment_school_indices=np.asarray([1, 2], dtype=np.int64),
            assignment_arrival_times=np.asarray([50.0, 100.0], dtype=np.float64),
            assignment_distance_km=np.asarray([9.0, 9.0], dtype=np.float64),
            assignment_service_time_min=np.asarray([4.0, 6.0], dtype=np.float64),
            assignment_stop_ptr=np.asarray([0, 1, 2], dtype=np.int64),
            assignment_stop_values=np.asarray([1, 1], dtype=np.int64),
        )

        report = routing_solution_json_from_bird_solution(instance, solution)

        self.assertEqual(report.metadata.backend, "bird")
        self.assertEqual(report.metadata.total_students_served, 2)
        self.assertEqual(len(report.solution), 2)
        self.assertEqual(report.solution[0].bus_name, "bird_bus_1")
        self.assertEqual(report.solution[0].origin_node_id, 100)
        self.assertEqual(report.solution[0].destination_node_id, 201)
        self.assertEqual(report.solution[0].school_name, "School A")
        self.assertEqual(report.solution[0].stop_node_ids, [101])
        self.assertEqual(report.solution[0].student_names, ["bird-student-a"])
        self.assertEqual(report.solution[0].start_time, 41.0)
        self.assertEqual(report.solution[0].end_time, 50.0)
        self.assertEqual(report.solution[0].time_spent, 9.0)
        self.assertEqual(report.solution[1].origin_node_id, 201)
        self.assertEqual(report.solution[1].destination_node_id, 202)
        self.assertEqual(report.solution[1].stop_node_ids, [102])
        self.assertEqual(report.solution[1].student_names, ["bird-student-b"])
        self.assertEqual(report.solution[1].start_time, 91.0)
        self.assertEqual(report.solution[1].time_spent, 9.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bird_solution.json"
            report.save(path)
            loaded = RoutingSolutionJson.load(path)
        self.assertEqual(loaded.solution[1].school_name, "School B")

    def test_formulation3_solution_json_contains_route_details(self) -> None:
        formulation = Formulation3(problem_data=_build_formulation_problem(), rounds=1)
        depot_start = formulation.D_PLUS[0]
        stop = formulation.P[0]
        school = formulation.S[0]
        arc_list = list(formulation.A.keys())
        arc_depot_stop = arc_list.index((depot_start, stop))
        arc_stop_school = arc_list.index((stop, school))
        node_to_idx = {node: idx for idx, node in enumerate(formulation.N)}

        solution = Formulation3Solution(
            status=2,
            status_name="OPTIMAL",
            objective_value=9.0,
            runtime_seconds=0.1,
            variables={
                "z_bq": {(0, 0): 1.0},
                "x_bqij": {
                    (0, 0, arc_depot_stop): 1.0,
                    (0, 0, arc_stop_school): 1.0,
                },
                "a_mbq": {(0, 0, 0): 1.0},
                "T_bqi": {
                    (0, 0, node_to_idx[depot_start]): 10.0,
                    (0, 0, node_to_idx[school]): 19.0,
                },
            },
            meta={"backend": "python"},
        )

        report = routing_solution_json_from_formulation3_solution(formulation, solution)

        self.assertEqual(report.metadata.backend, "python")
        self.assertEqual(report.metadata.buses_used, 1)
        self.assertEqual(report.metadata.total_distance_km, 9.0)
        self.assertEqual(report.metadata.total_students_served, 1)
        self.assertEqual(len(report.solution), 1)
        self.assertEqual(report.solution[0].bus_name, "formulation-bus-1")
        self.assertEqual(report.solution[0].round, 0)
        self.assertEqual(report.solution[0].students_served, 1)
        self.assertEqual(report.solution[0].distance_km, 9.0)
        self.assertEqual(report.solution[0].origin_node_id, 100)
        self.assertEqual(report.solution[0].destination_node_id, 201)
        self.assertEqual(report.solution[0].school_name, "School A")
        self.assertEqual(report.solution[0].stop_node_ids, [101])
        self.assertEqual(report.solution[0].start_time, 10.0)
        self.assertEqual(report.solution[0].end_time, 19.0)
        self.assertEqual(report.solution[0].time_spent, 9.0)
        self.assertEqual(report.solution[0].student_names, ["formulation-student-a"])


if __name__ == "__main__":
    unittest.main()
