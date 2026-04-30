from __future__ import annotations

import importlib.util
import shutil
import subprocess
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
    BirdExportInstance,
    build_bird_export_instance,
    normalized_result_from_bird_solution,
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


def _make_problem_data() -> TinyProblemData:
    depot = Depot(name="Depot A", geographic_location=Point(0, 0), node_id=100)
    shared_stop = Stop(name="Shared Stop", geographic_location=Point(1, 0), node_id=101)
    school_a = School(
        name="School A",
        geographic_location=Point(2, 0),
        node_id=102,
        id="school-a",
        type=SchoolType.E,
        start_time=8 * 60,
    )
    school_b = School(
        name="School B",
        geographic_location=Point(2, 1),
        node_id=103,
        id="school-b",
        type=SchoolType.MS,
        start_time=9 * 60,
    )

    students = [
        Student(
            name="conv-a",
            geographic_location=Point(1, 0),
            school=school_a,
            stop=shared_stop,
            requires_monitor=False,
            requires_wheelchair=False,
        ),
        Student(
            name="conv-b",
            geographic_location=Point(1, 0),
            school=school_b,
            stop=shared_stop,
            requires_monitor=False,
            requires_wheelchair=False,
        ),
        Student(
            name="sped-a",
            geographic_location=Point(1, 0),
            school=school_a,
            stop=shared_stop,
            requires_monitor=True,
            requires_wheelchair=False,
        ),
        Student(
            name="wheelchair-a",
            geographic_location=Point(1, 0),
            school=school_a,
            stop=shared_stop,
            requires_monitor=True,
            requires_wheelchair=True,
        ),
    ]

    buses = [
        Bus(
            name="bus-c-1",
            capacity=40,
            range=25,
            has_wheelchair_access=False,
            depot=depot,
            type=BusType.C,
        ),
        Bus(
            name="bus-c-2",
            capacity=40,
            range=25,
            has_wheelchair_access=False,
            depot=depot,
            type=BusType.C,
        ),
        Bus(
            name="bus-b-1",
            capacity=30,
            range=25,
            has_wheelchair_access=True,
            depot=depot,
            type=BusType.B,
        ),
    ]

    graph = nx.MultiDiGraph()
    distances = {
        (100, 101): 5.0,
        (100, 102): 7.0,
        (100, 103): 9.0,
        (101, 102): 4.0,
        (101, 103): 6.0,
        (102, 101): 4.0,
        (103, 101): 6.0,
        (102, 103): 3.0,
        (103, 102): 3.0,
        (102, 100): 7.0,
        (103, 100): 9.0,
    }
    for (src, dst), length in distances.items():
        graph.add_edge(src, dst, key=0, length=length, path=[src, dst])

    return TinyProblemData(
        name="bird-adapter-tiny",
        _service_graph=graph,
        _stops=[shared_stop],
        _schools=[school_a, school_b],
        _depots=[depot],
        _students=students,
        _buses=buses,
    )


def _make_reassignment_problem_data() -> TinyProblemData:
    depot = Depot(name="Depot A", geographic_location=Point(0.0, 0.0), node_id=100)
    stop_a = Stop(name="Stop A", geographic_location=Point(0.0000, 0.0000), node_id=101)
    stop_b = Stop(name="Stop B", geographic_location=Point(0.0200, 0.0000), node_id=102)
    school = School(
        name="School A",
        geographic_location=Point(0.0400, 0.0000),
        node_id=103,
        id="school-a",
        type=SchoolType.E,
        start_time=8 * 60,
    )
    students = [
        Student(
            name="student-near-a",
            geographic_location=Point(0.0000, 0.0000),
            school=school,
            stop=stop_b,
            requires_monitor=False,
            requires_wheelchair=False,
        ),
        Student(
            name="student-near-b",
            geographic_location=Point(0.0200, 0.0000),
            school=school,
            stop=stop_b,
            requires_monitor=False,
            requires_wheelchair=False,
        ),
    ]
    buses = [
        Bus(
            name="bus-c-1",
            capacity=40,
            range=25,
            has_wheelchair_access=False,
            depot=depot,
            type=BusType.C,
        )
    ]
    graph = nx.MultiDiGraph()
    for (src, dst), length in {
        (100, 101): 5.0,
        (100, 102): 6.0,
        (101, 103): 4.0,
        (102, 103): 3.5,
        (101, 102): 1.0,
        (102, 101): 1.0,
        (103, 101): 4.0,
        (103, 102): 3.5,
    }.items():
        graph.add_edge(src, dst, key=0, length=length, path=[src, dst])

    return TinyProblemData(
        name="bird-adapter-reassignment",
        _service_graph=graph,
        _stops=[stop_a, stop_b],
        _schools=[school],
        _depots=[depot],
        _students=students,
        _buses=buses,
    )


class BirdAdapterTests(unittest.TestCase):
    def test_conventional_export_duplicates_shared_stop_by_school(self) -> None:
        problem_data = _make_problem_data()

        instance = build_bird_export_instance(
            problem_data,
            BirdAdapterConfig(cohort="conventional", bus_type="C"),
        )

        self.assertEqual(instance.bus_type, "C")
        self.assertEqual(instance.fleet_size, 2)
        self.assertEqual(instance.bus_capacity, 40)
        self.assertEqual(len(instance.demand_rows), 2)
        self.assertEqual(
            [(row.source_stop_id, row.school_id, row.students) for row in instance.demand_rows],
            [
                ("Shared Stop", "school-a", 1),
                ("Shared Stop", "school-b", 1),
            ],
        )
        self.assertEqual(
            [row.student_names for row in instance.demand_rows],
            [["conv-a"], ["conv-b"]],
        )
        np.testing.assert_array_equal(instance.demand_school_indices, [1, 2])
        self.assertEqual(instance.travel_distance_km.shape, (5, 5))
        self.assertTrue(np.all(np.isfinite(np.diag(instance.travel_distance_km))))

    def test_sped_export_excludes_wheelchair_students(self) -> None:
        problem_data = _make_problem_data()

        instance = build_bird_export_instance(
            problem_data,
            BirdAdapterConfig(cohort="sped_no_wheelchair", bus_type="B"),
        )

        self.assertEqual(len(instance.demand_rows), 1)
        self.assertEqual(instance.demand_rows[0].school_id, "school-a")
        self.assertEqual(instance.demand_rows[0].students, 1)
        self.assertEqual(instance.demand_rows[0].student_names, ["sped-a"])
        self.assertEqual(instance.bus_capacity, 30)
        self.assertEqual(instance.fleet_size, 1)

    @unittest.skipUnless(importlib.util.find_spec("gurobipy") is not None, "gurobipy not installed")
    def test_optional_stop_reassignment_uses_student_locations(self) -> None:
        problem_data = _make_reassignment_problem_data()

        instance = build_bird_export_instance(
            problem_data,
            BirdAdapterConfig(
                cohort="conventional",
                bus_type="C",
                reassign_stops=True,
                stop_assignment_lambda=1.0e4,
            ),
        )

        self.assertTrue(instance.stop_assignment_enabled)
        self.assertEqual(instance.stop_assignment_lambda, 1.0e4)
        self.assertEqual(len(instance.demand_rows), 2)
        self.assertEqual(
            [(row.source_stop_id, row.student_names) for row in instance.demand_rows],
            [
                ("Stop A", ["student-near-a"]),
                ("Stop B", ["student-near-b"]),
            ],
        )

    def test_mixed_fleet_requires_explicit_bus_type(self) -> None:
        problem_data = _make_problem_data()

        with self.assertRaisesRegex(ValueError, "bus_type must be provided"):
            build_bird_export_instance(
                problem_data,
                BirdAdapterConfig(cohort="conventional"),
            )

    def test_instance_round_trip_and_solution_normalization(self) -> None:
        problem_data = _make_problem_data()
        instance = build_bird_export_instance(
            problem_data,
            BirdAdapterConfig(cohort="conventional", bus_type="C", lambda_value=4321.0),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            instance_path = Path(tmpdir) / "bird_instance.npz"
            instance.save(instance_path)
            loaded_instance = BirdExportInstance.load(instance_path)

            self.assertEqual([school.id for school in loaded_instance.schools], ["school-a", "school-b"])
            self.assertEqual([row.external_stop_id for row in loaded_instance.demand_rows], ["school-a:Shared Stop", "school-b:Shared Stop"])
            self.assertEqual(loaded_instance.lambda_value, 4321.0)
            self.assertFalse(loaded_instance.stop_assignment_enabled)
            self.assertEqual(loaded_instance.stop_assignment_lambda, 1.0e4)
            self.assertIsNone(loaded_instance.max_walking_distance_km)
            self.assertEqual([row.student_names for row in loaded_instance.demand_rows], [["conv-a"], ["conv-b"]])

            solution = BirdBackendSolution(
                status="OPTIMAL",
                objective_value=1.0,
                runtime_seconds=2.5,
                buses_used=1,
                total_distance_km=20.0,
                total_service_time_min=18.0,
                assignment_bus_ids=np.asarray([1, 1], dtype=np.int64),
                assignment_orders=np.asarray([0, 1], dtype=np.int64),
                assignment_school_indices=np.asarray([1, 2], dtype=np.int64),
                assignment_arrival_times=np.asarray([450.0, 510.0], dtype=np.float64),
                assignment_distance_km=np.asarray([9.0, 11.0], dtype=np.float64),
                assignment_service_time_min=np.asarray([8.0, 10.0], dtype=np.float64),
                assignment_stop_ptr=np.asarray([0, 1, 2], dtype=np.int64),
                assignment_stop_values=np.asarray([1, 1], dtype=np.int64),
            )

            normalized = normalized_result_from_bird_solution(loaded_instance, solution)

        self.assertEqual(normalized.backend, "bird")
        self.assertEqual(normalized.buses_used, 1)
        self.assertEqual(normalized.total_distance_km, 20.0)
        self.assertEqual([route.school_id for route in normalized.routes], ["school-a", "school-b"])
        self.assertEqual([route.stop_ids for route in normalized.routes], [["Shared Stop"], ["Shared Stop"]])
        self.assertEqual(len(normalized.itineraries), 1)
        self.assertEqual(normalized.itineraries[0].route_orders, [0, 1])

    def test_julia_lbh_driver_solves_exported_instance(self) -> None:
        julia = shutil.which("julia")
        if julia is None:
            self.skipTest("julia executable not available")

        problem_data = _make_problem_data()
        instance = build_bird_export_instance(
            problem_data,
            BirdAdapterConfig(cohort="conventional", bus_type="C"),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            instance_path = Path(tmpdir) / "bird_instance.npz"
            solution_path = Path(tmpdir) / "bird_solution.npz"
            instance.save(instance_path)

            subprocess.run(
                [
                    julia,
                    "--project=julia",
                    "experiments/solve_bird_backend_julia.jl",
                    "--instance",
                    str(instance_path),
                    "--solution",
                    str(solution_path),
                    "--method",
                    "lbh",
                ],
                check=True,
                cwd=Path(__file__).resolve().parents[1],
            )

            solution = BirdBackendSolution.load(solution_path)
            normalized = normalized_result_from_bird_solution(instance, solution)

        self.assertEqual(solution.status, "OPTIMAL")
        self.assertGreaterEqual(solution.buses_used, 1)
        self.assertEqual(normalized.backend, "bird")
        self.assertGreaterEqual(len(normalized.routes), 1)


if __name__ == "__main__":
    unittest.main()
