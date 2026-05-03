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
    assign_students_to_existing_stops,
    bird_stop_assignments,
    bird_student_assignments,
    build_bird_export_instance,
    normalized_result_from_bird_solution,
    summarize_bird_solution_for_mcdp,
)
from formulation.common import (
    Bus,
    BusType,
    Attributes,
    Depot,
    MPH_TO_KM_PER_MIN,
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
    def base_graph(self) -> nx.MultiDiGraph:
        return self._service_graph

    @property
    def service_graph(self) -> nx.MultiDiGraph:
        return self._service_graph

    @property
    def stops(self) -> tuple[Stop, ...]:
        return tuple(self._stops)

    @property
    def schools(self) -> tuple[School, ...]:
        return tuple(self._schools)

    @property
    def depots(self) -> tuple[Depot, ...]:
        return tuple(self._depots)

    @property
    def students(self) -> tuple[Student, ...]:
        return tuple(self._students)

    @property
    def buses(self) -> tuple[Bus, ...]:
        return tuple(self._buses)


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
            id="conv-a",
            name="conv-a",
            geographic_location=Point(1, 0),
            school=school_a,
            stop=shared_stop,
            attributes=Attributes(special_ed=False, wheelchair_user=False),
        ),
        Student(
            id="conv-b",
            name="conv-b",
            geographic_location=Point(1, 0),
            school=school_b,
            stop=shared_stop,
            attributes=Attributes(special_ed=False, wheelchair_user=False),
        ),
        Student(
            id="sped-a",
            name="sped-a",
            geographic_location=Point(1, 0),
            school=school_a,
            stop=shared_stop,
            attributes=Attributes(special_ed=True, wheelchair_user=False),
        ),
        Student(
            id="wheelchair-a",
            name="wheelchair-a",
            geographic_location=Point(1, 0),
            school=school_a,
            stop=shared_stop,
            attributes=Attributes(special_ed=True, wheelchair_user=True),
        ),
    ]

    buses = [
        Bus(
            id="bus-c-1",
            name="bus-c-1",
            capacity=40,
            range=25,
            wheelchair_capacity=0,
            depot=depot,
            type=BusType.C,
        ),
        Bus(
            id="bus-c-2",
            name="bus-c-2",
            capacity=40,
            range=25,
            wheelchair_capacity=0,
            depot=depot,
            type=BusType.C,
        ),
        Bus(
            id="bus-b-1",
            name="bus-b-1",
            capacity=30,
            range=25,
            wheelchair_capacity=2,
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
            id="student-near-a",
            name="student-near-a",
            geographic_location=Point(0.0000, 0.0000),
            school=school,
            stop=stop_b,
            attributes=Attributes(special_ed=False, wheelchair_user=False),
        ),
        Student(
            id="student-near-b",
            name="student-near-b",
            geographic_location=Point(0.0200, 0.0000),
            school=school,
            stop=stop_b,
            attributes=Attributes(special_ed=False, wheelchair_user=False),
        ),
    ]
    buses = [
        Bus(
            id="bus-c-1",
            name="bus-c-1",
            capacity=40,
            range=25,
            wheelchair_capacity=0,
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


def _make_fleet_aware_problem_data() -> TinyProblemData:
    depot_a = Depot(name="Depot A", geographic_location=Point(0, 0), node_id=100)
    depot_b = Depot(name="Depot B", geographic_location=Point(0, 1), node_id=104)
    shared_stop = Stop(name="Shared Stop", geographic_location=Point(1, 0), node_id=101)
    school = School(
        name="School A",
        geographic_location=Point(2, 0),
        node_id=102,
        id="school-a",
        type=SchoolType.E,
        start_time=8 * 60,
    )
    students = [
        Student(
            id="wheelchair-a",
            name="wheelchair-a",
            geographic_location=Point(1, 0),
            school=school,
            stop=shared_stop,
            attributes=Attributes(special_ed=True, wheelchair_user=True),
        ),
        Student(
            id="sped-a",
            name="sped-a",
            geographic_location=Point(1, 0),
            school=school,
            stop=shared_stop,
            attributes=Attributes(special_ed=True, wheelchair_user=False),
        ),
        Student(
            id="conv-a",
            name="conv-a",
            geographic_location=Point(1, 0),
            school=school,
            stop=shared_stop,
            attributes=Attributes(special_ed=False, wheelchair_user=False),
        ),
    ]
    buses = [
        Bus(
            id="c-1",
            name="C01",
            capacity=1,
            range=25,
            wheelchair_capacity=0,
            depot=depot_a,
            type=BusType.C,
        ),
        Bus(
            id="m-wc",
            name="M01",
            capacity=2,
            range=25,
            wheelchair_capacity=2,
            depot=depot_b,
            type=BusType.BWC,
        ),
        Bus(
            id="m-sped",
            name="M02",
            capacity=2,
            range=25,
            wheelchair_capacity=0,
            depot=depot_a,
            type=BusType.B,
        ),
    ]

    graph = nx.MultiDiGraph()
    for (src, dst), length in {
        (100, 101): 5.0,
        (104, 101): 6.0,
        (101, 102): 4.0,
        (102, 100): 7.0,
        (102, 104): 8.0,
        (102, 101): 4.0,
    }.items():
        graph.add_edge(src, dst, key=0, length=length, path=[src, dst])

    return TinyProblemData(
        name="bird-adapter-fleet-aware",
        _service_graph=graph,
        _stops=[shared_stop],
        _schools=[school],
        _depots=[depot_a, depot_b],
        _students=students,
        _buses=buses,
    )


def _make_grade_split_problem_data(bus_count: int = 2) -> TinyProblemData:
    depot = Depot(name="Depot A", geographic_location=Point(0, 0), node_id=100)
    shared_stop = Stop(name="Shared Stop", geographic_location=Point(1, 0), node_id=101)
    school = School(
        name="School A",
        geographic_location=Point(2, 0),
        node_id=102,
        id="school-a",
        type=SchoolType.E,
        start_time=8 * 60,
    )
    students = [
        Student(
            id="conv-k",
            name="conv-k",
            geographic_location=Point(1, 0),
            school=school,
            stop=shared_stop,
            attributes=Attributes(special_ed=False, wheelchair_user=False),
            grade="K",
        ),
        Student(
            id="conv-1",
            name="conv-1",
            geographic_location=Point(1, 0),
            school=school,
            stop=shared_stop,
            attributes=Attributes(special_ed=False, wheelchair_user=False),
            grade="1",
        ),
    ]
    buses = [
        Bus(
            id=f"c-{idx}",
            name=f"C{idx:02d}",
            capacity=10,
            range=25,
            wheelchair_capacity=0,
            depot=depot,
            type=BusType.C,
        )
        for idx in range(1, bus_count + 1)
    ]
    graph = nx.MultiDiGraph()
    for (src, dst), length in {
        (100, 101): 1.0,
        (101, 102): 1.0,
        (102, 100): 1.0,
        (102, 101): 1.0,
    }.items():
        graph.add_edge(src, dst, key=0, length=length, path=[src, dst])

    return TinyProblemData(
        name="bird-adapter-grade-split",
        _service_graph=graph,
        _stops=[shared_stop],
        _schools=[school],
        _depots=[depot],
        _students=students,
        _buses=buses,
    )


def _make_arrival_window_problem_data() -> TinyProblemData:
    depot = Depot(name="Depot A", geographic_location=Point(0, 0), node_id=100)
    stop_a = Stop(name="Stop A", geographic_location=Point(1, 0), node_id=101)
    stop_b = Stop(name="Stop B", geographic_location=Point(2, 0), node_id=102)
    school_a = School(
        name="School A",
        geographic_location=Point(3, 0),
        node_id=201,
        id="school-a",
        type=SchoolType.E,
        start_time=500,
    )
    school_b = School(
        name="School B",
        geographic_location=Point(4, 0),
        node_id=202,
        id="school-b",
        type=SchoolType.E,
        start_time=520,
    )
    students = [
        Student(
            id="conv-a",
            name="conv-a",
            geographic_location=Point(1, 0),
            school=school_a,
            stop=stop_a,
            attributes=Attributes(special_ed=False, wheelchair_user=False),
            grade="K",
        ),
        Student(
            id="conv-b",
            name="conv-b",
            geographic_location=Point(2, 0),
            school=school_b,
            stop=stop_b,
            attributes=Attributes(special_ed=False, wheelchair_user=False),
            grade="1",
        ),
    ]
    buses = [
        Bus(
            id="bus-c-1",
            name="bus-c-1",
            capacity=40,
            range=25,
            wheelchair_capacity=0,
            depot=depot,
            type=BusType.C,
        )
    ]
    graph = nx.MultiDiGraph()
    for (src, dst), length in {
        (100, 101): 1.0,
        (100, 102): 1.0,
        (101, 201): 10.0,
        (102, 202): 20.0,
        (201, 102): 5.0,
        (202, 101): 100.0,
        (201, 100): 1.0,
        (202, 100): 1.0,
    }.items():
        graph.add_edge(src, dst, key=0, length=length, path=[src, dst])

    return TinyProblemData(
        name="bird-adapter-arrival-window",
        _service_graph=graph,
        _stops=[stop_a, stop_b],
        _schools=[school_a, school_b],
        _depots=[depot],
        _students=students,
        _buses=buses,
    )


def _make_unreachable_stop_problem_data() -> TinyProblemData:
    depot = Depot(name="Depot A", geographic_location=Point(0, 0), node_id=100)
    reachable_stop = Stop(
        name="Reachable Stop", geographic_location=Point(1, 0), node_id=101
    )
    unreachable_stop = Stop(
        name="Unreachable Stop", geographic_location=Point(2, 0), node_id=102
    )
    school = School(
        name="School A",
        geographic_location=Point(3, 0),
        node_id=103,
        id="school-a",
        type=SchoolType.E,
        start_time=8 * 60,
    )
    students = [
        Student(
            id="reachable",
            name="reachable",
            geographic_location=Point(1, 0),
            school=school,
            stop=reachable_stop,
            attributes=Attributes(special_ed=False, wheelchair_user=False),
            grade="K",
        ),
        Student(
            id="unreachable",
            name="unreachable",
            geographic_location=Point(2, 0),
            school=school,
            stop=unreachable_stop,
            attributes=Attributes(special_ed=False, wheelchair_user=False),
            grade="K",
        ),
    ]
    buses = [
        Bus(
            id="bus-c-1",
            name="bus-c-1",
            capacity=40,
            range=25,
            wheelchair_capacity=0,
            depot=depot,
            type=BusType.C,
        )
    ]
    graph = nx.MultiDiGraph()
    for (src, dst), length in {
        (100, 101): 1.0,
        (100, 102): 1.0,
        (101, 103): 1.0,
        (103, 100): 1.0,
    }.items():
        graph.add_edge(src, dst, key=0, length=length, path=[src, dst])

    return TinyProblemData(
        name="bird-adapter-unreachable-stop",
        _service_graph=graph,
        _stops=[reachable_stop, unreachable_stop],
        _schools=[school],
        _depots=[depot],
        _students=students,
        _buses=buses,
    )


class BirdAdapterTests(unittest.TestCase):
    def test_default_speed_is_km_per_minute(self) -> None:
        self.assertAlmostEqual(
            BirdAdapterConfig().speed_km_per_minute,
            40 / MPH_TO_KM_PER_MIN,
        )

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
            [
                (row.source_stop_id, row.school_id, row.students)
                for row in instance.demand_rows
            ],
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

    def test_fleet_aware_export_includes_bus_records_and_group_split_demand(
        self,
    ) -> None:
        problem_data = _make_fleet_aware_problem_data()

        instance = build_bird_export_instance(
            problem_data,
            BirdAdapterConfig(
                cohort="all",
                fleet_aware=True,
                conventional_spillover=True,
                stop_time_per_wheelchair_student=2.5,
            ),
        )

        self.assertTrue(instance.fleet_aware)
        self.assertTrue(instance.conventional_spillover)
        self.assertEqual(instance.stop_time_per_wheelchair_student, 2.5)
        self.assertEqual(instance.bus_names, ["C01", "M01", "M02"])
        np.testing.assert_array_equal(instance.bus_capacities, [1, 2, 2])
        np.testing.assert_array_equal(instance.bus_depot_indices, [1, 2, 1])
        np.testing.assert_array_equal(instance.bus_has_monitor, [0, 1, 1])
        np.testing.assert_array_equal(instance.bus_wheelchair_capacities, [0, 2, 0])
        self.assertEqual(instance.bus_type_names, ["C", "BWC", "B"])
        self.assertEqual(
            [
                (
                    row.external_stop_id,
                    row.service_group,
                    row.students,
                    row.special_ed_students,
                    row.wheelchair_students,
                )
                for row in instance.demand_rows
            ],
            [
                ("wheelchair:school-a:Shared Stop", "wheelchair", 1, 1, 1),
                ("sped:school-a:Shared Stop", "sped", 1, 1, 0),
                ("conventional:school-a:Shared Stop", "conventional", 1, 0, 0),
            ],
        )

    def test_route_assigned_monitor_policy_marks_all_buses_monitor_capable(
        self,
    ) -> None:
        problem_data = _make_fleet_aware_problem_data()

        instance = build_bird_export_instance(
            problem_data,
            BirdAdapterConfig(
                cohort="all",
                fleet_aware=True,
                monitor_policy="route_assigned",
            ),
        )

        self.assertEqual(instance.monitor_policy, "route_assigned")
        np.testing.assert_array_equal(instance.bus_has_monitor, [1, 1, 1])
        self.assertEqual(
            [row.special_ed_students for row in instance.demand_rows],
            [1, 1, 0],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            instance_path = Path(tmpdir) / "bird_instance.npz"
            instance.save(instance_path)
            loaded_instance = BirdExportInstance.load(instance_path)

        self.assertEqual(loaded_instance.monitor_policy, "route_assigned")
        np.testing.assert_array_equal(loaded_instance.bus_has_monitor, [1, 1, 1])
        self.assertEqual(
            [row.special_ed_students for row in loaded_instance.demand_rows],
            [1, 1, 0],
        )

    def test_mcdp_summary_counts_overlapping_categories_and_type_telemetry(
        self,
    ) -> None:
        problem_data = _make_fleet_aware_problem_data()
        instance = build_bird_export_instance(
            problem_data,
            BirdAdapterConfig(
                cohort="all",
                fleet_aware=True,
                monitor_policy="route_assigned",
                speed_km_per_minute=1.0,
            ),
        )
        solution = BirdBackendSolution(
            status="OPTIMAL",
            objective_value=1.0,
            runtime_seconds=2.5,
            buses_used=2,
            total_distance_km=37.0,
            total_service_time_min=12.0,
            assignment_bus_ids=np.asarray([1, 2], dtype=np.int64),
            assignment_orders=np.asarray([0, 0], dtype=np.int64),
            assignment_school_indices=np.asarray([1, 1], dtype=np.int64),
            assignment_arrival_times=np.asarray([450.0, 455.0], dtype=np.float64),
            assignment_distance_km=np.asarray([10.0, 12.0], dtype=np.float64),
            assignment_service_time_min=np.asarray([5.0, 7.0], dtype=np.float64),
            assignment_stop_ptr=np.asarray([0, 2, 3], dtype=np.int64),
            assignment_stop_values=np.asarray([2, 3, 1], dtype=np.int64),
        )

        summary = summarize_bird_solution_for_mcdp(instance, solution)

        self.assertEqual(summary["students_served"], 3)
        self.assertEqual(summary["sped_students_served"], 2)
        self.assertEqual(summary["wheelchair_students_served"], 1)
        self.assertEqual(summary["students_unserved"], 0)
        self.assertEqual(summary["sped_students_unserved"], 0)
        self.assertEqual(summary["wheelchair_students_unserved"], 0)
        self.assertEqual(summary["stops_used"], 3)
        self.assertEqual(summary["monitor_buses"], 2)

        by_type = summary["by_type"]
        self.assertEqual(by_type["C"]["buses_used"], 1)
        self.assertEqual(by_type["C"]["students_served"], 2)
        self.assertEqual(by_type["C"]["sped_students_served"], 1)
        self.assertAlmostEqual(by_type["C"]["distance_km"], 17.0)
        self.assertAlmostEqual(by_type["C"]["runtime_s"], 1020.0)
        self.assertEqual(by_type["BWC"]["buses_used"], 1)
        self.assertEqual(by_type["BWC"]["students_served"], 1)
        self.assertEqual(by_type["BWC"]["sped_students_served"], 1)
        self.assertEqual(by_type["BWC"]["wheelchair_students_served"], 1)
        self.assertAlmostEqual(by_type["BWC"]["distance_km"], 20.0)
        self.assertAlmostEqual(by_type["BWC"]["runtime_s"], 1260.0)

    def test_export_splits_same_stop_school_service_group_by_grade(self) -> None:
        problem_data = _make_grade_split_problem_data()

        instance = build_bird_export_instance(
            problem_data,
            BirdAdapterConfig(cohort="conventional", fleet_aware=True),
        )

        self.assertEqual(
            [
                (
                    row.external_stop_id,
                    row.service_group,
                    row.grade,
                    row.students,
                    row.student_names,
                )
                for row in instance.demand_rows
            ],
            [
                (
                    "conventional:K:school-a:Shared Stop",
                    "conventional",
                    "K",
                    1,
                    ["conv-k"],
                ),
                (
                    "conventional:1:school-a:Shared Stop",
                    "conventional",
                    "1",
                    1,
                    ["conv-1"],
                ),
            ],
        )

    def test_arrival_buffers_resolve_validate_and_round_trip(self) -> None:
        problem_data = _make_problem_data()

        instance = build_bird_export_instance(
            problem_data,
            BirdAdapterConfig(
                cohort="conventional",
                bus_type="C",
                school_dwell_time=10.0,
                earliest_arrival_buffer=60.0,
                stop_time_per_wheelchair_student=3.0,
            ),
        )

        self.assertEqual(instance.stop_time_per_wheelchair_student, 3.0)
        self.assertEqual(instance.latest_arrival_buffer, 10.0)
        self.assertEqual(instance.earliest_arrival_buffer, 60.0)
        np.testing.assert_array_equal(
            instance.school_latest_arrival_buffers, [10.0, 10.0]
        )
        np.testing.assert_array_equal(
            instance.school_earliest_arrival_buffers, [60.0, 60.0]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            instance_path = Path(tmpdir) / "bird_instance.npz"
            instance.save(instance_path)
            loaded_instance = BirdExportInstance.load(instance_path)

        self.assertEqual(loaded_instance.latest_arrival_buffer, 10.0)
        self.assertEqual(loaded_instance.earliest_arrival_buffer, 60.0)
        self.assertEqual(loaded_instance.stop_time_per_wheelchair_student, 3.0)
        np.testing.assert_array_equal(
            loaded_instance.school_latest_arrival_buffers, [10.0, 10.0]
        )
        np.testing.assert_array_equal(
            loaded_instance.school_earliest_arrival_buffers, [60.0, 60.0]
        )

        default_instance = build_bird_export_instance(
            problem_data,
            BirdAdapterConfig(
                cohort="conventional",
                bus_type="C",
                school_dwell_time=7.0,
            ),
        )
        self.assertEqual(default_instance.latest_arrival_buffer, 7.0)
        self.assertEqual(default_instance.earliest_arrival_buffer, 7.0)

        with self.assertRaisesRegex(ValueError, "earliest_arrival_buffer"):
            build_bird_export_instance(
                problem_data,
                BirdAdapterConfig(
                    cohort="conventional",
                    bus_type="C",
                    earliest_arrival_buffer=5.0,
                    latest_arrival_buffer=10.0,
                ),
            )

    def test_julia_loader_uses_wheelchair_stop_time_component(self) -> None:
        julia = shutil.which("julia")
        if julia is None:
            self.skipTest("julia executable not available")

        problem_data = _make_fleet_aware_problem_data()
        instance = build_bird_export_instance(
            problem_data,
            BirdAdapterConfig(
                cohort="all",
                fleet_aware=True,
                constant_stop_time=1.0,
                stop_time_per_student=2.0,
                stop_time_per_wheelchair_student=3.0,
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            instance_path = Path(tmpdir) / "bird_instance.npz"
            instance.save(instance_path)
            result = subprocess.run(
                [
                    julia,
                    "--project=julia",
                    "-e",
                    (
                        'include("julia/src/BirdBackend.jl"); '
                        "using .BirdBackend; "
                        "data = load_instance(ARGS[1]); "
                        "print(BirdBackend.stop_time(data, data.stops[1][1]))"
                    ),
                    str(instance_path),
                ],
                check=True,
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
            )

        self.assertEqual(float(result.stdout.strip()), 6.0)

    @unittest.skipUnless(
        importlib.util.find_spec("gurobipy") is not None, "gurobipy not installed"
    )
    def test_optional_stop_reassignment_uses_student_locations(self) -> None:
        problem_data = _make_reassignment_problem_data()
        assigned_stops = assign_students_to_existing_stops(
            list(problem_data.students),
            list(problem_data.stops),
            lambda_value=1.0e4,
            max_walking_distance_km=None,
        )

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
        self.assertEqual([stop.name for stop in assigned_stops], ["Stop A", "Stop B"])
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

            self.assertEqual(
                [school.id for school in loaded_instance.schools],
                ["school-a", "school-b"],
            )
            self.assertEqual(
                [row.external_stop_id for row in loaded_instance.demand_rows],
                ["school-a:Shared Stop", "school-b:Shared Stop"],
            )
            self.assertEqual(loaded_instance.lambda_value, 4321.0)
            self.assertEqual(loaded_instance.method, "lbh")
            self.assertFalse(loaded_instance.stop_assignment_enabled)
            self.assertEqual(loaded_instance.stop_assignment_lambda, 1.0e4)
            self.assertIsNone(loaded_instance.max_walking_distance_km)
            self.assertEqual(
                [row.student_names for row in loaded_instance.demand_rows],
                [["conv-a"], ["conv-b"]],
            )

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
            student_assignments = bird_student_assignments(loaded_instance, solution)
            stop_assignments = bird_stop_assignments(loaded_instance, solution)

        self.assertEqual(normalized.backend, "bird")
        self.assertEqual(normalized.buses_used, 1)
        self.assertEqual(normalized.total_distance_km, 20.0)
        self.assertEqual(
            [route.school_id for route in normalized.routes], ["school-a", "school-b"]
        )
        self.assertEqual(
            [route.stop_ids for route in normalized.routes],
            [["Shared Stop"], ["Shared Stop"]],
        )
        self.assertEqual(len(normalized.itineraries), 1)
        self.assertEqual(normalized.itineraries[0].route_orders, [0, 1])
        self.assertEqual(
            [
                (
                    row["student_name"],
                    row["bus_name"],
                    row["route_order"],
                    row["school_id"],
                    row["assigned_stop_id"],
                    row["service_group"],
                )
                for row in student_assignments
            ],
            [
                ("conv-a", "bird_bus_1", 0, "school-a", "Shared Stop", "conventional"),
                ("conv-b", "bird_bus_1", 1, "school-b", "Shared Stop", "conventional"),
            ],
        )
        self.assertEqual(
            [
                (
                    row["bus_name"],
                    row["route_order"],
                    row["school_id"],
                    row["assigned_stop_id"],
                    row["student_names"],
                )
                for row in stop_assignments
            ],
            [
                ("bird_bus_1", 0, "school-a", "Shared Stop", ["conv-a"]),
                ("bird_bus_1", 1, "school-b", "Shared Stop", ["conv-b"]),
            ],
        )

    def test_julia_lbh_driver_solves_exported_instance(self) -> None:
        julia = shutil.which("julia")
        if julia is None:
            self.skipTest("julia executable not available")

        problem_data = _make_problem_data()
        instance = build_bird_export_instance(
            problem_data,
            BirdAdapterConfig(cohort="conventional", bus_type="C", method="lbh"),
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

    def test_julia_lbh_driver_respects_fleet_aware_bus_budget(self) -> None:
        julia = shutil.which("julia")
        if julia is None:
            self.skipTest("julia executable not available")

        problem_data = _make_fleet_aware_problem_data()
        instance = build_bird_export_instance(
            problem_data,
            BirdAdapterConfig(cohort="all", fleet_aware=True, method="lbh"),
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
                ],
                check=True,
                cwd=Path(__file__).resolve().parents[1],
            )

            solution = BirdBackendSolution.load(solution_path)
            normalized = normalized_result_from_bird_solution(instance, solution)

        self.assertEqual(solution.status, "OPTIMAL")
        self.assertEqual(solution.buses_used, 3)
        self.assertEqual(solution.assignment_bus_ids.tolist(), [2, 3, 1])
        self.assertEqual(
            {itinerary.bus_id for itinerary in normalized.itineraries},
            {"C01", "M01", "M02"},
        )
        self.assertTrue(normalized.metadata["fleet_aware"])

    def test_julia_scenario_driver_respects_fleet_aware_bus_budget(self) -> None:
        julia = shutil.which("julia")
        if julia is None:
            self.skipTest("julia executable not available")

        problem_data = _make_fleet_aware_problem_data()
        instance = build_bird_export_instance(
            problem_data,
            BirdAdapterConfig(cohort="all", fleet_aware=True, method="scenario"),
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
                ],
                check=True,
                cwd=Path(__file__).resolve().parents[1],
            )

            solution = BirdBackendSolution.load(solution_path)
            normalized = normalized_result_from_bird_solution(instance, solution)

        self.assertEqual(solution.status, "OPTIMAL")
        self.assertEqual(solution.buses_used, 3)
        self.assertEqual(set(solution.assignment_bus_ids.tolist()), {1, 2, 3})
        self.assertEqual(
            {itinerary.bus_id for itinerary in normalized.itineraries},
            {"C01", "M01", "M02"},
        )
        self.assertTrue(normalized.metadata["fleet_aware"])

    def test_julia_lbh_driver_reports_partial_fleet_aware_assignments(self) -> None:
        julia = shutil.which("julia")
        if julia is None:
            self.skipTest("julia executable not available")

        problem_data = _make_fleet_aware_problem_data()
        instance = build_bird_export_instance(
            problem_data,
            BirdAdapterConfig(
                cohort="all",
                bus_type="BWC",
                fleet_aware=True,
                allow_partial=True,
                method="lbh",
            ),
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
                ],
                check=True,
                cwd=Path(__file__).resolve().parents[1],
            )

            solution = BirdBackendSolution.load(solution_path)
            normalized = normalized_result_from_bird_solution(instance, solution)

        self.assertEqual(solution.status, "PARTIAL")
        self.assertEqual(solution.buses_used, 1)
        self.assertEqual(solution.unassigned_school_indices.tolist(), [1, 1])
        self.assertEqual(solution.unassigned_stop_indices.tolist(), [2, 3])
        self.assertEqual(normalized.metadata["unassigned_student_count"], 2)
        self.assertEqual(
            normalized.metadata["unassigned_students"],
            ["conv-a", "sped-a"],
        )
        self.assertEqual(
            [
                row["service_group"]
                for row in normalized.metadata["unassigned_demand_rows"]
            ],
            ["sped", "conventional"],
        )

    def test_julia_scenario_driver_reports_partial_fleet_aware_assignments(
        self,
    ) -> None:
        julia = shutil.which("julia")
        if julia is None:
            self.skipTest("julia executable not available")

        problem_data = _make_fleet_aware_problem_data()
        instance = build_bird_export_instance(
            problem_data,
            BirdAdapterConfig(
                cohort="all",
                bus_type="BWC",
                fleet_aware=True,
                allow_partial=True,
                method="scenario",
            ),
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
                ],
                check=True,
                cwd=Path(__file__).resolve().parents[1],
            )

            solution = BirdBackendSolution.load(solution_path)
            normalized = normalized_result_from_bird_solution(instance, solution)

        self.assertEqual(solution.status, "PARTIAL")
        self.assertEqual(solution.buses_used, 1)
        self.assertEqual(normalized.metadata["unassigned_student_count"], 2)
        self.assertEqual(
            normalized.metadata["unassigned_students"],
            ["conv-a", "sped-a"],
        )
        self.assertEqual(
            [
                row["service_group"]
                for row in normalized.metadata["unassigned_demand_rows"]
            ],
            ["sped", "conventional"],
        )

    def test_julia_scenario_driver_reports_partial_nonfleet_unreachable_stops(
        self,
    ) -> None:
        julia = shutil.which("julia")
        if julia is None:
            self.skipTest("julia executable not available")

        problem_data = _make_unreachable_stop_problem_data()
        instance = build_bird_export_instance(
            problem_data,
            BirdAdapterConfig(
                cohort="conventional",
                bus_type="C",
                allow_partial=True,
                constant_stop_time=0.0,
                stop_time_per_student=0.0,
                speed_km_per_minute=1.0,
                method="scenario",
            ),
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
                    "--seed",
                    "1",
                ],
                check=True,
                cwd=Path(__file__).resolve().parents[1],
            )

            solution = BirdBackendSolution.load(solution_path)
            normalized = normalized_result_from_bird_solution(instance, solution)

        self.assertEqual(solution.status, "PARTIAL")
        self.assertEqual(solution.buses_used, 1)
        self.assertEqual(solution.unassigned_school_indices.tolist(), [1])
        self.assertEqual(solution.unassigned_stop_indices.tolist(), [2])
        self.assertEqual(normalized.metadata["unassigned_students"], ["unreachable"])

    def test_julia_lbh_driver_reports_partial_nonfleet_unreachable_stops(
        self,
    ) -> None:
        julia = shutil.which("julia")
        if julia is None:
            self.skipTest("julia executable not available")

        problem_data = _make_unreachable_stop_problem_data()
        instance = build_bird_export_instance(
            problem_data,
            BirdAdapterConfig(
                cohort="conventional",
                bus_type="C",
                allow_partial=True,
                constant_stop_time=0.0,
                stop_time_per_student=0.0,
                speed_km_per_minute=1.0,
                method="lbh",
            ),
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
                    "--seed",
                    "1",
                ],
                check=True,
                cwd=Path(__file__).resolve().parents[1],
            )

            solution = BirdBackendSolution.load(solution_path)
            normalized = normalized_result_from_bird_solution(instance, solution)

        self.assertEqual(solution.status, "PARTIAL")
        self.assertEqual(solution.buses_used, 1)
        self.assertEqual(solution.unassigned_school_indices.tolist(), [1])
        self.assertEqual(solution.unassigned_stop_indices.tolist(), [2])
        self.assertEqual(normalized.metadata["unassigned_students"], ["unreachable"])

    def test_julia_lbh_driver_does_not_mix_grades_within_one_school_route(self) -> None:
        julia = shutil.which("julia")
        if julia is None:
            self.skipTest("julia executable not available")

        problem_data = _make_grade_split_problem_data(bus_count=1)
        instance = build_bird_export_instance(
            problem_data,
            BirdAdapterConfig(
                cohort="conventional",
                fleet_aware=True,
                allow_partial=True,
                method="lbh",
            ),
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
                    "--seed",
                    "1",
                ],
                check=True,
                cwd=Path(__file__).resolve().parents[1],
            )

            solution = BirdBackendSolution.load(solution_path)
            stop_assignments = bird_stop_assignments(instance, solution)
            normalized = normalized_result_from_bird_solution(instance, solution)

        self.assertEqual(solution.status, "PARTIAL")
        self.assertEqual(solution.buses_used, 1)
        self.assertEqual(len(stop_assignments), 1)
        self.assertEqual(normalized.metadata["unassigned_student_count"], 1)
        served_grades = {row["grade"] for row in stop_assignments}
        unassigned_grades = {
            row["grade"] for row in normalized.metadata["unassigned_demand_rows"]
        }
        self.assertEqual(len(served_grades), 1)
        self.assertEqual(len(unassigned_grades), 1)
        self.assertTrue(served_grades.isdisjoint(unassigned_grades))

    def test_julia_scenario_driver_does_not_mix_grades_within_one_school_route(
        self,
    ) -> None:
        julia = shutil.which("julia")
        if julia is None:
            self.skipTest("julia executable not available")

        problem_data = _make_grade_split_problem_data(bus_count=1)
        instance = build_bird_export_instance(
            problem_data,
            BirdAdapterConfig(
                cohort="conventional",
                fleet_aware=True,
                allow_partial=True,
                method="scenario",
            ),
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
                    "--seed",
                    "1",
                ],
                check=True,
                cwd=Path(__file__).resolve().parents[1],
            )

            solution = BirdBackendSolution.load(solution_path)
            stop_assignments = bird_stop_assignments(instance, solution)
            normalized = normalized_result_from_bird_solution(instance, solution)

        self.assertEqual(solution.status, "PARTIAL")
        self.assertEqual(solution.buses_used, 1)
        self.assertEqual(len(stop_assignments), 1)
        self.assertEqual(normalized.metadata["unassigned_student_count"], 1)
        served_grades = {row["grade"] for row in stop_assignments}
        unassigned_grades = {
            row["grade"] for row in normalized.metadata["unassigned_demand_rows"]
        }
        self.assertEqual(len(served_grades), 1)
        self.assertEqual(len(unassigned_grades), 1)
        self.assertTrue(served_grades.isdisjoint(unassigned_grades))

    def test_julia_lbh_driver_uses_early_arrival_window_for_bus_reuse(self) -> None:
        julia = shutil.which("julia")
        if julia is None:
            self.skipTest("julia executable not available")

        problem_data = _make_arrival_window_problem_data()
        base_config = dict(
            cohort="conventional",
            fleet_aware=True,
            allow_partial=True,
            school_dwell_time=10.0,
            constant_stop_time=0.0,
            stop_time_per_student=0.0,
            speed_km_per_minute=1.0,
            method="lbh",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            fixed_instance = build_bird_export_instance(
                problem_data,
                BirdAdapterConfig(**base_config),
            )
            fixed_instance_path = tmp_path / "bird_fixed_instance.npz"
            fixed_solution_path = tmp_path / "bird_fixed_solution.npz"
            fixed_instance.save(fixed_instance_path)
            subprocess.run(
                [
                    julia,
                    "--project=julia",
                    "experiments/solve_bird_backend_julia.jl",
                    "--instance",
                    str(fixed_instance_path),
                    "--solution",
                    str(fixed_solution_path),
                    "--seed",
                    "1",
                ],
                check=True,
                cwd=Path(__file__).resolve().parents[1],
            )
            fixed_solution = BirdBackendSolution.load(fixed_solution_path)

            wide_instance = build_bird_export_instance(
                problem_data,
                BirdAdapterConfig(
                    **base_config,
                    earliest_arrival_buffer=60.0,
                ),
            )
            wide_instance_path = tmp_path / "bird_wide_instance.npz"
            wide_solution_path = tmp_path / "bird_wide_solution.npz"
            wide_instance.save(wide_instance_path)
            subprocess.run(
                [
                    julia,
                    "--project=julia",
                    "experiments/solve_bird_backend_julia.jl",
                    "--instance",
                    str(wide_instance_path),
                    "--solution",
                    str(wide_solution_path),
                    "--seed",
                    "1",
                ],
                check=True,
                cwd=Path(__file__).resolve().parents[1],
            )
            wide_solution = BirdBackendSolution.load(wide_solution_path)
            wide_assignments = bird_stop_assignments(wide_instance, wide_solution)

        self.assertEqual(fixed_solution.status, "PARTIAL")
        self.assertEqual(len(fixed_solution.assignment_school_indices), 1)
        self.assertEqual(len(fixed_solution.unassigned_school_indices), 1)

        self.assertEqual(wide_solution.status, "OPTIMAL")
        self.assertEqual(wide_solution.buses_used, 1)
        self.assertEqual(wide_solution.assignment_school_indices.tolist(), [1, 2])
        self.assertEqual([row["grade"] for row in wide_assignments], ["K", "1"])
        self.assertEqual(len(wide_solution.assignment_arrival_times), 2)
        self.assertLess(
            wide_solution.assignment_arrival_times[0],
            wide_solution.assignment_arrival_times[1],
        )
        self.assertGreaterEqual(
            wide_solution.assignment_arrival_times[1],
            wide_solution.assignment_arrival_times[0] + 10.0 + 5.0 + 20.0 - 1e-6,
        )

    def test_julia_scenario_driver_allows_grade_change_after_school(self) -> None:
        julia = shutil.which("julia")
        if julia is None:
            self.skipTest("julia executable not available")

        problem_data = _make_arrival_window_problem_data()
        instance = build_bird_export_instance(
            problem_data,
            BirdAdapterConfig(
                cohort="conventional",
                fleet_aware=True,
                allow_partial=True,
                school_dwell_time=10.0,
                earliest_arrival_buffer=60.0,
                constant_stop_time=0.0,
                stop_time_per_student=0.0,
                speed_km_per_minute=1.0,
                method="scenario",
            ),
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
                    "--seed",
                    "1",
                ],
                check=True,
                cwd=Path(__file__).resolve().parents[1],
            )

            solution = BirdBackendSolution.load(solution_path)
            stop_assignments = bird_stop_assignments(instance, solution)

        self.assertEqual(solution.status, "OPTIMAL")
        self.assertEqual(solution.buses_used, 1)
        self.assertEqual(solution.assignment_school_indices.tolist(), [1, 2])
        self.assertEqual([row["grade"] for row in stop_assignments], ["K", "1"])


if __name__ == "__main__":
    unittest.main()
