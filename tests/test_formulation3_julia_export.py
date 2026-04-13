from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
from shapely import Point

from formulation.common import (
    Bus,
    Depot,
    ProblemData,
    School,
    SchoolType,
    Stop,
    Student,
)
from formulation.formulation_3.formulation3_gurobipy import build_model_from_definition
from formulation.formulation_3.julia_export import (
    build_formulation3_numeric_instance,
    export_formulation3_instance,
)
from formulation.formulation_3.problem3_definition import (
    MILES_TO_KILOMETERS,
    MPH_TO_KILOMETERS_PER_MINUTE,
    Formulation3,
)


@dataclass
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


def _make_tiny_problem(rounds: int = 2) -> Formulation3:
    depot = Depot(name="Depot A", geographic_location=Point(0, 0), node_id=100)
    stop = Stop(name="Stop A", geographic_location=Point(1, 0), node_id=101)
    school = School(
        name="School A",
        geographic_location=Point(2, 0),
        node_id=102,
        id="school-a",
        type=SchoolType.E,
        start_time=8 * 60,
    )

    student_a = Student(
        name="student-a",
        geographic_location=Point(1, 1),
        school=school,
        stop=stop,
        requires_monitor=False,
        requires_wheelchair=False,
    )
    student_b = Student(
        name="student-b",
        geographic_location=Point(1, -1),
        school=school,
        stop=stop,
        requires_monitor=True,
        requires_wheelchair=True,
    )

    bus = Bus(
        name="bus-a",
        capacity=40,
        range=25,
        has_wheelchair_access=True,
        depot=depot,
    )

    graph = nx.MultiDiGraph()
    graph.add_edge(100, 101, key=0, length=10.0, path=[100, 101])
    graph.add_edge(101, 102, key=0, length=20.0, path=[101, 102])
    graph.add_edge(102, 100, key=0, length=30.0, path=[102, 100])
    graph.add_edge(102, 101, key=0, length=15.0, path=[102, 101])

    problem_data = TinyProblemData(
        name="tiny",
        _service_graph=graph,
        _stops=[stop],
        _schools=[school],
        _depots=[depot],
        _students=[student_a, student_b],
        _buses=[bus],
    )
    return Formulation3(problem_data=problem_data, rounds=rounds)


class Formulation3JuliaExportTests(unittest.TestCase):
    def test_numeric_instance_has_expected_dimensions_and_one_based_indices(self) -> None:
        problem = _make_tiny_problem(rounds=2)

        instance = build_formulation3_numeric_instance(problem)

        self.assertEqual(instance.nB, 1)
        self.assertEqual(instance.nM, 2)
        self.assertEqual(instance.nP, 1)
        self.assertEqual(instance.nS, 1)
        self.assertEqual(instance.nQ, 2)
        self.assertEqual(instance.nTau, 3)
        self.assertTrue(np.all(instance.arc_src >= 1))
        self.assertTrue(np.all(instance.arc_dst >= 1))
        self.assertTrue(np.all(instance.pickup_node_of_m >= 1))
        self.assertTrue(np.all(instance.school_node_of_m >= 1))
        self.assertTrue(np.all(instance.school_index_of_m >= 1))
        self.assertTrue(np.all(instance.tau_of_m >= 1))
        self.assertEqual(instance.student_to_pickup.rows.tolist(), [1, 1])
        self.assertEqual(instance.student_to_pickup.cols.tolist(), [1, 2])
        self.assertEqual(instance.student_to_school.rows.tolist(), [1, 1])
        self.assertEqual(instance.student_to_school.cols.tolist(), [1, 2])
        self.assertEqual(instance.student_to_school_type.rows.tolist(), [1, 1])
        self.assertEqual(instance.student_to_school_type.cols.tolist(), [1, 2])
        self.assertEqual(instance.is_flagged_m.tolist(), [0, 1])
        self.assertEqual(instance.needs_wheelchair_m.tolist(), [0, 1])
        self.assertEqual(instance.bus_names.tolist(), ["bus-a"])
        self.assertEqual(instance.student_names.tolist(), ["student-a", "student-b"])
        np.testing.assert_allclose(instance.range_b, [25 * MILES_TO_KILOMETERS])
        self.assertAlmostEqual(problem.d_ij(problem.D_PLUS[0], problem.P[0]), 10.0)
        self.assertAlmostEqual(
            problem.t_ij(problem.D_PLUS[0], problem.P[0]),
            10.0 / (40 * MPH_TO_KILOMETERS_PER_MINUTE),
        )

    def test_npz_export_contains_expected_keys_and_shapes(self) -> None:
        problem = _make_tiny_problem(rounds=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "formulation3_instance.npz"
            export_formulation3_instance(problem, output_path)
            payload = np.load(output_path, allow_pickle=False)

            self.assertEqual(int(payload["schema_version"]), 1)
            self.assertEqual(int(payload["nB"]), 1)
            self.assertEqual(int(payload["nM"]), 2)
            self.assertEqual(int(payload["nQ"]), 2)
            self.assertEqual(payload["arc_src"].shape, (problem.A.__len__(),))
            self.assertEqual(payload["bus_start_arc_rows"].shape, payload["bus_start_arc_cols"].shape)
            self.assertEqual(payload["node_out_arc_rows"].shape, payload["node_out_arc_cols"].shape)
            self.assertTrue(np.all(payload["bus_start_arc_rows"] >= 1))
            self.assertTrue(np.all(payload["bus_start_arc_cols"] >= 1))
            self.assertEqual(payload["pickup_node_p"].tolist(), [1])

    def test_exported_index_data_matches_current_gurobi_builder(self) -> None:
        problem = _make_tiny_problem(rounds=2)

        instance = build_formulation3_numeric_instance(problem)
        model, variables = build_model_from_definition(problem)

        self.assertEqual(instance.pickup_node_of_m.tolist(), [idx + 1 for idx in variables["meta"]["p_idx"]])
        self.assertEqual(instance.school_node_of_m.tolist(), [idx + 1 for idx in variables["meta"]["s_idx"]])
        self.assertEqual(len(variables["z_b"]), instance.nB)
        self.assertEqual(len(variables["z_bq"]), instance.nB * instance.nQ)
        self.assertEqual(len(variables["x_bqij"]), instance.nB * instance.nQ * instance.nA)
        self.assertEqual(len(variables["a_mbq"]), instance.nM * instance.nB * instance.nQ)
        model.dispose()


if __name__ == "__main__":
    unittest.main()
