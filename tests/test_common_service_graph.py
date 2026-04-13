from __future__ import annotations

import unittest

import pandas as pd
from shapely import Point

from formulation.common import (
    Depot,
    ProblemDataReal,
    School,
    SchoolType,
    Stop,
    Student,
)


def _make_stop(name: str, node_id: int) -> Stop:
    return Stop(name=name, geographic_location=Point(node_id, 0), node_id=node_id)


def _make_school(name: str, node_id: int, school_type: SchoolType) -> School:
    return School(
        name=name,
        geographic_location=Point(node_id, 1),
        node_id=node_id,
        id=name.lower(),
        type=school_type,
        start_time=8 * 60,
    )


def _make_depot(name: str, node_id: int) -> Depot:
    return Depot(name=name, geographic_location=Point(node_id, -1), node_id=node_id)


def _make_student(name: str, stop: Stop, school: School) -> Student:
    return Student(
        name=name,
        geographic_location=Point(stop.node_id, school.node_id),
        school=school,
        stop=stop,
        requires_monitor=False,
        requires_wheelchair=False,
    )


def _make_problem_data(
    *,
    stops: list[Stop],
    schools: list[School],
    students: list[Student],
    depots: list[Depot] | None = None,
    prune: int | None = None,
    lengths: dict[tuple[int, int], float] | None = None,
) -> ProblemDataReal:
    problem_data = object.__new__(ProblemDataReal)
    object.__setattr__(problem_data, "name", "test")
    object.__setattr__(problem_data, "prune", prune)
    object.__setattr__(problem_data, "use_r5", False)
    object.__setattr__(problem_data, "_stops_cached", stops)
    object.__setattr__(problem_data, "_schools_cached", schools)
    object.__setattr__(problem_data, "_students_cached", students)
    object.__setattr__(problem_data, "_depots_cached", depots or [])

    path_lengths = lengths or {}

    def shortest_path(start: int, end: int, weight: str = "length"):
        return path_lengths.get((start, end), 1.0), [start, end]

    object.__setattr__(problem_data, "_get_shortest_path_osm", shortest_path)
    return problem_data


class CommonServiceGraphTests(unittest.TestCase):
    def test_stop_to_school_edges_follow_student_school_types(self) -> None:
        stop_e = _make_stop("Stop E", 1)
        stop_h = _make_stop("Stop H", 2)
        school_e = _make_school("School E", 10, SchoolType.E)
        school_h = _make_school("School H", 11, SchoolType.HS)
        depot = _make_depot("Depot", 20)
        students = [
            _make_student("student-e", stop_e, school_e),
            _make_student("student-h", stop_h, school_h),
        ]

        graph = _make_problem_data(
            stops=[stop_e, stop_h],
            schools=[school_e, school_h],
            students=students,
            depots=[depot],
        )._make_service_graph()

        self.assertTrue(graph.has_edge(stop_e.node_id, school_e.node_id))
        self.assertFalse(graph.has_edge(stop_e.node_id, school_h.node_id))
        self.assertTrue(graph.has_edge(stop_h.node_id, school_h.node_id))
        self.assertFalse(graph.has_edge(stop_h.node_id, school_e.node_id))

    def test_empty_stop_has_no_outgoing_school_edges(self) -> None:
        stop_used = _make_stop("Stop Used", 1)
        stop_empty = _make_stop("Stop Empty", 2)
        school_e = _make_school("School E", 10, SchoolType.E)
        students = [_make_student("student-e", stop_used, school_e)]

        graph = _make_problem_data(
            stops=[stop_used, stop_empty],
            schools=[school_e],
            students=students,
        )._make_service_graph()

        self.assertFalse(graph.has_edge(stop_empty.node_id, school_e.node_id))

    def test_school_to_stop_edges_remain_unfiltered(self) -> None:
        stop_e = _make_stop("Stop E", 1)
        school_e = _make_school("School E", 10, SchoolType.E)
        school_h = _make_school("School H", 11, SchoolType.HS)
        students = [_make_student("student-e", stop_e, school_e)]

        graph = _make_problem_data(
            stops=[stop_e],
            schools=[school_e, school_h],
            students=students,
        )._make_service_graph()

        self.assertFalse(graph.has_edge(stop_e.node_id, school_h.node_id))
        self.assertTrue(graph.has_edge(school_h.node_id, stop_e.node_id))

    def test_same_node_edges_are_zero_length(self) -> None:
        stop_a = _make_stop("Stop A", 1)
        stop_b = _make_stop("Stop B", 1)

        graph = _make_problem_data(
            stops=[stop_a, stop_b],
            schools=[],
            students=[],
        )._make_service_graph()

        self.assertEqual(graph.edges[1, 1, 0]["length"], 0)
        self.assertEqual(graph.edges[1, 1, 0]["path"], [1, 1])

    def test_prune_only_applies_to_stop_to_stop_edges(self) -> None:
        stop_a = _make_stop("Stop A", 1)
        stop_b = _make_stop("Stop B", 2)
        school_e = _make_school("School E", 10, SchoolType.E)
        students = [_make_student("student-e", stop_a, school_e)]

        graph = _make_problem_data(
            stops=[stop_a, stop_b],
            schools=[school_e],
            students=students,
            prune=5,
            lengths={
                (stop_a.node_id, stop_b.node_id): 10.0,
                (stop_b.node_id, stop_a.node_id): 10.0,
                (stop_a.node_id, school_e.node_id): 10.0,
                (school_e.node_id, stop_a.node_id): 10.0,
            },
        )._make_service_graph()

        self.assertFalse(graph.has_edge(stop_a.node_id, stop_b.node_id))
        self.assertTrue(graph.has_edge(stop_a.node_id, school_e.node_id))

    def test_r5_itinerary_lookup_is_keyed_by_from_to_ids(self) -> None:
        lookup = ProblemDataReal._r5_itinerary_lookup(
            pd.DataFrame(
                {
                    "from_id": ["place_0", "place_0", "place_1"],
                    "to_id": ["place_1", "place_1", "place_2"],
                    "distance": [12.5, 99.0, 7.0],
                    "geometry": ["geom-a", "geom-b", "geom-c"],
                }
            )
        )

        self.assertEqual(lookup[("place_0", "place_1")]["distance"], 12.5)
        self.assertEqual(lookup[("place_0", "place_1")]["geometry"], "geom-a")
        self.assertEqual(lookup[("place_1", "place_2")]["distance"], 7.0)


if __name__ == "__main__":
    unittest.main()
