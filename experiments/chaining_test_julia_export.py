from __future__ import annotations

import argparse
from pathlib import Path

from formulation.common import Bus, Depot, ProblemDataToy, School, SchoolType, Stop, Student
from formulation.formulation_3.julia_export import export_formulation3_instance
from formulation.formulation_3.problem3_definition import Formulation3
from formulation.toy_network import make_graph


def build_problem(rounds: int = 1) -> Formulation3:
    graph = make_graph(size=(4, 2))

    depot_point = 0
    depot = Depot(
        name="Depot 1",
        geographic_location=graph.nodes[depot_point]["location"],
        node_id=0,
    )
    bus = Bus(
        name="Bus 1",
        capacity=41,
        range=1000,
        depot=depot,
        has_wheelchair_access=True,
    )

    stop_node = 1
    stop = Stop(
        name="Stop 1",
        geographic_location=graph.nodes[stop_node]["location"],
        node_id=1,
    )

    school_1_node = 2
    school_1 = School(
        name="School 1",
        geographic_location=graph.nodes[school_1_node]["location"],
        node_id=2,
        id=1,
        type=SchoolType.E,
        start_time=8 * 60,
    )

    school_2_node = 3
    school_2 = School(
        name="School 2",
        geographic_location=graph.nodes[school_2_node]["location"],
        node_id=3,
        id=2,
        type=SchoolType.E,
        start_time=8 * 60 + 15,
    )

    students_1 = [
        Student(
            name=f"Student {i}",
            geographic_location=graph.nodes[stop_node]["location"],
            stop=stop,
            school=school_1,
            requires_monitor=False,
            requires_wheelchair=False,
        )
        for i in range(20)
    ]

    students_2 = [
        Student(
            name=f"Student {i + 20}",
            geographic_location=graph.nodes[stop_node]["location"],
            stop=stop,
            school=school_2,
            requires_monitor=False,
            requires_wheelchair=False,
        )
        for i in range(20)
    ]

    problem_data = ProblemDataToy(
        name="chaining_test",
        base_graph=graph,
        _depots=[depot],
        _buses=[bus],
        _stops=[stop],
        _schools=[school_1, school_2],
        _students=students_1 + students_2,
    )

    return Formulation3(problem_data=problem_data, rounds=rounds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output_path = export_formulation3_instance(build_problem(rounds=args.rounds), args.output)
    print(output_path)


if __name__ == "__main__":
    main()
