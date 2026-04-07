import os
from pathlib import Path

from formulation.common import (
    Depot,
    Bus,
    ProblemDataToy,
    School,
    SchoolType,
    Stop,
    Student,
)
from formulation.formulation_3.problem3_definition import Formulation3
from formulation.toy_network import make_graph
from formulation.formulation_3.formulation3_gurobipy import (
    build_model_from_definition,
    make_report,
    solve_problem,
    plot_bus_routes,
)

CURRENT_FILE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    graph = make_graph(size=(4, 1))

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
        start_time=8 * 60,  # 8:00 AM in minutes
    )

    school_2_node = 3
    school_2 = School(
        name="School 2",
        geographic_location=graph.nodes[school_2_node]["location"],
        node_id=3,
        id=2,
        type=SchoolType.E,
        start_time=8 * 60 + 15,  # 8:15 AM in minutes
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
            name=f"Student {i+20}",
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

    toy_data = Formulation3(problem_data=problem_data, rounds=1)

    model, vals = build_model_from_definition(toy_data)
    print(f"Model built for problem: {toy_data.problem_data.name}")

    solve_problem(model)
    print(f"Problem solved: {toy_data.problem_data.name}")

    print(
        {
            "problem_name": toy_data.problem_data.name,
            "rounds": toy_data.rounds,
            "objective_value": model.ObjVal,
            "runtime_seconds": model.Runtime,
            # TODO: ask riccardo about gurobi vars/values and how to extract them
            # "results": vals,
        }
    )

    plot_bus_routes(
        prob=model,
        formulation=toy_data,
        model_vars=vals,
        save_path=CURRENT_FILE_DIR
        / "outputs"
        / f"{toy_data.problem_data.name}_rounds_{toy_data.rounds}_routes.png",
        per_round=True,
    )

    report = make_report(model, toy_data, vals)
    report_file = (
        CURRENT_FILE_DIR
        / "outputs"
        / f"{toy_data.problem_data.name}_rounds_{toy_data.rounds}.txt"
    )
    with open(report_file, "w+", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    main()
