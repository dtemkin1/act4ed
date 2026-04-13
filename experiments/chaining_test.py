import os
from pathlib import Path

from gurobipy import GRB

from experiments.helpers import make_point_from_node_id
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
    output_dir = CURRENT_FILE_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    graph = make_graph(size=(4, 1))

    depot_point = 0
    depot = Depot(
        name="Depot 1",
        geographic_location=make_point_from_node_id(graph, depot_point),
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
        geographic_location=make_point_from_node_id(graph, stop_node),
        node_id=1,
    )

    school_1_node = 2
    school_1 = School(
        name="School 1",
        geographic_location=make_point_from_node_id(graph, school_1_node),
        node_id=2,
        id=1,
        type=SchoolType.E,
        start_time=8 * 60,  # 8:00 AM in minutes
    )

    school_2_node = 3
    school_2 = School(
        name="School 2",
        geographic_location=make_point_from_node_id(graph, school_2_node),
        node_id=3,
        id=2,
        type=SchoolType.E,
        start_time=8 * 60 + 15,  # 8:15 AM in minutes
    )

    students_1 = [
        Student(
            name=f"Student {i}",
            geographic_location=make_point_from_node_id(graph, stop_node),
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
            geographic_location=make_point_from_node_id(graph, stop_node),
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

    objective_value = model.ObjVal if model.SolCount > 0 else None

    print(
        {
            "problem_name": toy_data.problem_data.name,
            "rounds": toy_data.rounds,
            "objective_value": objective_value,
            "runtime_seconds": model.Runtime,
            # TODO: ask riccardo about gurobi vars/values and how to extract them
            # "results": vals,
        }
    )

    if model.Status == GRB.INFEASIBLE:
        iis_path = (
            output_dir / f"{toy_data.problem_data.name}_rounds_{toy_data.rounds}.ilp"
        )
        model.computeIIS()
        model.write(str(iis_path))
        print(f"Wrote IIS to {iis_path}")
        return

    plot_bus_routes(
        prob=model,
        formulation=toy_data,
        model_vars=vals,
        save_path=output_dir
        / f"{toy_data.problem_data.name}_rounds_{toy_data.rounds}_routes.png",
        per_round=True,
    )

    report = make_report(model, toy_data, vals)
    report_file = (
        output_dir / f"{toy_data.problem_data.name}_rounds_{toy_data.rounds}.txt"
    )
    with open(report_file, "w+", encoding="utf-8") as f:
        f.write(report)

    model.close()


if __name__ == "__main__":
    main()
