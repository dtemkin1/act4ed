import os
from pathlib import Path

from formulation.common.classes import Student
from formulation.common.problems import ProblemDataToy
from formulation.formulation_3.definition import Formulation3
from formulation.formulation_3.gurobipy import (
    build_model_from_definition,
    make_report,
    plot_bus_routes,
    solve_problem,
)
from experiments.helpers import make_osm_in_km, setup


CURRENT_FILE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# where
PLACE_NAME = "Framingham, Massachusetts, USA"


def main() -> None:
    problem_data_original = setup("framingham", PLACE_NAME)
    osm_graph_km = make_osm_in_km(problem_data_original.osm_graph)
    print("Problem data loaded!")

    # pick fuller at first since mcc is right next to it
    fuller = next(
        school for school in problem_data_original.schools if school.id == "FUL"
    )

    # only kids living relatively close
    nearby_students: list[Student] = []
    for student in problem_data_original.students:
        distance = problem_data_original.service_graph.edges[
            student.stop.node_id, fuller.node_id, 0
        ]["length"]

        # within 1 km of school
        if distance <= 1.0 and student.school == fuller:
            nearby_students.append(student)
    print(f"Number of nearby students: {len(nearby_students)}")

    stops_with_students = list(set(student.stop for student in nearby_students))

    # only use 1 buses total, wheelchair accessible
    buses_to_use = list(
        filter(lambda b: b.has_wheelchair_access, problem_data_original.buses)
    )[:1]

    problem_data = ProblemDataToy(
        "no_chaining_toy",
        _base_graph=osm_graph_km,
        _stops=stops_with_students,
        _schools=[fuller],
        _depots=problem_data_original.depots,
        _students=nearby_students,
        _buses=buses_to_use,
    )

    # formulation time baby
    no_chaining = Formulation3(
        problem_data=problem_data,
        rounds=1,
    )
    print("No-chaining formulation created")

    no_chaining_model, no_chaining_vars = build_model_from_definition(no_chaining)
    print("No-chaining model built")

    solve_problem(no_chaining_model)
    print("No-chaining problem solved!")

    report_no_chaining = make_report(no_chaining_model, no_chaining, no_chaining_vars)

    with open(
        CURRENT_FILE_DIR / ".." / "outputs" / "report_no_chaining.txt",
        "w+",
        encoding="utf-8",
    ) as f:
        f.write(report_no_chaining)
    print("No-chaining report written")

    plot_bus_routes(
        no_chaining_model,
        no_chaining,
        no_chaining_vars,
        CURRENT_FILE_DIR / ".." / "outputs" / "no_chaining_routes.png",
    )
    print("No-chaining routes plotted")
    no_chaining_model.close()


if __name__ == "__main__":
    main()
