import os
from pathlib import Path

from formulation.common import ProblemDataToy, Student

from formulation.formulation_3.problem3_definition import Formulation3
from formulation.formulation_3.formulation3 import (
    build_model_from_definition,
    make_report,
    plot_bus_routes,
    solve_problem,
)
from experiments.helpers import setup


CURRENT_FILE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# where
PLACE_NAME = "Framingham, Massachusetts, USA"


def main() -> None:
    problem_data_original = setup("framingham", PLACE_NAME)
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
        base_graph=problem_data_original.osm_graph,
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

    no_chaining_model = build_model_from_definition(no_chaining)
    print("No-chaining model built")

    solve_problem(no_chaining_model[0])
    print("No-chaining problem solved!")

    report_no_chaining = make_report(
        no_chaining_model[0], no_chaining, no_chaining_model[1]
    )

    with open(
        CURRENT_FILE_DIR / "outputs" / "report_no_chaining.txt", "w+", encoding="utf-8"
    ) as f:
        f.write(report_no_chaining)
    print("No-chaining report written")

    plot_bus_routes(
        no_chaining_model[0],
        no_chaining,
        no_chaining_model[1],
        CURRENT_FILE_DIR / "outputs" / "no_chaining_routes.png",
    )
    print("No-chaining routes plotted")

    print("Now doing chaining formulation...")
    mcc = next(school for school in problem_data_original.schools if school.id == "MCC")

    # only kids living relatively close
    both_nearby_students: list[Student] = []
    for student in problem_data_original.students:
        distance_fuller = problem_data_original.service_graph.edges[
            student.stop.node_id, fuller.node_id, 0
        ]["length"]
        distance_mcc = problem_data_original.service_graph.edges[
            student.stop.node_id, mcc.node_id, 0
        ]["length"]

        # within 1 km of school
        if distance_fuller <= 1.0 and student.school == fuller:
            both_nearby_students.append(student)
        if distance_mcc <= 1.0 and student.school == mcc:
            both_nearby_students.append(student)

    print(f"Number of both nearby students: {len(both_nearby_students)}")

    stops_with_students = list(set(student.stop for student in both_nearby_students))

    problem_data = ProblemDataToy(
        "chaining_toy",
        base_graph=problem_data_original.osm_graph,
        _stops=stops_with_students,
        _schools=[fuller, mcc],
        _depots=problem_data_original.depots,
        _students=both_nearby_students,
        _buses=buses_to_use,
    )
    chaining = Formulation3(
        problem_data=problem_data,
        rounds=3,
    )
    print("Chaining formulation created")

    chaining_model = build_model_from_definition(chaining)
    print("Chaining model built")

    solve_problem(chaining_model[0])
    print("Chaining problem solved!")

    report_chaining = make_report(chaining_model[0], chaining, chaining_model[1])

    with open(
        CURRENT_FILE_DIR / "outputs" / "report_chaining.txt", "w+", encoding="utf-8"
    ) as f:
        f.write(report_chaining)
    plot_bus_routes(
        chaining_model[0],
        chaining,
        chaining_model[1],
        CURRENT_FILE_DIR / "outputs" / "chaining_routes.png",
    )


if __name__ == "__main__":
    main()
