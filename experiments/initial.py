import os
from pathlib import Path

from formulation.common import ProblemData, Student, Stop

from formulation.formulation_3.problem3_definition import Formulation3
from formulation.formulation_3.formulation3 import (
    build_model_from_definition,
    make_report,
    plot_bus_routes,
    solve_problem,
)


CURRENT_FILE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# where
PLACE_NAME = "Framingham, Massachusetts, USA"
BOUNDARY_BUFFER_M = 1000

# data files
DEPOT_CSV = CURRENT_FILE_DIR / "data" / "depot.csv"
SCHOOLS_CSV = CURRENT_FILE_DIR / "data" / "schools.csv"
STOPS_CSV = CURRENT_FILE_DIR / "data" / "stops.csv"
STUDENTS_CSV = CURRENT_FILE_DIR / "data" / "students.csv"
BUSES_CSV = CURRENT_FILE_DIR / "data" / "buses.csv"

# thresholds
MAX_WALK_TIME_S = 15 * 60  # 15 mins
MAX_WALK_DIST_M = 1000  # 1 km (should we do miles? idk)

# base street network
NETWORK_TYPE = "drive"

# outputs
GRAPHML_FILE = CURRENT_FILE_DIR / "outputs" / "framingham_graph.graphml"
PAIRWISE_CSV = CURRENT_FILE_DIR / "outputs" / "depot_schools_stops_pairwise.csv"
STUDENT_ASSIGN_CSV = CURRENT_FILE_DIR / "outputs" / "student_to_stop_or_school.csv"


def setup() -> ProblemData:
    try:
        problem_data = ProblemData.load("framingham")
    except FileNotFoundError:
        problem_data = ProblemData(
            name="framingham",
            schools_path=SCHOOLS_CSV,
            stops_path=STOPS_CSV,
            students_path=STUDENTS_CSV,
            depots_path=DEPOT_CSV,
            buses_path=BUSES_CSV,
            place_name=PLACE_NAME,
            boundary_buffer_m=BOUNDARY_BUFFER_M,
        )

        problem_data.save()

    return problem_data


def main() -> None:
    problem_data = setup()
    print("Problem data loaded!")

    problem_data.sanity_checks()

    # pick a random school
    random_school = problem_data.schools[0]
    print(f"Random school: {random_school}")

    # only kids living relatively close
    nearby_students: list[Student] = []
    for student in problem_data.students:
        distance = problem_data.service_graph.edges[student.stop, random_school, 0][
            "length"
        ]

        if (
            distance <= 1500 and student.school == random_school
        ):  # convert lat/long distance to miles
            nearby_students.append(student)
    print(f"Number of nearby students: {len(nearby_students)}")

    stops_with_students: list[Stop] = []

    # only stops with nearby students
    for stop in problem_data.stops:
        if any(student.stop == stop for student in nearby_students):
            stops_with_students.append(stop)
    print(f"Number of stops with nearby students: {len(stops_with_students)}")

    problem_data.students = nearby_students
    problem_data.stops = stops_with_students
    problem_data.schools = [random_school]

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
    plot_bus_routes(
        no_chaining_model[0],
        no_chaining,
        no_chaining_model[1],
        CURRENT_FILE_DIR / "outputs" / "no_chaining_routes.png",
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
