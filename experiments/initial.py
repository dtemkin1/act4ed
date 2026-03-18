import os
from pathlib import Path


from formulation.common import (
    ProblemData,
)

# from formulation.formulation_3.problem3_definition import Formulation3
# from formulation.formulation_3.formulation3 import (
#     build_model_from_definition,
#     make_report,
#     plot_bus_routes,
#     solve_problem,
# )


CURRENT_FILE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
os.chdir(CURRENT_FILE_DIR)

# where
PLACE_NAME = "Framingham, Massachusetts, USA"
BOUNDARY_BUFFER_M = 1000

# data files
DEPOT_CSV = CURRENT_FILE_DIR / Path("data/depot.csv")
SCHOOLS_CSV = CURRENT_FILE_DIR / Path("data/schools.csv")
STOPS_CSV = CURRENT_FILE_DIR / Path("data/stops.csv")
STUDENTS_CSV = CURRENT_FILE_DIR / Path("data/students.csv")
BUSES_CSV = CURRENT_FILE_DIR / Path("data/buses.csv")

# thresholds
MAX_WALK_TIME_S = 15 * 60  # 15 mins
MAX_WALK_DIST_M = 1000  # 1 km (should we do miles? idk)

# base street network
NETWORK_TYPE = "drive"

# outputs
GRAPHML_FILE = CURRENT_FILE_DIR / Path("outputs/framingham_graph.graphml")
PAIRWISE_CSV = CURRENT_FILE_DIR / Path("outputs/depot_schools_stops_pairwise.csv")
STUDENT_ASSIGN_CSV = CURRENT_FILE_DIR / Path("outputs/student_to_stop_or_school.csv")


def main() -> None:
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

    # problem_data = ProblemData.load("framingham_test")

    # # pick a random school
    # random_school = schools[0]
    # print(f"Random school: {random_school}")

    # # only kids living within 0.5 miles of the school
    # nearby_students: list[Student] = []
    # for student in students:
    #     distance = ox.distance.euclidean(
    #         graph.nodes[student.location]["x"],
    #         graph.nodes[student.location]["y"],
    #         graph.nodes[random_school.location]["x"],
    #         graph.nodes[random_school.location]["y"],
    #     )
    #     if (
    #         distance
    #         <= 0.5
    #         * 69.172
    #         * math.cos(graph.nodes[random_school.location]["x"] * math.pi / 180.0)
    #         and student.school == random_school
    #     ):  # convert lat/long distance to miles
    #         nearby_students.append(student)
    # print(f"Number of nearby students: {len(nearby_students)}")

    # stops_with_students: list[Stop] = []

    # for stop in stops:
    #     if any(student.stop == stop for student in nearby_students):
    #         stops_with_students.append(stop)
    # print(f"Number of stops with nearby students: {len(stops_with_students)}")

    # # formulation time baby
    # no_chaining = Formulation3(
    #     graph=graph,
    #     rounds=1,
    #     schools=[random_school],
    #     stops=stops_with_students,
    #     students=nearby_students,
    #     depots=depots,
    #     buses=buses,
    # )
    # print("No chaining formulation created")

    # # chaining = copy.replace(no_chaining, rounds=3, Q=list(range(3)), Q_MAX=2)
    # # print("Chaining formulation created")

    # no_chaining_model = build_model_from_definition(no_chaining)
    # print("No chaining model built")

    # # chaining_model = build_model_from_definition(chaining)
    # # print("Chaining model built")

    # solve_problem(no_chaining_model[0])
    # # solve_problem(chaining_model[0])

    # report_no_chaining = make_report(
    #     no_chaining_model[0], no_chaining, no_chaining_model[1]
    # )
    # # report_chaining = make_report(chaining_model[0], chaining, chaining_model[1])

    # with open("outputs/report_no_chaining.txt", "w+", encoding="utf-8") as f:
    #     f.write(report_no_chaining)
    # plot_bus_routes(
    #     no_chaining_model[0],
    #     no_chaining,
    #     no_chaining_model[1],
    #     "outputs/no_chaining_routes.png",
    # )

    # with open("outputs/report_chaining.txt", "w+", encoding="utf-8") as f:
    #     f.write(report_chaining)
    # plot_bus_routes(
    #     chaining_model[0],
    #     chaining,
    #     chaining_model[1],
    #     "outputs/chaining_routes.png",
    # )


if __name__ == "__main__":
    main()
