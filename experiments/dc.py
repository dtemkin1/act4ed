import os
from pathlib import Path

from formulation.common import ProblemData


CURRENT_FILE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# where
PLACE_NAME = "Washington, District of Columbia, USA"
BOUNDARY_BUFFER_M = 0

# data files
DEPOT_CSV = CURRENT_FILE_DIR / "data" / "depot.csv"
SCHOOLS_CSV = CURRENT_FILE_DIR / "data" / "schools.csv"
STOPS_CSV = CURRENT_FILE_DIR / "data" / "stops.csv"
STUDENTS_CSV = CURRENT_FILE_DIR / "data" / "students.csv"
BUSES_CSV = CURRENT_FILE_DIR / "data" / "buses.csv"

# base street network
NETWORK_TYPE = "drive"


def setup() -> ProblemData:
    try:
        problem_data = ProblemData.load("washington")
    except FileNotFoundError:
        problem_data = ProblemData(
            name="washington",
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

    osm_graph = problem_data.osm_graph

    # get total length
    total_length_m = sum(data["length"] for _, _, data in osm_graph.edges(data=True))

    print(f"Total length of graph: {total_length_m / 1000:.2f} km")


if __name__ == "__main__":
    main()
