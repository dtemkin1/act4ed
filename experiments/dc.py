import os
from pathlib import Path

from experiments.helpers import setup


CURRENT_FILE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# where
PLACE_NAME = "Washington, District of Columbia, USA"


def main() -> None:
    problem_data = setup("washington", PLACE_NAME)
    print("Problem data loaded!")

    osm_graph = problem_data.osm_graph

    # get total length
    total_length_km = sum(data["length"] for _, _, data in osm_graph.edges(data=True))

    print(f"Total length of graph: {total_length_km:.2f} km")


if __name__ == "__main__":
    main()
