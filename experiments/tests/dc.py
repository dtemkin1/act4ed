from experiments.helpers import setup

# where
PLACE_NAME = "Washington, District of Columbia, USA"


def main() -> None:
    problem_data = setup("washington", PLACE_NAME)
    print("Problem data loaded!")

    osm_graph = problem_data.osm_graph

    # get total length
    total_length_m = sum(data["length"] for _, _, data in osm_graph.edges(data=True))
    total_length_km = total_length_m / 1000.0

    print(f"Total length of graph: {total_length_km:.2f} km")


if __name__ == "__main__":
    main()
