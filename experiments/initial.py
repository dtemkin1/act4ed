from functools import cache
import math
import os
import datetime as dt

import geopandas as gpd
import pandas as pd
import networkx as nx
import osmnx as ox
from matplotlib import pyplot as plt

from shapely.geometry import Polygon

from formulation.common import School, Stop, Student, Depot, Bus, SchoolType
from formulation.formulation_3.problem3_definition import Formulation3
from formulation.formulation_3.formulation3 import (
    build_model_from_definition,
    make_report,
    plot_bus_routes,
    solve_problem,
)

try:
    os.environ["JAVA_HOME"] = os.path.join("C:", "Program Files", "Java", "jdk-25.0.2")
    import shapely
except ImportError as exc:
    raise ImportError(
        "Shapely not found. Please install it with 'pip install shapely' and ensure Java is properly configured."
    ) from exc

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(CURRENT_FILE_DIR)

# where
PLACE_NAME = "Framingham, Massachusetts, USA"
BOUNDARY_BUFFER_M = 1000

# data files
DEPOT_CSV = "data/depot.csv"
SCHOOLS_CSV = "data/schools.csv"
STOPS_CSV = "data/stops.csv"
STUDENTS_CSV = "data/students.csv"
BUSES_CSV = "data/buses.csv"

# thresholds
MAX_WALK_TIME_S = 15 * 60  # 15 mins
MAX_WALK_DIST_M = 1000  # 1 km (should we do miles? idk)

# base street network
NETWORK_TYPE = "drive"

# outputs
GRAPHML_FILE = "outputs/framingham_graph.graphml"
PAIRWISE_CSV = "outputs/depot_schools_stops_pairwise.csv"
STUDENT_ASSIGN_CSV = "outputs/student_to_stop_or_school.csv"


def make_framingham_poly():
    # get boundary polygon (similar to analysis.ipynb)
    framingham_gdf = ox.geocode_to_gdf(PLACE_NAME)

    # project to utm for meters-based buffering
    framingham_projected = framingham_gdf.to_crs(framingham_gdf.estimate_utm_crs())
    framingham_projected["geometry"] = framingham_projected.buffer(BOUNDARY_BUFFER_M)

    # project back to original crs for osmnx
    framingham_buffered = framingham_projected.to_crs(framingham_gdf.crs)
    framingham_buffered_poly = framingham_buffered.geometry.iloc[0]

    # download street network
    G = ox.graph_from_polygon(framingham_buffered_poly, network_type=NETWORK_TYPE)

    # simplify
    G = ox.truncate.largest_component(G, strongly=False)
    # G = ox.simplify_graph(G)

    # remove self-loops
    G.remove_edges_from(list(nx.selfloop_edges(G)))

    # save as graphml for :sparkles: later :sparkles:
    os.makedirs(os.path.dirname(GRAPHML_FILE), exist_ok=True)
    ox.save_graphml(G, GRAPHML_FILE)

    return G, framingham_gdf, framingham_buffered_poly


def sanity_checks(G: nx.MultiDiGraph, framingham_gdf: gpd.GeoDataFrame):
    # sanity checks <3

    # nodes v edges
    print("Number of nodes:", len(G.nodes))
    print("Number of edges:", len(G.edges))

    # degree distribution
    degrees = [deg for _, deg in G.degree()]
    print("Min degree:", min(degrees))
    print("Max degree:", max(degrees))
    print("Mean degree:", sum(degrees) / len(degrees))

    # attributes
    print(
        "Node attributes:", [node for n, node in enumerate(G.nodes(data=True)) if n < 1]
    )
    print(
        "Edge attributes:",
        [edge for e, edge in enumerate(G.edges(data=True)) if e < 1],
    )

    # plot map rq with outline of city boundary
    _, ax = ox.plot_graph(
        G, node_size=5, edge_linewidth=0.5, figsize=(8, 8), show=False, close=False
    )
    framingham_gdf.boundary.plot(ax=ax, color="red", linewidth=2)
    # plt.show()

    with open("outputs/sanity_checks.png", "wb+") as f:
        plt.savefig(f, dpi=300)

    # note for self: make sure only one node per intersection/dead end,
    # and that there are no duplicate edges or goofy artifacts


@cache
def get_nearest_node_id(G: nx.MultiDiGraph, long_lat: tuple[float, float]) -> int:
    """Get the nearest node in the graph to a given point."""
    return ox.distance.nearest_nodes(G, long_lat[0], long_lat[1])


def get_schools(G: nx.MultiDiGraph) -> list[School]:
    schools_df = pd.read_csv(SCHOOLS_CSV)
    return_schools = []
    for _, row in schools_df.iterrows():
        nearest_node_id = get_nearest_node_id(G, (row["lon"], row["lat"]))
        start_time = dt.datetime.strptime(row["start_time"], "%H:%M").time()
        school = School(
            name=row["id"],
            location=nearest_node_id,
            type=SchoolType[row["type"].upper()],
            # mins from midnight
            start_time=start_time.hour * 60 + start_time.minute,
        )
        return_schools.append(school)
    return return_schools


def get_depots(G: nx.MultiDiGraph) -> list[Depot]:
    depots_df = pd.read_csv(DEPOT_CSV)
    return_depots = []
    for _, row in depots_df.iterrows():
        nearest_node_id = get_nearest_node_id(G, (row["lon"], row["lat"]))
        depot = Depot(
            name=row["id"],
            location=nearest_node_id,
        )
        return_depots.append(depot)
    return return_depots


def get_stops(G: nx.MultiDiGraph) -> list[Stop]:
    stops_df = pd.read_csv(STOPS_CSV)
    return_stops = []
    for _, row in stops_df.iterrows():
        nearest_node_id = get_nearest_node_id(G, (row["lon"], row["lat"]))
        stop = Stop(
            name=row["id"],
            location=nearest_node_id,
        )
        return_stops.append(stop)
    return return_stops


def get_nearest_stop(
    G: nx.MultiDiGraph,
    nearest_node_id: int,
    all_stops: list[Stop],
) -> Stop:
    nearest_stop = min(
        all_stops,
        key=lambda stop: ox.distance.euclidean(
            G.nodes[stop.location]["x"],
            G.nodes[stop.location]["y"],
            G.nodes[nearest_node_id]["x"],
            G.nodes[nearest_node_id]["y"],
        ),
    )
    return nearest_stop


def get_students(
    G: nx.MultiDiGraph, all_schools: list[School], all_stops: list[Stop]
) -> list[Student]:
    students_df = pd.read_csv(STUDENTS_CSV)
    return_students = []
    for _, row in students_df.iterrows():
        nearest_node_id = get_nearest_node_id(G, (row["lon"], row["lat"]))

        school = next(s for s in all_schools if s.name == row["school_id"])

        # find nearest stop to student
        nearest_stop = get_nearest_stop(G, nearest_node_id, all_stops)

        this_student = Student(
            name=f"Student {row['id']}",
            location=nearest_node_id,
            school=school,
            stop=nearest_stop,
            requires_monitor=bool(row["is_sp_ed"]),
            requires_wheelchair=bool(row["is_wheelchair_user"]),
        )
        return_students.append(this_student)
    return return_students


def get_buses(
    all_depots: list[Depot],
) -> list[Bus]:
    buses_df = pd.read_csv(BUSES_CSV)
    return_buses = []
    for _, row in buses_df.iterrows():
        depot = next(d for d in all_depots if d.name == row["depot_name"])
        bus = Bus(
            name=row["id"],
            capacity=row["capacity"],
            range=row["range"],
            depot=depot,
            has_wheelchair_access=bool(row["has_wheelchair_access"]),
        )
        return_buses.append(bus)
    return return_buses


def main() -> None:
    graph, gdf, _ = make_framingham_poly()
    sanity_checks(graph, gdf)

    schools = get_schools(graph)
    print(f"Imported {len(schools)} schools")
    stops = get_stops(graph)
    print(f"Imported {len(stops)} stops")
    depots = get_depots(graph)
    print(f"Imported {len(depots)} depots")
    students = get_students(graph, schools, stops)
    print(f"Imported {len(students)} students")
    buses = get_buses(depots)
    print(f"Imported {len(buses)} buses")

    # pick a random school
    random_school = schools[0]
    print(f"Random school: {random_school}")

    # only kids living within 0.5 miles of the school
    nearby_students: list[Student] = []
    for student in students:
        distance = ox.distance.euclidean(
            graph.nodes[student.location]["x"],
            graph.nodes[student.location]["y"],
            graph.nodes[random_school.location]["x"],
            graph.nodes[random_school.location]["y"],
        )
        if (
            distance
            <= 0.5
            * 69.172
            * math.cos(graph.nodes[random_school.location]["x"] * math.pi / 180.0)
            and student.school == random_school
        ):  # convert lat/long distance to miles
            nearby_students.append(student)
    print(f"Number of nearby students: {len(nearby_students)}")

    stops_with_students: list[Stop] = []

    for stop in stops:
        if any(student.stop == stop for student in nearby_students):
            stops_with_students.append(stop)
    print(f"Number of stops with nearby students: {len(stops_with_students)}")

    # formulation time baby
    no_chaining = Formulation3(
        graph=graph,
        rounds=1,
        schools=[random_school],
        stops=stops_with_students,
        students=nearby_students,
        depots=depots,
        buses=buses,
    )
    print("No chaining formulation created")

    chaining = Formulation3(
        graph=graph,
        rounds=3,
        schools=[random_school],
        stops=stops_with_students,
        students=nearby_students,
        depots=depots,
        buses=buses,
    )
    print("Chaining formulation created")

    no_chaining_model = build_model_from_definition(no_chaining)
    print("No chaining model built")

    chaining_model = build_model_from_definition(chaining)
    print("Chaining model built")

    solve_problem(no_chaining_model[0])
    solve_problem(chaining_model[0])

    report_no_chaining = make_report(
        no_chaining_model[0], no_chaining, no_chaining_model[1]
    )
    report_chaining = make_report(chaining_model[0], chaining, chaining_model[1])

    with open("outputs/report_no_chaining.txt", "w+", encoding="utf-8") as f:
        f.write(report_no_chaining)
    plot_bus_routes(
        no_chaining_model[0],
        no_chaining,
        no_chaining_model[1],
        "outputs/no_chaining_routes.png",
    )

    with open("outputs/report_chaining.txt", "w+", encoding="utf-8") as f:
        f.write(report_chaining)
    plot_bus_routes(
        chaining_model[0],
        chaining,
        chaining_model[1],
        "outputs/chaining_routes.png",
    )


if __name__ == "__main__":
    main()
