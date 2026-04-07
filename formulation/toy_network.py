import networkx as nx

from formulation.common import (
    Bus,
    Depot,
    School,
    SchoolType,
    Stop,
    Student,
    ProblemDataToy,
)
from shapely import Point


def make_graph(size: int = 10) -> nx.MultiDiGraph:
    graph = nx.grid_2d_graph(size, size, create_using=nx.MultiDiGraph)
    # Add weights to the edges
    for u, v in graph.edges():
        graph.edges[u, v]["length"] = 1000.0  # 1 km between adjacent nodes
    return graph


def make_schools(
    graph: nx.MultiDiGraph, num_schools: int = 3, types: list[SchoolType] | None = None
) -> list[School]:
    if types is None:
        types = list(SchoolType.__members__.values())
    all_nodes = list(graph.nodes)
    schools: list[School] = []
    for i in range(num_schools):
        point = Point(all_nodes[i % len(all_nodes)])
        school = School(
            name=f"School {i}",
            geographic_location=point,
            node_id=all_nodes[i % len(all_nodes)],
            type=types[i % len(types)],
            start_time=8 * 60 + i * 15,  # staggered start times
            id=str(i),
        )
        schools.append(school)
    return schools


def make_depots(graph: nx.MultiDiGraph, num_depots: int = 1) -> list[Depot]:
    all_nodes = list(graph.nodes)
    depots: list[Depot] = []
    for i in range(num_depots):
        point = Point(all_nodes[i % len(all_nodes)])
        depot = Depot(
            name=f"Depot {i}",
            geographic_location=point,
            node_id=all_nodes[i % len(all_nodes)],
        )
        depots.append(depot)
    return depots


def make_stops(graph: nx.MultiDiGraph, num_stops: int = 5) -> list[Stop]:
    all_nodes = list(graph.nodes)
    stops: list[Stop] = []
    for i in range(num_stops):
        point = Point(all_nodes[i % len(all_nodes)])
        stop = Stop(
            name=f"Stop {i}",
            geographic_location=point,
            node_id=all_nodes[i % len(all_nodes)],
        )
        stops.append(stop)
    return stops


def make_students(
    graph: nx.MultiDiGraph,
    num_students: int = 20,
    schools: list[School] | None = None,
    stops: list[Stop] | None = None,
) -> list[Student]:
    if schools is None:
        schools = make_schools(graph)
    if stops is None:
        stops = make_stops(graph)
    num_schools = len(schools)
    all_nodes = list(graph.nodes)
    students: list[Student] = []
    for i in range(num_students):
        home_location = Point(all_nodes[i % len(all_nodes)])
        student = Student(
            name=f"Student {i}",
            geographic_location=home_location,
            school=schools[i % num_schools],
            stop=stops[i % len(stops)],
            requires_wheelchair=(i % 5 == 0),
            requires_monitor=(i % 4 == 0),
        )
        students.append(student)
    return students


def make_buses(
    graph: nx.MultiDiGraph,
    num_buses: int = 3,
    capacities: list[int] | None = None,
    ranges: list[int] | None = None,
    depots: list[Depot] | None = None,
) -> list[Bus]:
    if capacities is None:
        capacities = [40] * num_buses
    if depots is None:
        depots = make_depots(graph)
    if ranges is None:
        ranges = [10] * num_buses
    buses: list[Bus] = []
    for i in range(num_buses):
        bus = Bus(
            name=f"Bus {i}",
            capacity=capacities[i % len(capacities)],
            range=ranges[i % len(ranges)],
            depot=depots[i % len(depots)],
            has_wheelchair_access=(i % 2 == 0),  # every other bus has wheelchair access
        )
        buses.append(bus)
    return buses


def make_toy_problem_data() -> ProblemDataToy:
    graph = make_graph()
    schools = make_schools(graph)
    depots = make_depots(graph)
    stops = make_stops(graph)
    students = make_students(graph, schools=schools, stops=stops)
    buses = make_buses(graph, depots=depots)

    return ProblemDataToy(
        "toy_network",
        base_graph=graph,
        schools=schools,
        depots=depots,
        stops=stops,
        students=students,
        buses=buses,
    )
