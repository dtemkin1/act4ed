from typing import cast
import random

import networkx as nx

from formulation.common import (
    Bus,
    Depot,
    NodeId,
    School,
    SchoolType,
    Stop,
    Student,
    ProblemDataToy,
)
from shapely import Point

random.seed(42)  # for reproducibility <3


def make_graph(size: tuple[int, int] = (10, 10)) -> "nx.MultiDiGraph[NodeId]":
    graph_2d = nx.grid_2d_graph(size[0], size[1], create_using=nx.MultiDiGraph)
    graph_2d = cast("nx.MultiDiGraph[tuple[int, int]]", graph_2d)
    # Add weights to the edges
    for u, v in graph_2d.edges():
        graph_2d.edges[u, v, 0]["length"] = 1.0  # 1 km between adjacent nodes

    graph: "nx.MultiDiGraph[NodeId]" = nx.MultiDiGraph()
    mapping: dict[tuple[int, int], NodeId] = {}
    for i, node in enumerate(graph_2d.nodes()):
        mapping[node] = i
        x = node[0]
        y = node[1]
        graph.add_node(i, x=x, y=y, location=(x, y))
    for u, v in graph_2d.edges():
        graph.add_edge(mapping[u], mapping[v], length=graph_2d.edges[u, v, 0]["length"])

    return graph


def make_schools(
    graph: "nx.MultiDiGraph[NodeId]",
    num_schools: int = 3,
    types: list[SchoolType] | None = None,
) -> list[School]:
    if types is None:
        types = list(SchoolType.__members__.values())
    all_nodes = list(graph.nodes)
    schools: list[School] = []
    for i in range(num_schools):
        node_id = all_nodes[random.randint(0, len(all_nodes) - 1)]
        point = Point(graph.nodes[node_id]["location"])
        school = School(
            name=f"School {i}",
            geographic_location=point,
            node_id=node_id,
            type=types[i % len(types)],
            start_time=8 * 60 + i * 15,  # staggered start times
            id=str(i),
        )
        schools.append(school)
    return schools


def make_depots(graph: "nx.MultiDiGraph[NodeId]", num_depots: int = 1) -> list[Depot]:
    all_nodes = list(graph.nodes)
    depots: list[Depot] = []
    for i in range(num_depots):
        node_id = all_nodes[random.randint(0, len(all_nodes) - 1)]
        point = Point(graph.nodes[node_id]["location"])
        depot = Depot(
            name=f"Depot {i}",
            geographic_location=point,
            node_id=node_id,
        )
        depots.append(depot)
    return depots


def make_stops(graph: "nx.MultiDiGraph[NodeId]", num_stops: int = 5) -> list[Stop]:
    all_nodes = list(graph.nodes)
    stops: list[Stop] = []
    for i in range(num_stops):
        node_id = all_nodes[random.randint(0, len(all_nodes) - 1)]
        point = Point(graph.nodes[node_id]["location"])
        stop = Stop(
            name=f"Stop {i}",
            geographic_location=point,
            node_id=node_id,
        )
        stops.append(stop)
    return stops


def make_students(
    graph: "nx.MultiDiGraph[NodeId]",
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
        node_id = all_nodes[random.randint(0, len(all_nodes) - 1)]
        home_location = Point(graph.nodes[node_id]["location"])
        # get nearest stop
        stop = min(
            stops,
            key=lambda s: home_location.distance(s.geographic_location),
        )
        random_school = schools[random.randint(0, num_schools - 1)]
        student = Student(
            name=f"Student {i}",
            geographic_location=home_location,
            school=random_school,
            stop=stop,
            requires_wheelchair=(i % 5 == 0),
            requires_monitor=(i % 4 == 0),
        )
        students.append(student)
    return students


def make_buses(
    graph: "nx.MultiDiGraph[NodeId]",
    num_buses: int = 3,
    capacities: list[int] | None = None,
    ranges: list[float] | None = None,
    depots: list[Depot] | None = None,
) -> list[Bus]:
    if capacities is None:
        capacities = [40] * num_buses
    if depots is None:
        depots = make_depots(graph)
    if ranges is None:
        ranges = [len(graph.nodes)] * num_buses
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


def make_toy_problem_data(
    name: str,
    size: tuple[int, int] | None,
    num_schools: int | None,
    num_depots: int | None,
    num_stops: int | None,
    num_students: int | None,
    num_buses: int | None,
) -> ProblemDataToy:
    graph = make_graph(size=size)
    schools = make_schools(graph, num_schools=num_schools)
    depots = make_depots(graph, num_depots=num_depots)
    stops = make_stops(graph, num_stops=num_stops)
    students = make_students(
        graph, num_students=num_students, schools=schools, stops=stops
    )
    buses = make_buses(graph, num_buses=num_buses, depots=depots)

    return ProblemDataToy(
        name=name,
        base_graph=graph,
        _schools=schools,
        _depots=depots,
        _stops=stops,
        _students=students,
        _buses=buses,
    )
