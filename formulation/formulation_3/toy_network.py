import networkx as nx

from formulation.common import Bus, Depot, School, SchoolType, Stop, Student


def make_graph(size: int = 10) -> nx.DiGraph:
    graph = nx.grid_2d_graph(size, size, create_using=nx.DiGraph)
    # Add weights to the edges
    for u, v in graph.edges():
        graph.edges[u, v]["weight"] = 1.0
    return graph


def make_schools(
    graph: nx.DiGraph, num_schools: int = 3, types: list[SchoolType] | None = None
) -> list[School]:
    if types is None:
        types = list(SchoolType.__members__.values())
    all_nodes = graph.nodes
    return [
        # School(
        #     name=f"School {i}",
        #     geographic_location=all_nodes[i % len(all_nodes)],
        #     node_id=all_nodes[i % len(all_nodes)],
        #     type=types[i % len(types)],
        #     start_time=8 * 60 + i * 15,  # staggered start times
        # )
        # for i in range(num_schools)
    ]


def make_depots(graph: nx.DiGraph, num_depots: int = 1) -> list[Depot]:
    all_nodes = list(graph.nodes)
    return [
        # Depot(name=f"Depot {i}", node_id=all_nodes[i % len(all_nodes)])
        # for i in range(num_depots)
    ]


def make_stops(graph: nx.DiGraph, num_stops: int = 5) -> list[Stop]:
    all_nodes = list(graph.nodes)
    return [
        # Stop(name=f"Stop {i}", node_id=all_nodes[i % len(all_nodes)])
        # for i in range(num_stops)
    ]


def make_students(
    graph: nx.DiGraph,
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
    return [
        # Student(
        #     name=f"Student {i}",
        #     node_id=all_nodes[i % len(all_nodes)],
        #     school=schools[i % num_schools],
        #     stop=stops[i % len(stops)],
        #     requires_monitor=(i % 4 == 0),  # every 4th student requires a monitor
        #     requires_wheelchair=(i % 6 == 0),  # every 6th student requires a wheelchair
        # )
        # for i in range(num_students)
    ]


def make_buses(
    graph: nx.DiGraph,
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
    return [
        Bus(
            name=f"Bus {i}",
            capacity=capacities[i % len(capacities)],
            range=ranges[i % len(ranges)],
            depot=depots[i % len(depots)],
            has_wheelchair_access=(i % 2 == 0),  # every other bus has wheelchair access
        )
        for i in range(num_buses)
    ]
