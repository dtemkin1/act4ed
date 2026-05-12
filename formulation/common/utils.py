from collections.abc import Hashable
from dataclasses import replace
from functools import cache

import networkx as nx

from formulation.common.classes import (Bus, Depot, NodeId, Place, School,
                                        Student)
from formulation.common.constants import (BUS_SPEED_NOT_HIGHWAY,
                                          BUS_SPEED_SCHOOL_ZONE, METERS_PER_KM,
                                          MPH_TO_KM_PER_MIN)


@cache
def make_place_copy[T: Place](place: T, suffix: str = "copy") -> T:
    return replace(place, name=place.name + f" ({suffix})")


@cache
def make_school_copy(school: School) -> School:
    return make_place_copy(school)


@cache
def make_depot_end_copy(depot: Depot) -> Depot:
    return make_place_copy(depot, "end copy")


@cache
def make_depot_start_copy(depot: Depot) -> Depot:
    return make_place_copy(depot, "start copy")


@cache
def get_shortest_path[T: Hashable](
    graph: "nx.MultiDiGraph[T]", start: T, end: T, weight: str = "length"
) -> tuple[float, list[T]]:
    """returns the length and path of the shortest path between start and end"""
    return nx.bidirectional_dijkstra(graph, source=start, target=end, weight=weight)


def meters_to_kilometers(distance_meters: float) -> float:
    return float(distance_meters) / METERS_PER_KM


def ensure_service_graph_kilometers(graph: "nx.MultiDiGraph[NodeId]") -> None:
    unit = graph.graph.get("distance_unit")
    if unit == "km":
        return

    lengths = [
        float(data["length"])
        for _, _, _, data in graph.edges(keys=True, data=True)
        if "length" in data
    ]
    needs_conversion = unit == "m" or (
        unit is None and any(length > 100.0 for length in lengths)
    )

    if needs_conversion:
        for _, _, _, data in graph.edges(keys=True, data=True):
            if "length" in data:
                data["length"] = meters_to_kilometers(float(data["length"]))

    graph.graph["distance_unit"] = "km"


def p_m(m: Student):
    """pickup stop of student m"""
    return m.stop


def s_m(m: Student):
    """school of student m"""
    return m.school


def tau_m(m: Student):
    """type of school of student m"""
    return m.school.type


def f_m(m: Student):
    """1 if student m if flagged"""
    return 1 if m.attributes.special_ed or m.attributes.wheelchair_user else 0


def depot_b(b: Bus):
    """depot of bus b"""
    return b.depot


def C_b(b: Bus):
    """capacity of bus b"""
    return b.capacity


def Wh_b(b: Bus):
    """1 if bus b has wheelchair access"""
    return 1 if b.wheelchair_capacity > 0 else 0


def R_b(b: Bus):
    """range of bus b in km"""
    return b.range_km


def h_s(s: School):
    """start time of school s in minutes from midnight"""
    return s.start_time


def slack_s(s: School):
    """required slack time for school s in minutes, same in our case"""
    return 0


def l_s(s: School):
    """latest allowable arrival time at school s in minutes from midnight"""
    return h_s(s) - slack_s(s)


@cache
def get_paths_between_nodes(
    nodes: tuple[NodeId, ...], service_graph: "nx.MultiDiGraph[NodeId]"
) -> list[tuple[NodeId, ...]]:
    """utility function to get paths between consecutive nodes in a list"""
    paths = []
    for k in range(len(nodes) - 1):
        edge_data = service_graph.get_edge_data(
            nodes[k], nodes[k + 1], key=0, default=None
        )
        if edge_data is not None:
            paths.append(tuple(edge_data["path"]))

    return paths


@cache
def get_travel_time(
    path: tuple[NodeId, ...],
    base_graph: "nx.MultiDiGraph[NodeId]",
) -> float:
    """utility function to get travel time along a path, used for caching travel times"""

    travel_time = 0.0
    unit = base_graph.graph.get("distance_unit")
    for k in range(len(path) - 1):
        travel_time += get_time_between_nodes(path[k], path[k + 1], base_graph, unit)

    return travel_time


@cache
def get_time_between_nodes(
    node1: NodeId,
    node2: NodeId,
    base_graph: "nx.MultiDiGraph[NodeId]",
    unit: str | None = None,
) -> float:
    """utility function to get travel time between two nodes, used for caching travel times"""

    edge_data = base_graph.get_edge_data(node1, node2, 0)
    speed = BUS_SPEED_NOT_HIGHWAY  # default speed if no edge data
    if edge_data is not None:
        is_school_zone: bool = edge_data.get("hazard", "") == "school_zone"
        is_highway: bool = edge_data.get("highway", "") == "motorway"
        maxspeed = edge_data.get("maxspeed", "40 mph")
        if isinstance(maxspeed, list):
            maxspeed = maxspeed[0]
            speed_limit_mph: float = float(
                maxspeed.split()[0]
            )  # in the format '30 mph'
            speed_limit = speed_limit_mph / MPH_TO_KM_PER_MIN

            if is_school_zone:
                speed = min(BUS_SPEED_SCHOOL_ZONE, speed_limit)
            elif is_highway:
                speed = speed_limit
            else:
                speed = min(BUS_SPEED_NOT_HIGHWAY, speed_limit)

        length_km = (
            edge_data["length"] if unit == "km" else (edge_data["length"] / 1000.0)
        )
        return length_km / speed
    else:
        return 0.0
