from dataclasses import replace
from functools import cache
from collections.abc import Hashable

import networkx as nx

from formulation.common.classes import Place, School, Bus, Student, Depot


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
    return 1 if m.requires_monitor or m.requires_wheelchair else 0


def depot_b(b: Bus):
    """depot of bus b"""
    return b.depot


def C_b(b: Bus):
    """capacity of bus b"""
    return b.capacity


def Wh_b(b: Bus):
    """1 if bus b has wheelchair access"""
    return 1 if b.has_wheelchair_access else 0


def R_b(b: Bus):
    """range of bus b in km"""
    return b.range_km


def h_s(s: School):
    """start time of school s in minutes from midnight"""
    return s.start_time


def slack_s(s: School):
    """required slack time for school s in minutes, same in our case"""
    return 30


def l_s(s: School):
    """latest allowable arrival time at school s in minutes from midnight"""
    return h_s(s) - slack_s(s)
