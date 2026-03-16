from functools import cache
import networkx as nx
from attr import dataclass
from enum import IntEnum


class SchoolType(IntEnum):
    ELEMENTARY = 0
    MIDDLE = 1
    HIGH = 2


@dataclass(frozen=True)
class Stop:
    name: str
    location: tuple[int, int]

    def __str__(self):
        return self.name


@dataclass(frozen=True)
class Depot:
    name: str
    location: tuple[int, int]

    def __str__(self):
        return self.name


@dataclass(frozen=True)
class School:
    name: str
    location: tuple[int, int]
    type: SchoolType
    start_time: int  # in minutes from midnight

    def __str__(self):
        return self.name


@dataclass(frozen=True)
class Bus:
    name: str
    capacity: int
    range: int
    has_wheelchair_access: bool
    depot: Depot

    def __str__(self):
        return self.name


@dataclass(frozen=True)
class Student:
    name: str
    location: tuple[int, int]
    school: School
    stop: Stop
    requires_monitor: bool
    requires_wheelchair: bool

    def __str__(self):
        return self.name


type N_TYPE = School | Depot | Stop

TAU = list(SchoolType)
"""school types"""


@cache
def make_school_copy(school: School) -> School:
    return School(
        name=school.name + " (copy)",
        location=school.location,
        type=school.type,
        start_time=school.start_time,
    )


@cache
def make_depot_copy(depot: Depot, suffix: str) -> Depot:
    return Depot(name=depot.name + f" ({suffix})", location=depot.location)


@cache
def make_depot_start_copy(depot: Depot) -> Depot:
    return make_depot_copy(depot, "start")


@cache
def make_depot_end_copy(depot: Depot) -> Depot:
    return make_depot_copy(depot, "end")


def get_accumulated_time(
    G: nx.DiGraph, path: list[tuple[int, int]], attr: str = "weight"
) -> float:
    total_time = 0.0
    for i in range(len(path) - 1):
        start_node = path[i]
        end_node = path[i + 1]
        edge_weight = G.edges[start_node, end_node][attr]
        total_time += edge_weight
    return total_time


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
    """range of bus b in miles"""
    return b.range


def h_s(s: School):
    """start time of school s in minutes from midnight"""
    return s.start_time


def slack_s(s: School):
    """required slack time for school s in minutes, same in our case"""
    return 30


def l_s(s: School):
    """latest allowable arrival time at school s in minutes from midnight"""
    return h_s(s) - slack_s(s)
