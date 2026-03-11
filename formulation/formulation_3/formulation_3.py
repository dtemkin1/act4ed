import random
from enum import IntEnum
from functools import cache
from typing import Optional

import cvxpy as cp
import networkx as nx
from attr import dataclass

# DATA CLASSES


class SchoolType(IntEnum):
    NONE = 0
    ELEMENTARY = 1
    MIDDLE = 2
    HIGH = 3


@dataclass(frozen=True)
class Stop:
    name: str
    location: tuple[int, int]


@dataclass(frozen=True)
class Depot:
    name: str
    location: tuple[int, int]


@dataclass(frozen=True)
class School:
    name: str
    location: tuple[int, int]
    type: SchoolType
    start_time: int  # in minutes from midnight


@dataclass(frozen=True)
class Bus:
    id: int
    capacity: int
    range: int
    has_wheelchair_access: bool
    depot: Depot


@dataclass(frozen=True)
class Student:
    name: str
    location: tuple[int, int]
    school: School
    stop: Stop
    requires_monitor: bool
    requires_wheelchair: bool


type N_TYPE = School | Depot | Stop


# CONSTANTS

SIZE = 10

# boarding dwell time per student
ALPHA = 0.3

# alighting dwell time per student
BETA = 0.5

# max ride time in minutes
H_RIDE = 120

# ratio of students served
PHI = 1.0

# non-negative time separation used for precedence
EPSILON = 1e-5

# small nonnegative per-round tie-breaker penalty
LAMBDA_ROUND = 1.0

# capacity multiplier for different school types
KAPPA = {
    SchoolType.NONE: 0.0,
    SchoolType.ELEMENTARY: 1.5,
    SchoolType.MIDDLE: 1.0,
    SchoolType.HIGH: 1.0,
}

# rounds a bus is allowed to have
Q: list[int] = list(range(1))

# Maximum number of sequential rounds allowed per bus (handpicked).
Q_MAX = len(Q) - 1

# HELPERS


def setup_toy_data(size: int = SIZE, multiplier: float = 1.0) -> nx.DiGraph:
    """
    Setup a SIZE x SIZE DiGraph with random edge weights

    Args:
        size (int, optional): The size of the grid graph. Defaults to SIZE.
        multiplier (float, optional): A multiplier for the edge weights. Defaults to 1.0.

    Returns:
        nx.DiGraph: The generated grid graph with random edge weights.
    """
    G: nx.DiGraph = nx.grid_2d_graph(size, size, create_using=nx.DiGraph)

    # assign random weights to the edges
    for u, v in G.edges():
        G.edges[u, v]["weight"] = random.random() * multiplier

    return G


def make_schools(num_schools: int = 5) -> list[School]:
    # add 5 schools at random positions, each either elementary, middle, or high school
    school_types = [SchoolType.ELEMENTARY, SchoolType.MIDDLE, SchoolType.HIGH]
    schools: list[School] = []
    for i in range(num_schools):
        school = School(
            name=f"School {i + 1}",
            location=(random.randint(0, SIZE - 1), random.randint(0, SIZE - 1)),
            type=school_types[i] if i < 3 else random.choice(school_types),
            start_time=random.randint(
                7 * 60, 9 * 60
            ),  # schools start between 7 and 9 AM in minutes from midnight
        )
        schools.append(school)

    return schools


def make_bus_stops(schools: Optional[list[School]] = None, num_stops: int = 20):

    if schools is None:
        schools = []

    # add 20 bus stops at random positions
    bus_stops: list[Stop] = []
    for i in range(num_stops):
        x = random.randint(0, SIZE - 1)
        y = random.randint(0, SIZE - 1)

        # make sure bus stops don't overlap with schools
        while any(school.location == (x, y) for school in schools):
            x = random.randint(0, SIZE - 1)
            y = random.randint(0, SIZE - 1)

        bus_stops.append(Stop(name=f"Bus Stop {i+1}", location=(x, y)))

    return bus_stops


def make_depots(
    num_depots: int = 1,
    schools: Optional[list[School]] = None,
    bus_stops: Optional[list[Stop]] = None,
):
    if schools is None:
        schools = []
    if bus_stops is None:
        bus_stops = []
    # depots
    depots: list[Depot] = []
    for i in range(num_depots):
        x = random.randint(0, SIZE - 1)
        y = random.randint(0, SIZE - 1)

        # make sure depots don't overlap with schools or bus stops
        while any(school.location == (x, y) for school in schools) or any(
            stop.location == (x, y) for stop in bus_stops
        ):
            x = random.randint(0, SIZE - 1)
            y = random.randint(0, SIZE - 1)

        depots.append(Depot(name=f"Depot {i+1}", location=(x, y)))

    return depots


def get_nearest_stop(location: tuple[int, int], bus_stops: list[Stop]) -> Stop:
    min_distance = float("inf")
    nearest_stop = None
    for stop in bus_stops:
        distance = (
            (location[0] - stop.location[0]) ** 2
            + (location[1] - stop.location[1]) ** 2
        ) ** 0.5
        if distance < min_distance:
            min_distance = distance
            nearest_stop = stop

    assert (
        nearest_stop is not None
    ), "There should be at least one bus stop to find the nearest stop."
    return nearest_stop


def make_students(
    schools: Optional[list[School]] = None, bus_stops: Optional[list[Stop]] = None
) -> list[Student]:
    if schools is None:
        schools = []
    if bus_stops is None:
        bus_stops = []
    # add 10 students at random positions, going to random schools
    students: list[Student] = []
    for i in range(6):
        location = (random.randint(0, SIZE - 1), random.randint(0, SIZE - 1))
        student = Student(
            name=f"Student {i + 1}",
            location=location,
            school=random.choice(schools),
            stop=get_nearest_stop(location, bus_stops),
            requires_monitor=random.choice(
                [True, False, False, False, False]
            ),  # some students require monitoring
            requires_wheelchair=random.choice(
                [True, False, False, False, False, False, False, False, False]
            ),  # some students require wheelchair accessibility
        )
        students.append(student)  # store the student object for later reference

    return students


def make_buses(depots: list[Depot]) -> list[Bus]:
    # add 5 buses with random capacities, ranges, and wheelchair access
    buses = []
    for i in range(6):
        bus = Bus(
            id=i + 1,
            capacity=10,
            range=50,
            # ensure at least buses has wheelchair access
            has_wheelchair_access=(
                True if i < 2 else random.choice([True, False, False, False])
            ),
            depot=random.choice(depots),
        )
        buses.append(bus)

    return buses


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


@cache
def p_m(m: Student):
    """pickup stop of student m"""
    return m.stop


@cache
def s_m(m: Student):
    """school of student m"""
    return m.school


@cache
def tau_m(m: Student):
    """type of school of student m"""
    return m.school.type


@cache
def f_m(m: Student):
    """1 if student m if flagged"""
    return 1 if m.requires_monitor or m.requires_wheelchair else 0


@cache
def depot_b(b: Bus):
    """depot of bus b"""
    return b.depot


@cache
def C_b(b: Bus):
    """capacity of bus b"""
    return b.capacity


@cache
def Wh_b(b: Bus):
    """1 if bus b has wheelchair access"""
    return 1 if b.has_wheelchair_access else 0


@cache
def R_b(b: Bus):
    """range of bus b in miles"""
    return b.range


def t_ij(A: dict[tuple[N_TYPE, N_TYPE], float], i: N_TYPE, j: N_TYPE) -> float:
    """travel time from node i to node j in minutes"""
    return A[(i, j)]


def d_ij(A: dict[tuple[N_TYPE, N_TYPE], float], i: N_TYPE, j: N_TYPE) -> float:
    """shortest distance from node i to node j in miles, for now same as above"""
    return t_ij(A, i, j)


@cache
def h_s(s: School):
    """start time of school s in minutes from midnight"""
    return s.start_time


@cache
def slack_s(s: School):
    """required slack time for school s in minutes, same in our case"""
    return 30


@cache
def l_s(s: School):
    """latest allowable arrival time at school s in minutes from midnight"""
    return h_s(s) - slack_s(s)


def Thorizon(schools: list[School]):
    """a safe time-horizon upper bound used to bound T_bqi and set Big-M"""
    max_arrival_time = max(l_s(school) for school in schools)
    return max_arrival_time + H_RIDE + 60  # add max ride time and some buffer


def get_accumulated_time(G: nx.DiGraph, path: list[tuple[int, int]]) -> float:
    total_time = 0.0
    for i in range(len(path) - 1):
        start_node = path[i]
        end_node = path[i + 1]
        edge_weight = G.edges[start_node, end_node]["weight"]
        total_time += edge_weight
    return total_time


def get_directed_arcs(G: nx.DiGraph, N: list[N_TYPE]):
    A: dict[tuple[N_TYPE, N_TYPE], float] = {}
    A_PATH: dict[tuple[N_TYPE, N_TYPE], list[tuple[int, int]]] = {}
    for start_node in N:
        start_location = start_node.location
        for end_node in N:
            end_location = end_node.location
            if not A.get((start_node, end_node)):
                shortest_path = nx.shortest_path(
                    G, source=start_location, target=end_location, weight="weight"
                )
                # set to the weight of the shortest path
                A[(start_node, end_node)] = get_accumulated_time(G, shortest_path)
                A_PATH[(start_node, end_node)] = shortest_path
            if not A.get((end_node, start_node)):
                shortest_path = nx.shortest_path(
                    G, source=end_location, target=start_location, weight="weight"
                )
                # set to the weight of the shortest path
                A[(end_node, start_node)] = get_accumulated_time(G, shortest_path)
                A_PATH[(end_node, start_node)] = shortest_path

    return A, A_PATH


# DECISION VARIABLES


def make_decision_variables(
    B: list[Bus],
    A: dict[tuple[N_TYPE, N_TYPE], float],
    N: list[N_TYPE],
    M: list[Student],
    S: list[School],
) -> dict[str, cp.Variable]:

    z_b = cp.Variable((len(B)), boolean=True)  # 1 if bus b is used (nonempty tour)

    z_bq = cp.Variable(
        (len(B), len(Q)), boolean=True
    )  # 1 if bus b is assigned to round q

    sigma_bq = cp.Variable(
        (len(B), len(Q)), integer=True, nonneg=True
    )  # Type of bus b in round q (elem, middle, high, etc.)

    x_bqij = cp.Variable(
        (len(B), len(Q), len(A)), boolean=True
    )  # 1 if bus b in round q travels from node i to node j

    v_bqi = cp.Variable(
        (len(B), len(Q), len(N)), boolean=True
    )  # 1 if bus b in round q visits node i

    a_mbq = cp.Variable(
        (len(M), len(B), len(Q)), boolean=True
    )  # 1 if student m is assigned to bus b in round q

    T_bqi = cp.Variable(
        (len(B), len(Q), len(N)), nonneg=True, bounds=[0, Thorizon(S)]
    )  # time bus b in round q arrives at node i

    L_bqi = cp.Variable(
        (len(B), len(Q), len(N)), integer=True, nonneg=True
    )  # load of bus b in round q after serving node i

    e_bqs = cp.Variable(
        (len(B), len(Q), len(S)), boolean=True
    )  # 1 if bus b in round q ends at school s

    r_bmon = cp.Variable(
        (len(B)), boolean=True
    )  # 1 if bus b as a monitor (ie serves a flagged student)

    return {
        "z_b": z_b,
        "z_bq": z_bq,
        "sigma_bq": sigma_bq,
        "x_bqij": x_bqij,
        "v_bqi": v_bqi,
        "a_mbq": a_mbq,
        "T_bqi": T_bqi,
        "L_bqi": L_bqi,
        "e_bqs": e_bqs,
        "r_bmon": r_bmon,
    }


def make_big_m(schools: list[School], buses: list[Bus]) -> dict[str, float]:
    # big-M for time linking
    M_TIME = Thorizon(schools)

    # big-M for capacity linking
    M_CAPACITY = max(C_b(bus) for bus in buses)

    # big-M for school type linking
    M_TYPE = max(SchoolType)

    return {
        "M_TIME": M_TIME,
        "M_CAPACITY": M_CAPACITY,
        "M_TYPE": M_TYPE,
    }


def make_objective(
    x_bqij: cp.Variable,
    z_bq: cp.Variable,
    r_bmon: cp.Variable,
    A: dict[tuple[N_TYPE, N_TYPE], float],
) -> cp.Minimize:
    return cp.Minimize(
        cp.sum(
            [
                cp.sum(
                    [
                        cp.sum(d_ij(A, *path) * x_bqij[:, :, ij])
                        for ij, path in enumerate(A.keys())
                    ]
                ),
                cp.multiply(LAMBDA_ROUND, cp.sum(z_bq)),
                cp.sum(r_bmon),
            ]
        )
    )


def constraints_student_assignments(
    a_mbq: cp.Variable,
    z_bq: cp.Variable,
    z_b: cp.Variable,
    M: list[Student],
    B: list[Bus],
) -> list[cp.Constraint]:

    constraints: list[cp.Constraint] = []

    # Each student is served by at most one (bus, round)
    for m in range(len(M)):
        constraints.append(cp.sum(a_mbq[m, :, :]) <= 1)

    # Minimum pickup / coverage requirement
    constraints.append(cp.sum(a_mbq[:, :, :]) >= PHI * len(M))

    # If student assigned to (b,q), that bus-round is used
    for m in range(len(M)):
        for b in range(len(B)):
            for q in range(len(Q)):
                constraints.append(a_mbq[m, b, q] <= z_bq[b, q])

    # Round must carry at least one student
    for b in range(len(B)):
        for q in range(len(Q)):
            constraints.append(z_bq[b, q] <= cp.sum(a_mbq[:, b, q]))

    # If bus is used, it must carry at least one student (in some round)
    for b in range(len(B)):
        constraints.append(z_b[b] <= cp.sum(a_mbq[:, b, :]))

    return constraints


def constraints_routing(
    x_bqij: cp.Variable,
    z_bq: cp.Variable,
    e_bqs: cp.Variable,
    v_bqi: cp.Variable,
    a_mbq: cp.Variable,
    B: list[Bus],
    A: dict[tuple[N_TYPE, N_TYPE], float],
    N: list[N_TYPE],
    M: list[Student],
    S: list[School],
    S_PLUS: list[School],
    P: list[Stop],
) -> list[cp.Constraint]:
    constraints: list[cp.Constraint] = []

    # ROUTING / TOUR STRUCTURE

    for b, bus in enumerate(B):
        constraints.append(
            cp.sum(
                [
                    x_bqij[b, 0, ij]
                    for ij, path in enumerate(A.keys())
                    if path[0] == make_depot_start_copy(depot_b(bus))
                ]
            )
            == z_bq[b, 0]
        )

    for b, bus in enumerate(B):
        for q in range(1, len(Q)):
            constraints.append(
                cp.sum(
                    [
                        x_bqij[b, q, ij]
                        for ij, path in enumerate(A.keys())
                        if path[0] == make_depot_start_copy(depot_b(bus))
                    ]
                )
                == 0
            )

    # round-end school selection (if next round is used, current round must end at exactly one school)
    for b in range(len(B)):
        for q in range(len(Q) - 1):
            constraints.append(cp.sum(e_bqs[b, q, :]) == z_bq[b, q + 1])
        constraints.append(cp.sum(e_bqs[b, Q_MAX, :]) == 0)

    # start of each round (first starts at depot; rest start at the previous round's end school copy s^+)
    for b in range(len(B)):
        for s in range(len(S)):
            # don't use school copy nodes for round 0
            constraints.append(
                cp.sum(
                    [
                        x_bqij[b, 0, ij]
                        for ij, path in enumerate(A.keys())
                        if path[0] in S_PLUS
                    ]
                )
                == 0
            )
            # if round q > 0 starts at school copy s^+, then round q-1 must end at school s
            for q in range(1, len(Q)):
                constraints.append(
                    cp.sum(
                        [
                            x_bqij[b, q, ij]
                            for ij, path in enumerate(A.keys())
                            if path[0] == S_PLUS[s]
                        ]
                    )
                    == e_bqs[b, q - 1, s]
                )
            for q in range(len(Q)):
                # don't let paths end at school copy s^+
                constraints.append(
                    cp.sum(
                        [
                            x_bqij[b, q, ij]
                            for ij, path in enumerate(A.keys())
                            if path[1] == S_PLUS[s]
                        ]
                    )
                    == 0
                )

    for b, bus in enumerate(B):
        for q in range(len(Q) - 1):
            constraints.append(
                cp.sum(
                    [
                        x_bqij[b, q, ij]
                        for ij, path in enumerate(A.keys())
                        if path[1] == make_depot_end_copy(depot_b(bus))
                    ]
                )
                == z_bq[b, q] - z_bq[b, q + 1]
            )
    for b, bus in enumerate(B):
        constraints.append(
            cp.sum(
                [
                    x_bqij[b, len(Q) - 1, ij]
                    for ij, path in enumerate(A.keys())
                    if path[1] == make_depot_end_copy(depot_b(bus))
                ]
            )
            == z_bq[b, len(Q) - 1]
        )

    # flow conservation at pickup stops
    for b in range(len(B)):
        for q in range(len(Q)):
            for i, node in enumerate(N):
                if node in P:
                    constraints.append(
                        cp.sum(
                            [
                                x_bqij[b, q, ij]
                                for ij, path in enumerate(A.keys())
                                if path[0] == node
                            ]
                        )
                        == v_bqi[b, q, i]
                    )
                    constraints.append(
                        cp.sum(
                            [
                                x_bqij[b, q, ij]
                                for ij, path in enumerate(A.keys())
                                if path[1] == node
                            ]
                        )
                        == v_bqi[b, q, i]
                    )

    # Stop visit only if someone is assigned from that stop in that round
    for b in range(len(B)):
        for q in range(len(Q)):
            for i, node in enumerate(N):
                if node in P:
                    constraints.append(
                        v_bqi[b, q, i]
                        <= cp.sum(
                            [
                                a_mbq[m, b, q]
                                for m, student in enumerate(M)
                                if p_m(student) == node
                            ]
                        )
                    )

    # flow conservation at schools (allow school to be end of a non-last round via e_{b,q,s})
    for b in range(len(B)):
        for q in range(len(Q)):
            for i, node in enumerate(N):
                if node in S:
                    assert isinstance(node, School)
                    constraints.append(
                        cp.sum(
                            [
                                x_bqij[b, q, ij]
                                for ij, path in enumerate(A.keys())
                                if path[1] == node
                            ]
                        )
                        == v_bqi[b, q, i]
                    )
                    constraints.append(
                        cp.sum(
                            [
                                x_bqij[b, q, ij]
                                for ij, path in enumerate(A.keys())
                                if path[0] == node
                            ]
                        )
                        == v_bqi[b, q, i] - e_bqs[b, q, S.index(node)]
                    )

    for b in range(len(B)):
        for q in range(len(Q)):
            for i, node in enumerate(N):
                if node in S or node in P:
                    constraints.append(v_bqi[b, q, i] <= z_bq[b, q])

    # Each assigned student forces visiting their pickup and their school (in the same round)
    for m, student in enumerate(M):
        for b in range(len(B)):
            for q in range(len(Q)):
                constraints.append(a_mbq[m, b, q] <= v_bqi[b, q, N.index(p_m(student))])
                constraints.append(a_mbq[m, b, q] <= v_bqi[b, q, N.index(s_m(student))])

    return constraints


def constraints_time_anchoring(
    T_bqi: cp.Variable,
    e_bqs: cp.Variable,
    a_mbq: cp.Variable,
    B: list[Bus],
    N: list[N_TYPE],
    S: list[School],
    S_PLUS: list[School],
    M: list[Student],
    M_TIME: float,
):
    constraints: list[cp.Constraint] = []
    # TIME ANCHORING
    for b, bus in enumerate(B):
        bus_depot = depot_b(bus)
        # make start copy of depot for time anchoring
        bus_start_depot = make_depot_start_copy(bus_depot)
        constraints.append(T_bqi[b, 0, N.index(bus_start_depot)] == 0)

    # round-to-round time chaining (start next round at s^+ after finishing previous round at s)
    for b in range(len(B)):
        for q in range(1, len(Q)):
            for s, school in enumerate(S):
                constraints.append(
                    T_bqi[b, q, N.index(S_PLUS[s])]
                    >= T_bqi[b, q - 1, N.index(S[s])]
                    + BETA
                    * cp.sum(
                        [
                            a_mbq[m, b, q - 1]
                            for m, student in enumerate(M)
                            if s_m(student) == school
                        ]
                    )
                    - M_TIME * (1 - e_bqs[b, q - 1, s])
                )

    return constraints


def constraints_time_propogation(
    T_bqi: cp.Variable,
    x_bqij: cp.Variable,
    a_mbq: cp.Variable,
    B: list[Bus],
    A: dict[tuple[N_TYPE, N_TYPE], float],
    N: list[N_TYPE],
    M: list[Student],
    M_TIME: float,
):
    constraints: list[cp.Constraint] = []

    # TIME PROPOGATION

    # with explicit dwell times
    for b in range(len(B)):
        for q in range(len(Q)):
            for ij, path in enumerate(A):
                constraints.append(
                    T_bqi[b, q, N.index(path[1])]
                    >= T_bqi[b, q, N.index(path[0])]
                    + t_ij(A, *path)
                    + ALPHA
                    * cp.sum(
                        [
                            a_mbq[m, b, q]
                            for m, student in enumerate(M)
                            if p_m(student) == path[0]
                        ]
                    )
                    + BETA
                    * cp.sum(
                        [
                            a_mbq[m, b, q]
                            for m, student in enumerate(M)
                            if s_m(student) == path[0]
                        ]
                    )
                    - M_TIME * (1 - x_bqij[b, q, ij])
                )

    return constraints


def constraints_school_latest_arrival(
    T_bqi: cp.Variable,
    a_mbq: cp.Variable,
    v_bqi: cp.Variable,
    B: list[Bus],
    N: list[N_TYPE],
    S: list[School],
    M: list[Student],
    M_TIME: float,
):
    constraints: list[cp.Constraint] = []

    # SCHOOL LATEST ARRIVAL

    # School latest-arrival (delivery-by) constraints
    for b in range(len(B)):
        for q in range(len(Q)):
            for school in S:
                constraints.append(
                    T_bqi[b, q, N.index(school)]
                    + BETA
                    * cp.sum(
                        [
                            a_mbq[m, b, q]
                            for m, student in enumerate(M)
                            if s_m(student) == school
                        ]
                    )
                    <= l_s(school) + M_TIME * (1 - v_bqi[b, q, N.index(school)])
                )

    return constraints
