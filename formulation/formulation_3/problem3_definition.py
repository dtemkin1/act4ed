from dataclasses import dataclass, field
import networkx as nx

from formulation.common import (
    Bus,
    C_b,
    Depot,
    School,
    SchoolType,
    Student,
    Stop,
    Node_Type,
    ProblemData,
    l_s,
    make_depot_end_copy,
    make_depot_start_copy,
    make_school_copy,
)


@dataclass(slots=True)
class Formulation3:
    """
    formulation definition for the trip-chaining with multiple rounds formulation,
    designed to be flexible enough to allow for easy experimentation with different
    constraints and objective function terms.

    see formulation_3.ipynb for details.
    """

    problem_data: ProblemData
    # vals
    rounds: int

    # constants
    ALPHA = 0.3
    """boarding dwell time per student"""
    BETA = 0.5
    """alighting dwell time per student"""
    H_RIDE = 120
    """max ride time in minutes"""
    PHI = 1.0
    """ratio of students served"""
    EPSILON = 1e-5
    """non-negative time separation used for precedence"""
    LAMBDA_ROUND = 1.0
    """small nonnegative per-round tie-breaker penalty"""
    KAPPA = {SchoolType.E: 1.0, SchoolType.MS: 0.67, SchoolType.HS: 0.67}
    """capacity multiplier for different school types"""

    # sets
    G: nx.MultiDiGraph = field(init=False)
    """road network graph"""
    P: list[Stop] = field(init=False, default_factory=list)
    """pickup stop nodes"""
    S: list[School] = field(init=False, default_factory=list)
    """school nodes"""
    S_PLUS: list[School] = field(init=False, default_factory=list)
    """school start-copy nodes"""
    D: list[Depot] = field(init=False, default_factory=list)
    """depot nodes"""
    D_PLUS: list[Depot] = field(init=False, default_factory=list)
    """depot start-copy nodes"""
    D_MINUS: list[Depot] = field(init=False, default_factory=list)
    """depot end-copy nodes"""
    N: list[Node_Type] = field(init=False, default_factory=list)
    """all nodes"""
    B: list[Bus] = field(init=False, default_factory=list)
    """buses"""
    M: list[Student] = field(init=False, default_factory=list)
    """students"""
    F: list[Student] = field(init=False, default_factory=list)
    """students needing monitor, eg special education or wheelchair"""
    W: list[Student] = field(init=False, default_factory=list)
    """students needing wheelchair access"""

    # utility
    A: dict[tuple[Node_Type, Node_Type], float] = field(
        init=False, default_factory=dict
    )
    """arc travel times in minutes"""
    A_PATH: dict[tuple[Node_Type, Node_Type], list[list[Node_Type]]] = field(
        init=False, default_factory=dict
    )
    """arc shortest paths"""
    Q: list[int] = field(init=False, default_factory=list)
    """rounds a bus is allowed to have"""
    Q_MAX: int = field(init=False)
    """maximum number of sequential rounds allowed per bus (handpicked)."""

    # big M for constraints
    M_TIME: float = field(init=False)
    """big M for time constraints, set to max possible travel time in minutes"""
    M_CAPACITY: int = field(init=False)
    """big M for capacity constraints, set to max bus capacity"""
    M_TYPE: SchoolType = field(init=False)
    """big M for bus type constraints, set to max school type"""

    # derived vals
    T_horizon: float = field(init=False)
    """time horizon in minutes, used to bound T_bqi and set big M for time constraints"""

    def __post_init__(self):
        self.G = self.problem_data.service_graph
        self.P = self.problem_data.stops
        self.S = self.problem_data.schools
        self.S_PLUS = [make_school_copy(school) for school in self.problem_data.schools]
        self.D = self.problem_data.depots
        self.D_PLUS = [
            make_depot_start_copy(depot) for depot in self.problem_data.depots
        ]
        self.D_MINUS = [
            make_depot_end_copy(depot) for depot in self.problem_data.depots
        ]
        self.N = self.P + self.S + self.S_PLUS + self.D + self.D_PLUS + self.D_MINUS
        self.B = self.problem_data.buses
        self.M = self.problem_data.students
        self.F = [
            student
            for student in self.problem_data.students
            if student.requires_monitor or student.requires_wheelchair
        ]
        self.W = [
            student
            for student in self.problem_data.students
            if student.requires_wheelchair
        ]

        self.Q = list(range(self.rounds))
        self.Q_MAX = self.rounds - 1

        self.A = {}
        self.A_PATH = {}
        for i in self.N:
            for j in self.N:
                if i != j:
                    self.A[(i, j)] = self.problem_data.service_graph[i.node_id][
                        j.node_id
                    ][0]["length"]
                    self.A[(j, i)] = self.problem_data.service_graph[j.node_id][
                        i.node_id
                    ][0]["length"]
                    self.A_PATH[(i, j)] = self.problem_data.service_graph[i.node_id][
                        j.node_id
                    ][0]["path"]
                    self.A_PATH[(j, i)] = self.problem_data.service_graph[j.node_id][
                        i.node_id
                    ][0]["path"]

        self.T_horizon = self._T_horizon()

        self.M_TIME = self.T_horizon
        self.M_CAPACITY = max(C_b(bus) for bus in self.problem_data.buses)
        self.M_TYPE = max(SchoolType)

    def t_ij(self, i: Node_Type, j: Node_Type) -> float:
        """travel time from node i to node j in minutes"""
        return self.A[(i, j)]

    def d_ij(self, i: Node_Type, j: Node_Type) -> float:
        """shortest distance from node i to node j in miles, for now same as above"""
        return self.t_ij(i, j)

    def _T_horizon(self):
        """a safe time-horizon upper bound used to bound T_bqi and set Big-M"""
        max_arrival_time = max(l_s(school) for school in self.problem_data.schools)
        return max_arrival_time + self.H_RIDE + 60  # add max ride time and some buffer

    def C_CAP_B(self, b: Bus):
        """capacity upper bound for bus b, used in big-M constraints"""
        return C_b(b) * max(
            self.KAPPA[school.type] for school in self.problem_data.schools
        )


if __name__ == "__main__":
    pass
