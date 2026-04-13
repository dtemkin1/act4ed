from dataclasses import dataclass, field, replace
from functools import cached_property
import networkx as nx
from dataclasses_json import dataclass_json

from formulation.common import (
    Bus,
    C_b,
    Depot,
    NodeId,
    School,
    SchoolType,
    Student,
    Stop,
    Place,
    ProblemData,
    ProblemDataReal,
    l_s,
    make_depot_end_copy,
    make_depot_start_copy,
    make_school_copy,
)

MPH_TO_KM_PER_MIN = 37.282
"divide the mph value by 37.282"


@dataclass_json
@dataclass(frozen=True)
class ExperimentConfig:
    problem_data_pkl: str
    # vals
    rounds: int

    # constants
    ALPHA: float = 0.3
    """boarding dwell time per student"""
    BETA: float = 0.5
    """alighting dwell time per student"""
    H_RIDE: float = 120
    """max ride time in minutes"""
    PHI: float = 1.0
    """ratio of students served"""
    EPSILON: float = 1e-5
    """non-negative time separation used for precedence"""
    LAMBDA_ROUND: float = 1.0
    """small nonnegative per-round tie-breaker penalty"""
    KAPPA: dict[SchoolType, float] = field(
        default_factory=lambda: {
            SchoolType.E: 1.0,
            SchoolType.MS: 0.67,
            SchoolType.HS: 0.67,
        },
    )
    """capacity multiplier for different school types"""

    # FILTERS
    consider_wheelchair_students: bool = True
    consider_monitor_students: bool = True
    schools_to_consider: list[str] = field(default_factory=list)
    students_to_consider: list[str] = field(default_factory=list)
    buses_to_consider: list[str] = field(default_factory=list)

    @cached_property
    def filtered_problem_data(self):
        # filter problem data according to config
        problem_data = ProblemDataReal.load_path(self.problem_data_pkl)
        filtered_students = problem_data.students
        if self.students_to_consider:
            filtered_students = [
                student
                for student in filtered_students
                if student.id in self.students_to_consider
            ]

        if not self.consider_wheelchair_students:
            filtered_students = [
                student
                for student in filtered_students
                if not student.requires_wheelchair
            ]

        if not self.consider_monitor_students:
            filtered_students = [
                student for student in filtered_students if not student.requires_monitor
            ]

        filtered_schools = problem_data.schools
        if self.schools_to_consider:
            filtered_schools = [
                school
                for school in filtered_schools
                if school.id in self.schools_to_consider
            ]

        filtered_buses = problem_data.buses
        if self.buses_to_consider:
            filtered_buses = [
                bus for bus in filtered_buses if bus.id in self.buses_to_consider
            ]

        # create new problem data with filters applied
        new_problem_data = replace(
            problem_data,
            students=filtered_students,
            schools=filtered_schools,
            buses=filtered_buses,
        )
        del new_problem_data.service_graph
        return new_problem_data

    def make_formulation(self):
        return Formulation3(
            problem_data=self.filtered_problem_data,
            rounds=self.rounds,
            ALPHA=self.ALPHA,
            BETA=self.BETA,
            H_RIDE=self.H_RIDE,
            PHI=self.PHI,
            EPSILON=self.EPSILON,
            LAMBDA_ROUND=self.LAMBDA_ROUND,
            KAPPA=self.KAPPA,
        )


@dataclass_json
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
    ALPHA: float = 0.3
    """boarding dwell time per student"""
    BETA: float = 0.5
    """alighting dwell time per student"""
    H_RIDE: float = 120
    """max ride time in minutes"""
    PHI: float = 1.0
    """ratio of students served"""
    EPSILON: float = 1e-5
    """non-negative time separation used for precedence"""
    LAMBDA_ROUND: float = 1.0
    """small nonnegative per-round tie-breaker penalty"""
    KAPPA: dict[SchoolType, float] = field(
        default_factory=lambda: {
            SchoolType.E: 1.0,
            SchoolType.MS: 0.67,
            SchoolType.HS: 0.67,
        },
    )
    """capacity multiplier for different school types"""

    # https://malegislature.gov/Laws/GeneralLaws/PartI/TitleXIV/Chapter90/Section17
    BUS_SPEED_NOT_HIGHWAY: float = 40.0 / MPH_TO_KM_PER_MIN
    BUS_SPEED_SCHOOL_ZONE: float = 20.0 / MPH_TO_KM_PER_MIN

    # sets
    G: "nx.MultiDiGraph[NodeId]" = field(init=False)
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
    N: list[Place] = field(init=False, default_factory=list)
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
    A: dict[tuple[Place, Place], float] = field(init=False, default_factory=dict)
    """arc travel times in minutes"""
    A_PATH: dict[tuple[Place, Place], list[NodeId]] = field(
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

        self.N = self.P + self.S + self.S_PLUS + self.D_PLUS + self.D_MINUS
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
                if i == j or i.node_id == j.node_id:
                    continue

                ij_edge_data = self.problem_data.service_graph.get_edge_data(
                    i.node_id, j.node_id, key=0, default=None
                )
                if ij_edge_data is not None:
                    self.A[i, j] = ij_edge_data["length"]
                    self.A_PATH[i, j] = ij_edge_data["path"]

        self.T_horizon = self._time_horizon()

        self.M_TIME = self.T_horizon
        self.M_CAPACITY = self._max_capacity()
        self.M_TYPE = max(SchoolType)

    def t_ij(self, i: Place, j: Place) -> float:
        """travel time from node i to node j in minutes"""

        return self.d_ij(i, j) / self.BUS_SPEED_NOT_HIGHWAY

    def d_ij(self, i: Place, j: Place) -> float:
        """shortest distance from node i to node j in km"""

        return self.A[i, j]

    def _max_capacity(self):
        return max(C_b(bus) for bus in self.problem_data.buses)

    def _time_horizon(self):
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
