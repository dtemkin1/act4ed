"""
1- Specify the fixed functionality f_y
2- Specify and generate the discretization of the implementation space to get (i_1,..., i_N)
3- For each i_x ∈ (i_1,..., i_N)
3.1 - Run any simulation/optimization needed and measure the compute time ts_yx
3.2 - Fill the co-design model (<- Here, we generate a catalog that contains only i_x)
3.3 - Solve the co-design query to obtain r_yx and measure the compute time tq_yx
"""

import csv
import os
from pathlib import Path
from typing import Any, NamedTuple

from gurobipy import GRB, Var, tupledict

from formulation.common.problems import ProblemDataToy
from formulation.common.toy import (
    make_buses,
    make_depots,
    make_graph,
    make_stops,
    make_schools,
    make_students,
)
from formulation.formulation_3.definition import Formulation3
from formulation.formulation_3.gurobipy import (
    build_model_from_definition,
    solve_problem,
)

CURRENT_FILE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

SAMPLING_OUTPUT = CURRENT_FILE_DIR / "outputs" / "sampling.csv"

# TOY DATA
# TODO: make these more realistic, maybe by sampling from the real data? also add some randomness to them

BUS_COSTS = 5000
BUS_RANGE = 200
BUS_CAPACITIES = 50
BUS_WHEELCHAIRS = True

GRID = make_graph((50, 50))
DEPOTS = make_depots(GRID, num_depots=1)
BUSES = make_buses(
    GRID, num_buses=10, capacities=[BUS_CAPACITIES], ranges=[BUS_RANGE], depots=DEPOTS
)
SCHOOLS = make_schools(GRID, num_schools=5)
STOPS = make_stops(GRID, num_stops=50)
STUDENTS = make_students(GRID, num_students=250, schools=SCHOOLS, stops=STOPS)

MAX_ROUNDS = 2

PROBLEM_DATA = ProblemDataToy(
    name="toy_problem",
    _base_graph=GRID,
    _schools=SCHOOLS,
    _depots=DEPOTS,
    _stops=STOPS,
    _students=STUDENTS,
    _buses=BUSES,
)


class SamplingTable(NamedTuple):
    implementation: list[int]
    """which buses were used"""

    fixed_functionality: list[int]
    """what students are transported"""

    minimal_resources: tuple[float, int, int, int, int, int]
    """total distance, rounds used, buses with monitors, total buses, total bus capacity, wheelchair accessible buses"""

    compute_time_simulation: float
    """time to run the simulation/optimization for this implementation"""

    compute_time_query: float
    """time to solve the co-design query for this implementation"""


def make_sampling_csv(rows: list[SamplingTable]) -> None:
    HEADINGS = [
        "implementation",
        "fixed_functionality",
        "minimal_resources",
        "compute_time_simulation",
        "compute_time_query",
    ]

    with open(SAMPLING_OUTPUT, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows([HEADINGS] + rows)


def get_rows() -> list[SamplingTable]:
    implementations: list[SamplingTable] = []
    phi_vals = [x / 100.0 for x in range(0, 101, 5)]

    # functionality if which students we transport
    for i, phi in enumerate(phi_vals):
        for r in range(MAX_ROUNDS):
            problem_data = PROBLEM_DATA
            formulation = Formulation3(
                problem_data=problem_data,
                rounds=r + 1,
                PHI=phi,
            )

            model, vals = build_model_from_definition(formulation)
            # dont run on local with 64 gb...
            solve_problem(model)

            if model.Status == GRB.INFEASIBLE:
                print(
                    f"Implementation {i} with phi={phi} and rounds={r+1} is infeasible."
                )
                model.close()
                continue

            B = formulation.B
            A = formulation.A
            Q = formulation.Q

            z_b: tupledict[tuple[Any], Var] = vals["z_b"]
            x_bqij: tupledict[tuple[Any, ...], Var] = vals["x_bqij"]
            r_bmon: tupledict[tuple[Any, ...], Var] = vals["r_bmon"]
            a_mbq: tupledict[tuple[Any, ...], Var] = vals["a_mbq"]

            total_distance = 0.0
            buses_with_monitors = 0
            buses_total = 0
            total_bus_capacity = 0
            buses_with_wheelchair_access = 0

            for b, bus in enumerate(B):
                if z_b[b].X > 0.5:
                    buses_total += 1
                    total_bus_capacity += bus.capacity
                    if bus.has_wheelchair_access:
                        buses_with_wheelchair_access += 1
                    if r_bmon[b].X > 0.5:
                        buses_with_monitors += 1
                for q in Q:
                    route = []
                    for ij, path in enumerate(A.keys()):
                        if x_bqij[b, q, ij].X > 0.5:
                            route.append(path)
                    route_by_start = {path[0]: path for path in route}
                    route_destinations = {path[1] for path in route}
                    route_start = next(
                        (
                            path[0]
                            for path in route
                            if path[0] not in route_destinations
                        ),
                        route[0][0] if route else None,
                    )
                    ordered_route = []
                    current_node = route_start
                    while current_node in route_by_start:
                        next_path = route_by_start[current_node]
                        ordered_route.append(next_path)
                        current_node = next_path[1]

                    route_distance = sum(
                        formulation.d_ij(*path) for path in ordered_route
                    )
                    total_distance += route_distance

            result = SamplingTable(
                implementation=[b for b in range(len(B)) if z_b[b].X > 0.5],
                fixed_functionality=[
                    student
                    for m, student in enumerate(problem_data.students)
                    if any(a_mbq[m, b, q].X > 0.5 for b in range(len(B)) for q in Q)
                ],
                minimal_resources=(
                    total_distance,
                    r + 1,
                    buses_with_monitors,
                    buses_total,
                    total_bus_capacity,
                    buses_with_wheelchair_access,
                ),
                compute_time_simulation=model.Runtime,
                # TODO: ask riccardo how to get query compute time here...
                compute_time_query=0.0,
            )
            implementations.append(result)
            model.close()

    return implementations


def main() -> None:
    rows = get_rows()
    make_sampling_csv(rows)


if __name__ == "__main__":
    main()
