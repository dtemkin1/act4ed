from dataclasses import replace
import os
from pathlib import Path
from typing import Any
from gurobipy import GRB, Var, tupledict
import yaml


from formulation.common import ProblemDataToy
from formulation.formulation_3.formulation3_gurobipy import (
    build_model_from_definition,
    solve_problem,
)
from formulation.formulation_3.problem3_definition import Formulation3
from formulation.toy_network import (
    make_buses,
    make_graph,
    make_depots,
    make_schools,
    make_stops,
    make_students,
)

CURRENT_FILE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

FLEET_OUTPUT = CURRENT_FILE_DIR / ".." / "outputs" / "fleet_bus_count.dpc.yaml"
ROUTING_OUTPUT = CURRENT_FILE_DIR / ".." / "outputs" / "routing_service.dpc.yaml"

BUS_COSTS = 50000
BUS_RANGE = 300
BUS_CAPACITIES = 50
BUS_WHEELCHAIRS = True

GRID = make_graph((10, 10))
DEPOTS = make_depots(GRID, num_depots=1)
BUSES = make_buses(
    GRID, num_buses=10, capacities=[BUS_CAPACITIES], ranges=[BUS_RANGE], depots=DEPOTS
)
SCHOOLS = make_schools(GRID, num_schools=2)
STOPS = make_stops(GRID, num_stops=4)
STUDENTS = make_students(GRID, num_students=10, schools=SCHOOLS, stops=STOPS)

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


def generate_fleet_yaml():
    buses = PROBLEM_DATA.buses
    implementations: dict[str, dict[str, list[str]]] = {}

    for i in range(len(buses)):
        guideline_name = f"guideline_{i}"
        implementations[guideline_name] = {
            "f_max": [
                # fleet size
                str(i),
                # student capacity
                str(sum(bus.capacity for bus in buses[:i])),
                # wheelchair buses
                str(sum(1 for bus in buses[:i] if bus.has_wheelchair_access)),
            ],
            "r_min": [
                str(sum(BUS_COSTS for _ in buses[:i])) + " USD",
            ],
        }

    return {
        "F": ["Nat", "Nat", "Nat"],
        "R": ["USD"],
        "implementations": implementations,
    }


def get_all_combos(num: int) -> list[list[int]]:
    if num == 0:
        return [[]]
    smaller_combos = get_all_combos(num - 1)
    return smaller_combos + [combo + [num - 1] for combo in smaller_combos]


def generate_routing_yaml():
    implementations: dict[str, dict[str, list[str]]] = {}
    groups_of_students = get_all_combos(len(PROBLEM_DATA.students))

    for i, group in enumerate(groups_of_students):
        for r in range(MAX_ROUNDS):
            guideline_name = f"guideline_{i}_{r}"
            problem_data = replace(
                PROBLEM_DATA, _students=[PROBLEM_DATA.students[j] for j in group]
            )
            formulation = Formulation3(
                problem_data=problem_data,
                rounds=r + 1,
            )

            model, vals = build_model_from_definition(formulation)
            solve_problem(model)

            if model.Status == GRB.INFEASIBLE:
                continue

            B = formulation.B
            A = formulation.A
            Q = formulation.Q

            z_b: tupledict[tuple[Any], Var] = vals["z_b"]
            x_bqij: tupledict[tuple[Any, ...], Var] = vals["x_bqij"]
            r_bmon: tupledict[tuple[Any, ...], Var] = vals["r_bmon"]

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

            implementations[guideline_name] = {
                "f_max": [str(len(group))],
                "r_min": [
                    # distance
                    str(total_distance) + " km",
                    # rounds used
                    str(r + 1),
                    # buses with monitors
                    str(buses_with_monitors),
                    # buses total
                    str(buses_total),
                    # total bus capacity
                    str(total_bus_capacity),
                    # buses with wheelchair access
                    str(buses_with_wheelchair_access),
                ],
            }

    return {
        "F": ["Nat"],  # students served
        "R": [
            "km",  # distance
            "Nat",  # rounds used
            "Nat",  # buses with monitors
            "Nat",  # buses total
            "Nat",  # total bus capacity
            "Nat",  # buses with wheelchair access
        ],
        "implementations": implementations,
    }


def main() -> None:
    fleet_yaml = generate_fleet_yaml()
    routing_yaml = generate_routing_yaml()

    with open(FLEET_OUTPUT, "w+", encoding="utf-8") as f:
        yaml.dump(fleet_yaml, f)

    with open(ROUTING_OUTPUT, "w+", encoding="utf-8") as f:
        yaml.dump(routing_yaml, f)


if __name__ == "__main__":
    main()
