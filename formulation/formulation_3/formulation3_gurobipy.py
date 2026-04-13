from typing import Any
import datetime as dt
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB, GurobiError, tupledict, Var
from matplotlib import pyplot as plt
import osmnx as ox

from loguru import logger

from formulation.common import (
    TAU,
    C_b,
    NodeId,
    ProblemDataReal,
    ProblemDataToy,
    R_b,
    Wh_b,
    depot_b,
    f_m,
    l_s,
    make_depot_end_copy,
    make_depot_start_copy,
    p_m,
    s_m,
    tau_m,
)
from formulation.formulation_3.problem3_definition import Formulation3

METERS_PER_MILE = 1609.344


def build_model_from_definition(
    problem: Formulation3,
) -> tuple[gp.Model, dict[str, Any]]:
    B = problem.B
    M = problem.M
    S = problem.S
    S_PLUS = problem.S_PLUS
    P = problem.P
    A = problem.A
    N = problem.N
    Q = problem.Q
    Q_MAX = problem.Q_MAX
    PHI = problem.PHI
    LAMBDA_ROUND = problem.LAMBDA_ROUND
    BETA = problem.BETA
    M_TIME = problem.M_TIME
    M_CAPACITY = problem.M_CAPACITY
    ALPHA = problem.ALPHA
    t_ij = problem.t_ij
    d_ij = problem.d_ij
    T_horizon = problem.T_horizon
    EPSILON = problem.EPSILON
    H_RIDE = problem.H_RIDE
    KAPPA = problem.KAPPA
    C_CAP_B = problem.C_CAP_B

    B_idx = range(len(B))
    M_idx = range(len(M))
    S_idx = range(len(S))
    Q_idx = range(len(Q))
    N_idx = range(len(N))
    TAU_idx = range(len(TAU))

    A_list = list(A.keys())
    A_idx = range(len(A_list))

    node_to_idx = {node: idx for idx, node in enumerate(N)}
    school_to_idx = {school: idx for idx, school in enumerate(S)}
    school_type_to_idx = {school_type: idx for idx, school_type in enumerate(TAU)}

    arcs_from_node = {i: [] for i in N_idx}
    arcs_to_node = {i: [] for i in N_idx}
    for arc_idx, path in enumerate(A_list):
        start_idx = node_to_idx[path[0]]
        end_idx = node_to_idx[path[1]]
        arcs_from_node[start_idx].append(arc_idx)
        arcs_to_node[end_idx].append(arc_idx)

    pickup_node_indices = [node_to_idx[stop] for stop in P]
    school_node_indices = [node_to_idx[school] for school in S]
    school_copy_node_indices = [node_to_idx[school_copy] for school_copy in S_PLUS]

    pickup_students_by_node = {i: [] for i in N_idx}
    dropoff_students_by_node = {i: [] for i in N_idx}
    students_by_school = {s: [] for s in S_idx}
    p_idx: list[int] = []
    s_idx: list[int] = []
    tau_idx_for_student: list[int] = []
    flagged_student_indices: list[int] = []
    wheelchair_student_indices: list[int] = []

    for m, student in enumerate(M):
        pickup_node_idx = node_to_idx[p_m(student)]
        school_node_idx = node_to_idx[s_m(student)]
        school_idx = school_to_idx[s_m(student)]

        p_idx.append(pickup_node_idx)
        s_idx.append(school_node_idx)
        tau_idx_for_student.append(school_type_to_idx[tau_m(student)])

        pickup_students_by_node[pickup_node_idx].append(m)
        dropoff_students_by_node[school_node_idx].append(m)
        students_by_school[school_idx].append(m)

        if f_m(student) == 1:
            flagged_student_indices.append(m)
        if student.requires_wheelchair:
            wheelchair_student_indices.append(m)

    depot_start_arcs_by_bus = {
        b: arcs_from_node[node_to_idx[make_depot_start_copy(depot_b(bus))]]
        for b, bus in enumerate(B)
    }
    depot_start_in_arcs_by_bus = {
        b: arcs_to_node[node_to_idx[make_depot_start_copy(depot_b(bus))]]
        for b, bus in enumerate(B)
    }
    depot_end_arcs_by_bus = {
        b: arcs_to_node[node_to_idx[make_depot_end_copy(depot_b(bus))]]
        for b, bus in enumerate(B)
    }
    depot_end_out_arcs_by_bus = {
        b: arcs_from_node[node_to_idx[make_depot_end_copy(depot_b(bus))]]
        for b, bus in enumerate(B)
    }
    school_copy_out_arcs = {
        s: arcs_from_node[school_copy_node_indices[s]] for s in S_idx
    }
    school_copy_in_arcs = {s: arcs_to_node[school_copy_node_indices[s]] for s in S_idx}

    distance_by_arc = {arc_idx: d_ij(*path) for arc_idx, path in enumerate(A_list)}
    travel_time_by_arc = {arc_idx: t_ij(*path) for arc_idx, path in enumerate(A_list)}
    arc_start_idx = {
        arc_idx: node_to_idx[path[0]] for arc_idx, path in enumerate(A_list)
    }
    arc_end_idx = {arc_idx: node_to_idx[path[1]] for arc_idx, path in enumerate(A_list)}
    service_node_indices = pickup_node_indices + school_node_indices
    range_limit_by_bus = {b: R_b(bus) * METERS_PER_MILE for b, bus in enumerate(B)}
    capacity_by_bus = {b: C_b(bus) for b, bus in enumerate(B)}
    cap_upper_by_bus = {b: C_CAP_B(bus) for b, bus in enumerate(B)}

    model = gp.Model("formulation3_gurobi")
    model.Params.OutputFlag = 1
    model.Params.LogToConsole = 1

    try:
        model.Params.LogFile = f"gurobi-{dt.datetime.now()}.log"
    except GurobiError:
        logger.warning("could not set gurobi log file")

    logger.info("model initialized :3")

    z_b = model.addVars(B_idx, vtype=GRB.BINARY, name="z_b")
    logger.info("z_b vars added")
    z_bq = model.addVars(B_idx, Q_idx, vtype=GRB.BINARY, name="z_bq")
    logger.info("z_bq vars added")
    y_bqtau = model.addVars(B_idx, Q_idx, TAU_idx, vtype=GRB.BINARY, name="y_bqtau")
    logger.info("y_bqtau vars added")
    v_bqi = model.addVars(B_idx, Q_idx, N_idx, vtype=GRB.BINARY, name="v_bqi")
    logger.info("v_bqi vars added")
    a_mbq = model.addVars(M_idx, B_idx, Q_idx, vtype=GRB.BINARY, name="a_mbq")
    logger.info("a_mbq vars added")
    T_bqi = model.addVars(
        B_idx,
        Q_idx,
        N_idx,
        vtype=GRB.CONTINUOUS,
        lb=0.0,
        ub=T_horizon,
        name="T_bqi",
    )
    logger.info("T_bqi vars added")
    L_bqi = model.addVars(
        B_idx,
        Q_idx,
        N_idx,
        vtype=GRB.INTEGER,
        lb=0.0,
        name="L_bqi",
    )
    logger.info("L_bqi vars added")
    e_bqs = model.addVars(B_idx, Q_idx, S_idx, vtype=GRB.BINARY, name="e_bqs")
    logger.info("e_bqs vars added")
    r_bmon = model.addVars(B_idx, vtype=GRB.BINARY, name="r_bmon")
    logger.info("r_bmon vars added")
    logger.info(f"vars we need to make: {len(B_idx) * len(Q_idx) * len(A_idx)} :(")
    try:
        x_bqij = model.addVars(B_idx, Q_idx, A_idx, vtype=GRB.BINARY, name="x_bqij")
        logger.info("x_bqij vars added")
    except Exception as exc:
        raise Exception("var x_bqij failed </3") from exc

    logger.info("vars added to model")

    students_per_bus_round = {
        (b, q): a_mbq.sum("*", b, q) for b in B_idx for q in Q_idx
    }
    students_per_bus = {(b): a_mbq.sum("*", b, "*") for b in B_idx}
    pickup_assignments = {
        (b, q, node_idx): gp.quicksum(
            a_mbq[m, b, q] for m in pickup_students_by_node[node_idx]
        )
        for b in B_idx
        for q in Q_idx
        for node_idx in pickup_node_indices
    }
    dropoff_assignments = {
        (b, q, node_idx): gp.quicksum(
            a_mbq[m, b, q] for m in dropoff_students_by_node[node_idx]
        )
        for b in B_idx
        for q in Q_idx
        for node_idx in school_node_indices
    }
    school_assignments = {
        (b, q, s): gp.quicksum(a_mbq[m, b, q] for m in students_by_school[s])
        for b in B_idx
        for q in Q_idx
        for s in S_idx
    }
    type_capacity_multiplier = {
        (b, q): gp.quicksum(
            KAPPA[TAU[tau_idx]] * y_bqtau[b, q, tau_idx] for tau_idx in TAU_idx
        )
        for b in B_idx
        for q in Q_idx
    }
    logger.info("helpers made to help")

    model.setObjective(
        gp.quicksum(
            distance_by_arc[arc_idx] * x_bqij[b, q, arc_idx]
            for b in B_idx
            for q in Q_idx
            for arc_idx in A_idx
        )
        + LAMBDA_ROUND * gp.quicksum(z_bq[b, q] for b in B_idx for q in Q_idx)
        + gp.quicksum(r_bmon[b] for b in B_idx),
        GRB.MINIMIZE,
    )
    logger.info("objective set")

    # STUDENT ASSIGNMENT
    model.addConstrs((a_mbq.sum(m, "*", "*") <= 1 for m in M_idx))

    model.addConstr(a_mbq.sum("*", "*", "*") >= PHI * len(M))

    model.addConstrs(
        (a_mbq[m, b, q] <= z_bq[b, q] for m in M_idx for b in B_idx for q in Q_idx)
    )

    model.addConstrs(
        (z_bq[b, q] <= students_per_bus_round[b, q] for b in B_idx for q in Q_idx)
    )

    model.addConstrs((z_b[b] <= students_per_bus[b] for b in B_idx))
    logger.info("main constraints added")

    # ROUTING / TOUR STRUCTURE
    for b in B_idx:
        model.addConstr(
            gp.quicksum(x_bqij[b, 0, arc_idx] for arc_idx in depot_start_arcs_by_bus[b])
            == z_bq[b, 0]
        )

    for b in B_idx:
        for q in range(1, len(Q)):
            model.addConstr(
                gp.quicksum(
                    x_bqij[b, q, arc_idx] for arc_idx in depot_start_arcs_by_bus[b]
                )
                == 0
            )
    for b in B_idx:
        for q in Q_idx:
            model.addConstr(
                gp.quicksum(
                    x_bqij[b, q, arc_idx] for arc_idx in depot_start_in_arcs_by_bus[b]
                )
                == 0
            )
            model.addConstr(
                gp.quicksum(
                    x_bqij[b, q, arc_idx] for arc_idx in depot_end_out_arcs_by_bus[b]
                )
                == 0
            )

    for b in B_idx:
        for q in range(len(Q) - 1):
            model.addConstr(
                gp.quicksum(e_bqs[b, q, s] for s in S_idx) == z_bq[b, q + 1]
            )
        model.addConstr(gp.quicksum(e_bqs[b, Q_MAX, s] for s in S_idx) == 0)

    for b in B_idx:
        model.addConstr(
            gp.quicksum(
                x_bqij[b, 0, arc_idx]
                for s in S_idx
                for arc_idx in school_copy_out_arcs[s]
            )
            == 0
        )
        for s in S_idx:
            for q in range(1, len(Q)):
                model.addConstr(
                    gp.quicksum(
                        x_bqij[b, q, arc_idx] for arc_idx in school_copy_out_arcs[s]
                    )
                    == e_bqs[b, q - 1, s]
                )
            for q in Q_idx:
                model.addConstr(
                    gp.quicksum(
                        x_bqij[b, q, arc_idx] for arc_idx in school_copy_in_arcs[s]
                    )
                    == 0
                )

    for b in B_idx:
        for q in range(len(Q) - 1):
            model.addConstr(
                gp.quicksum(
                    x_bqij[b, q, arc_idx] for arc_idx in depot_end_arcs_by_bus[b]
                )
                == z_bq[b, q] - z_bq[b, q + 1]
            )
        model.addConstr(
            gp.quicksum(
                x_bqij[b, len(Q) - 1, arc_idx] for arc_idx in depot_end_arcs_by_bus[b]
            )
            == z_bq[b, len(Q) - 1]
        )

    for b in B_idx:
        for q in Q_idx:
            for node_idx in pickup_node_indices:
                model.addConstr(
                    gp.quicksum(
                        x_bqij[b, q, arc_idx] for arc_idx in arcs_from_node[node_idx]
                    )
                    == v_bqi[b, q, node_idx]
                )
                model.addConstr(
                    gp.quicksum(
                        x_bqij[b, q, arc_idx] for arc_idx in arcs_to_node[node_idx]
                    )
                    == v_bqi[b, q, node_idx]
                )

    for node_idx in pickup_node_indices:
        for b in B_idx:
            for q in Q_idx:
                model.addConstr(
                    v_bqi[b, q, node_idx] <= pickup_assignments[b, q, node_idx]
                )

    for b in B_idx:
        for q in Q_idx:
            for s, school in enumerate(S):
                school_node_idx = school_node_indices[s]
                model.addConstr(
                    gp.quicksum(
                        x_bqij[b, q, arc_idx]
                        for arc_idx in arcs_to_node[school_node_idx]
                    )
                    == v_bqi[b, q, school_node_idx]
                )
                model.addConstr(
                    gp.quicksum(
                        x_bqij[b, q, arc_idx]
                        for arc_idx in arcs_from_node[school_node_idx]
                    )
                    == v_bqi[b, q, school_node_idx] - e_bqs[b, q, s]
                )

    for b in B_idx:
        for q in Q_idx:
            for node_idx in service_node_indices:
                model.addConstr(v_bqi[b, q, node_idx] <= z_bq[b, q])

    for m in M_idx:
        pickup_node_idx = p_idx[m]
        school_node_idx = s_idx[m]
        for b in B_idx:
            for q in Q_idx:
                model.addConstr(a_mbq[m, b, q] <= v_bqi[b, q, pickup_node_idx])
                model.addConstr(a_mbq[m, b, q] <= v_bqi[b, q, school_node_idx])

    logger.info("routing/tour structure constraints made")

    # TIME ANCHORING
    for b, bus in enumerate(B):
        start_depot_idx = node_to_idx[make_depot_start_copy(depot_b(bus))]
        model.addConstr(T_bqi[b, 0, start_depot_idx] == 0)

    for b in B_idx:
        for q in range(1, len(Q)):
            for s, school in enumerate(S):
                model.addConstr(
                    T_bqi[b, q, school_copy_node_indices[s]]
                    >= T_bqi[b, q - 1, school_node_indices[s]]
                    + BETA * school_assignments[b, q - 1, s]
                    - M_TIME * (1 - e_bqs[b, q - 1, s])
                )
    logger.info("time anchoring constraints")

    # TIME PROPAGATION
    for b in B_idx:
        for q in Q_idx:
            for arc_idx in A_idx:
                start_idx = arc_start_idx[arc_idx]
                end_idx = arc_end_idx[arc_idx]
                model.addConstr(
                    T_bqi[b, q, end_idx]
                    >= T_bqi[b, q, start_idx]
                    + travel_time_by_arc[arc_idx]
                    + ALPHA * pickup_assignments.get((b, q, start_idx), 0)
                    + BETA * dropoff_assignments.get((b, q, start_idx), 0)
                    - M_TIME * (1 - x_bqij[b, q, arc_idx])
                )
    logger.info("time propogation constraints")

    # SCHOOL LATEST ARRIVAL
    for b in B_idx:
        for q in Q_idx:
            for s, school in enumerate(S):
                school_node_idx = school_node_indices[s]
                model.addConstr(
                    T_bqi[b, q, school_node_idx] + BETA * school_assignments[b, q, s]
                    <= l_s(school) + M_TIME * (1 - v_bqi[b, q, school_node_idx])
                )
    logger.info("school latest arrival constraints")

    # PICKUP BEFORE DROPOFF AND MAX RIDE TIME
    for m in M_idx:
        pickup_node_idx = p_idx[m]
        school_node_idx = s_idx[m]
        for b in B_idx:
            for q in Q_idx:
                model.addConstr(
                    T_bqi[b, q, school_node_idx]
                    >= T_bqi[b, q, pickup_node_idx]
                    + EPSILON
                    - M_TIME * (1 - a_mbq[m, b, q])
                )
                model.addConstr(
                    T_bqi[b, q, school_node_idx] - T_bqi[b, q, pickup_node_idx]
                    <= H_RIDE + M_TIME * (1 - a_mbq[m, b, q])
                )
    logger.info("add pickup before dropoff constraints")

    # DISTANCE RANGE CONSTRAINTS
    for b, bus in enumerate(B):
        model.addConstr(
            gp.quicksum(
                distance_by_arc[arc_idx] * x_bqij[b, q, arc_idx]
                for q in Q_idx
                for arc_idx in A_idx
            )
            <= range_limit_by_bus[b] * z_b[b]
        )
    logger.info("distance range constraints")

    # LOAD / CAPACITY CONSTRAINTS PER ROUND
    for b, bus in enumerate(B):
        depot_start_idx = node_to_idx[make_depot_start_copy(depot_b(bus))]
        depot_end_idx = node_to_idx[make_depot_end_copy(depot_b(bus))]
        model.addConstr(L_bqi[b, 0, depot_start_idx] == 0)

        for q in Q_idx:
            model.addConstr(L_bqi[b, q, depot_end_idx] == 0)

        for q in Q_idx:
            for school_copy_idx in school_copy_node_indices:
                model.addConstr(L_bqi[b, q, school_copy_idx] == 0)

        for q in Q_idx:
            for arc_idx in A_idx:
                start_idx = arc_start_idx[arc_idx]
                end_idx = arc_end_idx[arc_idx]
                load_change = pickup_assignments.get(
                    (b, q, end_idx), 0
                ) - dropoff_assignments.get((b, q, end_idx), 0)
                model.addConstr(
                    L_bqi[b, q, end_idx]
                    >= L_bqi[b, q, start_idx]
                    + load_change
                    - M_CAPACITY * (1 - x_bqij[b, q, arc_idx])
                )
                model.addConstr(
                    L_bqi[b, q, end_idx]
                    <= L_bqi[b, q, start_idx]
                    + load_change
                    + M_CAPACITY * (1 - x_bqij[b, q, arc_idx])
                )

        for q in Q_idx:
            for node_idx in N_idx:
                model.addConstr(
                    L_bqi[b, q, node_idx]
                    <= capacity_by_bus[b] * type_capacity_multiplier[b, q]
                )

        for q in Q_idx:
            for s, _school in enumerate(S):
                model.addConstr(
                    L_bqi[b, q, school_node_indices[s]]
                    <= cap_upper_by_bus[b] * (1 - e_bqs[b, q, s])
                )
    logger.info("load/capacity constraints")

    # MONITOR FEASIBILITY PER BUS
    for b in B_idx:
        model.addConstr(
            r_bmon[b]
            <= gp.quicksum(
                a_mbq[m, b, q] for m in flagged_student_indices for q in Q_idx
            )
        )
        for m in flagged_student_indices:
            for q in Q_idx:
                model.addConstr(r_bmon[b] >= a_mbq[m, b, q])
    logger.info("monitor constraints")

    # WHEELCHAIR CONSTRAINTS
    for m in wheelchair_student_indices:
        for b, bus in enumerate(B):
            for q in Q_idx:
                model.addConstr(a_mbq[m, b, q] <= Wh_b(bus))
    logger.info("wheelchair constraints")

    # SCHOOL LEVEL
    for b in B_idx:
        for q in Q_idx:
            model.addConstr(y_bqtau.sum(b, q, "*") == z_bq[b, q])
            for m in M_idx:
                model.addConstr(a_mbq[m, b, q] <= y_bqtau[b, q, tau_idx_for_student[m]])
    logger.info("school level constraints")

    meta = {
        "A_list": A_list,
        "N": N,
        "B": B,
        "M": M,
        "Q": Q,
        "S": S,
        "TAU": TAU,
        "node_to_idx": node_to_idx,
        "p_idx": p_idx,
        "s_idx": s_idx,
    }

    return model, {
        "z_b": z_b,
        "z_bq": z_bq,
        "y_bqtau": y_bqtau,
        "x_bqij": x_bqij,
        "v_bqi": v_bqi,
        "a_mbq": a_mbq,
        "T_bqi": T_bqi,
        "L_bqi": L_bqi,
        "e_bqs": e_bqs,
        "r_bmon": r_bmon,
        "meta": meta,
    }


def solve_problem(model: gp.Model) -> None:
    print("Solving MILP")
    model.optimize()

    print()
    print("Status:", _gurobi_status_name(model.Status))
    if model.SolCount > 0:
        print("Objective:", model.ObjVal)
    else:
        print("Objective: unavailable")


def _gurobi_status_name(status: int) -> str:
    status_names = {
        GRB.LOADED: "LOADED",
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.CUTOFF: "CUTOFF",
        GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
    }
    return status_names.get(status, str(status))


def make_report(prob: gp.Model, formulation: Formulation3, model_vars: dict[str, Any]):
    # Extract and print the route

    B = formulation.B
    M = formulation.M
    S = formulation.S
    A = formulation.A
    Q = formulation.Q

    # z_b = model_vars["z_b"]
    z_bq: tupledict[tuple[Any, ...], Var] = model_vars["z_bq"]
    y_bqtau: tupledict[tuple[Any, ...], Var] = model_vars["y_bqtau"]
    x_bqij: tupledict[tuple[Any, ...], Var] = model_vars["x_bqij"]
    # v_bqi = model_vars["v_bqi"]
    a_mbq: tupledict[tuple[Any, ...], Var] = model_vars["a_mbq"]
    # T_bqi = model_vars["T_bqi"]
    # L_bqi = model_vars["L_bqi"]
    # e_bqs = model_vars["e_bqs"]
    r_bmon: tupledict[tuple[Any, ...], Var] = model_vars["r_bmon"]

    result_string = ""

    if prob.status not in [GRB.OPTIMAL]:
        result_string += (
            f"⚠️ Solver did not find a feasible solution (status={prob.status}).\n"
        )
    else:
        for b, bus in enumerate(B):
            result_string += f"{bus} (capacity {C_b(bus)}, range {R_b(bus)}, wheelchair access {Wh_b(bus) == 1}, monitor needed: {r_bmon[b].X > 0.5})\n"
            for q in range(len(Q)):
                assert z_bq[b, q].X is not None
                if z_bq[b, q].X > 0.5:
                    result_string += f"  Round {q}:\n"
                    route = []
                    students_on_bus = []
                    schools_served = []
                    for ij, path in enumerate(A.keys()):
                        if x_bqij[b, q, ij].X > 0.5:
                            route.append(path)
                            for node in path:
                                if (
                                    node in S
                                    and node not in schools_served
                                    and any(
                                        a_mbq[m, b, q].X > 0.5 and s_m(M[m]) == node
                                        for m in range(len(M))
                                    )
                                ):
                                    schools_served.append(node)
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

                    for m, student in enumerate(M):
                        if a_mbq[m, b, q].X > 0.5:
                            students_on_bus.append(student)
                    result_string += f"    Total travel distance: {sum(formulation.d_ij(*path) for path in ordered_route):.2f} meters\n"
                    school_type = TAU[
                        max(
                            (tau for tau in range(len(TAU))),
                            key=lambda tau: (
                                y_bqtau[b, q, tau].X
                                if y_bqtau[b, q, tau].X is not None
                                else 0
                            ),
                        )
                    ]
                    result_string += f"    Bus type: {school_type.name}\n"
                    result_string += f"    Students on bus this round:\n      {'\n      '.join(str(student) for student in students_on_bus)}\n"
                    result_string += f"    Schools served:\n      {'\n      '.join(str(school) for school in schools_served)}\n"

                    # Sort route by travel time from depot start
                    # depot_start = make_depot_start_copy(depot_b(bus))

                    # make sure route is in right order, where end of one path is the start of the next
                    # ordered_route = []
                    # current_node = depot_start
                    # while len(ordered_route) < len(route):
                    #     for path in route:
                    #         if path[0] == current_node:
                    #             ordered_route.append(path)
                    #             current_node = path[1]
                    #             break

                    result_string += "    Route:\n"
                    for path in ordered_route:
                        result_string += f"      {path[0]} -> {path[1]}\n"

    return result_string


def plot_bus_routes(
    prob: gp.Model,
    formulation: Formulation3,
    model_vars: dict[str, Any],
    save_path: Path | None = None,
    per_round: bool = False,
):
    # raise NotImplementedError("plotting not implemented yet :(")

    G = formulation.G
    B = formulation.B
    A = formulation.A
    A_PATH = formulation.A_PATH
    Q = formulation.Q

    schools = formulation.problem_data.schools
    bus_stops = formulation.problem_data.stops
    depots = formulation.problem_data.depots
    school_colors = {
        school.type: color for school, color in zip(schools, ["green", "blue", "red"])
    }

    z_bq: tupledict[tuple[Any, ...], Var] = model_vars["z_bq"]
    x_bqij: tupledict[tuple[Any, ...], Var] = model_vars["x_bqij"]

    problem_data = formulation.problem_data

    if isinstance(problem_data, ProblemDataReal):
        graph = problem_data.osm_graph
        pos: dict[NodeId, tuple[float, float]] = {
            node: (
                graph.nodes[node]["x"],
                graph.nodes[node]["y"],
            )
            for node in G.nodes()
        }
    elif isinstance(problem_data, ProblemDataToy):
        graph = problem_data.base_graph
        graph.graph["crs"] = "EPSG:3857"  # uses meters
        pos = {
            node: (
                graph.nodes[node]["x"],
                graph.nodes[node]["y"],
            )
            for node in graph.nodes()
        }
    else:
        raise NotImplementedError(
            "Plotting only implemented for ProblemDataReal and ProblemDataToy currently"
        )

    if prob.status not in [GRB.OPTIMAL]:
        print("No feasible solution to visualize.")
    else:
        # Visualize the routes on the graph
        if per_round:
            fig, axes = plt.subplots(nrows=1, ncols=len(Q), figsize=(12, 8))
            if len(Q) == 1:
                axes = [axes]
        else:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
            axes = [ax]

        # fig, ax = ox.plot_graph(graph, ax=ax, show=False)
        for q, ax in enumerate(axes):
            qs = range(len(Q)) if not per_round else [q]
            _, ax = ox.plot_graph(graph, ax=ax, show=False, bbox=(0, 0, 4000, 1000))
            for b, _ in enumerate(B):
                for q in qs:
                    if z_bq[b, q].X > 0.5:
                        for ij, path in enumerate(A.keys()):
                            if x_bqij[b, q, ij].X > 0.5:
                                path_edges = A_PATH[path]
                                if path_edges:
                                    ox.plot_graph_route(
                                        graph,
                                        path_edges,
                                        ax=ax,
                                        orig_dest_size=0,
                                        show=False,
                                        # route_color="r"
                                    )

            # plot schools, students, and bus stops
            for school in schools:
                ax.scatter(
                    pos[school.node_id][0],
                    pos[school.node_id][1],
                    c=school_colors[school.type],
                    marker="s",
                    label={school.name},
                )
            for bus_stop in bus_stops:
                ax.scatter(
                    pos[bus_stop.node_id][0],
                    pos[bus_stop.node_id][1],
                    c="yellow",
                    marker="^",
                    label=bus_stop.name,
                )
            for depot in depots:
                ax.scatter(
                    pos[depot.node_id][0],
                    pos[depot.node_id][1],
                    c="purple",
                    marker="X",
                    label=depot.name,
                )

            ax.title.set_text("Optimized School Bus Routes")
            ax.legend(loc="upper right", fontsize="small")

        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight")
        # plt.show()
