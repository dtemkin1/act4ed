from typing import Any

import gurobipy as gp
from gurobipy import GRB

from formulation.common import (
    TAU,
    C_b,
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
    depot_end_arcs_by_bus = {
        b: arcs_to_node[node_to_idx[make_depot_end_copy(depot_b(bus))]]
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
    range_limit_by_bus = {
        b: R_b(bus) * METERS_PER_MILE for b, bus in enumerate(B)
    }
    capacity_by_bus = {b: C_b(bus) for b, bus in enumerate(B)}
    cap_upper_by_bus = {b: C_CAP_B(bus) for b, bus in enumerate(B)}

    model = gp.Model("formulation3_gurobi")

    z_b = model.addVars(B_idx, vtype=GRB.BINARY, name="z_b")
    z_bq = model.addVars(B_idx, Q_idx, vtype=GRB.BINARY, name="z_bq")
    y_bqtau = model.addVars(B_idx, Q_idx, TAU_idx, vtype=GRB.BINARY, name="y_bqtau")
    x_bqij = model.addVars(B_idx, Q_idx, A_idx, vtype=GRB.BINARY, name="x_bqij")
    v_bqi = model.addVars(B_idx, Q_idx, N_idx, vtype=GRB.BINARY, name="v_bqi")
    a_mbq = model.addVars(M_idx, B_idx, Q_idx, vtype=GRB.BINARY, name="a_mbq")
    T_bqi = model.addVars(
        B_idx,
        Q_idx,
        N_idx,
        vtype=GRB.CONTINUOUS,
        lb=0.0,
        ub=T_horizon,
        name="T_bqi",
    )
    L_bqi = model.addVars(
        B_idx,
        Q_idx,
        N_idx,
        vtype=GRB.INTEGER,
        lb=0.0,
        name="L_bqi",
    )
    e_bqs = model.addVars(B_idx, Q_idx, S_idx, vtype=GRB.BINARY, name="e_bqs")
    r_bmon = model.addVars(B_idx, vtype=GRB.BINARY, name="r_bmon")

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
        for q in range(len(Q) - 1):
            model.addConstr(gp.quicksum(e_bqs[b, q, s] for s in S_idx) == z_bq[b, q + 1])
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
                gp.quicksum(x_bqij[b, q, arc_idx] for arc_idx in depot_end_arcs_by_bus[b])
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
                        x_bqij[b, q, arc_idx] for arc_idx in arcs_to_node[school_node_idx]
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

    # SCHOOL LATEST ARRIVAL
    for b in B_idx:
        for q in Q_idx:
            for s, school in enumerate(S):
                school_node_idx = school_node_indices[s]
                model.addConstr(
                    T_bqi[b, q, school_node_idx]
                    + BETA * school_assignments[b, q, s]
                    <= l_s(school) + M_TIME * (1 - v_bqi[b, q, school_node_idx])
                )

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
                load_change = pickup_assignments.get((b, q, end_idx), 0) - dropoff_assignments.get(
                    (b, q, end_idx), 0
                )
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

    # WHEELCHAIR CONSTRAINTS
    for m in wheelchair_student_indices:
        for b, bus in enumerate(B):
            for q in Q_idx:
                model.addConstr(a_mbq[m, b, q] <= Wh_b(bus))

    # SCHOOL LEVEL
    for b in B_idx:
        for q in Q_idx:
            model.addConstr(y_bqtau.sum(b, q, "*") == z_bq[b, q])
            for m in M_idx:
                model.addConstr(
                    a_mbq[m, b, q] <= y_bqtau[b, q, tau_idx_for_student[m]]
                )

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
    model.Params.OutputFlag = 1
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
