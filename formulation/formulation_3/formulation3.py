import networkx as nx
import matplotlib.pyplot as plt
import cvxpy as cp
import numpy as np

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
from formulation.formulation_3.toy_network import (
    make_buses,
    make_depots,
    make_graph,
    make_schools,
    make_stops,
    make_students,
)


def build_model_from_definition(
    problem: Formulation3, formulation_name: str
) -> tuple[cp.Problem, dict[str, cp.Variable]]:
    B = problem.B
    M = problem.M
    S = problem.S
    S_PLUS = problem.S_PLUS
    D = problem.D
    D_PLUS = problem.D_PLUS
    D_MINUS = problem.D_MINUS
    P = problem.P
    A = problem.A
    N = problem.N
    F = problem.F
    W = problem.W
    Q = problem.Q
    Q_MAX = problem.Q_MAX
    PHI = problem.PHI
    LAMBDA_ROUND = problem.LAMBDA_ROUND
    BETA = problem.BETA
    M_TIME = problem.M_TIME
    M_CAPACITY = problem.M_CAPACITY
    M_TYPE = problem.M_TYPE
    ALPHA = problem.ALPHA
    t_ij = problem.t_ij
    d_ij = problem.d_ij
    T_horizon = problem.T_horizon
    EPSILON = problem.EPSILON
    H_RIDE = problem.H_RIDE
    KAPPA = problem.KAPPA
    C_CAP_B = problem.C_CAP_B

    # DECISION VARIABLES

    z_b = cp.Variable((len(B)), boolean=True)  # 1 if bus b is used (nonempty tour)

    z_bq = cp.Variable(
        (len(B), len(Q)), boolean=True
    )  # 1 if bus b is assigned to round q

    y_bqtau = cp.Variable(
        (len(B), len(Q), len(TAU)), boolean=True
    )  # 1 if bus b in round q is of type tau

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
        (len(B), len(Q), len(N)), nonneg=True, bounds=[0, T_horizon]
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

    # HEAD HONCHO OBJECTIVE AND CONSTRAINTS

    objective = cp.Minimize(
        cp.sum(
            [
                cp.sum(
                    [
                        cp.sum(d_ij(*path) * x_bqij[:, :, ij])
                        for ij, path in enumerate(A.keys())
                    ]
                ),
                cp.multiply(LAMBDA_ROUND, cp.sum(z_bq)),
                cp.sum(r_bmon),
            ]
        )
    )

    constraints: list[cp.Constraint] = []

    # CONSTRAINTS GO HERE
    # -------------------

    # STUDENT ASSIGNMENT

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
                    assert isinstance(node, type(S[0]))
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

    # TIME PROPOGATION

    # with explicit dwell times
    for b in range(len(B)):
        for q in range(len(Q)):
            for ij, path in enumerate(A):
                constraints.append(
                    T_bqi[b, q, N.index(path[1])]
                    >= T_bqi[b, q, N.index(path[0])]
                    + t_ij(*path)
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

    # PICKUP BEFORE DROPOFF AND MAX RIDE TIME

    for b in range(len(B)):
        for q in range(len(Q)):
            for m, student in enumerate(M):
                constraints.append(
                    T_bqi[b, q, N.index(s_m(student))]
                    >= T_bqi[b, q, N.index(p_m(student))]
                    + EPSILON
                    - M_TIME * (1 - a_mbq[m, b, q])
                )
                constraints.append(
                    T_bqi[b, q, N.index(s_m(student))]
                    - T_bqi[b, q, N.index(p_m(student))]
                    <= H_RIDE + M_TIME * (1 - a_mbq[m, b, q])
                )

    # DISTANCE RANGE CONSTRAINTS

    for b, bus in enumerate(B):
        constraints.append(
            cp.sum([d_ij(*path) * x_bqij[b, :, ij] for ij, path in enumerate(A.keys())])
            <= R_b(bus) * z_b[b]
        )

    # LOAD / CAPACITY CONSTRAINTS PER ROUND

    for b, bus in enumerate(B):
        depot_start = make_depot_start_copy(depot_b(bus))
        depot_end = make_depot_end_copy(depot_b(bus))
        # load at beginning is 0
        constraints.append(L_bqi[b, 0, N.index(depot_start)] == 0)

        for q in range(len(Q)):
            constraints.append(L_bqi[b, q, N.index(depot_end)] == 0)

        for q in range(len(Q)):
            for s, school in enumerate(S):
                constraints.append(L_bqi[b, q, N.index(S_PLUS[s])] == 0)

        for q in range(len(Q)):
            for ij, path in enumerate(A):
                # if a round ends at school s, the bus must be empty after servicing s
                constraints.append(
                    L_bqi[b, q, N.index(path[1])]
                    >= L_bqi[b, q, N.index(path[0])]
                    + cp.sum(
                        [
                            a_mbq[m, b, q]
                            for m, student in enumerate(M)
                            if p_m(student) == path[1]
                        ]
                    )
                    - cp.sum(
                        [
                            a_mbq[m, b, q]
                            for m, student in enumerate(M)
                            if s_m(student) == path[1]
                        ]
                    )
                    - M_CAPACITY * (1 - x_bqij[b, q, ij])
                )
                constraints.append(
                    L_bqi[b, q, N.index(path[1])]
                    <= L_bqi[b, q, N.index(path[0])]
                    + cp.sum(
                        [
                            a_mbq[m, b, q]
                            for m, student in enumerate(M)
                            if p_m(student) == path[1]
                        ]
                    )
                    - cp.sum(
                        [
                            a_mbq[m, b, q]
                            for m, student in enumerate(M)
                            if s_m(student) == path[1]
                        ]
                    )
                    + M_CAPACITY * (1 - x_bqij[b, q, ij])
                )
                # if a round ends at school s, the bus must be empty after servicing s
        for q in range(len(Q)):
            for i in range(len(N)):
                constraints.append(
                    L_bqi[b, q, i]
                    <= C_b(bus)
                    * cp.sum(
                        [
                            KAPPA[school_type] * y_bqtau[b, q, tau]
                            for tau, school_type in enumerate(TAU)
                        ]
                    )
                )
        for q in range(len(Q)):
            for i in range(len(N)):
                constraints.append(L_bqi[b, q, i] >= 0)

        for q in range(len(Q)):
            for s, school in enumerate(S):
                constraints.append(
                    L_bqi[b, q, N.index(school)] <= C_CAP_B(bus) * (1 - e_bqs[b, q, s])
                )

    # MONITOR FEASIBILITY PER BUS

    for b in range(len(B)):
        for q in range(len(Q)):
            # If there are no flagged students, no monitor is needed.
            constraints.append(
                r_bmon[b]
                <= cp.sum(
                    [a_mbq[m, b, q] for m, student in enumerate(F) if f_m(student) == 1]
                )
            )
            # If a flag student is on the bus, there should be one monitor
            for flagged in F:
                constraints.append(r_bmon[b] >= a_mbq[M.index(flagged), b, q])

    # WHEELCHAIR CONSTRAINTS

    # if a student requiring a wheelchair is assigned a bus, the bus must be wheelchair-accessible
    for m, student in enumerate(W):
        for b, bus in enumerate(B):
            for q in range(len(Q)):
                constraints.append(a_mbq[M.index(student), b, q] <= Wh_b(bus))

    # SCHOOL LEVEL

    # if a bus is used in a round, it should have a school type assigned to that round,
    # and it should match the school type of any student assigned to that bus-round
    for b in range(len(B)):
        for q in range(len(Q)):
            constraints.append(cp.sum(y_bqtau[b, q, :]) == z_bq[b, q])
            for m, student in enumerate(M):
                constraints.append(
                    a_mbq[m, b, q] <= y_bqtau[b, q, TAU.index(tau_m(student))]
                )

    return cp.Problem(objective, constraints), {
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
    }


def solve_problem(prob: cp.Problem):
    print("Solving MILP")
    try:
        # Try with verbose to see progress; set a reasonable time limit
        prob.solve(
            solver=cp.GLPK_MI,
            verbose=True,
            glpk={"msg_lev": "GLP_MSG_ON", "tm_lim": 2 * 60 * 1000},
        )  # 2-min limit
    except Exception as e:
        print(f"⚠️  GLPK_MI failed or unavailable: {e}")
        print("Trying CBC as fallback...")
        try:
            prob.solve(solver=cp.CBC, verbose=True, maximumSeconds=120)
        except Exception as e2:
            print(f"⚠️  CBC also failed: {e2}")
            print("Trying default solver...")
            prob.solve(verbose=True)

    print()
    print("Status:", prob.status)
    print("Objective:", prob.value)


def make_report(
    prob: cp.Problem, formulation: Formulation3, model_vars: dict[str, cp.Variable]
):
    # Extract and print the route

    B = formulation.B
    M = formulation.M
    S = formulation.S
    A = formulation.A
    Q = formulation.Q

    # z_b = model_vars["z_b"]
    z_bq = model_vars["z_bq"]
    y_bqtau = model_vars["y_bqtau"]
    x_bqij = model_vars["x_bqij"]
    # v_bqi = model_vars["v_bqi"]
    a_mbq = model_vars["a_mbq"]
    # T_bqi = model_vars["T_bqi"]
    # L_bqi = model_vars["L_bqi"]
    # e_bqs = model_vars["e_bqs"]
    r_bmon = model_vars["r_bmon"]

    if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        print(f"⚠️ Solver did not find a feasible solution (status={prob.status}).")
    else:
        assert (
            z_bq.value is not None
            and y_bqtau.value is not None
            and x_bqij.value is not None
            and a_mbq.value is not None
        )
        for b, bus in enumerate(B):
            print(
                f"\{bus} (capacity {C_b(bus)}, range {R_b(bus)}, wheelchair access {Wh_b(bus) == 1}, monitor needed: {r_bmon[b].value > 0.5})"
            )
            for q in range(len(Q)):
                if z_bq[b, q].value > 0.5:
                    print(f"  Round {q}:")
                    route = []
                    students_on_bus = []
                    schools_served = []
                    for ij, path in enumerate(A.keys()):
                        if x_bqij[b, q, ij].value > 0.5:
                            route.append(path)
                            for node in path:
                                if (
                                    node in S
                                    and node not in schools_served
                                    and any(
                                        a_mbq[m, b, q].value > 0.5 and s_m(M[m]) == node
                                        for m in range(len(M))
                                    )
                                ):
                                    schools_served.append(node)
                    for m, student in enumerate(M):
                        if a_mbq[m, b, q].value > 0.5:
                            students_on_bus.append(student)
                    print(
                        f"    Total travel time (excluding dwell): {sum(formulation.d_ij(*path) for path in route):.2f} minutes"
                    )
                    print(
                        f"    Bus type: {TAU[np.argmax(y_bqtau[b,q,:])].name if z_bq[b, q].value > 0.5 else 'N/A'}"
                    )
                    print(
                        f"    Students on bus this round:\n      {'\n      '.join(str(student) for student in students_on_bus)}"
                    )
                    print(
                        f"    Schools served:\n      {'\n      '.join(str(school) for school in schools_served)}"
                    )

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

                    print("    Route:")
                    for path in route:
                        print(f"      {path[0]} -> {path[1]}")


if __name__ == "__main__":
    # Example usage
    # Create a sample problem definition (you can replace this with actual data)

    graph = make_graph(size=5)
    schools = make_schools(graph, num_schools=2)
    depots = make_depots(graph, num_depots=1)
    stops = make_stops(graph, num_stops=5)
    students = make_students(graph, num_students=10, schools=schools, stops=stops)
    buses = make_buses(graph, num_buses=2)

    problem3 = Formulation3(
        graph=graph,
        rounds=1,
        schools=schools,
        depots=depots,
        buses=buses,
        students=students,
        stops=stops,
    )
    print("Problem definition created.")
    print(problem3)

    # Build the model from the problem definition
    model, model_variables = build_model_from_definition(
        problem3, formulation_name="Formulation 3"
    )
    # Generate a report by solving the model
    solve_problem(model)
    make_report(model, problem3, model_variables)
