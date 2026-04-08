from dataclasses import dataclass
from typing import Any
import datetime as dt
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB, tupledict, Var
from matplotlib import pyplot as plt
import osmnx as ox

from formulation.common import (
    TAU,
    C_b,
    NodeId,
    ProblemDataReal,
    ProblemDataToy,
    R_b,
    Wh_b,
    s_m,
)
from formulation.formulation_3.problem3_definition import Formulation3
from solution import Formulation3Solution, _coerce_reporting_inputs


def make_report(
    prob: gp.Model | Formulation3Solution,
    formulation: Formulation3,
    model_vars: dict[str, Any] | None = None,
    rounds: int | None = None,
):
    # Extract and print the route
    prob, model_vars = _coerce_reporting_inputs(prob, model_vars)

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
            total_rounds = len(Q) if rounds is None else min(rounds, len(Q))
            for q in range(total_rounds):
                assert z_bq[b, q].X is not None
                if z_bq[b, q].X > 0.5:
                    result_string += f"\tRound {q}:\n"
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
                    result_string += f"\t\tTotal travel distance: {sum(formulation.d_ij(*path) for path in ordered_route):.2f} meters\n"
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
                    result_string += f"\t\tBus type: {school_type.name}\n"
                    result_string += f"\t\tStudents on bus this round:\n\t\t{'\n\t\t'.join(str(student) for student in students_on_bus)}\n"
                    result_string += f"\t\tSchools served:\n\t\t{'\n\t\t'.join(str(school) for school in schools_served)}\n"

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

                    result_string += "\t\tRoute:\n"
                    for path in ordered_route:
                        result_string += f"\t\t{path[0]} -> {path[1]}\n"

    return result_string


def plot_bus_routes(
    prob: gp.Model | Formulation3Solution,
    formulation: Formulation3,
    model_vars: dict[str, Any] | None = None,
    save_path: Path | None = None,
    per_round: bool = False,
) -> tuple[plt.Figure, list[plt.Axes]]:
    # raise NotImplementedError("plotting not implemented yet :(")
    prob, model_vars = _coerce_reporting_inputs(prob, model_vars)

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
                graph.nodes[node]["location"][0],
                graph.nodes[node]["location"][1],
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

        return fig, axes
