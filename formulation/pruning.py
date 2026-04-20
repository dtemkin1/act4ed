from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from formulation.common import ProblemData, School

KILOMETERS_TO_METERS = 1000.0


@dataclass(frozen=True)
class SchoolPruneSummary:
    school_id: str | int
    school_name: str
    student_count: int
    stop_count: int
    direct_stop_arc_count: int
    strong_connectivity_prune_m: float | None
    all_direct_arcs_prune_m: float


def summarize_school_pruning(
    problem_data: ProblemData,
    school: School | str | int,
) -> SchoolPruneSummary:
    """
    Summarize school-specific stop-stop pruning thresholds.

    `strong_connectivity_prune_m` is the smallest prune threshold that makes the
    school's directed stop graph strongly connected.

    `all_direct_arcs_prune_m` is the conservative threshold that preserves every
    direct stop-stop arc in the unpruned school subproblem.

    Exact summaries require an unpruned base `problem_data`. If the loaded cache
    already has `prune` applied, the long arcs are gone and the result is no
    longer exact.
    """

    _require_unpruned(problem_data)
    school_problem, resolved_school = _school_problem(problem_data, school)
    stop_graph = _stop_graph_from_school_problem(school_problem)
    weights_m = sorted(
        {data["length_m"] for _u, _v, data in stop_graph.edges(data=True)}
    )

    return SchoolPruneSummary(
        school_id=resolved_school.id,
        school_name=resolved_school.name,
        student_count=len(school_problem.students),
        stop_count=len(school_problem.stops),
        direct_stop_arc_count=stop_graph.number_of_edges(),
        strong_connectivity_prune_m=_min_threshold_for_strong_connectivity(
            stop_graph, weights_m
        ),
        all_direct_arcs_prune_m=0.0 if not weights_m else weights_m[-1],
    )


def school_stop_graph(
    problem_data: ProblemData,
    school: School | str | int,
) -> nx.DiGraph:
    """
    Build the unpruned directed stop-stop graph for a single school.

    Edge weights are stored in meters as `length_m`.
    """

    _require_unpruned(problem_data)
    school_problem, _resolved_school = _school_problem(problem_data, school)
    return _stop_graph_from_school_problem(school_problem)


def _stop_graph_from_school_problem(
    school_problem: ProblemData,
) -> nx.DiGraph:
    graph = nx.DiGraph()
    stop_ids = {stop.node_id for stop in school_problem.stops}

    for stop in school_problem.stops:
        graph.add_node(stop.node_id, stop=stop)

    for start_id, end_id, data in school_problem.service_graph.edges(data=True):
        if start_id not in stop_ids or end_id not in stop_ids or start_id == end_id:
            continue

        length_m = float(data["length"]) * KILOMETERS_TO_METERS
        existing = graph.get_edge_data(start_id, end_id)
        if existing is None or length_m < existing["length_m"]:
            graph.add_edge(start_id, end_id, length_m=length_m)

    return graph


def _school_problem(
    problem_data: ProblemData,
    school: School | str | int,
) -> tuple[ProblemData, School]:
    school_id = school.id if isinstance(school, School) else school

    if len(problem_data.schools) == 1 and problem_data.schools[0].id == school_id:
        return problem_data, problem_data.schools[0]

    school_problem = problem_data.restrict_to_school(school_id)
    return school_problem, school_problem.schools[0]


def _require_unpruned(problem_data: ProblemData) -> None:
    prune = getattr(problem_data, "prune", None)
    if prune is not None:
        raise ValueError(
            "exact school prune summaries require unpruned problem data; "
            "load the cache with prune=None first"
        )


def _min_threshold_for_strong_connectivity(
    stop_graph: nx.DiGraph,
    weights_m: list[float],
) -> float | None:
    if stop_graph.number_of_nodes() <= 1:
        return 0.0

    if stop_graph.number_of_edges() == 0:
        return None

    if not nx.is_strongly_connected(stop_graph):
        return None

    lo = 0
    hi = len(weights_m) - 1
    answer = weights_m[-1]

    while lo <= hi:
        mid = (lo + hi) // 2
        threshold_m = weights_m[mid]
        threshold_graph = nx.DiGraph()
        threshold_graph.add_nodes_from(stop_graph.nodes)
        threshold_graph.add_edges_from(
            (u, v)
            for u, v, data in stop_graph.edges(data=True)
            if data["length_m"] <= threshold_m
        )

        if nx.is_strongly_connected(threshold_graph):
            answer = threshold_m
            hi = mid - 1
        else:
            lo = mid + 1

    return answer
