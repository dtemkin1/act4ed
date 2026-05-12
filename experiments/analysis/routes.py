# average/distribution of travel time
# (across all buses, and across all students, across different demographics
# (income bins, race, english proficiency, car ownership))


import statistics
from typing import Callable, overload

from experiments.existing_data.bird_routes import (BIRD_CONFIG,
                                                   get_bird_routes,
                                                   get_bird_routes_json)
from experiments.existing_data.current_routes import (RouteResult,
                                                      get_existing_routes)
from experiments.existing_data.utils import get_assigned_students
from experiments.helpers import setup_framingham
from formulation.common.classes import Place, Stop, Student
from formulation.common.constants import KM_PER_MILE
from formulation.common.problems import FilteredProblemData, ProblemData
from formulation.common.utils import get_travel_time
from formulation.normalized_result import (NormalizedBusItinerary,
                                           NormalizedRoute,
                                           NormalizedRoutingResult,
                                           RoutingSolutionJson,
                                           RoutingSolutionRow)


def get_dwell_time(students_at_stop: tuple[Student, ...], place: Place) -> float:
    if isinstance(place, Stop) and len(students_at_stop) > 0:
        return (
            BIRD_CONFIG.stop_time_per_student * len(students_at_stop)
            + BIRD_CONFIG.stop_time_per_wheelchair_student
            * sum(
                1 for student in students_at_stop if student.attributes.wheelchair_user
            )
            + BIRD_CONFIG.constant_stop_time
        )

    return 0.0


def get_route_time(route: tuple[Place, ...], problem_data: ProblemData) -> float:
    _, path = problem_data.get_shortest_paths_base(
        tuple(place.node_id for place in route)
    )
    return get_travel_time(
        tuple(path),
        problem_data.base_graph,
    )


def get_student_time_on_bus(
    student: Student, route: tuple[Place, ...], problem_data: ProblemData
) -> float:
    """note: does not account for time spent at intersection, only time spent traveling"""
    stop_gets_on = student.stop
    school_gets_off = student.school

    valid_starts = [i for i, p in enumerate(route) if p == stop_gets_on]
    valid_ends = [i for i, p in enumerate(route) if p == school_gets_off]

    start_idx = -1
    end_idx = -1
    for s in valid_starts:
        for e in valid_ends:
            if s < e:
                if start_idx == -1 or (e - s) < (end_idx - start_idx):
                    start_idx = s
                    end_idx = e

    if start_idx == -1 or end_idx == -1:
        return 0.0

    all_places_after = route[start_idx:]
    places_visited = route[start_idx : end_idx + 1]
    node_ids_visiting = tuple(place.node_id for place in places_visited)

    overall_dwell_time = 0.0
    for place in places_visited[:-1]:  # exclude school where student gets off
        overall_dwell_time += get_dwell_time(
            tuple(
                student
                for student in problem_data.students
                if student.stop == place and student.school in all_places_after
            ),
            place,
        )

    _, full_path = problem_data.get_shortest_paths_base(node_ids_visiting)

    travel_time = (
        get_travel_time(
            tuple(full_path),
            problem_data.base_graph,
        )
        + overall_dwell_time
    )

    return travel_time


def avg_time_on_bus_for_students(
    students: tuple[Student, ...], route: tuple[Place, ...], problem_data: ProblemData
) -> float:
    total_time: list[float] = []
    for student in students:
        time_on_bus = get_student_time_on_bus(student, route, problem_data)
        if time_on_bus != 0.0:
            total_time.append(time_on_bus)

    avg_time = statistics.mean(total_time) if len(total_time) > 0 else 0.0
    return avg_time


def stats_time_on_bus_for_students_find_route(
    students: tuple[Student, ...],
    routes: list[RouteResult] | NormalizedRoutingResult | RoutingSolutionJson,
    problem_data: ProblemData,
) -> tuple[
    float, float, float, float
]:  # (average time, standard deviation, minimum time, maximum time)

    total_time: list[float] = []
    for student in students:
        route = get_route_for_student(student, routes)
        if route is not None:
            if isinstance(route, RouteResult):
                time_on_bus = get_student_time_on_bus(
                    student, route.all_places, problem_data
                )
                total_time.append(time_on_bus)
            elif isinstance(route, RoutingSolutionRow):
                assert isinstance(routes, RoutingSolutionJson)
                total_route: list[Place] = []
                bus_name = route.bus_name
                route_orders = list(
                    filter(lambda r: r.bus_name == bus_name, routes.solution)
                )

                for i, route_json in enumerate(route_orders):
                    stop_nodes = route_json.stop_node_ids or []

                    origin = list(
                        filter(
                            lambda d: d.node_id == route_json.origin_node_id,
                            (problem_data.depots + problem_data.schools),
                        )
                    )

                    stops: list[Stop] = []
                    for stop_id in stop_nodes:
                        stop_filter = list(
                            filter(
                                lambda s: s.node_id == stop_id
                                and s in problem_data.stops,
                                problem_data.stops,
                            )
                        )
                        if len(stop_filter) > 1:
                            # if student's stop is in stop_filter, use that. else, use the first one
                            if student.stop in stop_filter:
                                stop_filter = [student.stop]
                            else:
                                stop_filter = [
                                    list(
                                        filter(lambda s: s != student.stop, stop_filter)
                                    )[0]
                                ]
                        stops.extend(stop_filter)
                    destination = list(
                        filter(
                            lambda s: s.node_id == route_json.destination_node_id,
                            (problem_data.stops + problem_data.schools),
                        )
                    )

                    places = origin + stops + destination
                    while total_route and places and total_route[-1] == places[0]:
                        # if the last place in the current total route matches the first place in the new route, we can chain them together
                        places = places[1:]

                    total_route.extend(places)

                time_on_bus = get_student_time_on_bus(
                    student, tuple(total_route), problem_data
                )
                total_time.append(time_on_bus)
            elif isinstance(route, NormalizedBusItinerary):
                assert isinstance(routes, NormalizedRoutingResult)
                total_route: list[Place] = []
                bird_routes: list[NormalizedRoute] = list(
                    filter(lambda r: r.bus_id == route.bus_id, routes.routes)
                )
                for i, bird_route in enumerate(bird_routes):
                    depot = problem_data.depots[0]
                    stops: list[Stop] = []
                    for stop_id in bird_route.stop_ids:
                        stop_filter = filter(
                            lambda s: s.name == stop_id, problem_data.stops
                        )
                        stops.append(list(stop_filter)[0])
                    school = list(
                        filter(
                            lambda s: s.id == bird_route.school_id,
                            problem_data.schools,
                        )
                    )[0]

                    places = (
                        ([depot] if bird_route.order == 0 else []) + stops + [school]
                    )
                    total_route += places

                time_on_bus = get_student_time_on_bus(
                    student, tuple(total_route), problem_data
                )
                total_time.append(time_on_bus)

    avg_time = statistics.mean(total_time) if len(total_time) > 0 else 0.0
    std_dev = statistics.stdev(total_time) if len(total_time) > 1 else 0.0
    min_time = min(total_time) if len(total_time) > 0 else 0.0
    max_time = max(total_time) if len(total_time) > 0 else 0.0
    return avg_time, std_dev, min_time, max_time


@overload
def get_route_for_student(
    student: Student, routes: list[RouteResult]
) -> RouteResult | None: ...


@overload
def get_route_for_student(
    student: Student, routes: RoutingSolutionJson
) -> RoutingSolutionRow | None: ...


@overload
def get_route_for_student(
    student: Student, routes: NormalizedRoutingResult
) -> NormalizedBusItinerary | None: ...


def get_route_for_student(
    student: Student,
    routes: list[RouteResult] | NormalizedRoutingResult | RoutingSolutionJson,
) -> RouteResult | NormalizedBusItinerary | RoutingSolutionRow | None:

    if isinstance(routes, list):
        for route in routes:
            if student in route.students_served:
                return route

    if isinstance(routes, NormalizedRoutingResult):
        for itinerary in routes.itineraries:
            for route_order in itinerary.route_orders:
                route = routes.routes[route_order]
                if (
                    student.stop.name in route.stop_ids
                    and student.school.id == route.school_id
                ):
                    return itinerary

    if isinstance(routes, RoutingSolutionJson):
        for solution in routes.solution:
            if (
                student.stop.node_id in (solution.stop_node_ids or [])
                and student.school.name == solution.school_name
            ):
                return solution

    return None


def get_relevant_stats(
    routes: list[RouteResult] | NormalizedRoutingResult | RoutingSolutionJson,
    problem_data: ProblemData,
    student_filters: list[tuple[str, Callable[[Student], bool]]],
) -> None:

    for filter_name, student_filter in student_filters:
        filtered_students = tuple(filter(student_filter, problem_data.students))
        avg_time, std_dev, min_time, max_time = (
            stats_time_on_bus_for_students_find_route(
                filtered_students, routes, problem_data
            )
        )
        print(
            f"Average time on bus for {filter_name}: {avg_time:.2f} minutes (±{std_dev:.2f}), min: {min_time:.2f} minutes, max: {max_time:.2f} minutes)"
        )


def is_student_served(
    student: Student,
    routes: NormalizedRoutingResult | RoutingSolutionJson | list[RouteResult],
) -> bool:
    return get_route_for_student(student, routes) is not None


RELEVANT_STATS: list[tuple[str, Callable[[Student], bool]]] = [
    ("all students", lambda s: True),
    (
        "non english at home",
        lambda s: not s.demographics.english_at_home if s.demographics else False,
    ),
    (
        "non car owning",
        lambda s: not s.demographics.owns_car if s.demographics else False,
    ),
    (
        "special needs",
        lambda s: s.attributes.special_ed or s.attributes.wheelchair_user,
    ),
]


def students_over_1_5_miles_away(student: Student, problem_data: ProblemData) -> bool:
    if student.stop is None or student.school is None:
        return False

    edge_data = problem_data.service_graph.get_edge_data(
        student.stop.node_id, student.school.node_id, 0
    )
    if edge_data is None:
        return False

    distance = edge_data["length"]
    return distance > (1.5 * KM_PER_MILE)


def main() -> None:
    framingham_problem_data = setup_framingham(precompute_cache=True)

    assigned_students = get_assigned_students(
        framingham_problem_data.schools, framingham_problem_data.stops
    )

    filtered_problem_data = FilteredProblemData(
        "framingham_filtered_assigned",
        base_problem_data=framingham_problem_data,
        _students=assigned_students,
    )

    current_routes = get_existing_routes(filtered_problem_data)

    print("STATS FOR CURRENT STUDENTS")
    get_relevant_stats(current_routes, filtered_problem_data, RELEVANT_STATS)

    filtered_bird_results = get_bird_routes_json("assigned_students_routes")

    print("STATS FOR BIRD ROUTES (ASSIGNED STUDENTS)")
    get_relevant_stats(filtered_bird_results, framingham_problem_data, RELEVANT_STATS)

    # only for students living over 1.5mi away (actual driving distance)
    filtered_distance_students: tuple[Student, ...] = tuple(
        filter(
            lambda s: framingham_problem_data.service_graph.get_edge_data(
                s.stop.node_id, s.school.node_id, 0
            )["length"]
            > (1.5 * KM_PER_MILE),
            framingham_problem_data.students,
        )
    )

    far_mile_students_routes = get_bird_routes_json("1_5_mile_students_routes")

    print("STATS FOR BIRD ROUTES (STUDENTS OVER 1.5 MILES AWAY)")
    students_over_1_5_miles_away_filter = filter(
        lambda s: students_over_1_5_miles_away(s, framingham_problem_data),
        framingham_problem_data.students,
    )
    print(
        f"Number of students over 1.5 miles away: {len(tuple(students_over_1_5_miles_away_filter))}"
    )
    get_relevant_stats(
        far_mile_students_routes, framingham_problem_data, RELEVANT_STATS
    )

    all_bird_results = get_bird_routes("all_students_routes")
    served_students = tuple(
        filter(
            lambda s: is_student_served(s, all_bird_results),
            framingham_problem_data.students,
        )
    )

    print(
        f"Students served by BIRD new routes (1.5 miles): {len(served_students)} / {len(filtered_distance_students)}"
    )

    print("STATS FOR BIRD ROUTES (ALL STUDENTS)")
    get_relevant_stats(all_bird_results, framingham_problem_data, RELEVANT_STATS)


if __name__ == "__main__":
    main()
