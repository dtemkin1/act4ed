# average/distribution of travel time
# (across all buses, and across all students, across different demographics
# (income bins, race, english proficiency, car ownership))


from functools import cache
import statistics
from typing import Callable, overload
from dataclasses import replace

from experiments.existing_data.bird_routes import (
    BIRD_CONFIG,
    get_bird_routes,
    get_bird_routes_json,
)
from experiments.existing_data.current_routes import RouteResult, get_existing_routes
from experiments.existing_data.utils import get_assigned_students
from experiments.helpers import setup_framingham
from formulation.common.classes import Place, School, Stop, Student
from formulation.common.problems import (
    FilteredProblemData,
    ProblemData,
)
from formulation.common.utils import get_travel_time
from formulation.normalized_result import (
    NormalizedBusItinerary,
    NormalizedRoutingResult,
    RoutingSolutionJson,
    RoutingSolutionRow,
)


def get_dwell_time(students_at_stop: tuple[Student, ...], place: Place) -> float:
    if isinstance(place, Stop):
        return (
            BIRD_CONFIG.stop_time_per_student * len(students_at_stop)
            + BIRD_CONFIG.constant_stop_time
        )
    elif isinstance(place, School):
        return BIRD_CONFIG.school_dwell_time

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
    all_places_after = route[route.index(stop_gets_on) :]
    places_visited = route[route.index(stop_gets_on) : route.index(school_gets_off) + 1]
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
    total_time = 0.0
    for student in students:
        time_on_bus = get_student_time_on_bus(student, route, problem_data)
        total_time += time_on_bus

    avg_time = total_time / len(students) if len(students) > 0 else 0.0
    return avg_time


def stats_time_on_bus_for_students_find_route(
    students: tuple[Student, ...],
    routes: list[RouteResult] | NormalizedRoutingResult | RoutingSolutionJson,
    problem_data: ProblemData,
) -> tuple[float, float]:  # (average time, standard deviation)

    total_time: list[float] = []
    for student in students:
        route = get_route_for_student(student, routes)
        if route:
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
                    depot = problem_data.depots[0]
                    stops: list[Stop] = []
                    for stop_id in route_json.stop_node_ids or []:
                        stop_filter = filter(
                            lambda s: s.node_id == stop_id, problem_data.stops
                        )
                        stops.extend(list(stop_filter))
                    school = list(
                        filter(
                            lambda s: s.name == route_json.school_name,
                            problem_data.schools,
                        )
                    )[0]

                    places = ((depot,) if i == 0 else ()) + tuple(stops) + (school,)
                    total_route += places

                time_on_bus = get_student_time_on_bus(
                    student, tuple(total_route), problem_data
                )
                total_time.append(time_on_bus)
            else:
                assert isinstance(routes, NormalizedRoutingResult)
                total_route: list[Place] = []
                for i, route_order in enumerate(route.route_orders):
                    bird_route = routes.routes[route_order]
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

                    places = ([depot] if i == 0 else []) + stops + [school]
                    total_route += places

                time_on_bus = get_student_time_on_bus(
                    student, tuple(total_route), problem_data
                )
                total_time.append(time_on_bus)

    avg_time = statistics.mean(total_time) if len(total_time) > 0 else 0.0
    std_dev = statistics.stdev(total_time) if len(total_time) > 1 else 0.0
    return avg_time, std_dev


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
            assert isinstance(route, RouteResult)
            if student.stop in route.stops and student.school == route.school:
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

    for (filter_name, student_filter), identifier in zip(
        student_filters, student_filters
    ):
        filtered_students = tuple(filter(student_filter, problem_data.students))
        avg_time, std_dev = stats_time_on_bus_for_students_find_route(
            filtered_students, routes, problem_data
        )
        print(
            f"Average time on bus for {filter_name}: {avg_time:.2f} minutes (±{std_dev:.2f})"
        )


def is_student_served(
    student: Student, routes: NormalizedRoutingResult | list[RouteResult]
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


def main() -> None:
    framingham_problem_data = setup_framingham(precompute_cache=True)

    assigned_students = get_assigned_students(
        framingham_problem_data.schools, framingham_problem_data.stops
    )

    filtered_problem_data = FilteredProblemData(
        "framingham_filtered",
        base_problem_data=framingham_problem_data,
        _students=assigned_students,
    )

    current_routes = get_existing_routes(filtered_problem_data)

    print("STATS FOR CURRENT STUDENTS")
    get_relevant_stats(current_routes, filtered_problem_data, RELEVANT_STATS)

    # current_routes_average_time, current_routes_std_dev = (
    #     stats_time_on_bus_for_students_find_route(
    #         filtered_problem_data.students, current_routes, filtered_problem_data
    #     )
    # )
    # print(f"Current routes: {len(current_routes)}")
    # print(
    #     f"Average time on bus for current routes: {current_routes_average_time:.2f} minutes (±{current_routes_std_dev:.2f})"
    # )

    for current_route in current_routes:
        time = get_travel_time(current_route.path, filtered_problem_data.base_graph)
        print(f"Route {current_route.bus.name} time: {time:.2f} minutes")

    filtered_bird_results = get_bird_routes_json(
        "existing_student_routes",
    )

    print("STATS FOR BIRD ROUTES (EXISTING STUDENTS)")
    get_relevant_stats(filtered_bird_results, filtered_problem_data, RELEVANT_STATS)

    # filtered_bird_results_avg_time, filtered_bird_results_std_dev = (
    #     stats_time_on_bus_for_students_find_route(
    #         filtered_problem_data.students, filtered_bird_results, filtered_problem_data
    #     )
    # )

    # filtered_route_names: set[str] = set()
    # for solution in filtered_bird_results.solution:
    #     filtered_route_names.add(solution.bus_name)

    # print(f"BIRD itineraries (filtered): {len(filtered_route_names)}")
    # print(
    #     f"Average time on bus for BIRD filtered routes: {filtered_bird_results_avg_time:.2f} minutes (±{filtered_bird_results_std_dev:.2f})"
    # )

    # for route_name in filtered_route_names:
    #     route_orders = list(
    #         filter(lambda r: r.bus_name == route_name, filtered_bird_results.solution)
    #     )
    #     for round, route_json in enumerate(route_orders):
    #         depot = filtered_problem_data.depots[0]
    #         stops: list[Stop] = []
    #         for stop_id in route_json.stop_node_ids or []:
    #             stop_filter = filter(
    #                 lambda s: s.node_id == stop_id, filtered_problem_data.stops
    #             )
    #             stops.extend(list(stop_filter))
    #         school = list(
    #             filter(
    #                 lambda s: s.name == route_json.school_name,
    #                 filtered_problem_data.schools,
    #             )
    #         )[0]

    #         places = ((depot,) if round == 0 else ()) + tuple(stops) + (school,)
    #         time = get_route_time(places, filtered_problem_data)

    #         identifier = (
    #             route_name
    #             if len(route_orders) <= 1
    #             else route_name + " (round " + str(round + 1) + ")"
    #         )

    #         print(f"BIRD Route (filtered) {identifier} time: {time:.2f} minutes")

    config_allow_partial = replace(BIRD_CONFIG, allow_partial=True)

    all_bird_results = get_bird_routes(
        "new_routes",
        framingham_problem_data,
        config=config_allow_partial,
        save_results=False,
    )
    served_students = tuple(
        filter(
            lambda s: is_student_served(s, all_bird_results),
            framingham_problem_data.students,
        )
    )

    print(
        f"Percentage of students served by BIRD new routes: {len(served_students)} / {len(framingham_problem_data.students)}"
    )

    print("STATS FOR BIRD ROUTES (ALL STUDENTS)")
    get_relevant_stats(all_bird_results, framingham_problem_data, RELEVANT_STATS)

    # all_bird_results_avg_time, all_bird_results_std_dev = (
    #     stats_time_on_bus_for_students_find_route(
    #         served_students, all_bird_results, framingham_problem_data
    #     )
    # )
    # print(f"BIRD itineraries (all students): {len(all_bird_results.itineraries)}")
    # print(
    #     f"Average time on bus for BIRD all students routes: {all_bird_results_avg_time:.2f} minutes (±{all_bird_results_std_dev:.2f})"
    # )
    # for bird_itinerary in all_bird_results.itineraries:
    #     for round, route_order in enumerate(bird_itinerary.route_orders):
    #         route = all_bird_results.routes[route_order]
    #         depot = framingham_problem_data.depots[0]
    #         stops: list[Stop] = []
    #         for stop_id in route.stop_ids:
    #             stop_filter = filter(
    #                 lambda s: s.name == stop_id, framingham_problem_data.stops
    #             )
    #             stops.extend(list(stop_filter))
    #         school = list(
    #             filter(
    #                 lambda s: s.name == route.school_id,
    #                 framingham_problem_data.schools,
    #             )
    #         )[0]

    #         places = (depot,) + tuple(stops) + (school,)
    #         time = get_route_time(places, framingham_problem_data)

    #         identifier = (
    #             bird_itinerary.bus_id
    #             if len(bird_itinerary.route_orders) == 1
    #             else bird_itinerary.bus_id + " (round " + str(round + 1) + ")"
    #         )

    #         print(f"BIRD Route (all_students) {identifier} time: {time:.2f} minutes")


if __name__ == "__main__":
    main()
