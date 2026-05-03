# average/distribution of travel time
# (across all buses, and across all students, across different demographics
# (income bins, race, english proficiency, car ownership))


from experiments.existing_data.bird_routes import (
    BIRD_CONFIG,
    EXISTING_ROUTES_OUTPUT,
    get_bird_routes,
)
from experiments.existing_data.current_routes import get_existing_routes
from experiments.existing_data.utils import get_assigned_students
from experiments.helpers import setup_framingham
from formulation.common.classes import Place, Stop
from formulation.common.problems import (
    FilteredProblemData,
    ProblemData,
)
from formulation.common.utils import get_travel_time


def get_route_time(route: list[Place], problem_data: ProblemData) -> float:
    _, path = problem_data.get_shortest_paths_base(
        tuple(place.node_id for place in route)
    )
    return get_travel_time(
        tuple(path),
        problem_data.base_graph,
    )


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

    current_routes = get_existing_routes(framingham_problem_data)

    for current_route in current_routes:
        time = get_travel_time(current_route.path, framingham_problem_data.base_graph)
        print(f"Route {current_route.bus.name} time: {time:.2f} minutes")

    filtered_bird_results = get_bird_routes(
        filtered_problem_data,
        config=BIRD_CONFIG,
        out_dir=EXISTING_ROUTES_OUTPUT,
        save_results=False,
    )

    for bird_itinerary in filtered_bird_results.itineraries:
        for round, route_order in enumerate(bird_itinerary.route_orders):
            route = filtered_bird_results.routes[route_order]
            depot = filtered_problem_data.depots[0]
            stops: list[Stop] = []
            for stop_id in route.stop_ids:
                stop_filter = filter(
                    lambda s: s.name == stop_id, filtered_problem_data.stops
                )
                stops.append(list(stop_filter)[0])
            school = list(
                filter(
                    lambda s: s.id == route.school_id,
                    filtered_problem_data.schools,
                )
            )[0]

            places = [depot] + stops + [school]
            time = get_route_time(places, filtered_problem_data)

            identifier = (
                bird_itinerary.bus_id
                if len(bird_itinerary.route_orders) == 1
                else bird_itinerary.bus_id + " (round " + str(round + 1) + ")"
            )

            print(f"BIRD Route (filtered) {identifier} time: {time:.2f} minutes")

    all_bird_results = get_bird_routes(
        framingham_problem_data,
        config=BIRD_CONFIG,
        out_dir=EXISTING_ROUTES_OUTPUT,
        save_results=False,
    )

    for bird_itinerary in all_bird_results.itineraries:
        for round, route_order in enumerate(bird_itinerary.route_orders):
            route = all_bird_results.routes[route_order]
            depot = framingham_problem_data.depots[0]
            stops: list[Stop] = []
            for stop_id in route.stop_ids:
                stop_filter = filter(
                    lambda s: s.name == stop_id, framingham_problem_data.stops
                )
                stops.append(list(stop_filter)[0])
            school = list(
                filter(
                    lambda s: s.id == route.school_id,
                    framingham_problem_data.schools,
                )
            )[0]

            places = [depot] + stops + [school]
            time = get_route_time(places, framingham_problem_data)

            identifier = (
                bird_itinerary.bus_id
                if len(bird_itinerary.route_orders) == 1
                else bird_itinerary.bus_id + " (round " + str(round + 1) + ")"
            )

            print(f"BIRD Route (all_students) {identifier} time: {time:.2f} minutes")


if __name__ == "__main__":
    main()
