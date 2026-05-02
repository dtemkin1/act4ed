# average/distribution of travel time
# (across all buses, and across all students, across different demographics
# (income bins, race, english proficiency, car ownership))


from experiments.existing_data.bird_routes import (
    BIRD_CONFIG,
    EXISTING_ROUTES_OUTPUT,
    get_bird_routes,
)
from experiments.helpers import setup_framingham


def main() -> None:
    problem_data = setup_framingham(precompute_cache=True)

    bird_results = get_bird_routes(
        problem_data,
        config=BIRD_CONFIG,
        out_dir=EXISTING_ROUTES_OUTPUT,
        save_results=False,
    )

    for i, route in enumerate(bird_results.routes):
        print(f"Route {i}: {route.distance_km} km")


if __name__ == "__main__":
    main()
