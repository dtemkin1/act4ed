# plots bird routes similarly to other map

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import osmnx as ox
import matplotlib as mpl

from experiments.existing_data.bird_routes import (
    get_bird_routes_json,
)
from experiments.existing_data.utils import get_assigned_students
from experiments.helpers import OUTPUTS_FOLDER, setup_framingham
from formulation.common.classes import Depot, NodeId, Place, School, Stop
from formulation.common.problems import FilteredProblemData, ProblemData
from formulation.normalized_result import (
    NormalizedRoutingResult,
    RoutingSolutionJson,
    RoutingSolutionRow,
)


def plot_bird_routes(
    name: str,
    routes: NormalizedRoutingResult | RoutingSolutionJson,
    problem_data: ProblemData,
    save_fig: bool = True,
) -> tuple[Figure, Axes]:
    if isinstance(routes, NormalizedRoutingResult):
        return plot_bird_routes_normalized(name, routes, problem_data, save_fig)
    elif isinstance(routes, RoutingSolutionJson):
        return plot_bird_routes_json(name, routes, problem_data, save_fig)


def plot_bird_routes_json(
    name: str,
    routes_json: RoutingSolutionJson,
    problem_data: ProblemData,
    save_fig: bool = True,
) -> tuple[Figure, Axes]:
    graph = problem_data.base_graph

    if "crs" not in graph.graph:
        graph.graph["crs"] = "EPSG:3857"  # uses meters

    pos = {
        node: (
            graph.nodes[node]["x"],
            graph.nodes[node]["y"],
        )
        for node in graph.nodes()
    }

    fig, ax = ox.plot_graph(graph, node_size=8, show=False)

    schools_plotted = set()
    depots_plotted = set()
    stops_plotted = set()

    colormap = mpl.colormaps["hsv"]

    bus_names = set()
    for route in routes_json.solution:
        bus_names.add(route.bus_name)

    for i, bus_name in enumerate(bus_names):
        total_route: list[Place] = []
        all_nodes: list[NodeId] = []

        bus_routes: list[RoutingSolutionRow] = list(
            filter(lambda r: r.bus_name == bus_name, routes_json.solution)
        )
        for route in bus_routes:
            depot = problem_data.depots[0]
            stops: list[Stop] = []
            for stop_id in route.stop_node_ids or []:
                stop_filter = filter(lambda s: s.node_id == stop_id, problem_data.stops)
                stops.extend(list(stop_filter))
            school = list(
                filter(
                    lambda s: s.name == route.school_name,
                    problem_data.schools,
                )
            )[0]

            places = [depot] + stops + [school]
            total_route += places

        _, all_nodes = problem_data.get_shortest_paths_base(
            tuple(map(lambda p: p.node_id, total_route))
        )

        ox.plot_graph_route(
            graph,
            list(all_nodes),
            route_color=colormap(i / len(bus_names)),  # type: ignore
            orig_dest_size=0,
            ax=ax,
            route_alpha=0.2,
            show=False,
        )

        for place in total_route:
            if isinstance(place, Depot) and place not in depots_plotted:
                ax.scatter(
                    pos[place.node_id][0],
                    pos[place.node_id][1],
                    c="black",
                    marker="X",
                    label=place.name if place not in depots_plotted else "",
                    zorder=4,
                    s=16,
                )
                depots_plotted.add(place)

            if isinstance(place, School) and place not in schools_plotted:
                ax.scatter(
                    pos[place.node_id][0],
                    pos[place.node_id][1],
                    c="red",
                    marker="s",
                    label=place.name if place not in schools_plotted else "",
                    zorder=3,
                    s=16,
                )
                schools_plotted.add(place)

            if isinstance(place, Stop) and place.node_id not in stops_plotted:
                ax.scatter(
                    pos[place.node_id][0],
                    pos[place.node_id][1],
                    c="tab:blue",
                    marker="o",
                    s=8,
                )
                stops_plotted.add(place.node_id)

    ax.title.set_text("BiRD School Bus Routes")
    ax.legend(loc="upper right", fontsize="small")

    if save_fig:
        fig.savefig(OUTPUTS_FOLDER / f"bird_{name}.png", bbox_inches="tight")

    return fig, ax


def plot_bird_routes_normalized(
    name: str,
    routes: NormalizedRoutingResult,
    problem_data: ProblemData,
    save_fig: bool = True,
) -> tuple[Figure, Axes]:

    graph = problem_data.base_graph

    if "crs" not in graph.graph:
        graph.graph["crs"] = "EPSG:3857"  # uses meters

    pos = {
        node: (
            graph.nodes[node]["x"],
            graph.nodes[node]["y"],
        )
        for node in graph.nodes()
    }

    fig, ax = ox.plot_graph(graph, node_size=8, show=False)

    schools_plotted = set()
    depots_plotted = set()
    stops_plotted = set()

    colormap = mpl.colormaps["hsv"]

    total_route: list[Place] = []
    all_nodes: list[NodeId] = []
    for i, route in enumerate(routes.itineraries):
        for j, route_order in enumerate(route.route_orders):
            bird_route = routes.routes[route_order]
            depot = problem_data.depots[0]
            stops: list[Stop] = []
            for stop_id in bird_route.stop_ids:
                stop_filter = filter(lambda s: s.name == stop_id, problem_data.stops)
                stops.append(list(stop_filter)[0])
            school = list(
                filter(
                    lambda s: s.id == bird_route.school_id,
                    problem_data.schools,
                )
            )[0]

            places = ([depot] if j == 0 else []) + stops + [school]
            total_route += places

        _, all_nodes = problem_data.get_shortest_paths_base(
            tuple(map(lambda p: p.node_id, total_route))
        )

        ox.plot_graph_route(
            graph,
            list(all_nodes),
            route_color=colormap(i / len(routes.itineraries)),  # type: ignore
            orig_dest_size=0,
            ax=ax,
            route_alpha=0.2,
            show=False,
        )

    for place in total_route:
        if isinstance(place, Depot) and place not in depots_plotted:
            ax.scatter(
                pos[place.node_id][0],
                pos[place.node_id][1],
                c="black",
                marker="X",
                label=place.name if place not in depots_plotted else "",
                zorder=4,
                s=16,
            )
            depots_plotted.add(place)

        if isinstance(place, School) and place not in schools_plotted:
            ax.scatter(
                pos[place.node_id][0],
                pos[place.node_id][1],
                c="red",
                marker="s",
                label=place.name if place not in schools_plotted else "",
                zorder=3,
                s=16,
            )
            schools_plotted.add(place)

        if isinstance(place, Stop) and place.node_id not in stops_plotted:
            ax.scatter(
                pos[place.node_id][0],
                pos[place.node_id][1],
                c="tab:blue",
                marker="o",
                s=8,
            )
            stops_plotted.add(place.node_id)

    ax.title.set_text("BiRD School Bus Routes")
    ax.legend(loc="upper right", fontsize="small")

    if save_fig:
        fig.savefig(OUTPUTS_FOLDER / f"bird_{name}.png", bbox_inches="tight")

    return fig, ax


def main() -> None:
    # load existing routes
    framingham_problem_data = setup_framingham(precompute_cache=True)
    assigned_students = get_assigned_students(
        framingham_problem_data.schools, framingham_problem_data.stops
    )

    filtered_problem_data = FilteredProblemData(
        "framingham_filtered",
        base_problem_data=framingham_problem_data,
        _students=assigned_students,
    )

    filtered_bird_results = get_bird_routes_json(
        "existing_student_routes",
    )

    plot_bird_routes(
        "existing_student_routes",
        filtered_bird_results,
        problem_data=filtered_problem_data,
        save_fig=True,
    )


if __name__ == "__main__":
    main()
