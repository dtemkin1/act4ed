# plots bird routes similarly to other map

import itertools

import networkx as nx
import osmnx as ox
import pandas as pd
from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from experiments.existing_data.bird_routes import (BIRD_CONFIG,
                                                   get_bird_routes,
                                                   get_bird_routes_json)
from experiments.helpers import OUTPUTS_FOLDER, setup_framingham
from formulation.common.classes import Depot, NodeId, Place, School, Stop
from formulation.common.problems import ProblemData
from formulation.normalized_result import (NormalizedRoutingResult,
                                           RoutingSolutionJson,
                                           RoutingSolutionRow)


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

    edges = pd.Series(nx.edges(graph))
    edges_weight = edges.map(lambda edge: 0.5)  # default weight of 0.5 for all edges
    edges_colors = edges.map(
        lambda edge: colors.to_hex("darkgray")
    )  # default color for all edges

    node_colors: dict[NodeId, str] = {node: "darkgray" for node in nx.nodes(graph)}

    bus_names: set[str] = set()
    for route in routes_json.solution:
        bus_names.add(route.bus_name)

    total_routes: dict[str, list[Place]] = {bus_name: [] for bus_name in bus_names}

    for bus_name in bus_names:
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

        total_routes[bus_name] = total_route

        _, all_nodes = problem_data.get_shortest_paths_base(
            tuple(map(lambda p: p.node_id, total_route))
        )

        for u, v in itertools.pairwise(all_nodes):
            if (edges == (u, v)).any():
                index = (edges == (u, v)).idxmax()
                edges_weight[
                    index
                ] += 0.5  # increase weight by 1.0 for each route that uses this edge
                edges_colors[index] = colors.to_hex(
                    "w"
                )  # change color to orange for edges
            if u in node_colors:
                node_colors[u] = "w"
            if v in node_colors:
                node_colors[v] = "w"

    fig, ax = ox.plot_graph(
        graph,
        node_size=0,
        node_color=[colors.to_hex(color) for color in node_colors.values()],
        edge_color=edges_colors.tolist(),
        edge_linewidth=edges_weight.tolist(),
        show=False,
    )

    schools_plotted = set()
    depots_plotted = set()

    for total_route in total_routes.values():
        for place in total_route:
            if isinstance(place, Depot) and place not in depots_plotted:
                ax.scatter(
                    pos[place.node_id][0],
                    pos[place.node_id][1],
                    c="tab:blue",
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
                    c="tab:red",
                    marker="s",
                    label=place.name if place not in schools_plotted else "",
                    zorder=3,
                    s=16,
                )
                schools_plotted.add(place)

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

    edges = pd.Series(nx.edges(graph))
    edges_weight = edges.map(lambda edge: 0.5)  # default weight of 0.5 for all edges
    edges_colors = edges.map(
        lambda edge: colors.to_hex("darkgray")
    )  # default color for all edges

    node_colors: dict[NodeId, str] = {node: "darkgray" for node in nx.nodes(graph)}

    bus_names = set[str]()
    for itinerary in routes.itineraries:
        bus_names.add(itinerary.bus_id)
    bus_names = list(sorted(bus_names))

    total_routes: dict[str, list[Place]] = {bus_name: [] for bus_name in bus_names}

    for route in routes.itineraries:
        total_route: list[Place] = []
        bus_id = route.bus_id
        for j, route_order in enumerate(route.route_orders):
            bird_route = list(
                filter(
                    lambda r: r.bus_id == bus_id and r.order == route_order,
                    routes.routes,
                )
            )[0]
            depot = problem_data.depots[0]
            stops: list[Stop] = []
            for stop_id in bird_route.stop_ids:
                stop_filter = filter(lambda s: s.name == stop_id, problem_data.stops)
                stops.extend(list(stop_filter))
            school = list(
                filter(
                    lambda s: s.id == bird_route.school_id,
                    problem_data.schools,
                )
            )[0]

            places = ([depot] if j == 0 else []) + stops + [school]

            while (
                total_route and places and total_route[-1].node_id == places[0].node_id
            ):
                places = places[1:]

            total_route += places

        total_routes[bus_id] = total_route

        _, all_nodes = problem_data.get_shortest_paths_base(
            tuple(map(lambda p: p.node_id, total_route))
        )

        for u, v in itertools.pairwise(all_nodes):
            if (edges == (u, v)).any():
                index = (edges == (u, v)).idxmax()
                edges_weight[
                    index
                ] += 0.5  # increase weight by 1.0 for each route that uses this edge
                edges_colors[index] = colors.to_hex(
                    "w"
                )  # change color to orange for edges
            if u in node_colors:
                node_colors[u] = "w"
            if v in node_colors:
                node_colors[v] = "w"

    fig, ax = ox.plot_graph(
        graph,
        node_size=0,
        node_color=[colors.to_hex(color) for color in node_colors.values()],
        edge_color=edges_colors.tolist(),
        edge_linewidth=edges_weight.tolist(),
        show=False,
    )

    schools_plotted = set()
    depots_plotted = set()

    for total_route in total_routes.values():
        for place in total_route:
            if isinstance(place, Depot) and place not in depots_plotted:
                ax.scatter(
                    pos[place.node_id][0],
                    pos[place.node_id][1],
                    c="tab:blue",
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
                    c="tab:red",
                    marker="s",
                    label=place.name if place not in schools_plotted else "",
                    zorder=3,
                    s=16,
                )
                schools_plotted.add(place)

    ax.legend(loc="upper right", fontsize="small")

    if save_fig:
        fig.savefig(OUTPUTS_FOLDER / f"bird_{name}.png", bbox_inches="tight")

    return fig, ax


def main() -> None:
    # load existing routes
    framingham_problem_data = setup_framingham(precompute_cache=True)

    filtered_bird_results = get_bird_routes_json(
        "assigned_students_routes",
    )

    plot_bird_routes(
        "assigned_students_routes",
        filtered_bird_results,
        problem_data=framingham_problem_data,
        save_fig=True,
    )

    all_students_distance = get_bird_routes_json(
        "1_5_mile_students_routes",
    )

    plot_bird_routes(
        "all_students_over_distance",
        all_students_distance,
        problem_data=framingham_problem_data,
        save_fig=True,
    )

    all_students = get_bird_routes_json(
        "all_students_routes",
    )

    plot_bird_routes(
        "all_students",
        all_students,
        problem_data=framingham_problem_data,
        save_fig=True,
    )


if __name__ == "__main__":
    main()
