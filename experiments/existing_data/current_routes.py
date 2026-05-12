import itertools
import json
from dataclasses import dataclass
from datetime import time
from typing import TypedDict

import networkx as nx
import osmnx as ox
import pandas as pd
from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from experiments.existing_data.utils import (RawBusRoutes,
                                             get_raw_assigned_buses)
from experiments.helpers import OUTPUTS_FOLDER, setup_framingham
from formulation.common.classes import (Bus, Depot, NodeId, Place, School,
                                        Stop, Student)
from formulation.common.problems import ProblemData

OUTPUT_ROUTES = OUTPUTS_FOLDER / "existing_routes.json"


class RouteResultExport(TypedDict):
    bus_name: str
    destination_node_id: NodeId
    distance_km: float
    end_time: float
    origin_node_id: NodeId
    round: int
    school_name: str
    start_time: float
    stop_node_ids: list[NodeId]
    student_names: list[str]
    students_served: int
    time_spent: float


@dataclass
class RouteResult:
    bus: Bus
    school: School
    stops: tuple[Stop, ...]
    students_served: tuple[Student, ...]
    depot: Depot
    distance_km: float
    round: int
    end_time: time
    start_time: time
    path: tuple[NodeId, ...]

    @property
    def all_places(self) -> tuple[Place, ...]:
        """
        Returns a tuple of all node ids in the route, including the depot, stops, and school.
        Assumes the route goes from depot to stops to school in order.
        """
        return (self.depot,) + tuple(self.stops) + (self.school,)

    @property
    def export(self) -> RouteResultExport:
        return RouteResultExport(
            bus_name=self.bus.name,
            destination_node_id=self.school.node_id,
            distance_km=self.distance_km,
            end_time=self.end_time.hour * 60.0 + self.end_time.minute,
            origin_node_id=self.depot.node_id,
            round=self.round,
            school_name=self.school.name,
            start_time=self.start_time.hour * 60.0 + self.start_time.minute,
            stop_node_ids=[stop.node_id for stop in self.stops],
            student_names=[student.name for student in self.students_served],
            students_served=len(self.students_served),
            time_spent=(self.end_time.hour * 60.0 + self.end_time.minute)
            - (self.start_time.hour * 60.0 + self.start_time.minute),
        )


class SolutionMetadata(TypedDict):
    backend: str
    buses_used: int
    objective_value: float
    runtime_seconds: float
    status: str
    total_distance_km: float
    total_students_served: int


def ordered_routes(raw_buses: set[RawBusRoutes]) -> dict[str, list[str]]:
    """
    Orders the raw bus routes by time, and returns a dictionary of the form {bus_id: [list of stop_ids in order]}.
    """

    raw_buses_list = sorted(
        raw_buses, key=lambda x: x.bus_name
    )  # sort by bus name for consistency

    bus_to_stops_times: dict[str, list[tuple[str, time]]] = {}
    for bus_route in raw_buses_list:
        if bus_route.bus_name not in bus_to_stops_times:
            bus_to_stops_times[bus_route.bus_name] = []
        bus_to_stops_times[bus_route.bus_name].append(
            (bus_route.stop_name, bus_route.time)
        )

    # sort the stops for each bus by time
    bus_to_stops: dict[str, list[str]] = {}
    for bus_name, stops in bus_to_stops_times.items():
        bus_to_stops[bus_name] = [
            stop_id for stop_id, _ in sorted(stops, key=lambda x: x[1])
        ]

    return bus_to_stops


def add_depot_and_school_to_routes(
    raw_buses: set[RawBusRoutes],
    bus_to_stops: dict[str, list[str]],
    bus_to_school: dict[str, str],
    problem_data: ProblemData,
) -> list[RouteResult]:
    """
    Adds the depot and school to the bus routes, assuming the depot is always first and the school is always last.

    Returns: A dictionary of the form {bus_id: [depot_id, stop_id, ..., school_id]}.
    """
    bus_to_stops_with_depot_and_school: list[RouteResult] = []
    school_name_to_school = {school.name: school for school in problem_data.schools}
    bus_name_to_bus = {bus.name: bus for bus in problem_data.buses}
    stop_names_to_stops = {stop.name: stop for stop in problem_data.stops}

    for bus_name, stop_names in bus_to_stops.items():
        school_name = bus_to_school[bus_name]

        if school_name not in school_name_to_school:
            print(
                f"Warning: School {school_name} for bus {bus_name} not found in problem data. Skipping this bus."
            )
            continue
        if bus_name not in bus_name_to_bus:
            print(
                f"Warning: Bus {bus_name} not found in problem data. Skipping this bus."
            )
            continue

        bus = bus_name_to_bus[bus_name]
        depot = bus.depot
        school = school_name_to_school[school_name]

        stops: list[Stop] = []
        for stop_name in stop_names:
            if stop_name not in stop_names_to_stops:
                print(
                    f"Warning: Stop {stop_name} for bus {bus_name} not found in problem data. Skipping this stop."
                )
                continue
            stops.append(stop_names_to_stops[stop_name])

        start_time = min(
            bus_route.time for bus_route in raw_buses if bus_route.bus_name == bus_name
        )
        end_time = max(
            bus_route.time for bus_route in raw_buses if bus_route.bus_name == bus_name
        )

        # get students served by this bus, using students at the stop who are going to the school destination
        students_served = []
        for student in problem_data.students:
            if student.stop in stops and student.school == school:
                students_served.append(student)

        path_locations = [depot] + stops + [school]
        total_distance_m, all_nodes = problem_data.get_shortest_paths_base(
            tuple(place.node_id for place in path_locations)
        )
        total_distance_km = total_distance_m / 1000.0

        route_result = RouteResult(
            bus=bus,
            school=school,
            stops=tuple(stops),
            students_served=tuple(students_served),
            depot=depot,
            distance_km=total_distance_km,
            round=0,
            end_time=end_time,
            start_time=start_time,
            path=tuple(all_nodes),
        )

        bus_to_stops_with_depot_and_school.append(route_result)
    return bus_to_stops_with_depot_and_school


def get_solution_metadata(
    bus_to_stops_with_depot_and_school: list[RouteResult],
) -> SolutionMetadata:
    total_distance_km = sum(
        route.distance_km for route in bus_to_stops_with_depot_and_school
    )
    total_students_served = sum(
        len(route.students_served) for route in bus_to_stops_with_depot_and_school
    )

    return SolutionMetadata(
        backend="existing_data",
        buses_used=len(bus_to_stops_with_depot_and_school),
        objective_value=0.0,  # Placeholder value
        runtime_seconds=0.0,  # Placeholder value
        status="completed",
        total_distance_km=total_distance_km,
        total_students_served=total_students_served,
    )


def get_existing_routes(
    problem_data: ProblemData,
) -> list[RouteResult]:
    raw_buses, bus_to_school = get_raw_assigned_buses()
    bus_to_stops = ordered_routes(raw_buses)

    bus_to_stops_with_depot_and_school = add_depot_and_school_to_routes(
        raw_buses, bus_to_stops, bus_to_school, problem_data
    )
    return bus_to_stops_with_depot_and_school


def plot_existing_routes(
    routes: list[RouteResult], problem_data: ProblemData, save_fig: bool = True
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

    # in correct order for osmnx...
    edges = pd.Series(nx.edges(graph))
    edges_weight = edges.map(lambda edge: 0.5)  # default weight of 0.5 for all edges
    edges_colors = edges.map(
        lambda edge: colors.to_hex("darkgray")
    )  # default color for all edges

    node_colors: dict[NodeId, str] = {node: "darkgray" for node in nx.nodes(graph)}

    for route in routes:
        all_nodes = route.path

        # plot the route by increasing edge weight for each edge in the route, so that overlapping routes are visible
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

    for route in routes:
        depot = route.depot
        school = route.school

        if depot not in depots_plotted:
            ax.scatter(
                pos[depot.node_id][0],
                pos[depot.node_id][1],
                c="tab:blue",
                marker="X",
                label=depot.name if depot not in depots_plotted else "",
                zorder=4,
                s=16,
            )
            depots_plotted.add(depot)

        if school not in schools_plotted:
            ax.scatter(
                pos[school.node_id][0],
                pos[school.node_id][1],
                c="tab:red",
                marker="s",
                label=school.name if school not in schools_plotted else "",
                zorder=3,
                s=16,
            )
            schools_plotted.add(school)

    ax.legend(loc="upper right", fontsize="small")

    if save_fig:
        fig.savefig(OUTPUTS_FOLDER / "existing_routes.png", bbox_inches="tight")

    return fig, ax


def main() -> None:
    problem_data = setup_framingham()

    existing_routes = get_existing_routes(problem_data)
    metadata = get_solution_metadata(existing_routes)

    print(
        "There are currently {} routes, serving a total of {} students with a total distance of {:.2f} km.".format(
            metadata["buses_used"],
            metadata["total_students_served"],
            metadata["total_distance_km"],
        )
    )

    plot_existing_routes(existing_routes, problem_data)

    # Save the routes to a JSON file
    with open(OUTPUT_ROUTES, "w") as f:
        existing_routes_export = [route.export for route in existing_routes]
        json.dump(
            {"metadata": metadata, "solution": existing_routes_export},
            f,
            indent=4,
        )


if __name__ == "__main__":
    main()
