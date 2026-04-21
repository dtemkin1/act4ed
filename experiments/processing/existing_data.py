"""
This file contains code to load the existing bus routes from the data, and plot them on a map.
The main function is `get_existing_routes`, which returns a list of `RouteResult` objects,
each representing a bus route with the depot, stops, and school.

The `plot_existing_routes` function takes these routes and plots them on a map using OSMnx and Matplotlib.
"""

import json
import os
from pathlib import Path
from datetime import time
from typing import NamedTuple, TypedDict

import pandas as pd
import osmnx as ox
import matplotlib.pyplot as plt
import matplotlib as mpl

from experiments.helpers import setup_framingham
from formulation.common.problems import ProblemData
from formulation.common.classes import Stop

CURRENT_FILE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

ASSIGNED_STUDENTS = CURRENT_FILE_DIR / ".." / "data" / "assigned_students.csv"
BUSES = CURRENT_FILE_DIR / ".." / "data" / "buses.csv"

OUTPUT_ROUTES = CURRENT_FILE_DIR / ".." / "outputs" / "existing_routes.json"


class RawBusRoutes(NamedTuple):
    bus_name: str
    stop_name: str
    time: time


class RouteResult(TypedDict):
    bus_name: str
    destination_node_id: int
    distance_km: float
    end_time: float
    origin_node_id: int
    round: int
    school_name: str
    start_time: float
    stop_node_ids: list[int]
    student_names: list[str]
    students_served: int
    time_spent: float


class SolutionMetadata(TypedDict):
    backend: str
    buses_used: int
    objective_value: float
    runtime_seconds: float
    status: str
    total_distance_km: float
    total_students_served: int


def get_raw_assigned_buses() -> tuple[set[RawBusRoutes], dict[str, str]]:
    """
    Gets the raw assigned buses from the data, without filtering for only those that are in our problem data.

    Returns: A two-element tuple of the following:
            A set of tuples of the form (bus_id, stop_id, time).
            A dictionary of the form {bus_id: school_id}.
    """
    assigned_students = pd.read_csv(
        ASSIGNED_STUDENTS,
        encoding="utf-8",
        dtype={
            "Student_District ID": str,
            "Student_First Name": str,
            "Student_Last Name": str,
            "Student_Program": str,
            "Student_School": str,
            "BUS": str,
            "P/U D/O TIME": str,
            "BUS STOP": str,
        },
    )

    # filter if no id
    assigned_students = assigned_students[
        assigned_students["Student_District ID"] != ""
    ]

    # filter if no bus stop
    assigned_students = assigned_students[assigned_students["BUS STOP"] != ""]

    # filter for only morning times (each student has morning and afternoon) (in 24 hr time)
    assigned_students = assigned_students[
        assigned_students["P/U D/O TIME"].str.split(":").str[0].astype(int) < 12
    ]

    schools_to_bus = {
        (
            row["BUS"]
            if row["BUS"].startswith("M")
            else ("FRAM" + (len(row["BUS"]) < 2 and "0" or "") + row["BUS"])
        ): row["Student_School"]
        for _, row in assigned_students.iterrows()
    }

    return {
        RawBusRoutes(
            bus_name=(
                row["BUS"]
                if row["BUS"].startswith("M")
                else ("FRAM" + (len(row["BUS"]) < 2 and "0" or "") + row["BUS"])
            ),
            stop_name=row["BUS STOP"],
            time=time(
                int(row["P/U D/O TIME"].split(":")[0]),
                int(row["P/U D/O TIME"].split(":")[1]),
            ),
        )
        for _, row in assigned_students.iterrows()
    }, schools_to_bus


def ordered_routes(raw_buses: set[RawBusRoutes]) -> dict[str, list[str]]:
    """
    Orders the raw bus routes by time, and returns a dictionary of the form {bus_id: [list of stop_ids in order]}.
    """
    bus_to_stops_times: dict[str, list[tuple[str, time]]] = {}
    for bus_route in raw_buses:
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
        total_time = (end_time.hour * 60.0 + end_time.minute) - (
            start_time.hour * 60.0 + start_time.minute
        )

        # get students served by this bus, using students at the stop who are going to the school destination
        students_served = []
        for student in problem_data.students:
            if student.stop in stops and student.school == school:
                students_served.append(student)

        total_distance_km = 0.0
        for i in range(len(stops) - 1):
            stop_before = stops[i]
            stop_after = stops[i + 1]

            edge_data = problem_data.service_graph.get_edge_data(
                stop_before.node_id, stop_after.node_id, 0
            )
            if edge_data is None:
                print(
                    f"Warning: No edge found between stop {stop_before.name} and stop {stop_after.name} for bus {bus_name}. Skipping this edge."
                )
                continue

            total_distance_km += edge_data["length"]

        route_result = RouteResult(
            bus_name=bus_name,
            destination_node_id=school.node_id,
            distance_km=total_distance_km,
            end_time=end_time.hour * 60.0 + end_time.minute,
            origin_node_id=depot.node_id,
            round=1,  # only 1 round in real world data
            school_name=school_name,
            start_time=start_time.hour * 60.0 + start_time.minute,
            stop_node_ids=[stop.node_id for stop in stops],
            student_names=[student.name for student in students_served],
            students_served=len(students_served),
            time_spent=total_time,
        )

        bus_to_stops_with_depot_and_school.append(route_result)
    return bus_to_stops_with_depot_and_school


def get_solution_metadata(
    bus_to_stops_with_depot_and_school: list[RouteResult],
) -> SolutionMetadata:
    total_distance_km = sum(
        route["distance_km"] for route in bus_to_stops_with_depot_and_school
    )
    total_students_served = sum(
        route["students_served"] for route in bus_to_stops_with_depot_and_school
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
    routes: list[RouteResult], problem_data: ProblemData
) -> tuple[plt.Figure, plt.Axes]:
    graph = problem_data.base_graph
    service_graph = problem_data.service_graph
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

    for i, route in enumerate(routes):
        start = route["origin_node_id"]
        end = route["destination_node_id"]
        stop_node_ids = route["stop_node_ids"]

        all_places = [start] + stop_node_ids + [end]
        all_nodes = []
        for i in range(len(all_places) - 1):
            node1 = all_places[i]
            node2 = all_places[i + 1]
            edge_data = service_graph.get_edge_data(node1, node2, 0)
            if edge_data is None:
                print(f"Warning: No edge found between node {node1} and node {node2}.")
                continue

            path = list(edge_data["path"])
            while all_nodes and path and path[0] == all_nodes[-1]:
                path = path[1:]

            all_nodes.extend(path)

        ox.plot_graph_route(
            graph,
            all_nodes,
            route_color=colormap(i / len(routes)),
            orig_dest_size=0,
            ax=ax,
            route_alpha=0.2,
            show=False,
        )

        if start not in depots_plotted:
            ax.scatter(
                pos[start][0],
                pos[start][1],
                c="black",
                marker="X",
                label="Depot" if start not in depots_plotted else "",
                s=16,
            )
            depots_plotted.add(start)

        if end not in schools_plotted:
            ax.scatter(
                pos[end][0],
                pos[end][1],
                c="red",
                marker="s",
                label=route["school_name"] if end not in schools_plotted else "",
                s=16,
            )
            schools_plotted.add(end)

        for node_id in stop_node_ids:
            if node_id not in stops_plotted:
                ax.scatter(
                    pos[node_id][0],
                    pos[node_id][1],
                    c="tab:blue",
                    marker="o",
                    s=8,
                )
                stops_plotted.add(node_id)

    ax.title.set_text("Existing School Bus Routes")
    ax.legend(loc="upper right", fontsize="small")

    fig.savefig(
        CURRENT_FILE_DIR / "outputs" / "existing_routes.png", bbox_inches="tight"
    )
    return fig, ax


def main() -> None:
    problem_data = setup_framingham()

    existing_routes = get_existing_routes(problem_data)
    metadata = get_solution_metadata(existing_routes)

    plot_existing_routes(existing_routes, problem_data)

    # Save the routes to a JSON file
    with open(OUTPUT_ROUTES, "w") as f:
        json.dump(
            {"metadata": metadata, "solution": existing_routes},
            f,
            indent=4,
        )


if __name__ == "__main__":
    main()
