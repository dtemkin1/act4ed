import os
from pathlib import Path
from typing import Literal, overload

import pandas as pd
from networkx import MultiDiGraph
from shapely import Point

from formulation.common.classes import NodeId, Student
from formulation.common.problems import (ProblemDataReal,
                                         ProblemDataRealSurrogate)

CURRENT_FILE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# where
BOUNDARY_BUFFER_KM = 1.0

# data files
DATA_FOLDER = CURRENT_FILE_DIR / "data"
DEPOT_CSV = DATA_FOLDER / "depot.csv"
SCHOOLS_CSV = DATA_FOLDER / "schools.csv"
STOPS_CSV = DATA_FOLDER / "stops.csv"
STUDENTS_CSV = DATA_FOLDER / "students.csv"
BUSES_CSV = DATA_FOLDER / "buses.csv"

# thresholds
MAX_WALK_TIME_S = 15 * 60  # 15 mins
MAX_WALK_DIST_KM = 1.0  # 1 km

# base street network
NETWORK_TYPE = "drive"

# outputs
OUTPUTS_FOLDER = CURRENT_FILE_DIR / "outputs"
GRAPHML_FILE = OUTPUTS_FOLDER / "framingham_graph.graphml"
PAIRWISE_CSV = OUTPUTS_FOLDER / "depot_schools_stops_pairwise.csv"
STUDENT_ASSIGN_CSV = OUTPUTS_FOLDER / "student_to_stop_or_school.csv"

FRAMINGHAM_NAME = "Framingham, Massachusetts, USA"


def setup(
    problem_name: str,
    place_name: str,
    prune: int | None = None,
    schools_path: Path = SCHOOLS_CSV,
    stops_path: Path = STOPS_CSV,
    students_path: Path = STUDENTS_CSV,
    depots_path: Path = DEPOT_CSV,
    buses_path: Path = BUSES_CSV,
    boundary_buffer_km: float = BOUNDARY_BUFFER_KM,
    hexagonal: bool = False,
    save_path: Path | None = None,
    sanity_check: bool = False,
    precompute_cache: bool = True,
) -> ProblemDataReal | ProblemDataRealSurrogate:
    ProblemDataClass = ProblemDataRealSurrogate if hexagonal else ProblemDataReal
    should_save = False

    try:
        problem_data = ProblemDataClass.load(problem_name, prune)
    except FileNotFoundError:
        problem_data = ProblemDataClass(
            name=problem_name,
            schools_path=schools_path,
            stops_path=stops_path,
            students_path=students_path,
            depots_path=depots_path,
            buses_path=buses_path,
            place_name=place_name,
            boundary_buffer_km=boundary_buffer_km,
            prune=prune,
        )

        if sanity_check:
            problem_data.sanity_checks()

        should_save = True

    if precompute_cache and "_service_graph_cached" not in vars(problem_data):
        print(f"Precomputing service graph for cache: {problem_data.name}")
        _ = problem_data.service_graph
        should_save = True

    if should_save:
        problem_data.save(cache_dir=save_path)

    return problem_data


@overload
def setup_framingham(
    hexagonal: Literal[False] = False,
    prune: int | None = None,
    sanity_check: bool = False,
    precompute_cache: bool = True,
) -> ProblemDataReal: ...


@overload
def setup_framingham(
    hexagonal: Literal[True],
    prune: None = None,
    sanity_check: bool = False,
    precompute_cache: bool = True,
) -> ProblemDataRealSurrogate: ...


def setup_framingham(
    hexagonal: bool = False,
    prune: int | None = None,
    sanity_check: bool = False,
    precompute_cache: bool = True,
) -> ProblemDataReal | ProblemDataRealSurrogate:
    return setup(
        problem_name="framingham",
        place_name=FRAMINGHAM_NAME,
        hexagonal=hexagonal,
        prune=prune,
        sanity_check=sanity_check,
        precompute_cache=precompute_cache,
    )


def make_osm_in_km(graph: "MultiDiGraph[NodeId]") -> "MultiDiGraph[NodeId]":
    """Converts the OSM graph from meters to kilometers for easier interpretation"""

    graph_km = graph.copy()
    for u, v, key, data in graph_km.edges(keys=True, data=True):
        graph_km.edges[u, v, key]["length"] = data["length"] / 1000.0

    return graph_km


def make_point_from_node_id(graph: "MultiDiGraph[NodeId]", node_id: NodeId) -> Point:
    """Helper function to make a Point from a node id in the graph"""

    return Point(graph.nodes[node_id]["x"], graph.nodes[node_id]["y"])


def make_students_csv(students: tuple[Student, ...], path: Path | None = None) -> pd.DataFrame:
    """Helper function to create a CSV file of students from the problem data"""

    # id,lon,lat,school_id,is_sp_ed,is_wheelchair_user
    df = pd.DataFrame(
        [
            {
                "id": student.id,
                "name": student.name,
                "lon": student.geographic_location.x,
                "lat": student.geographic_location.y,
                "school_id": student.school.id,
                "is_sp_ed": student.attributes.special_ed,
                "is_wheelchair_user": student.attributes.wheelchair_user,
            }
            for student in students
        ]
    )

    if path is not None:
        df.to_csv(path, index=False)

    return df


if __name__ == "__main__":
    problem_data = setup_framingham()
