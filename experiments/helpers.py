import os
from pathlib import Path
from typing import Literal, overload

from networkx import MultiDiGraph
import pandas as pd
from shapely import Point

from formulation.common.classes import Attributes, NodeId, Student
from formulation.common.problems import ProblemDataReal, ProblemDataRealSurrogate

CURRENT_FILE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# where
BOUNDARY_BUFFER_KM = 1.0

# data files
DATA_FOLDER = CURRENT_FILE_DIR / "data"
DEPOT_CSV = DATA_FOLDER / "depot.csv"
SCHOOLS_CSV = DATA_FOLDER / "schools.csv"
STOPS_CSV = DATA_FOLDER / "stops.csv"
STUDENTS_CSV = DATA_FOLDER / "students_with_special_ed_inferred_w_grades.csv"
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


@overload
def setup(
    problem_name: str,
    place_name: str,
    prune: int | None = None,
    hexagonal: Literal[False] = False,
    save_path: Path | None = None,
    sanity_check: bool = False,
    precompute_cache: bool = True,
) -> ProblemDataReal: ...


@overload
def setup(
    problem_name: str,
    place_name: str,
    hexagonal: Literal[True],
    prune: None = None,
    save_path: Path | None = None,
    sanity_check: bool = False,
    precompute_cache: bool = True,
) -> ProblemDataRealSurrogate: ...


def setup(
    problem_name: str,
    place_name: str,
    hexagonal: bool = False,
    prune: int | None = None,
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
            schools_path=SCHOOLS_CSV,
            stops_path=STOPS_CSV,
            students_path=STUDENTS_CSV,
            depots_path=DEPOT_CSV,
            buses_path=BUSES_CSV,
            place_name=place_name,
            boundary_buffer_km=BOUNDARY_BUFFER_KM,
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
) -> ProblemDataReal: ...


@overload
def setup_framingham(
    hexagonal: Literal[True], prune: None = None, sanity_check: bool = False
) -> ProblemDataRealSurrogate: ...


def setup_framingham(
    hexagonal: bool = False, prune: int | None = None, sanity_check: bool = False
) -> ProblemDataReal | ProblemDataRealSurrogate:
    return setup(
        problem_name="framingham",
        place_name=FRAMINGHAM_NAME,
        hexagonal=hexagonal,
        prune=prune,
        sanity_check=sanity_check,
    )

def get_assigned_students(problem_data: ProblemDataReal) -> tuple[Student, ...]:
    """
    Uses the assigned_students.csv file to get a list of students
    with their assigned bus stops and schools. This is used for plotting the
    location of students, but this file does not include student addresses.
    """
    assigned_students = pd.read_csv(
        CURRENT_FILE_DIR / "data" / "assigned_students.csv",
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

    # filter for only students where we have a school match in our data
    school_names = set(school.name for school in problem_data.schools)
    assigned_students = assigned_students[
        assigned_students["Student_School"].isin(school_names)
    ]

    # filter for only students where we have a bus stop match in our data
    stop_names = set(stop.name for stop in problem_data.stops)
    assigned_students = assigned_students[
        assigned_students["BUS STOP"].isin(stop_names)
    ]

    students: list[Student] = []
    for _, row in assigned_students.iterrows():
        special_ed = "SPED" in row["Student_Program"]
        # am not sure this is how they mark it, follow up
        wheelchair_user = "WHEELCHAIR" in row["Student_Program"]
        student_id = str(
            row.get(
                "Student_District ID",
                f"{row['Student_First Name']} {row['Student_Last Name']}",
            )
        )
        grade = None
        for grade_column in ("Student_Grade", "Grade", "grade"):
            if grade_column in assigned_students.columns and not pd.isna(row[grade_column]):
                grade = str(row[grade_column])
                break
        stop = next(stop for stop in problem_data.stops if stop.name == row["BUS STOP"])

        student = Student(
            id=student_id,
            name=f"{row['Student_First Name']} {row['Student_Last Name']}",
            geographic_location=stop.geographic_location,
            school=next(
                school
                for school in problem_data.schools
                if school.name == row["Student_School"]
            ),
            stop=next(
                stop for stop in problem_data.stops if stop.name == row["BUS STOP"]
            ),
            attributes=Attributes(
                special_ed=special_ed, wheelchair_user=wheelchair_user
            ),
            grade=grade,
        )
        students.append(student)

    return tuple(students)


def plot_special_education_students(problem_data: ProblemDataReal) -> None:
    """
    Plots the location of special education students,
    colored by how far they are from their school.
    """

    students = get_assigned_students(problem_data)
    special_education_students = [
        student
        for student in students
        if student.attributes.special_ed or student.attributes.wheelchair_user
    ]

    # plot framingham graph with special education students highlighted
    gdf = problem_data.gdf
    fig, ax = plt.subplots()

    gdf.plot(ax=ax, color="white", edgecolor="black")

    # plot students, color based on how far they are from their school
    all_distances = [
        problem_data.service_graph.edges[
            student.stop.node_id, student.school.node_id, 0
        ]["length"]
        for student in special_education_students
        if student.school is not None
    ]
    color_gradient = plt.cm.get_cmap("RdYlGn_r")
    norm = plt.Normalize(vmin=min(all_distances), vmax=max(all_distances))
    sm = plt.cm.ScalarMappable(cmap=color_gradient, norm=norm)
    sm.set_array([])

    for student in special_education_students:
        distance = problem_data.service_graph.edges[
            student.stop.node_id, student.school.node_id, 0
        ]["length"]

        ax.scatter(
            student.geographic_location.x,
            student.geographic_location.y,
            color=color_gradient(norm(distance)),
        )

    ax.set_title("Location of Special Education Students")
    # add gradient legend
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Distance to School (km)")

    # remove axis borders and ticks
    ax.set_axis_off()
    fig.savefig(
        CURRENT_FILE_DIR / "outputs" / "special_education_students.png",
        dpi=300,
        bbox_inches="tight",
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


def make_students_csv(students: tuple[Student, ...], path: Path | None = None) -> None:
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
