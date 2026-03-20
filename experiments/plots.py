import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from formulation.common import ProblemData, Student


CURRENT_FILE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))


def get_assigned_students(problem_data: ProblemData) -> list[Student]:
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

    students = []
    for _, row in assigned_students.iterrows():
        requires_monitor = "SPED" in row["Student_Program"]
        # am not sure this is how they mark it, follow up
        requires_wheelchair = "WHEELCHAIR" in row["Student_Program"]
        stop = next(stop for stop in problem_data.stops if stop.name == row["BUS STOP"])

        student = Student(
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
            requires_monitor=requires_monitor,
            requires_wheelchair=requires_wheelchair,
        )
        students.append(student)

    return students


def plot_special_education_students(problem_data: ProblemData) -> None:
    """
    Plots the location of special education students,
    colored by how far they are from their school.
    """

    students = get_assigned_students(problem_data)
    special_education_students = [
        student
        for student in students
        if student.requires_monitor or student.requires_wheelchair
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
    cbar.set_label("Distance to School (m)")

    # remove axis borders and ticks
    ax.set_axis_off()
    fig.savefig(
        CURRENT_FILE_DIR / "outputs" / "special_education_students.png",
        dpi=300,
        bbox_inches="tight",
    )
