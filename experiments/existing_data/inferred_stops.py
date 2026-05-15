from dataclasses import replace
from random import random

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from experiments.existing_data.utils import (get_assigned_students,
                                             get_raw_assigned_students)
from experiments.helpers import (DATA_FOLDER, OUTPUTS_FOLDER,
                                 make_students_csv, setup_framingham)
from formulation.common.classes import Attributes, Student
from formulation.common.constants import METERS_PER_MILE
from formulation.common.problems import ProblemDataReal

ASSIGNED_STUDENTS = DATA_FOLDER / "assigned_students.csv"
BUSES = DATA_FOLDER / "buses.csv"
OUTPUT_ROUTES = OUTPUTS_FOLDER / "existing_routes.json"


def more_realistic_students(problem_data: ProblemDataReal) -> tuple[Student, ...]:
    """
    currently, student data has random special ed status. we can use the assigned_students.csv file
    to get a more realistic set of students with special ed status,
    and then update the problem data with this information.

    we do this by calculating the ratio of special ed students at each stop, then randomly assigning
    special ed status to students at that stop based on that ratio.
    """

    students = list(problem_data.students)
    new_students: list[Student] = []

    assigned_students = get_raw_assigned_students()

    stop_name_to_special_ed_ratio: dict[str, float] = {}
    for stop_name, group in assigned_students.groupby("BUS STOP"):
        special_ed_count = group["Student_Program"].str.contains("SPED").sum()
        total_count = len(group)
        stop_name_to_special_ed_ratio[str(stop_name)] = special_ed_count / total_count

    for student in students:
        stop_name = student.stop.name
        if stop_name in stop_name_to_special_ed_ratio:
            ratio = stop_name_to_special_ed_ratio[stop_name]
            new_student = replace(
                student,
                attributes=Attributes(
                    special_ed=random() < ratio,
                    wheelchair_user=student.attributes.wheelchair_user,
                ),
            )
            new_students.append(new_student)
        else:
            new_students.append(student)

    return tuple(new_students)


def plot_special_education_students(problem_data: ProblemDataReal) -> None:
    """
    Plots the location of special education students,
    colored by how far they are from their school.
    """

    students = get_assigned_students(problem_data.schools, problem_data.stops)
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
    all_distances = {
        student: problem_data.get_shortest_path_base(
            student.stop.node_id, student.school.node_id
        )[0]
        / METERS_PER_MILE  # convert to miles
        for student in special_education_students
        if student.school is not None
    }
    color_gradient = plt.get_cmap("RdYlGn_r")
    norm = Normalize(vmin=min(all_distances.values()), vmax=max(all_distances.values()))
    sm = plt.cm.ScalarMappable(cmap=color_gradient, norm=norm)
    sm.set_array([])

    for student, distance in all_distances.items():
        ax.scatter(
            student.geographic_location.x,
            student.geographic_location.y,
            color=color_gradient(norm(distance)),
        )

    # add gradient legend
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Distance to School (miles)")

    # remove axis borders and ticks
    ax.set_axis_off()
    fig.savefig(
        OUTPUTS_FOLDER / "special_education_students.pdf",
        bbox_inches="tight",
    )


def main() -> None:
    problem_data = setup_framingham()

    print(f"Number of assigned students: {len(get_raw_assigned_students())}")

    make_students_csv(
        more_realistic_students(problem_data),
        path=DATA_FOLDER / "students_with_special_ed_inferred.csv",
    )

    plot_special_education_students(problem_data)


if __name__ == "__main__":
    main()
