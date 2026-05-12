from datetime import time
from functools import cache
from typing import NamedTuple

import pandas as pd

from experiments.helpers import DATA_FOLDER
from formulation.common.classes import Attributes, School, Stop, Student

ASSIGNED_STUDENTS = DATA_FOLDER / "assigned_students.csv"


class RawBusRoutes(NamedTuple):
    bus_name: str
    stop_name: str
    time: time


@cache
def get_raw_assigned_students() -> pd.DataFrame:
    """
    Gets the raw assigned students from the data, without filtering for only those that are in our problem data.

    Returns: A pandas DataFrame with the assigned students.
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

    return assigned_students


def get_raw_assigned_buses() -> tuple[set[RawBusRoutes], dict[str, str]]:
    """
    Gets the raw assigned buses from the data, without filtering for only those that are in our problem data.

    Returns: A two-element tuple of the following:
            A set of tuples of the form (bus_id, stop_id, time).
            A dictionary of the form {bus_id: school_id}.
    """
    assigned_students = get_raw_assigned_students()

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


def get_assigned_students(
    schools: tuple[School, ...], stops: tuple[Stop, ...]
) -> tuple[Student, ...]:
    """
    Uses the assigned_students.csv file to get a list of students
    with their assigned bus stops and schools. This is used for plotting the
    location of students, but this file does not include student addresses.
    """
    assigned_students = get_raw_assigned_students()

    # filter for only students where we have a school match in our data
    school_names = set(school.name for school in schools)
    assigned_students = assigned_students[
        assigned_students["Student_School"].isin(school_names)
    ]

    # filter for only students where we have a bus stop match in our data
    stop_names = set(stop.name for stop in stops)
    assigned_students = assigned_students[
        assigned_students["BUS STOP"].isin(stop_names)
    ]

    students: list[Student] = []
    for _, row in assigned_students.iterrows():
        special_ed = "SPED" in row["Student_Program"]
        # am not sure this is how they mark it, follow up
        wheelchair_user = "WHEELCHAIR" in row["Student_Program"]
        stop = next(stop for stop in stops if stop.name == row["BUS STOP"])

        student = Student(
            id=row["Student_District ID"],
            name=f"{row['Student_First Name']} {row['Student_Last Name']}",
            geographic_location=stop.geographic_location,
            school=next(
                school for school in schools if school.name == row["Student_School"]
            ),
            stop=next(stop for stop in stops if stop.name == row["BUS STOP"]),
            attributes=Attributes(
                special_ed=special_ed, wheelchair_user=wheelchair_user
            ),
        )
        students.append(student)

    return tuple(students)
