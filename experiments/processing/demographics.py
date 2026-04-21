import os
from pathlib import Path
from typing import NamedTuple
import json

from experiments.helpers import get_assigned_students, setup_framingham
from formulation.common.classes import Stop, Student
from formulation.common.problems import ProblemDataReal

CURRENT_FILE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

DUMP_FILE = CURRENT_FILE_DIR / ".." / "outputs" / "demographics_of_stops.json"


class DemographicOfStop(NamedTuple):
    stop_id: int
    total_students: int
    special_ed_students: int
    wheelchair_users: int


def get_demographics_of_stops(
    problem_data: ProblemDataReal, assigned_students: tuple[Student, ...]
) -> dict[Stop, DemographicOfStop]:
    demographics_of_stops: dict[Stop, DemographicOfStop] = {}
    for stop in problem_data.stops:
        students_at_stop = [
            student for student in assigned_students if student.stop == stop
        ]
        demographics_of_stops[stop] = DemographicOfStop(
            stop_id=stop.node_id,
            total_students=len(students_at_stop),
            special_ed_students=sum(
                student.demographics.special_ed for student in students_at_stop
            ),
            wheelchair_users=sum(
                student.demographics.wheelchair_user for student in students_at_stop
            ),
        )
    return demographics_of_stops


def main() -> None:
    problem_data = setup_framingham()
    assigned_students = get_assigned_students(problem_data)

    demographics_of_stops = get_demographics_of_stops(problem_data, assigned_students)

    stops_no_data = [
        stop.name
        for stop in problem_data.stops
        if stop not in demographics_of_stops.keys()
    ]

    stops_data = {
        stop.name: demographics._asdict()
        for stop, demographics in demographics_of_stops.items()
    }

    with open(DUMP_FILE, "w") as f:
        json.dump({"no_data": stops_no_data, "data": stops_data}, f, indent=4)


if __name__ == "__main__":
    main()
