import json
import os
from pathlib import Path
from typing import NamedTuple

from experiments.existing_data.utils import get_assigned_students
from experiments.helpers import OUTPUTS_FOLDER, setup_framingham
from formulation.common.classes import Stop, Student
from formulation.common.problems import ProblemDataReal

CURRENT_FILE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

DUMP_FILE = OUTPUTS_FOLDER / "attributes_of_stops.json"


class AttributesOfStop(NamedTuple):
    stop_id: int
    total_students: int
    special_ed_students: int
    wheelchair_users: int


def get_attributes_of_stops(
    problem_data: ProblemDataReal, assigned_students: tuple[Student, ...]
) -> dict[Stop, AttributesOfStop]:
    attributes_of_stops: dict[Stop, AttributesOfStop] = {}
    for stop in problem_data.stops:
        students_at_stop = [
            student for student in assigned_students if student.stop == stop
        ]
        attributes_of_stops[stop] = AttributesOfStop(
            stop_id=stop.node_id,
            total_students=len(students_at_stop),
            special_ed_students=sum(
                student.attributes.special_ed for student in students_at_stop
            ),
            wheelchair_users=sum(
                student.attributes.wheelchair_user for student in students_at_stop
            ),
        )
    return attributes_of_stops


def main() -> None:
    problem_data = setup_framingham()
    assigned_students = get_assigned_students(problem_data.schools, problem_data.stops)

    attributes_of_stops = get_attributes_of_stops(problem_data, assigned_students)

    stops_no_data = [
        stop.name
        for stop in problem_data.stops
        if stop not in attributes_of_stops.keys()
    ]

    stops_data = {
        stop.name: attributes._asdict()
        for stop, attributes in attributes_of_stops.items()
    }

    with open(DUMP_FILE, "w") as f:
        json.dump({"no_data": stops_no_data, "data": stops_data}, f, indent=4)


if __name__ == "__main__":
    main()
