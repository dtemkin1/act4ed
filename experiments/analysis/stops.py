# get stops without students

from experiments.existing_data.utils import get_assigned_students
from experiments.helpers import setup_framingham


def main() -> None:
    framingham_data = setup_framingham(precompute_cache=True)

    assigned_students = get_assigned_students(
        framingham_data.schools, framingham_data.stops
    )
    assigned_stops = set(s.stop for s in assigned_students)
    unassigned_stops = set(s for s in framingham_data.stops if s not in assigned_stops)

    print(f"Number of unassigned stops: {len(unassigned_stops)}")


if __name__ == "__main__":
    main()
