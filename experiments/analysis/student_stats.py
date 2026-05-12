from collections.abc import Callable

from experiments.existing_data.utils import get_assigned_students
from experiments.helpers import setup_framingham
from formulation.common.classes import IncomeLevel, Student
from formulation.common.constants import KM_PER_MILE
from formulation.common.problems import ProblemData


def distance_to_school(student: Student, problem_data: ProblemData) -> float:
    stop = student.stop
    school = student.school

    length_m, _ = problem_data.get_shortest_path_base(stop.node_id, school.node_id)

    length_km = length_m / 1000.0
    return length_km


def filter_students(
    students: tuple[Student, ...], filter: Callable[[Student], bool]
) -> tuple[Student, ...]:
    return tuple(student for student in students if filter(student))


def sum_over_students(
    students: tuple[Student, ...], func: Callable[[Student], float | int | bool]
) -> float:
    return sum(func(student) for student in students)


def avg_over_students(
    students: tuple[Student, ...], func: Callable[[Student], float | int | bool]
) -> float:
    total = sum(func(student) for student in students)
    count = len(students)
    return total / count if count > 0 else 0.0


def analyze_time_per_demographic() -> None:
    framingham_data = setup_framingham(precompute_cache=True)

    students = framingham_data.students
    assigned_students = get_assigned_students(
        framingham_data.schools, framingham_data.stops
    )

    students_english_proficient = filter_students(
        students, lambda s: s.demographics.english_at_home if s.demographics else False
    )
    students_not_english_proficient = filter_students(
        students,
        lambda s: not s.demographics.english_at_home if s.demographics else False,
    )
    students_not_car_owning = filter_students(
        students,
        lambda s: not s.demographics.owns_car if s.demographics else False,
    )
    students_car_owning = filter_students(
        students,
        lambda s: (s.demographics.owns_car if s.demographics else False),
    )
    students_special_education = filter_students(
        students,
        lambda s: s.attributes.special_ed or s.attributes.wheelchair_user,
    )
    students_within_1mi = filter_students(
        students,
        lambda s: distance_to_school(s, framingham_data) <= (1.0 * KM_PER_MILE),
    )
    students_within_2mi = filter_students(
        students,
        lambda s: 1.0 < distance_to_school(s, framingham_data) <= (2.0 * KM_PER_MILE),
    )
    students_over_5mi = filter_students(
        students, lambda s: distance_to_school(s, framingham_data) > (5.0 * KM_PER_MILE)
    )

    avg_distance_to_school = avg_over_students(
        students, lambda s: distance_to_school(s, framingham_data)
    )
    avg_assigned_distance_to_school = avg_over_students(
        assigned_students, lambda s: distance_to_school(s, framingham_data)
    )
    avg_distance_to_school_english_proficient = avg_over_students(
        students_english_proficient, lambda s: distance_to_school(s, framingham_data)
    )
    avg_distance_to_school_not_english_proficient = avg_over_students(
        students_not_english_proficient,
        lambda s: distance_to_school(s, framingham_data),
    )
    avg_distance_to_school_not_car_owning = avg_over_students(
        students_not_car_owning, lambda s: distance_to_school(s, framingham_data)
    )
    avg_distance_to_school_car_owning = avg_over_students(
        students_car_owning, lambda s: distance_to_school(s, framingham_data)
    )
    avg_distance_to_school_special_education = avg_over_students(
        students_special_education, lambda s: distance_to_school(s, framingham_data)
    )
    avg_distance_to_school_1mi = avg_over_students(
        students_within_1mi, lambda s: distance_to_school(s, framingham_data)
    )
    avg_distance_to_school_2mi = avg_over_students(
        students_within_2mi, lambda s: distance_to_school(s, framingham_data)
    )
    avg_distance_to_school_over_5mi = avg_over_students(
        students_over_5mi, lambda s: distance_to_school(s, framingham_data)
    )

    print("Total number of students:", len(students))
    print("Number of assigned students:", len(assigned_students))
    print("Students within 1 mile of school:", len(students_within_1mi))
    print("Students within 2 miles of school:", len(students_within_2mi))
    print("Students over 5 miles from school:", len(students_over_5mi))
    print("English proficient students:", len(students_english_proficient))
    print("Not English proficient students:", len(students_not_english_proficient))
    print("Students from non-car-owning households:", len(students_not_car_owning))
    print("Students from car-owning households:", len(students_car_owning))
    print("Students with special education needs:", len(students_special_education))
    print("")

    print(f"Average distance to school: {avg_distance_to_school:.2f} km")
    print(
        f"Average distance to school (current bus-riding students): {avg_assigned_distance_to_school:.2f} km"
    )
    print(
        f"Average distance to school (English proficient): {avg_distance_to_school_english_proficient:.2f} km"
    )
    print(
        f"Average distance to school (Not English proficient): {avg_distance_to_school_not_english_proficient:.2f} km"
    )
    print(
        f"Average distance to school (Not car owning household): {avg_distance_to_school_not_car_owning:.2f} km"
    )
    print(
        f"Average distance to school (Car owning household): {avg_distance_to_school_car_owning:.2f} km"
    )
    print(
        f"Average distance to school (Special education needs): {avg_distance_to_school_special_education:.2f} km"
    )
    print(
        f"Average distance to school (Within 1 mile): {avg_distance_to_school_1mi:.2f} km"
    )
    print(
        f"Average distance to school (Within 2 miles): {avg_distance_to_school_2mi:.2f} km"
    )
    print(
        f"Average distance to school (Over 5 miles): {avg_distance_to_school_over_5mi:.2f} km"
    )


def analyze_route_distances_per_income() -> None:
    framingham_problem_data = setup_framingham(precompute_cache=True)
    assigned_students = get_assigned_students(
        framingham_problem_data.schools, framingham_problem_data.stops
    )

    # get average distance from school across income groups
    for income_bin in IncomeLevel:
        students_in_bin = tuple(
            student
            for student in framingham_problem_data.students
            if student.demographics and student.demographics.income_level == income_bin
        )
        assigned_students_in_bin = tuple(
            student
            for student in assigned_students
            if student.demographics and student.demographics.income_level == income_bin
        )

        distances_from_school: list[float] = []
        for student in assigned_students_in_bin:
            length_m, _ = framingham_problem_data.get_shortest_path_base(
                student.stop.node_id, student.school.node_id
            )
            distances_from_school.append(length_m / 1000.0)

        median_distance = sorted(distances_from_school)[len(distances_from_school) // 2]

        print(f"Income bin: {income_bin.name}")
        print(f"Total students: {len(students_in_bin)}")
        print(f"Assigned students: {len(assigned_students_in_bin)}")
        print(f"Median distance from school: {median_distance:.2f} km")
        print("")


def main() -> None:
    analyze_time_per_demographic()
    analyze_route_distances_per_income()


if __name__ == "__main__":
    main()
