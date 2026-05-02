from experiments.helpers import setup_framingham


def main() -> None:
    problem_data = setup_framingham(precompute_cache=True)
    for student in problem_data.students:
        print(
            f"Student {student.name} at stop {student.stop.name} has census data: {student.census_data}"
        )


if __name__ == "__main__":
    main()
