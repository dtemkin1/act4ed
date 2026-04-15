import os
from pathlib import Path

from formulation.formulation_3.problem3_definition import Formulation3
from formulation.formulation_3.formulation3_gurobipy import (
    build_model_from_definition,
    solve_problem,
)
from experiments.helpers import setup


CURRENT_FILE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# where
PLACE_NAME = "Framingham, Massachusetts, USA"


def main() -> None:
    problem_data = setup("framingham", PLACE_NAME, 1.0)

    # formulation time baby
    no_chaining = Formulation3(
        problem_data=problem_data,
        rounds=1,
    )

    print("No-chaining formulation created")

    no_chaining_model, no_chaining_vars = build_model_from_definition(no_chaining)
    print("No-chaining model built")

    solve_problem(no_chaining_model)
    print("No-chaining problem solved!")

    # report_no_chaining = make_report(
    #     no_chaining_model, no_chaining, no_chaining_vars
    # )

    # with open(
    #     CURRENT_FILE_DIR / ".." / "outputs" / "report_no_chaining_all.txt",
    #     "w+",
    #     encoding="utf-8",
    # ) as f:
    #     f.write(report_no_chaining)
    # print("No-chaining report written")

    # plot_bus_routes(
    #     no_chaining_model,
    #     no_chaining,
    #     no_chaining_vars,
    #     CURRENT_FILE_DIR / ".." / "outputs" / "no_chaining_routes_all.png",
    # )
    # print("No-chaining routes plotted")
    no_chaining_model.close()

    print("Now doing chaining formulation...")
    problem_data = setup("framingham", PLACE_NAME, 1000)
    chaining = Formulation3(
        problem_data=problem_data,
        rounds=3,
    )
    print("Chaining formulation created")

    chaining_model, chaining_vars = build_model_from_definition(chaining)
    print("Chaining model built")

    solve_problem(chaining_model)
    print("Chaining problem solved!")

    # report_chaining = make_report(chaining_model, chaining, chaining_vars)

    # with open(
    #     CURRENT_FILE_DIR / ".." / "outputs" / "report_chaining_all.txt", "w+", encoding="utf-8"
    # ) as f:
    #     f.write(report_chaining)

    # plot_bus_routes(
    #     chaining_model,
    #     chaining,
    #     chaining_vars,
    #     CURRENT_FILE_DIR / ".." / "outputs" / "chaining_routes_all.png",
    # )

    chaining_model.close()


if __name__ == "__main__":
    main()
