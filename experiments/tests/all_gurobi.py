import os
from pathlib import Path

from formulation.formulation_3.definition import Formulation3
from formulation.formulation_3.gurobipy import (
    build_model_from_definition,
    solve_problem,
)
from experiments.helpers import setup_framingham

CURRENT_FILE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# where
PLACE_NAME = "Framingham, Massachusetts, USA"


def main() -> None:
    output_dir = CURRENT_FILE_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    problem_data = setup_framingham(prune=1)

    # formulation time baby
    no_chaining = Formulation3(
        problem_data=problem_data,
        rounds=1,
    )

    print("No-chaining formulation created")

    no_chaining_bundle = build_model_from_definition(no_chaining)
    print("No-chaining model built")

    no_chaining_solution = solve_problem(no_chaining_bundle.model)
    print("No-chaining problem solved!")
    if no_chaining_solution is not None:
        no_chaining_solution.save(output_dir / "no_chaining_all.pkl")

    # report_no_chaining = make_report(
    #     no_chaining_bundle.model, no_chaining, no_chaining_bundle.variables
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
    no_chaining_bundle.model.close()

    print("Now doing chaining formulation...")
    problem_data = setup_framingham(prune=1)
    chaining = Formulation3(
        problem_data=problem_data,
        rounds=3,
    )
    print("Chaining formulation created")

    chaining_bundle = build_model_from_definition(chaining)
    print("Chaining model built")

    chaining_solution = solve_problem(chaining_bundle.model)
    print("Chaining problem solved!")
    if chaining_solution is not None:
        chaining_solution.save(output_dir / "chaining_all.pkl")

    # report_chaining = make_report(chaining_bundle.model, chaining, chaining_bundle.variables)

    # with open(
    #     CURRENT_FILE_DIR / ".." / "outputs" / "report_chaining_all.txt", "w+", encoding="utf-8"
    # ) as f:
    #     f.write(report_chaining)

    # plot_bus_routes(
    #     chaining_bundle.model,
    #     chaining,
    #     chaining_bundle.variables,
    #     CURRENT_FILE_DIR / ".." / "outputs" / "chaining_routes_all.png",
    # )

    chaining_bundle.model.close()


if __name__ == "__main__":
    main()
