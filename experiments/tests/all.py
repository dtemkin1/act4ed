import os
from pathlib import Path

from formulation.formulation_3.definition import ExperimentConfig, Formulation3
from formulation.formulation_3.cvxpy import (
    build_model_from_definition,
    make_report,
    plot_bus_routes,
    solve_problem,
)
from experiments.helpers import setup_framingham


CURRENT_FILE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# where
PLACE_NAME = "Framingham, Massachusetts, USA"


def experimental_config():
    no_chaining = ExperimentConfig(
        problem_data_pkl=str(
            CURRENT_FILE_DIR
            / ".."
            / ".."
            / "formulation"
            / "cache"
            / "framingham_problem_data.pkl"
        ),
        rounds=1,
    )

    return no_chaining


def main() -> None:
    problem_data = setup_framingham()

    # formulation time baby
    no_chaining = Formulation3(
        problem_data=problem_data,
        rounds=1,
    )
    print("No-chaining formulation created")

    no_chaining_model = build_model_from_definition(no_chaining)
    print("No-chaining model built")

    solve_problem(no_chaining_model[0])
    print("No-chaining problem solved!")

    report_no_chaining = make_report(
        no_chaining_model[0], no_chaining, no_chaining_model[1]
    )

    with open(
        CURRENT_FILE_DIR / ".." / "outputs" / "report_no_chaining_all.txt",
        "w+",
        encoding="utf-8",
    ) as f:
        f.write(report_no_chaining)
    print("No-chaining report written")

    plot_bus_routes(
        no_chaining_model[0],
        no_chaining,
        no_chaining_model[1],
        CURRENT_FILE_DIR / ".." / "outputs" / "no_chaining_routes_all.png",
    )
    print("No-chaining routes plotted")

    print("Now doing chaining formulation...")
    problem_data = setup_framingham()
    chaining = Formulation3(
        problem_data=problem_data,
        rounds=3,
    )
    print("Chaining formulation created")

    chaining_model = build_model_from_definition(chaining)
    print("Chaining model built")

    solve_problem(chaining_model[0])
    print("Chaining problem solved!")

    report_chaining = make_report(chaining_model[0], chaining, chaining_model[1])

    with open(
        CURRENT_FILE_DIR / ".." / "outputs" / "report_chaining_all.txt",
        "w+",
        encoding="utf-8",
    ) as f:
        f.write(report_chaining)

    plot_bus_routes(
        chaining_model[0],
        chaining,
        chaining_model[1],
        CURRENT_FILE_DIR / ".." / "outputs" / "chaining_routes_all.png",
    )


if __name__ == "__main__":
    main()
