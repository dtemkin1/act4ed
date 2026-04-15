import json
import os
from pathlib import Path

from formulation.formulation_3.problem3_definition import Formulation3
from formulation.toy_network import make_toy_problem_data
from formulation.formulation_3.formulation3_gurobipy import (
    build_model_from_definition,
    solve_problem,
    plot_bus_routes,
)

CURRENT_FILE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))


def get_problems_from_json(json_file: Path) -> list[Formulation3]:
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list), f"Expected a list of problem data, got {type(data)}"

    toy_problems: list[Formulation3] = []
    for item in data:
        assert isinstance(
            item, dict
        ), f"Expected each item to be a dict, got {type(item)}"
        assert "size" in item, "Each problem data must have a 'size' field"
        assert isinstance(
            item["size"], list
        ), f"Expected 'size' to be a list, got {type(item['size'])}"
        assert (
            len(item["size"]) == 2
        ), f"Expected 'size' to have 2 elements, got {len(item['size'])}"
        assert (
            "num_schools" in item
        ), "Each problem data must have a 'num_schools' field"
        assert "num_depots" in item, "Each problem data must have a 'num_depots' field"
        assert "num_stops" in item, "Each problem data must have a 'num_stops' field"
        assert (
            "num_students" in item
        ), "Each problem data must have a 'num_students' field"
        assert "num_buses" in item, "Each problem data must have a 'num_buses' field"
        prob_data = make_toy_problem_data(
            name=item.get("name", "toy_network"),
            size=item["size"],
            num_schools=item["num_schools"],
            num_depots=item["num_depots"],
            num_stops=item["num_stops"],
            num_students=item["num_students"],
            num_buses=item["num_buses"],
        )

        assert (
            "formulation_3" in item
        ), "Each problem data must have a 'formulation_3' field"
        assert isinstance(
            item["formulation_3"], list
        ), f"Expected 'formulation_3' to be a list, got {type(item['formulation_3'])}"

        for formulations in item["formulation_3"]:
            assert isinstance(
                formulations, dict
            ), f"Expected each formulation to be a dict, got {type(formulations)}"
            assert (
                "rounds" in formulations
            ), "Each formulation must have a 'rounds' field"

            formulation_3 = Formulation3(prob_data, **formulations)
            toy_problems.append(formulation_3)

    return toy_problems


def main() -> None:
    json_file = CURRENT_FILE_DIR / "experiments.json"
    toy_problems = get_problems_from_json(json_file)

    results = []
    for toy_data in toy_problems:
        print(
            f"Solving problem: {toy_data.problem_data.name} with rounds={toy_data.rounds}"
        )

        model, vals = build_model_from_definition(toy_data)
        print(f"Model built for problem: {toy_data.problem_data.name}")

        solve_problem(model)
        print(f"Problem solved: {toy_data.problem_data.name}")

        results.append(
            {
                "problem_name": toy_data.problem_data.name,
                "rounds": toy_data.rounds,
                "objective_value": model.ObjVal,
                "runtime_seconds": model.Runtime,
                # TODO: ask riccardo about gurobi vars/values and how to extract them
                # "results": vals,
            }
        )

        plot_bus_routes(
            prob=model,
            formulation=toy_data,
            model_vars=vals,
            save_path=CURRENT_FILE_DIR
            / "outputs"
            / f"{toy_data.problem_data.name}_rounds_{toy_data.rounds}_routes.png",
        )

    results_file = CURRENT_FILE_DIR / "outputs" / "experiment_results.json"
    with open(results_file, "w+", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
