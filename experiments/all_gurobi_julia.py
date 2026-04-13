import argparse
import os
from pathlib import Path
import subprocess

from formulation.formulation_3.julia_export import export_formulation3_instance
from formulation.formulation_3.problem3_definition import Formulation3
from formulation.formulation_3.formulation3_gurobipy import (
    Formulation3Solution,
    build_model_from_definition,
    solve_problem,
)
from experiments.helpers import setup


CURRENT_FILE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = CURRENT_FILE_DIR.parent
DEFAULT_PLACE_NAME = "Framingham, Massachusetts, USA"
DEFAULT_PROBLEM_NAME = "framingham"
DEFAULT_PRUNE = 800


def _solve_with_python(
    problem: Formulation3,
    *,
    label: str,
    output_dir: Path,
) -> Formulation3Solution:
    bundle = build_model_from_definition(problem)
    print(f"{label}: model built with Python/Gurobi")

    solution = solve_problem(bundle)
    if solution is None:
        raise RuntimeError(f"{label}: expected a Formulation3Solution from Python solve")

    solution.save(output_dir / f"{label}_all.pkl")
    solution.save(output_dir / f"{label}_all.npz")
    print(f"{label}: saved Python solution snapshot")
    return solution


def _solve_with_julia(
    problem: Formulation3,
    *,
    label: str,
    output_dir: Path,
) -> Formulation3Solution:
    instance_path = export_formulation3_instance(
        problem,
        output_dir / f"{label}_all_instance.npz",
    )
    solution_path = output_dir / f"{label}_all.npz"
    log_file = output_dir / f"{label}_all_julia.log"

    julia_cmd = os.environ.get("JULIA_CMD", "julia")
    julia_project = os.environ.get("JULIA_PROJECT", "julia")
    command = [
        julia_cmd,
        f"--project={julia_project}",
        str(PROJECT_ROOT / "experiments" / "solve_formulation3_julia.jl"),
        "--instance",
        str(instance_path),
        "--solution",
        str(solution_path),
        "--log-file",
        str(log_file),
    ]
    print(f"{label}: exported Julia instance to {instance_path}")
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)
    print(f"{label}: Julia solve finished")

    solution = Formulation3Solution.load(solution_path)
    solution.save(output_dir / f"{label}_all.pkl")
    print(f"{label}: saved Python-compatible snapshot from Julia solution")
    return solution


def _build_problem(
    *,
    problem_name: str,
    place_name: str,
    prune: int | None,
    rounds: int,
) -> Formulation3:
    problem_data = setup(problem_name, place_name, prune)
    return Formulation3(problem_data=problem_data, rounds=rounds)


def _run_case(
    *,
    backend: str,
    label: str,
    rounds: int,
    problem_name: str,
    place_name: str,
    prune: int | None,
    output_dir: Path,
) -> Formulation3Solution:
    formulation = _build_problem(
        problem_name=problem_name,
        place_name=place_name,
        prune=prune,
        rounds=rounds,
    )
    print(f"{label}: formulation created with rounds={rounds}")

    if backend == "python":
        return _solve_with_python(formulation, label=label, output_dir=output_dir)
    if backend == "julia":
        return _solve_with_julia(formulation, label=label, output_dir=output_dir)
    raise ValueError(f"unsupported backend: {backend}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=("python", "julia"),
        default="python",
        help="Model builder/solver backend to use.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CURRENT_FILE_DIR / "outputs",
    )
    parser.add_argument(
        "--problem-name",
        default=DEFAULT_PROBLEM_NAME,
    )
    parser.add_argument(
        "--place-name",
        default=DEFAULT_PLACE_NAME,
    )
    parser.add_argument(
        "--prune",
        type=int,
        default=DEFAULT_PRUNE,
    )
    parser.add_argument(
        "--no-chaining-rounds",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--chaining-rounds",
        type=int,
        default=3,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running all_gurobi with backend={args.backend}")

    _run_case(
        backend=args.backend,
        label="no_chaining",
        rounds=args.no_chaining_rounds,
        problem_name=args.problem_name,
        place_name=args.place_name,
        prune=args.prune,
        output_dir=output_dir,
    )

    _run_case(
        backend=args.backend,
        label="chaining",
        rounds=args.chaining_rounds,
        problem_name=args.problem_name,
        place_name=args.place_name,
        prune=args.prune,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
