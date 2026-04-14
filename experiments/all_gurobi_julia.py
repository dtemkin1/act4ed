import argparse
from collections import Counter
from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess

from formulation.common import SchoolType
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


@dataclass(frozen=True)
class RunScope:
    label_suffix: str
    description: str
    problem_data: object


def _slugify(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", value.strip()).strip("_").lower()
    return slug or "item"


def _build_run_scopes(
    problem_data,
    *,
    per_school: bool,
    per_school_type: bool,
) -> list[RunScope]:
    if per_school and per_school_type:
        raise ValueError("per-school and per-school-type modes are mutually exclusive")

    if not per_school and not per_school_type:
        return [RunScope(label_suffix="", description="all schools", problem_data=problem_data)]

    student_counts_by_school = Counter(student.school.id for student in problem_data.students)
    if per_school:
        scopes: list[RunScope] = []
        for school in problem_data.schools:
            student_count = student_counts_by_school.get(school.id, 0)
            if student_count == 0:
                print(f"Skipping school {school.id} ({school.name}): no students")
                continue
            scopes.append(
                RunScope(
                    label_suffix=f"school_{_slugify(str(school.id))}",
                    description=f"school {school.id} ({school.name}), students={student_count}",
                    problem_data=problem_data.restrict_to_school(school.id),
                )
            )
        if not scopes:
            raise RuntimeError("no schools with students were available for per-school runs")
        return scopes

    student_counts_by_type = Counter(student.school.type for student in problem_data.students)
    scopes = [
        RunScope(
            label_suffix=f"school_type_{school_type.name.lower()}",
            description=f"school type {school_type.name}, students={student_counts_by_type[school_type]}",
            problem_data=problem_data.restrict_to_school_type(school_type),
        )
        for school_type in SchoolType
        if student_counts_by_type[school_type] > 0
    ]
    if not scopes:
        raise RuntimeError("no school types with students were available for per-school-type runs")
    return scopes


def _scoped_label(base_label: str, scope: RunScope) -> str:
    return base_label if scope.label_suffix == "" else f"{base_label}_{scope.label_suffix}"


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
    problem_data,
    rounds: int,
) -> Formulation3:
    return Formulation3(problem_data=problem_data, rounds=rounds)


def _run_case(
    *,
    backend: str,
    label: str,
    description: str,
    rounds: int,
    problem_data,
    output_dir: Path,
) -> Formulation3Solution:
    formulation = _build_problem(
        problem_data=problem_data,
        rounds=rounds,
    )
    print(f"{label}: formulation created with rounds={rounds} ({description})")

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
    scope_group = parser.add_mutually_exclusive_group()
    scope_group.add_argument(
        "--per-school",
        action="store_true",
        help="Run separate no-chaining and chaining solves for each school with students.",
    )
    scope_group.add_argument(
        "--per-school-type",
        action="store_true",
        help="Run separate no-chaining and chaining solves for each school type with students.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    problem_data = setup(args.problem_name, args.place_name, args.prune)
    scopes = _build_run_scopes(
        problem_data,
        per_school=args.per_school,
        per_school_type=args.per_school_type,
    )

    print(f"Running all_gurobi with backend={args.backend}")
    print(f"Prepared {len(scopes)} run scope(s)")

    for scope in scopes:
        _run_case(
            backend=args.backend,
            label=_scoped_label("no_chaining", scope),
            description=scope.description,
            rounds=args.no_chaining_rounds,
            problem_data=scope.problem_data,
            output_dir=output_dir,
        )

        _run_case(
            backend=args.backend,
            label=_scoped_label("chaining", scope),
            description=scope.description,
            rounds=args.chaining_rounds,
            problem_data=scope.problem_data,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
