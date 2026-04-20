import argparse
from collections import Counter
from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess
from typing import Any

from formulation.bird_adapter import (
    BirdAdapterConfig,
    BirdBackendSolution,
    BirdExportInstance,
    export_bird_instance,
    normalized_result_from_bird_solution,
    routing_solution_json_from_bird_solution,
)
from formulation.common import SchoolType, FilteredProblemData
from formulation.formulation_3.julia_export import export_formulation3_instance
from formulation.formulation_3.problem3_definition import Formulation3
from formulation.formulation_3.formulation3_gurobipy import (
    Formulation3Solution,
    build_model_from_definition,
    solve_problem,
)
from formulation.normalized_result import (
    NormalizedRoutingResult,
    normalized_result_from_formulation3_solution,
    RoutingSolutionJson,
    routing_solution_json_from_formulation3_solution,
)
from experiments.helpers import setup
from helpers import ProblemDataReal


CURRENT_FILE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = CURRENT_FILE_DIR.parent
DEFAULT_PLACE_NAME = "Framingham, Massachusetts, USA"
DEFAULT_PROBLEM_NAME = "framingham"
DEFAULT_PRUNE = 3000


@dataclass(frozen=True)
class RunScope:
    label_suffix: str
    description: str
    problem_data: ProblemDataReal | FilteredProblemData


def _slugify(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", value.strip()).strip("_").lower()
    return slug or "item"


def _build_run_scopes(
    problem_data: ProblemDataReal,
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


def _save_normalized_result(
    normalized_result: NormalizedRoutingResult,
    *,
    label: str,
    output_dir: Path,
) -> Path:
    output_path = output_dir / f"{label}_all_normalized.json"
    normalized_result.save(output_path)
    print(f"{label}: saved normalized result to {output_path}")
    return output_path


def _save_routing_solution_json(
    routing_solution_json: RoutingSolutionJson,
    *,
    label: str,
    output_dir: Path,
) -> Path:
    output_path = output_dir / f"{label}_all_solution.json"
    routing_solution_json.save(output_path)
    print(f"{label}: saved route-level solution JSON to {output_path}")
    return output_path


def _solve_with_bird(
    problem_data: Any,
    *,
    label: str,
    output_dir: Path,
    cohort: str,
    bus_type: str | None,
    lambda_value: float,
    reassign_stops: bool,
    stop_assignment_lambda: float,
    max_walking_distance_km: float | None,
    method: str,
) -> NormalizedRoutingResult:
    adapter_config = BirdAdapterConfig(
        cohort=cohort,
        bus_type=bus_type,
        lambda_value=lambda_value,
        reassign_stops=reassign_stops,
        stop_assignment_lambda=stop_assignment_lambda,
        max_walking_distance_km=max_walking_distance_km,
    )
    instance_path = export_bird_instance(
        problem_data,
        output_dir / f"{label}_all_instance.npz",
        adapter_config,
    )
    solution_path = output_dir / f"{label}_all.npz"
    log_file = output_dir / f"{label}_all_julia.log"

    julia_cmd = os.environ.get("JULIA_CMD", "julia")
    julia_project = os.environ.get("JULIA_PROJECT", "julia")
    command = [
        julia_cmd,
        f"--project={julia_project}",
        str(PROJECT_ROOT / "experiments" / "solve_bird_backend_julia.jl"),
        "--instance",
        str(instance_path),
        "--solution",
        str(solution_path),
        "--method",
        method,
        "--log-file",
        str(log_file),
    ]
    print(f"{label}: exported Bird instance to {instance_path}")
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)
    print(f"{label}: Bird Julia solve finished")

    instance = BirdExportInstance.load(instance_path)
    solution = BirdBackendSolution.load(solution_path)
    normalized_result = normalized_result_from_bird_solution(instance, solution)
    routing_solution_json = routing_solution_json_from_bird_solution(instance, solution)
    _save_normalized_result(normalized_result, label=label, output_dir=output_dir)
    _save_routing_solution_json(routing_solution_json, label=label, output_dir=output_dir)
    return normalized_result


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
        solution = _solve_with_python(formulation, label=label, output_dir=output_dir)
        normalized_result = normalized_result_from_formulation3_solution(formulation, solution)
        routing_solution_json = routing_solution_json_from_formulation3_solution(
            formulation,
            solution,
        )
        _save_normalized_result(normalized_result, label=label, output_dir=output_dir)
        _save_routing_solution_json(routing_solution_json, label=label, output_dir=output_dir)
        return solution
    if backend == "julia":
        solution = _solve_with_julia(formulation, label=label, output_dir=output_dir)
        normalized_result = normalized_result_from_formulation3_solution(formulation, solution)
        routing_solution_json = routing_solution_json_from_formulation3_solution(
            formulation,
            solution,
        )
        _save_normalized_result(normalized_result, label=label, output_dir=output_dir)
        _save_routing_solution_json(routing_solution_json, label=label, output_dir=output_dir)
        return solution
    raise ValueError(f"unsupported backend: {backend}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=("python", "julia", "bird"),
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
    parser.add_argument(
        "--bird-cohort",
        choices=("conventional", "sped_no_wheelchair"),
        default="conventional",
        help="Student cohort to export when running the Bird backend.",
    )
    parser.add_argument(
        "--bird-bus-type",
        default=None,
        help="Homogeneous bus type slice for Bird runs, e.g. C, B, BWC, WC, or an enum value.",
    )
    parser.add_argument(
        "--bird-method",
        choices=("scenario", "lbh"),
        default="scenario",
        help="Bird solve method to use in Julia.",
    )
    parser.add_argument(
        "--bird-lambda",
        type=float,
        default=1.0e4,
        help="Per-route penalty lambda used by Bird scenario generation.",
    )
    parser.add_argument(
        "--bird-reassign-stops",
        action="store_true",
        help="Optionally reassign students to existing stops before exporting the Bird instance.",
    )
    parser.add_argument(
        "--bird-stop-assignment-lambda",
        type=float,
        default=1.0e4,
        help="Lambda used by the optional Bird student-to-stop reassignment model.",
    )
    parser.add_argument(
        "--bird-max-walking-km",
        type=float,
        default=None,
        help="Optional max walking distance for Bird stop reassignment; current stop remains feasible as fallback.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Pruning distance: {args.prune}")
    problem_data = setup(args.problem_name, args.place_name, args.prune)
    scopes = _build_run_scopes(
        problem_data,
        per_school=args.per_school,
        per_school_type=args.per_school_type,
    )

    print(f"Running all_gurobi with backend={args.backend}")
    print(f"Prepared {len(scopes)} run scope(s)")

    for scope in scopes:
        if args.backend == "bird":
            label = _scoped_label("bird", scope)
            print(
                f"{label}: Bird backend run ({scope.description}, cohort={args.bird_cohort}, bus_type={args.bird_bus_type}, method={args.bird_method}, lambda={args.bird_lambda}, reassign_stops={args.bird_reassign_stops})"
            )
            _solve_with_bird(
                scope.problem_data,
                label=label,
                output_dir=output_dir,
                cohort=args.bird_cohort,
                bus_type=args.bird_bus_type,
                lambda_value=args.bird_lambda,
                reassign_stops=args.bird_reassign_stops,
                stop_assignment_lambda=args.bird_stop_assignment_lambda,
                max_walking_distance_km=args.bird_max_walking_km,
                method=args.bird_method,
            )
            continue

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
