import os
import subprocess
from pathlib import Path

from experiments.existing_data.utils import get_assigned_students
from experiments.helpers import OUTPUTS_FOLDER, setup_framingham
from formulation.bird_adapter import (BirdAdapterConfig, BirdBackendSolution,
                                      BirdExportInstance, export_bird_instance,
                                      normalized_result_from_bird_solution)
from formulation.common.problems import FilteredProblemData, ProblemData
from formulation.normalized_result import (NormalizedRoutingResult,
                                           RoutingSolutionJson)

CURRENT_FILE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

BIRD_CONFIG = BirdAdapterConfig(
    cohort="all",
    fleet_aware=True,
    max_time_on_bus=60,
    school_dwell_time=10,
    earliest_arrival_buffer=40,
    bus_mph=30,
    method="scenario",
    conventional_spillover=True,
)
SOLVE_BIRD_BACKEND_PATH = CURRENT_FILE_DIR / ".." / "solve_bird_backend_julia.jl"


def get_bird_routes_json(
    bird_name: str,
) -> RoutingSolutionJson:

    out_dir = OUTPUTS_FOLDER / f"bird_run_{bird_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    solution = RoutingSolutionJson.load(out_dir / "bird_solution.json")
    return solution


def get_bird_routes(
    bird_name: str,
    bird_problem: ProblemData | None = None,
    config: BirdAdapterConfig | None = None,
    save_results: bool = False,
) -> NormalizedRoutingResult:

    out_dir = OUTPUTS_FOLDER / f"bird_run_{bird_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        normalized = NormalizedRoutingResult.load(out_dir / "bird_all_normalized.json")
        return normalized
    except Exception:
        pass

    # load if already exists
    try:
        instance = BirdExportInstance.load(out_dir / "bird_all_instance.npz")
        solution = BirdBackendSolution.load(out_dir / "bird_all.npz")

        normalized = normalized_result_from_bird_solution(instance, solution)
        return normalized
    except Exception:
        pass

    assert bird_problem is not None, "Must provide bird_problem if not already solved"
    assert config is not None, "Must provide config if not already solved"

    instance_path = export_bird_instance(
        bird_problem,
        out_dir / "bird_all_instance.npz",
        config,
    )
    solution_path = out_dir / "bird_all.npz"

    subprocess.run(
        [
            "julia",
            "--project=julia",
            str(SOLVE_BIRD_BACKEND_PATH),
            "--instance",
            str(instance_path),
            "--solution",
            str(solution_path),
        ],
        check=True,
    )

    instance = BirdExportInstance.load(instance_path)
    solution = BirdBackendSolution.load(solution_path)
    normalized = normalized_result_from_bird_solution(instance, solution)

    if save_results:
        normalized.save(out_dir / "bird_all_normalized.json")

    return normalized


def main() -> None:
    framingham_problem_data = setup_framingham(precompute_cache=True)
    assigned_students = get_assigned_students(
        framingham_problem_data.schools, framingham_problem_data.stops
    )

    # filtered_problem_data = FilteredProblemData(
    #     "framingham_filtered",
    #     base_problem_data=framingham_problem_data,
    #     _students=assigned_students,
    # )

    # config = BIRD_CONFIG

    normalized_result = get_bird_routes_json("assigned_student_routes")
    print(normalized_result)


if __name__ == "__main__":
    main()
