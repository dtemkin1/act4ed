import subprocess

from experiments.existing_data.utils import get_assigned_students
from experiments.helpers import OUTPUTS_FOLDER, setup_framingham

from formulation.common.problems import FilteredProblemData

from formulation.bird_adapter import (
    BirdAdapterConfig,
    BirdBackendSolution,
    BirdExportInstance,
    export_bird_instance,
    normalized_result_from_bird_solution,
)

BIRD_CONFIG = BirdAdapterConfig(
    cohort="conventional",
    bus_type="C",
    school_dwell_time=10,
    max_time_on_bus=60,
    method="scenario",
)


def main() -> None:
    framingham_problem_data = setup_framingham(precompute_cache=True)
    assigned_students = get_assigned_students(
        framingham_problem_data.schools,
        framingham_problem_data.stops,
    )

    filtered_problem_data = FilteredProblemData(
        "framingham_filtered",
        base_problem_data=framingham_problem_data,
        _students=assigned_students,
    )

    bird_problem = filtered_problem_data
    config = BIRD_CONFIG

    out_dir = OUTPUTS_FOLDER / "existing_bird"
    out_dir.mkdir(parents=True, exist_ok=True)

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
            "experiments/solve_bird_backend_julia.jl",
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
    normalized.save(out_dir / "bird_all_normalized.json")


if __name__ == "__main__":
    main()
