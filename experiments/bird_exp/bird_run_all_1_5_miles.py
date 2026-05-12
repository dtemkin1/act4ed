import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.helpers import setup
from experiments.tests.all_gurobi_julia import DEFAULT_PLACE_NAME
from formulation.bird_adapter import (BirdBackendSolution, BirdExportInstance,
                                      _bird_bus_display_name,
                                      export_bird_instance,
                                      normalized_result_from_bird_solution,
                                      routing_solution_json_from_bird_solution)
from formulation.common import KM_PER_MILE, FilteredProblemData

from .config import config

RUN_NAME = "run_1_5_mile_students" + f"_{'PARTIAL' if config.allow_partial else 'COMPLETE'}"

problem_data = setup(
    problem_name="framingham_full_students",
    place_name=DEFAULT_PLACE_NAME, 
    prune=None, 
    students_path=Path('experiments/data/students.csv')
)
school_name_to_code = pd.read_csv("experiments/data/schools.csv").set_index("name")["id"].to_dict()

# Create output folder
out_dir = Path(f"experiments/outputs/bird_{RUN_NAME}_routes")
out_dir.mkdir(parents=True, exist_ok=True)

# Consider the currently assigned students, i.e. all students that are currently served by FPS
filtered_distance_students = list(filter(
    lambda s: problem_data.service_graph.get_edge_data(
        s.stop.node_id, s.school.node_id, 0
    )["length"] >= (1.5 * KM_PER_MILE),
    problem_data.students
))
filtered_problem_data = FilteredProblemData(
    name=problem_data.name,
    base_problem_data=problem_data,
    _students=filtered_distance_students,
)

bird_problem = filtered_problem_data

print("Number of students considered:", len(bird_problem.students))
print("Number of SPED students", len([s for s in problem_data.students if s.attributes.special_ed]))

instance_path = export_bird_instance(
    bird_problem,
    out_dir / f"bird_{RUN_NAME}_instance.npz",
    config,
)
solution_path = out_dir / f"bird_{RUN_NAME}_solution.npz"

subprocess.run(
    [
        "julia",
        "--project=julia",
        "experiments/solve_bird_backend_julia.jl",
        "--instance",
        str(instance_path),
        "--solution",
        str(solution_path),
        "--timing-log",
        "--gurobi-verbose"
    ],
    check=True,
)

instance = BirdExportInstance.load(instance_path)
solution = BirdBackendSolution.load(solution_path)

normalized = normalized_result_from_bird_solution(instance, solution)
normalized.save(out_dir / "bird_all_normalized.json")

school_demand_rows = {
    school_idx: [
        row
        for row, idx in zip(instance.demand_rows, instance.demand_school_indices)
        if int(idx) == school_idx
    ]
    for school_idx in range(1, len(instance.schools) + 1)
}

rows = []
for bus_id in sorted(set(solution.assignment_bus_ids.tolist())):
    bus_rows = np.where(solution.assignment_bus_ids == bus_id)[0]
    bus_rows = sorted(bus_rows, key=lambda i: int(solution.assignment_orders[i]))
    if not bus_rows:
        continue

    student_names_for_bus = set()
    total_students = 0
    total_distance_km = 0.0
    arrival_summaries = []
    slacks = []

    for row_idx in bus_rows:
        q = int(solution.assignment_orders[row_idx])
        school_idx = int(solution.assignment_school_indices[row_idx])
        school = instance.schools[school_idx - 1]

        start = int(solution.assignment_stop_ptr[row_idx])
        end = int(solution.assignment_stop_ptr[row_idx + 1])
        local_stop_ids = solution.assignment_stop_values[start:end].tolist()

        demand_rows_for_school = school_demand_rows[school_idx]
        route_demand_rows = [demand_rows_for_school[local_stop_id - 1] for local_stop_id in local_stop_ids]

        student_names = [i for s in [row.student_names for row in route_demand_rows] for i in s]
        student_names_for_bus = student_names_for_bus.union(student_names)
        route_students = sum(row.students for row in route_demand_rows)
        total_students += route_students
        total_distance_km += float(solution.assignment_distance_km[row_idx])

        arrival = float(solution.assignment_arrival_times[row_idx])
        slack = float(school.start_time) - arrival
        stop_names = ", ".join(row.stop_name for row in route_demand_rows)

        arrival_summaries.append(
            f"round {q}: {school.name}, {slack:.1f} min before start "
            f"(T={arrival:.1f}), {route_students} students, stops=[{stop_names}]"
        )
        
        slacks.append(slack)

    rows.append(
        {
            "bus": f"bird_bus_{bus_id}",
            "bus_name": _bird_bus_display_name(instance, bus_id),
            "rounds_used": len(bus_rows),
            "students": total_students,
            # "student_ids": list(sorted(student_names_for_bus)),
            "distance_km": round(total_distance_km, 2),
            "arrival_vs_school_start": "; ".join(arrival_summaries),
            "slacks": slacks
        }
    )

bird_report_df = pd.DataFrame(rows).sort_values("bus").reset_index(drop=True)

# Solution printing
print(f"Buses used: {len(bird_report_df)}")
print("unassigned_students", normalized.metadata["unassigned_students"])
print("unassigned_demand_rows", normalized.metadata["unassigned_demand_rows"])
print("unassigned_student_count", normalized.metadata["unassigned_student_count"])
print("unassigned_demand_count", normalized.metadata["unassigned_demand_count"])

print("Saving solution ...")
report = routing_solution_json_from_bird_solution(instance, solution)
report.save(out_dir / "bird_solution.json")
