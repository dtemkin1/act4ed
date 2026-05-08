from pathlib import Path
import pandas as pd
from formulation.common import FilteredProblemData

from formulation.bird_adapter import (
    BirdAdapterConfig,
    BirdBackendSolution,
    BirdExportInstance,
    export_bird_instance,
    normalized_result_from_bird_solution,
    routing_solution_json_from_bird_solution,
    _bird_bus_display_name
)
from formulation.common import KM_PER_MILE
from experiments.existing_data.utils import get_assigned_students
import subprocess
import numpy as np

from experiments.helpers import setup
from experiments.tests.all_gurobi_julia import (
    DEFAULT_PLACE_NAME
)

RUN_NAME = "run_assigned_students"

problem_data = setup(
    problem_name="framingham_full_students",
    place_name=DEFAULT_PLACE_NAME, 
    prune=None, 
    students_path=Path('experiments/data/students.csv')
)
school_name_to_code = pd.read_csv("experiments/data/schools.csv").set_index("name")["id"].to_dict()

# Loading from a specific file
# cache_path = Path("formulation/cache/framingham_problem_data.pkl")
# problem_data = ProblemDataReal.load_path(cache_path)

# Create output folder
out_dir = Path(f"experiments/outputs/bird_{RUN_NAME}_routes")
out_dir.mkdir(parents=True, exist_ok=True)

# Consider the currently assigned students, i.e. all students that are currently served by FPS
assigned_students = get_assigned_students(problem_data.schools, problem_data.stops)
filtered_problem_data = FilteredProblemData(
    name=problem_data.name,
    base_problem_data=problem_data,
    _students=assigned_students,
)

bird_problem = filtered_problem_data

print("Number of students considered:", len(bird_problem.students))
print("Number of SPED students", len([s for s in problem_data.students if s.attributes.special_ed]))

config = BirdAdapterConfig(
    # ── Cohort / assignment mode ──────────────────────────────────────────────
    # Which student population to route.
    cohort="all",                   # "conventional"       – non-SPED, non-wheelchair students
    # cohort="sped_no_wheelchair",  # "sped_no_wheelchair" – SPED students who are not wheelchair users
    # cohort="sped_and_wheelchair", # "sped_and_wheelchair"– all SPED and/or wheelchair students
    # cohort="all",                 # "all"                – fleet-aware staged routing (wheelchair → SPED → conventional)

    # ── Bus selection ─────────────────────────────────────────────────────────
    # Homogeneous (old BiRD) mode – pick a single bus type for all routes:
    # bus_type="C",                 # "C"   – standard bus (no monitor / no wheelchair lift)
    # bus_type="B",                 # "B"   – monitor bus
    # bus_type="BWC",               # "BWC" – monitor + wheelchair lift
    # bus_type="WC",                # "WC"  – wheelchair lift only

    # Fleet-aware mode – uses concrete fleet buses/capacities/depots/monitor flags:
    fleet_aware=True,               # enable fleet-aware routing (default: False)
    # bus_type=None,                # None means use all bus types from the fleet
    conventional_spillover=False,   # allow conventional students onto remaining monitor buses (default: False)

    # ── Monitor Policy ─────────────────────────────────────────────────────────
    # monitor_policy="route_assigned",

    # ── Capacity / time constraints ───────────────────────────────────────────
    max_time_on_bus=60,             # max minutes a student may spend on the bus (default: 120)
    school_dwell_time=10,           # minutes the bus waits at the school after arriving (default: 0)
    earliest_arrival_buffer=40,     # bus must arrive ≥ N min before bell (default: None)
    # latest_arrival_buffer=10,     # bus must arrive ≤ N min before bell (default: None)

    # ── Stop dwell times ─────────────────────────────────────────────────────
    # constant_stop_time=0,             # fixed dwell per stop in minutes (default: 0)
    stop_time_per_student=0.5,          # additional dwell per boarding student in minutes (default: 0.3)
    stop_time_per_sped=1.0,           # additional dwell per boarding student in minutes (default: 0.3)
    stop_time_per_wheelchair_student=1, # additional dwell per wheelchair student in minutes (default: 0)

    # ── Routing parameters ───────────────────────────────────────────────────
    bus_mph=30,                     # assumed bus travel speed (default: _DEFAULT_BUS_MPH)
    # lambda_value=1.0e4,           # trade-off weight between distance and ride time (default: _DEFAULT_BIRD_LAMBDA_VALUE)

    # ── Partial assignment ───────────────────────────────────────────────────
    allow_partial=False,             # allow some students to remain unassigned (default: False)

    # ── Optional stop reassignment ───────────────────────────────────────────
    # reassign_stops=True,          # re-optimise stop assignments before routing (default: False)
    # stop_assignment_lambda=1.0e4, # weight for stop reassignment optimisation (default: _DEFAULT_STOP_ASSIGNMENT_LAMBDA)
    # max_walking_distance_km=1.0,  # max walk distance when reassigning stops in km (default: None)

    # ── Optimization method ──────────────────────────────────────────────────
    # method="lbh",                 # fast greedy route construction
    method="scenario",              # scenario mode; fleet-aware currently delegates to concrete fleet path
)

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
