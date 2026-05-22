# Routing MCDP Workflow

Run commands from the repository root.

## Regenerate Static Files

This rewrites fleet/catalogue support files, BiRD config and student-policy
posets, guidelines/catalogue defaults, and cost modules without running the
BiRD grid.

```bash
PYTHONPATH=. uv run python -c "from experiments.mcdp.routing_bird import write_static_catalogs; write_static_catalogs(routing_implementations=None)"
```

## Run the BiRD Grid

Grid constants live in `experiments/mcdp/routing_bird.py`:
`GRID_FLEET`, `GRID_METHODS`, `GRID_LAMBDAS`, `GRID_PARTIAL`,
`GRID_SPILLOVER`, `GRID_DWELL`, `GRID_ARRIVAL_WINDOWS`, and
`GRID_AVG_SPEEDS`. The grid also includes `GRID_STUDENT_POLICIES`: current
assignment, distance thresholds at 2, 1.5, 1, and 0.5 miles, and all students.
`GRID_METHODS` includes both `lbh` and `scenario`.

```bash
PYTHONPATH=. uv run python -m experiments.mcdp.routing_bird
```

Useful options:

- `--output-dir PATH`: where `.npz`, logs, partial results, and errors are written.
- `--routing-lib PATH`: MCDP library to update; defaults to `routing.mcdplib`.
- `--current-routes`: run only the `current_assignment` policy variant.
- `--workers N`: number of concurrent grid-point solves (defaults to number of cores / cores per solve).
- `--cpus-per-solve N`: Julia/Gurobi thread budget per solve (defaults to 4 CPUs).

The grid writes `routing.mcdplib/yaml_catalogs/routing_service.dpc.yaml` and
strict policy rows in `routing.mcdplib/yaml_catalogs/guidelines.dpc.yaml`.
When route implementations are available, the fleet catalogue is bounded by the
exact unique `used_*` bus-count vectors observed in the solved route catalogue
rather than the full bus inventory Cartesian product.

## Run MCDP Queries

Queries are `*.mcdp_query.yaml` files in `routing.mcdplib`. The query name is
the filename without `.mcdp_query.yaml`.

Currently we have:

- `routing_simple`: compact smoke query over cost, emissions, unserved students, unique stops used, BiRD configuration caps, and `student_policy`; it uses query-specific route, fleet, and guideline catalogues.
- `routing_policy_template`: adaptable policy query that puts the student policy, minimum service level, unserved caps, budget, and BiRD configuration directly in the query. Use this when experimenting with `guidelines` disconnected from `routing.mcdp`.

Run it with the MCDP Docker image:

```bash
docker run --rm -v "$PWD:$PWD" -w "$PWD" zupermind/mcdp:2025 \
  bash -lc 'mcdp-solve-query --nocache -d routing.mcdplib routing_simple'
```

Use `--nocache` after changing models, catalogues, or query files.

To create a new query, copy `routing.mcdplib/routing_simple.mcdp_query.yaml`,
rename it to `routing.mcdplib/<query_name>.mcdp_query.yaml`, and edit:

- `model`: usually ``"`routing"``.
- `min_f`: minimum required functionality, such as served students.
- `max_r`: resource caps, such as total cost, emissions, unserved counts,
  `unique_stops_used`, and BiRD config/student-policy poset values.
- `optimize_for`: resource objective list.

Then run:

```bash
docker run --rm -v "$PWD:$PWD" -w "$PWD" zupermind/mcdp:2025 \
  bash -lc 'mcdp-solve-query --nocache -d routing.mcdplib <query_name>'
```

## Plot Pareto Fronts

The Pareto plotting script reads the routing catalogue and cost config, then
writes CSVs and PDFs. Pareto fronts are drawn as horizontal/vertical step lines
because intermediate points between sampled route implementations were not
explored.

```bash
PYTHONPATH=. uv run python -m experiments.mcdp.plot_routing_bird_pareto
```

Optional paths:

```bash
PYTHONPATH=. uv run python -m experiments.mcdp.plot_routing_bird_pareto \
  --library routing.mcdplib \
  --costs experiments/mcdp/routing_costs.yaml \
  --output-dir experiments/outputs/mcdp_solve/pareto
```

Outputs include:

- `routing_bird_mcdp_points.csv`
- `routing_bird_mcdp_routing_simple_caps.csv`
- `routing_bird_mcdp_routing_simple_budget_feasible.csv`
- `routing_bird_mcdp_pareto_all.pdf`
- `routing_bird_mcdp_pareto_routing_simple_caps.pdf`
- `routing_bird_mcdp_pareto_routing_simple_budget_feasible.pdf`
