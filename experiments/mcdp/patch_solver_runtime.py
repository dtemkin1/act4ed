"""Patch solver_runtime_s into an existing routing_bird partial-result JSON.

Usage:
    python patch_solver_runtime.py --output-dir <path/to/output_dir>

Iterates over every entry in routing_bird_catalogue_partial_result.json,
loads the matching *_solution.npz file, and writes runtime_seconds back
as solver_runtime_s.  Entries whose solution file is missing are skipped
with a warning.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from formulation.bird_adapter import BirdBackendSolution

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "outputs" / "routing_bird_grid"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    partial_result_path: Path = args.output_dir / "routing_bird_catalogue_partial_result.json"
    data: dict = json.loads(partial_result_path.read_text(encoding="utf-8"))

    missing = 0
    patched = 0
    already_set = 0
    for label, entry in data.items():
        if entry.get("solver_runtime_s") is not None:
            already_set += 1
            continue
        solution_path = args.output_dir / f"{label}_solution.npz"
        if not solution_path.exists():
            print(f"[WARN] missing solution file for {label}")
            missing += 1
            continue
        solution = BirdBackendSolution.load(solution_path)
        entry["solver_runtime_s"] = solution.runtime_seconds
        patched += 1

    partial_result_path.write_text(
        json.dumps(data, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Done: patched={patched}, already_set={already_set}, missing={missing}")


if __name__ == "__main__":
    main()
