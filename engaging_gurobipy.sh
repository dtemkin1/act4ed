#!/bin/bash

# Job Flags
#SBATCH -p mit_normal
#SBATCH -c 24
#SBATCH --mem=375G

set -euo pipefail

if [[ -n "${PROJECT_ROOT:-}" ]]; then
	BASEDIR="${PROJECT_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
	BASEDIR="${SLURM_SUBMIT_DIR}"
else
	BASEDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fi

cd "${BASEDIR}"

CONTAINER="${ACT4ED_CONTAINER:-${JLAB_CONTAINER:-}}"

if [[ -n "$CONTAINER" ]]; then
	exec apptainer exec \
		${ACT4ED_APPTAINER_ARGS:-} \
		--bind "${BASEDIR}:/work" \
		--pwd /work \
		"$CONTAINER" \
		bash -lc 'set -euo pipefail; /usr/local/bin/uv sync --frozen; exec /usr/local/bin/uv run python -m experiments.all_gurobi "$@"' bash "$@"
fi

uv sync --frozen
exec uv run python -m experiments.all_gurobi "$@"