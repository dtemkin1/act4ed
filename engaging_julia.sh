#!/bin/bash

# Job Flags
#SBATCH -p mit_quicktest
#SBATCH --time=0:10:00
#SBATCH -c 24
#SBATCH --mem=375G

set -euo pipefail

BASEDIR=$(cd "$(dirname "$0")" && pwd)

cd "${BASEDIR}"

module load julia/1.9.1

export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export JULIA_PROJECT="${BASEDIR}/julia"
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-${BASEDIR}/.julia_depot}"

mkdir -p "${JULIA_DEPOT_PATH}"

julia --project="${JULIA_PROJECT}" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

uv run python -m experiments.all_gurobi_julia --backend julia "$@"
