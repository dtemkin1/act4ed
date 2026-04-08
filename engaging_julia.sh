#!/bin/bash

# Job Flags
#SBATCH -p mit_quicktest
#SBATCH --time=0:10:00
#SBATCH -c 24
#SBATCH --mem=124G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

if [[ -n "${PROJECT_ROOT:-}" ]]; then
  BASEDIR="${PROJECT_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  BASEDIR="${SLURM_SUBMIT_DIR}"
else
  BASEDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fi

cd "${BASEDIR}"

module load julia/1.10.4

export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-$HOME/.julia_depot}"
export JULIA_CLUSTER_PROJECT="${JULIA_CLUSTER_PROJECT:-${BASEDIR}/.julia_cluster_project}"

mkdir -p "${JULIA_DEPOT_PATH}"
mkdir -p "${JULIA_CLUSTER_PROJECT}"

cp "${BASEDIR}/julia/Project.toml" "${JULIA_CLUSTER_PROJECT}/Project.toml"
rm -rf "${JULIA_CLUSTER_PROJECT}/src"
cp -R "${BASEDIR}/julia/src" "${JULIA_CLUSTER_PROJECT}/src"
export JULIA_PROJECT="${JULIA_CLUSTER_PROJECT}"

julia --project="${JULIA_PROJECT}" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

uv run python -m experiments.all_gurobi_julia --backend julia "$@"
