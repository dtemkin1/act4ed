#!/bin/bash

# Job Flags
#SBATCH -p mit_normal
#SBATCH -c 24
#SBATCH --mem=375G

set -euo pipefail

BASEDIR=$(cd "$(dirname "$0")" && pwd)

cd "${BASEDIR}"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

uv run python -m experiments.all_gurobi --backend julia "$@"
