#!/bin/bash

# Job Flags
#SBATCH -p mit_preemptable
#SBATCH --time=10:00:00
#SBATCH -c 64
#SBATCH --mem=128G
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

CPUS_PER_SOLVE="${CPUS_PER_SOLVE:-4}"
TOTAL_CPUS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-64}}"
WORKERS="${WORKERS:-$(( TOTAL_CPUS / CPUS_PER_SOLVE ))}"
if (( WORKERS < 1 )); then
  WORKERS=1
fi

export CPUS_PER_SOLVE
export WORKERS
export JULIA_NUM_THREADS="${CPUS_PER_SOLVE}"
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-$HOME/.julia_depot}"
export JULIA_CLUSTER_PROJECT="${JULIA_CLUSTER_PROJECT:-${BASEDIR}/.julia_cluster_project}"

echo "Running routing_bird with ${WORKERS} concurrent solves and ${CPUS_PER_SOLVE} CPUs per solve"

export ACT4ED_CONTAINER=~/containers/ACT4ED.sif

CONTAINER="${ACT4ED_CONTAINER:-${JLAB_CONTAINER:-}}"

module load apptainer/1.4.2 >/dev/null 2>&1 || true

if [[ -n "$CONTAINER" ]]; then
  exec apptainer exec \
    ${ACT4ED_APPTAINER_ARGS:-} \
    --bind "${BASEDIR}:/work" \
    --bind "${JULIA_DEPOT_PATH}:${JULIA_DEPOT_PATH}" \
    --pwd /work \
    --env JULIA_NUM_THREADS="${JULIA_NUM_THREADS}" \
    --env CPUS_PER_SOLVE="${CPUS_PER_SOLVE}" \
    --env WORKERS="${WORKERS}" \
    --env JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH}" \
    --env JULIA_CLUSTER_PROJECT="${JULIA_CLUSTER_PROJECT}" \
    "$CONTAINER" \
    bash -lc 'set -euo pipefail; mkdir -p "$JULIA_DEPOT_PATH" "$JULIA_CLUSTER_PROJECT"; cp /work/julia/Project.toml "$JULIA_CLUSTER_PROJECT/Project.toml"; rm -rf "$JULIA_CLUSTER_PROJECT/src"; cp -R /work/julia/src "$JULIA_CLUSTER_PROJECT/src"; export JULIA_PROJECT="$JULIA_CLUSTER_PROJECT"; export JULIA_CMD="/opt/julia/bin/julia"; /opt/julia/bin/julia --threads="$JULIA_NUM_THREADS" --project="$JULIA_PROJECT" -e '\''using Pkg; Pkg.instantiate(); Pkg.precompile()'\''; /usr/local/bin/uv sync --frozen; exec /usr/local/bin/uv run python -m experiments.mcdp.routing_bird --workers "$WORKERS" --cpus-per-solve "$CPUS_PER_SOLVE" "$@"' bash "$@"
fi

module load julia/1.10.4 >/dev/null 2>&1 || true

mkdir -p "${JULIA_DEPOT_PATH}"
mkdir -p "${JULIA_CLUSTER_PROJECT}"

cp "${BASEDIR}/julia/Project.toml" "${JULIA_CLUSTER_PROJECT}/Project.toml"
rm -rf "${JULIA_CLUSTER_PROJECT}/src"
cp -R "${BASEDIR}/julia/src" "${JULIA_CLUSTER_PROJECT}/src"
export JULIA_PROJECT="${JULIA_CLUSTER_PROJECT}"
export JULIA_CMD="$(command -v julia)"

julia --threads="${JULIA_NUM_THREADS}" --project="${JULIA_PROJECT}" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
uv sync --frozen
exec uv run python -m experiments.mcdp.routing_bird \
  --workers "${WORKERS}" \
  --cpus-per-solve "${CPUS_PER_SOLVE}" \
  "$@"
