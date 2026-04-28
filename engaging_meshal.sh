#!/bin/bash

# Job Flags
#SBATCH -p mit_normal
#SBATCH -c 24
#SBATCH --mem=375G

BASEDIR=$(dirname $0)

cd ${BASEDIR}
uv run python -m experiments.mcdp.meshal