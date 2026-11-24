#!/bin/bash

# ==============================================================================
# SLURM RESOURCE ALLOCATION (SBATCH Directives)
# ==============================================================================
#SBATCH --job-name=wccfc-lorenz-rollout
#SBATCH --output=log/wccfc-lorenz-rollout_%j.log
#SBATCH --partition=GENOA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -e

# ==============================================================================
# ENVIRONMENT SETUP (on compute node)
# ==============================================================================

if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi

mkdir -p log
echo "Working directory: $(pwd)"

VENV_DIR="$(pwd)/.venv"
VENV_PY="${VENV_DIR}/bin/python"

# Serialize venv creation across parallel compute-node jobs
(
    flock -x 200
    if [ ! -x "${VENV_PY}" ]; then
        echo "Virtual environment not found. Creating .venv on compute node..."
        python3 -m venv "${VENV_DIR}" || python -m venv "${VENV_DIR}"
        "${VENV_PY}" -m ensurepip --upgrade --default-pip || true
        "${VENV_PY}" -m pip install --quiet --upgrade pip
        if [ -f "requirements.txt" ]; then
            echo "Installing requirements..."
            "${VENV_PY}" -m pip install --quiet -r requirements.txt
        fi
    fi
) 200>"${VENV_DIR}.lock"

source "${VENV_DIR}/bin/activate"
echo "Using Python: $(which python) ($(python --version 2>&1))"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export MPLBACKEND="Agg"

# Execute experiment
"${VENV_PY}" -u -m src.main --config ./exp/wccfc-lorenz-rollout/config.yaml --device cpu
