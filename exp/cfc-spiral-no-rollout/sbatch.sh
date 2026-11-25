#!/bin/bash

# ==============================================================================
# SLURM RESOURCE ALLOCATION (SBATCH Directives)
# ==============================================================================
#SBATCH --job-name=cfc-spiral-no-rollout
#SBATCH --output=log/cfc-spiral-no-rollout_%j.log
#SBATCH --partition=GENOA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -e

# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================

if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi

mkdir -p log
echo "Working directory: $(pwd)"

VENV_PY="$(pwd)/.venv/bin/python"
if [ ! -x "${VENV_PY}" ]; then
    echo "Virtual environment not found. Creating .venv..."
    python3 -m venv .venv || python -m venv .venv
    "${VENV_PY}" -m ensurepip --upgrade --default-pip || true
    "${VENV_PY}" -m pip install --quiet --upgrade pip
    if [ -f "requirements.txt" ]; then
        "${VENV_PY}" -m pip install --quiet -r requirements.txt
    fi
fi

source "$(pwd)/.venv/bin/activate"
echo "Using Python: $(which python) ($(python --version 2>&1))"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export MPLBACKEND="Agg"

# Execute experiment
"${VENV_PY}" -u -m src.main --config ./exp/cfc-spiral-no-rollout/config.yaml --device cpu
