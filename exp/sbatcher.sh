#!/bin/bash
set -e

# ==============================================================================
# SBATCHER: Submit all experiment sbatch.sh scripts under ./exp/ with one call.
#
# Usage (from repository root or anywhere):
#   bash sbatcher.sh           # Submit all SLURM jobs
#   bash sbatcher.sh --local   # Run all experiments sequentially in current shell
# ==============================================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

mkdir -p log

# Ensure .venv exists and dependencies are installed once before parallel sbatch
VENV_DIR="${REPO_ROOT}/.venv"
VENV_PY="${VENV_DIR}/bin/python"
if [ ! -x "${VENV_PY}" ]; then
    echo "Virtual environment not found at ${VENV_DIR}. Creating it..."
    python3 -m venv "${VENV_DIR}" || python -m venv "${VENV_DIR}"
    "${VENV_PY}" -m ensurepip --upgrade --default-pip || true
    "${VENV_PY}" -m pip install --quiet --upgrade pip
    if [ -f "${REPO_ROOT}/requirements.txt" ]; then
        echo "Installing requirements into ${VENV_DIR}..."
        "${VENV_PY}" -m pip install --quiet -r "${REPO_ROOT}/requirements.txt"
    fi
fi

EXPERIMENTS=(
    "cfc-lorenz-rollout"
    "cfc-lorenz-no-rollout"
    "wccfc-lorenz-rollout"
    "wccfc-lorenz-no-rollout"
    "cfc-duffing-rollout"
    "cfc-duffing-no-rollout"
    "wccfc-duffing-rollout"
    "wccfc-duffing-no-rollout"
)

if [ "${1:-}" == "--local" ]; then
    echo "Running all ${#EXPERIMENTS[@]} experiments sequentially in local mode..."
    source "${VENV_DIR}/bin/activate"
    export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
    export MPLBACKEND="Agg"
    for exp in "${EXPERIMENTS[@]}"; do
        cfg="./exp/${exp}/config.yaml"
        echo "------------------------------------------------------------"
        echo "Running ${exp} (${cfg}) ..."
        "${VENV_PY}" -u -m src.main --config "${cfg}" --device auto
    done
    exit 0
fi

JOB_IDS=()

for exp in "${EXPERIMENTS[@]}"; do
    script="./exp/${exp}/sbatch.sh"
    cfg="./exp/${exp}/config.yaml"
    if [ ! -f "${script}" ] || [ ! -f "${cfg}" ]; then
        echo "Error: Missing ${script} or ${cfg}"
        exit 1
    fi

    job_id=$(sbatch --parsable "${script}")
    echo "Submitted ${exp} (${script}) -> Job ID: ${job_id}"
    JOB_IDS+=("${job_id}")
done

echo "============================================================"
echo "Submitted ${#JOB_IDS[@]} SLURM jobs: ${JOB_IDS[*]}"
echo "Logs will be written to: ${REPO_ROOT}/log/"
echo "============================================================"
