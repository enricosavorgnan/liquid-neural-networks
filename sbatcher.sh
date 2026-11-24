#!/bin/bash
set -e

# ==============================================================================
# SBATCHER: Submit all experiment sbatch.sh scripts under ./exp/ with one call.
#
# Usage (from repository root or anywhere):
#   bash sbatcher.sh           # Submit all SLURM jobs
#   bash sbatcher.sh --local   # Run all experiment scripts sequentially locally
# ==============================================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

mkdir -p log

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
    for exp in "${EXPERIMENTS[@]}"; do
        script="./exp/${exp}/sbatch.sh"
        echo "------------------------------------------------------------"
        echo "Running ${script} ..."
        bash "${script}"
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
