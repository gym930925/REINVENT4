#!/bin/bash
# Thin wrapper that activates the reinvent4 conda environment and launches
# run.sh. Useful on clusters where you submit via sbatch or qsub.
#
# Usage:
#   bash local_submitter.sh              # run locally
#   sbatch local_submitter.sh            # submit to SLURM (add #SBATCH headers below)
#
# SLURM example headers (uncomment and adjust):
# #SBATCH --job-name=navidiv_demo
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=16G
# #SBATCH --time=4:00:00
# #SBATCH --output=slurm_%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Export NAVIDIV_ROOT if not already set.
# Adjust this path if NaviDiv is installed elsewhere.
if [ -z "${NAVIDIV_ROOT:-}" ]; then
    REINVENT4_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
    export NAVIDIV_ROOT="$(cd "$REINVENT4_ROOT/../NaviDiv" && pwd)"
fi

export PYTHONPATH="${PYTHONPATH:-}:${NAVIDIV_ROOT}/src/navidiv/reinvent"

echo "Activating reinvent4 environment ..."
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate reinvent4

echo "Launching run.sh ..."
bash "${SCRIPT_DIR}/run.sh"
