#!/bin/bash
# Launch the NaviDiv Streamlit app pre-configured with the bundled sample data.
#
# Usage:
#   conda activate NaviDiv
#   bash run_app.sh
#
# Or let the script activate the environment itself:
#   bash run_app.sh --activate

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLE_CSV="${SCRIPT_DIR}/sample_molecules.csv"

# Locate NaviDiv repo root
if [ -z "${NAVIDIV_ROOT:-}" ]; then
    REINVENT4_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
    NAVIDIV_ROOT="$(cd "$REINVENT4_ROOT/../NaviDiv" 2>/dev/null && pwd)" || true
fi

if [ -z "${NAVIDIV_ROOT:-}" ] || [ ! -d "$NAVIDIV_ROOT" ]; then
    echo "ERROR: Cannot find NaviDiv root. Set NAVIDIV_ROOT before running:"
    echo "  export NAVIDIV_ROOT=/path/to/NaviDiv && bash run_app.sh"
    exit 1
fi

APP="${NAVIDIV_ROOT}/app.py"
if [ ! -f "$APP" ]; then
    echo "ERROR: app.py not found at $APP"
    exit 1
fi

# Optional: activate conda env when --activate flag is passed
if [ "${1:-}" = "--activate" ]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate NaviDiv
fi

echo "Starting NaviDiv dashboard ..."
echo ""
echo "  When the browser opens, load this file:"
echo "  $SAMPLE_CSV"
echo ""
echo "  Suggested workflow:"
echo "    1. Load File"
echo "    2. Run t-SNE"
echo "    3. Run individual scorers (Scaffold, Ngram, Fragments, ...)"
echo "    4. Run All Scorers for a full report"
echo ""

streamlit run "$APP" --server.port 8501
