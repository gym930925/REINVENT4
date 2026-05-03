#!/bin/bash
# Run all NaviDiv diversity strategies against QED optimisation, then
# post-process each run with t-SNE and comprehensive diversity analysis.
#
# Usage (from any directory):
#   chmod +x run.sh
#   ./run.sh
#
# Prerequisites:
#   conda env 'reinvent4' with NaviDiv + REINVENT4 installed
#   NAVIDIV_ROOT exported, or let this script detect it automatically

set -euo pipefail

# ── path resolution ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Auto-detect NAVIDIV_ROOT as the NaviDiv package root on PYTHONPATH, or fall
# back to expecting it as a sibling of the REINVENT4 contrib tree.
if [ -z "${NAVIDIV_ROOT:-}" ]; then
    # Try: two levels up from contrib/tutorials/NaviDiv sits the REINVENT4 repo;
    # NaviDiv is expected as a sibling directory.
    REINVENT4_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
    NAVIDIV_ROOT="$(cd "$REINVENT4_ROOT/../NaviDiv" 2>/dev/null && pwd)" || true
fi

if [ -z "${NAVIDIV_ROOT:-}" ] || [ ! -d "$NAVIDIV_ROOT" ]; then
    echo "ERROR: Cannot find NaviDiv root. Set NAVIDIV_ROOT before running:"
    echo "  export NAVIDIV_ROOT=/path/to/NaviDiv"
    exit 1
fi

echo "NAVIDIV_ROOT = $NAVIDIV_ROOT"

# ── environment ──────────────────────────────────────────────────────────────
ENV_NAME="reinvent4"
export PYTHONPATH="${PYTHONPATH:-}:${NAVIDIV_ROOT}/src/navidiv/reinvent"

PRIOR="${SCRIPT_DIR}/priors/formed.prior"
CONFIG_PATH="${SCRIPT_DIR}/conf_folder"
WD="${SCRIPT_DIR}/runs/demo"

mkdir -p "$WD"

# ── run each diversity strategy ──────────────────────────────────────────────
for diversity_yaml in "$CONFIG_PATH"/diversity_scorer/*.yaml; do
    scorer_name="$(basename "$diversity_yaml" .yaml)"

    # skip the internal default used by test.yaml itself
    [ "$scorer_name" = "1_default" ] && continue

    echo ""
    echo "=== Running diversity_scorer=${scorer_name} ==="

    conda run -n "$ENV_NAME" \
        python3 "${NAVIDIV_ROOT}/src/navidiv/reinvent/run_reinvent_2.py" \
            --config-name test \
            --config-path "$CONFIG_PATH" \
            "name=${scorer_name}" \
            "wd=${WD}" \
            "input_generator.file_path=${SCRIPT_DIR}/InputGenerator_custom.py" \
            "reinvent_common.prior_filename=${PRIOR}" \
            "reinvent_common.agent_filename=${PRIOR}" \
            "reinvent_common.max_steps=100" \
            "diversity_scorer=${scorer_name}"

    CSV_OUT="${WD}/${scorer_name}/${scorer_name}_1.csv"

    # t-SNE projection
    conda run -n "$ENV_NAME" \
        python3 "${NAVIDIV_ROOT}/src/navidiv/get_tsne.py" \
            --df_path "$CSV_OUT" \
            --step 20

    # Comprehensive diversity analysis
    conda run -n "$ENV_NAME" \
        python3 "${NAVIDIV_ROOT}/src/navidiv/run_all_scorers.py" \
            --df_path "${CSV_OUT%.csv}_TSNE.csv" \
            --output_path "${WD}/${scorer_name}/scorer_output"
done

echo ""
echo "All runs complete. Results are in: $WD"
echo "Load any *_TSNE.csv in the NaviDiv app for interactive exploration:"
echo "  conda activate NaviDiv && streamlit run \$NAVIDIV_ROOT/app.py"
