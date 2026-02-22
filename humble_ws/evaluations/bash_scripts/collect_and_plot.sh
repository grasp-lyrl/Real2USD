#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_ROOT="${1:-${EVAL_ROOT}/results}"

python3 "${EVAL_ROOT}/collect_ablation_table.py" --results-root "$RESULTS_ROOT"
python3 "${EVAL_ROOT}/plot_eval.py" --results-root "$RESULTS_ROOT"

echo "[OK] Tables and plots refreshed: ${RESULTS_ROOT}"
