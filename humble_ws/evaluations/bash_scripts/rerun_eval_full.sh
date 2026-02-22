#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <RUN_DIR> <GT_JSON> <SCENE_NAME> [RUN_ID]"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$1"

"${SCRIPT_DIR}/run_eval_pair.sh" "$@"
"${SCRIPT_DIR}/collect_and_plot.sh" "${RUN_DIR%/}/results"

echo "[OK] Full rerun+report flow complete."
