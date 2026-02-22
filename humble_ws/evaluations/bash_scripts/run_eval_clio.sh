#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <CLIO_GRAPHML> <GT_JSON> <SCENE_NAME> [RUN_ID] [RESULTS_ROOT]"
  exit 1
fi

CLIO_GRAPHML="$1"
GT_JSON="$2"
SCENE_NAME="$3"
RUN_ID="${4:-clio_$(basename "${CLIO_GRAPHML%.*}")}"
RESULTS_ROOT="${5:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WS_ROOT="$(cd "${EVAL_ROOT}/.." && pwd)"
WS_SETUP="${WS_ROOT}/install/setup.bash"

if [[ ! -f "$CLIO_GRAPHML" ]]; then
  echo "[ERR] CLIO graphml not found: $CLIO_GRAPHML"
  exit 1
fi
if [[ ! -f "$GT_JSON" ]]; then
  echo "[ERR] GT json not found: $GT_JSON"
  exit 1
fi

if [[ -f /opt/ros/humble/setup.bash ]]; then
  set +u
  source /opt/ros/humble/setup.bash
  set -u
fi
if [[ -f "$WS_SETUP" ]]; then
  set +u
  source "$WS_SETUP"
  set -u
fi

CMD=(python3 "${EVAL_ROOT}/run_eval.py"
  --prediction-type clio
  --prediction-path "$CLIO_GRAPHML"
  --gt-json "$GT_JSON"
  --scene "$SCENE_NAME"
  --run-id "$RUN_ID")

if [[ -n "$RESULTS_ROOT" ]]; then
  CMD+=(--results-root "$RESULTS_ROOT")
fi

"${CMD[@]}"
echo "[OK] CLIO eval complete for ${SCENE_NAME} (${RUN_ID})"
