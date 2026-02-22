#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <RUN_DIR> <CLIO_GRAPHML> <GT_JSON> <SCENE_NAME> [RUN_ID] [RESULTS_ROOT]"
  exit 1
fi

RUN_DIR="$1"
CLIO_GRAPHML="$2"
GT_JSON="$3"
SCENE_NAME="$4"
RUN_ID="${5:-$(basename "$RUN_DIR")}"
RESULTS_ROOT="${6:-${RUN_DIR%/}/results}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WS_ROOT="$(cd "${EVAL_ROOT}/.." && pwd)"
WS_SETUP="${WS_ROOT}/install/setup.bash"

if [[ ! -f "${RUN_DIR}/scene_graph.json" ]]; then
  echo "[ERR] Missing ${RUN_DIR}/scene_graph.json"
  exit 1
fi
if [[ ! -f "$CLIO_GRAPHML" ]]; then
  echo "[ERR] Missing CLIO graphml: $CLIO_GRAPHML"
  exit 1
fi
if [[ ! -f "$GT_JSON" ]]; then
  echo "[ERR] Missing GT JSON: $GT_JSON"
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

python3 "${EVAL_ROOT}/run_eval.py" \
  --prediction-type scene_graph \
  --prediction-path "${RUN_DIR}/scene_graph.json" \
  --gt-json "$GT_JSON" \
  --run-dir "$RUN_DIR" \
  --scene "$SCENE_NAME" \
  --run-id "$RUN_ID" \
  --results-root "$RESULTS_ROOT"

python3 "${EVAL_ROOT}/run_eval.py" \
  --prediction-type clio \
  --prediction-path "$CLIO_GRAPHML" \
  --gt-json "$GT_JSON" \
  --run-dir "$RUN_DIR" \
  --scene "$SCENE_NAME" \
  --run-id "$RUN_ID" \
  --results-root "$RESULTS_ROOT"

python3 "${EVAL_ROOT}/compare_pipeline_clio.py" \
  --results-root "$RESULTS_ROOT" \
  --scene "$SCENE_NAME" \
  --run-id "$RUN_ID"

echo "[OK] Pipeline vs CLIO side-by-side complete under ${RESULTS_ROOT}/comparisons"
