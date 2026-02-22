#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <RUN_DIR> <GT_JSON> <SCENE_NAME> [RUN_ID]"
  exit 1
fi

RUN_DIR="$1"
GT_JSON="$2"
SCENE_NAME="$3"
RUN_ID="${4:-$(basename "$RUN_DIR")}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WS_ROOT="$(cd "${EVAL_ROOT}/.." && pwd)"
RESULTS_ROOT="${RUN_DIR%/}/results"
RUN_EVAL="${EVAL_ROOT}/run_eval.py"
PLOT_OVERLAY="${EVAL_ROOT}/plot_bbox_overlays.py"
WS_SETUP="${WS_ROOT}/install/setup.bash"

if [[ ! -d "$RUN_DIR" ]]; then
  echo "[ERR] RUN_DIR not found: $RUN_DIR"
  exit 1
fi
if [[ ! -f "$GT_JSON" ]]; then
  echo "[ERR] GT_JSON not found: $GT_JSON"
  exit 1
fi
if [[ ! -f "${RUN_DIR}/scene_graph.json" ]]; then
  echo "[ERR] Missing ${RUN_DIR}/scene_graph.json"
  exit 1
fi
if [[ ! -f "${RUN_DIR}/scene_graph_sam3d_only.json" ]]; then
  echo "[ERR] Missing ${RUN_DIR}/scene_graph_sam3d_only.json"
  exit 1
fi

if [[ -f /opt/ros/humble/setup.bash ]]; then
  set +u
  source /opt/ros/humble/setup.bash
  set -u
else
  echo "[WARN] ROS setup not found at /opt/ros/humble/setup.bash; continuing without sourcing it."
fi

if [[ -f "$WS_SETUP" ]]; then
  set +u
  source "$WS_SETUP"
  set -u
else
  echo "[WARN] Workspace setup not found at ${WS_SETUP}; continuing."
fi

python3 "$RUN_EVAL" \
  --prediction-json "${RUN_DIR}/scene_graph.json" \
  --gt-json "$GT_JSON" \
  --run-dir "$RUN_DIR" \
  --scene "$SCENE_NAME" \
  --run-id "$RUN_ID" \
  --results-root "$RESULTS_ROOT"

BY_RUN_MAIN="$(ls -t "${RESULTS_ROOT}/by_run/${SCENE_NAME}_${RUN_ID}_"*.json 2>/dev/null | head -n1 || true)"
if [[ -n "${BY_RUN_MAIN}" ]]; then
  python3 "$PLOT_OVERLAY" --by-run-json "$BY_RUN_MAIN"
fi

python3 "$RUN_EVAL" \
  --prediction-json "${RUN_DIR}/scene_graph_sam3d_only.json" \
  --gt-json "$GT_JSON" \
  --run-dir "$RUN_DIR" \
  --scene "${SCENE_NAME}_sam3d_only" \
  --run-id "$RUN_ID" \
  --results-root "$RESULTS_ROOT"

BY_RUN_SAM3D="$(ls -t "${RESULTS_ROOT}/by_run/${SCENE_NAME}_sam3d_only_${RUN_ID}_"*.json 2>/dev/null | head -n1 || true)"
if [[ -n "${BY_RUN_SAM3D}" ]]; then
  python3 "$PLOT_OVERLAY" --by-run-json "$BY_RUN_SAM3D"
fi

echo "[OK] Pair eval complete for run_id=${RUN_ID}"
