#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <RUN_DIR> <GT_JSON> <SCENE_NAME> [RUN_ID] [THRESHOLDS] [LABEL_MATCH_OPTIONS]"
  echo "Example: $0 /data/sam3d_queue/run_20260222_051228 evaluations/supervisely/lounge-0_voxel_pointcloud.pcd.json lounge-0-lidar"
  exit 1
fi

RUN_DIR="$1"
GT_JSON="$2"
SCENE_NAME="$3"
RUN_ID="${4:-$(basename "$RUN_DIR")}"
THRESHOLDS="${5:-0.05,0.10,0.15,0.20,0.25}"
LABEL_MATCH_OPTIONS="${6:-false,true}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WS_ROOT="$(cd "${EVAL_ROOT}/.." && pwd)"
RESULTS_ROOT="${RUN_DIR%/}/results"
WS_SETUP="${WS_ROOT}/install/setup.bash"

if [[ ! -f "${RUN_DIR}/scene_graph.json" ]]; then
  echo "[ERR] Missing ${RUN_DIR}/scene_graph.json"
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

python3 "${EVAL_ROOT}/sweep_eval.py" \
  --prediction-json "${RUN_DIR}/scene_graph.json" \
  --gt-json "${GT_JSON}" \
  --run-dir "${RUN_DIR}" \
  --scene "${SCENE_NAME}" \
  --run-id "${RUN_ID}" \
  --results-root "${RESULTS_ROOT}" \
  --thresholds "${THRESHOLDS}" \
  --label-match-options "${LABEL_MATCH_OPTIONS}"

echo "[OK] Sweep complete under ${RESULTS_ROOT}/sweeps"
