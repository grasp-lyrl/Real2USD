#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <RUN_DIR> <GT_JSON> <SCENE_NAME> [RUN_ID] [CLIO_GRAPHML]"
  echo "  RUN_DIR       directory with scene_graph.json and scene_graph_sam3d_only.json"
  echo "  GT_JSON       Supervisely GT (e.g. supervisely/lounge-0_voxel_pointcloud.pcd.json)"
  echo "  SCENE_NAME    scene name (e.g. lounge)"
  echo "  RUN_ID        optional; default: basename of RUN_DIR"
  echo "  CLIO_GRAPHML  optional; if set, also run eval on CLIO and include in results"
  exit 1
fi

RUN_DIR="$1"
GT_JSON="$2"
SCENE_NAME="$3"
RUN_ID="${4:-$(basename "$RUN_DIR")}"
CLIO_GRAPHML="${5:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WS_ROOT="$(cd "${EVAL_ROOT}/.." && pwd)"
RESULTS_ROOT="${RUN_DIR%/}/results"
RUN_EVAL="${EVAL_ROOT}/scripts/run_eval.py"
PLOT_OVERLAY="${EVAL_ROOT}/scripts/plot_overlays.py"
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
  --prediction-type scene_graph \
  --prediction-json "${RUN_DIR}/scene_graph.json" \
  --gt-json "$GT_JSON" \
  --run-dir "$RUN_DIR" \
  --scene "$SCENE_NAME" \
  --run-id "$RUN_ID" \
  --method-tag "pipeline_full" \
  --results-root "$RESULTS_ROOT"

BY_RUN_MAIN="$(ls -t "${RESULTS_ROOT}/by_run/${SCENE_NAME}_${RUN_ID}_pipeline_full_"*.json 2>/dev/null | head -n1 || true)"
if [[ -z "${BY_RUN_MAIN}" ]]; then
  BY_RUN_MAIN="$(ls -t "${RESULTS_ROOT}/by_run/${SCENE_NAME}_${RUN_ID}_"*.json 2>/dev/null | head -n1 || true)"
fi
if [[ -n "${BY_RUN_MAIN}" ]]; then
  python3 "$PLOT_OVERLAY" --by-run-json "$BY_RUN_MAIN" --run-dir "$RUN_DIR"
fi

python3 "$RUN_EVAL" \
  --prediction-type scene_graph \
  --prediction-json "${RUN_DIR}/scene_graph_sam3d_only.json" \
  --gt-json "$GT_JSON" \
  --run-dir "$RUN_DIR" \
  --scene "$SCENE_NAME" \
  --run-id "$RUN_ID" \
  --method-tag "sam3d_only" \
  --results-root "$RESULTS_ROOT"

BY_RUN_SAM3D="$(ls -t "${RESULTS_ROOT}/by_run/${SCENE_NAME}_${RUN_ID}_sam3d_only_"*.json 2>/dev/null | head -n1 || true)"
if [[ -n "${BY_RUN_SAM3D}" ]]; then
  python3 "$PLOT_OVERLAY" --by-run-json "$BY_RUN_SAM3D" --run-dir "$RUN_DIR"
fi

# Optional: eval CLIO and plot overlays
if [[ -n "${CLIO_GRAPHML}" ]]; then
  if [[ ! -f "$CLIO_GRAPHML" ]]; then
    echo "[ERR] CLIO_GRAPHML not found: $CLIO_GRAPHML"
    exit 1
  fi
  python3 "$RUN_EVAL" \
    --prediction-type clio \
    --prediction-json "$CLIO_GRAPHML" \
    --gt-json "$GT_JSON" \
    --run-dir "$RUN_DIR" \
    --scene "$SCENE_NAME" \
    --run-id "$RUN_ID" \
    --method-tag "clio" \
    --results-root "$RESULTS_ROOT"
  BY_RUN_CLIO="$(ls -t "${RESULTS_ROOT}/by_run/${SCENE_NAME}_${RUN_ID}_clio_"*.json 2>/dev/null | head -n1 || true)"
  if [[ -n "${BY_RUN_CLIO}" ]]; then
    python3 "$PLOT_OVERLAY" --by-run-json "$BY_RUN_CLIO" --run-dir "$RUN_DIR"
  fi
fi

echo "[OK] Eval complete for run_id=${RUN_ID}"
