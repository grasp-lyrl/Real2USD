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
if [[ ! -f "${RUN_DIR}/scene_graph_sam3d_only.json" ]]; then
  echo "[ERR] Missing ${RUN_DIR}/scene_graph_sam3d_only.json"
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

# 1) Pipeline full
python3 "${EVAL_ROOT}/scripts/run_eval.py" \
  --prediction-type scene_graph \
  --prediction-json "${RUN_DIR}/scene_graph.json" \
  --gt-json "$GT_JSON" \
  --run-dir "$RUN_DIR" \
  --scene "$SCENE_NAME" \
  --run-id "$RUN_ID" \
  --method-tag "pipeline_full" \
  --results-root "$RESULTS_ROOT"

# 2) SAM3D-only
python3 "${EVAL_ROOT}/scripts/run_eval.py" \
  --prediction-type scene_graph \
  --prediction-json "${RUN_DIR}/scene_graph_sam3d_only.json" \
  --gt-json "$GT_JSON" \
  --run-dir "$RUN_DIR" \
  --scene "$SCENE_NAME" \
  --run-id "$RUN_ID" \
  --method-tag "sam3d_only" \
  --results-root "$RESULTS_ROOT"

# 3) CLIO
python3 "${EVAL_ROOT}/scripts/run_eval.py" \
  --prediction-type clio \
  --prediction-json "$CLIO_GRAPHML" \
  --gt-json "$GT_JSON" \
  --run-dir "$RUN_DIR" \
  --scene "$SCENE_NAME" \
  --run-id "$RUN_ID" \
  --method-tag "clio" \
  --results-root "$RESULTS_ROOT"

# 4) Consolidated tables/plots over this results root
python3 "${EVAL_ROOT}/collect_ablation_table.py" --results-root "$RESULTS_ROOT"
python3 "${EVAL_ROOT}/plot_eval.py" --results-root "$RESULTS_ROOT"

# 5) Side-by-side comparison artifacts for the 3 methods
python3 "${EVAL_ROOT}/compare_three_methods.py" \
  --results-root "$RESULTS_ROOT" \
  --scene "$SCENE_NAME" \
  --run-id "$RUN_ID" \
  --method-tags "pipeline_full,sam3d_only,clio"

# 6) BBox overlays for each method (including CLIO)
for METHOD in pipeline_full sam3d_only clio; do
  JSON_PATH="$(python3 - <<PY
import json
from pathlib import Path
results_root=Path(${RESULTS_ROOT@Q})
scene=${SCENE_NAME@Q}
run_id=${RUN_ID@Q}
method=${METHOD@Q}
best=None
best_ts=""
for p in (results_root/'by_run').glob('*.json'):
    try:
        d=json.loads(p.read_text())
        m=d.get('metadata',{})
        if m.get('scene')!=scene or m.get('run_id')!=run_id or (m.get('method_tag') or '')!=method:
            continue
        ts=str(m.get('created_at',''))
        if ts>=best_ts:
            best=p; best_ts=ts
    except Exception:
        pass
print(str(best) if best else '')
PY
)"
  if [[ -n "$JSON_PATH" ]]; then
    python3 "${EVAL_ROOT}/scripts/plot_overlays.py" --by-run-json "$JSON_PATH" --run-dir "$RUN_DIR"
    python3 "${EVAL_ROOT}/pose_sensitivity_eval.py" --by-run-json "$JSON_PATH"
  fi
done

# 7) Consolidated lidar-vs-realsense decision report (if both sensors exist in results)
python3 "${EVAL_ROOT}/sensor_decision_report.py" --results-root "$RESULTS_ROOT" || true

echo "[OK] Full 3-method comparison complete."
echo "[OK] See: ${RESULTS_ROOT}/comparisons and ${RESULTS_ROOT}/plots"
