#!/usr/bin/env bash
# Sim val-10 paper campaign (see docs/EXPERIMENT_MATRIX.md). Idempotent: skips any run whose
# out-dir already has run.json. Default = FREE stage (gt detections + cluster). Pass --asset to
# also queue+collect the asset_icp runs (needs the SAM3D worker draining each sam3d_queue).
set -u
cd /home/chris.hsu/repos/Real2USD/humble_ws/src_Real2USD/r2s3d_core
# Render display: the GPU-backed Xorg session (2026-07-19 fix for AI-9). Was :1 under the old
# Xorg session; after the Wayland->Xorg switch the session X is :0 and needs the gdm Xauthority.
# Override via env if the session display changes again.
export DISPLAY=${R2S3D_DISPLAY:-:0}
export XAUTHORITY=${R2S3D_XAUTHORITY:-/run/user/1003/gdm/Xauthority}
EX="--extra procthor --extra mesh --extra registration --extra detector"
IDS="137 200 428 434 534 569 573 683 771 912"
DET=results/detections/procthor/gt
DO_ASSET=0; [ "${1:-}" = "--asset" ] && DO_ASSET=1

for id in $IDS; do
  echo "########## scene $id ##########"

  # 1) detections (gt-vocab) — prereq, no SAM3D
  if [ ! -d "$DET/$id" ]; then
    echo "[$id] detect gt ..."
    # NB detect.run appends the PROMPT to --out, so pass the PARENT (results/detections/procthor)
    # -> it writes results/detections/procthor/gt/<scene>. Passing $DET here would double to gt/gt.
    uv run $EX python -m r2s3d_core.detect.run --source procthor --scene $id \
      --prompt gt --split val --stride 1 --out results/detections/procthor \
      || { echo "[$id] DETECT FAIL"; continue; }
  else echo "[$id] detections present"; fi

  # 2) cluster (FREE, no SAM3D)
  OUT=results/paper/sim/cluster_gt_s$id
  if [ ! -f "$OUT/run.json" ]; then
    echo "[$id] cluster ..."
    uv run $EX python -m r2s3d_core.eval.run --source procthor --scene $id --split val \
      --method object_track_cluster --detections $DET --gt-mesh asset --stride 1 \
      --label-map clip --out $OUT --no-glb || echo "[$id] CLUSTER FAIL"
  else echo "[$id] cluster done"; fi

  # 3) asset_icp (SAM3D) — only with --asset; queues jobs, worker must drain, re-run to collect
  if [ "$DO_ASSET" = "1" ]; then
    OUT=results/paper/sim/asset_icp_gt_s$id
    if [ ! -f "$OUT/run.json" ] || [ -n "$(ls $OUT/sam3d_queue/input 2>/dev/null)" ]; then
      echo "[$id] asset_icp (queues SAM3D; worker must drain) ..."
      uv run $EX python -m r2s3d_core.eval.run --source procthor --scene $id --split val \
        --method object_track_icp --detections $DET --gt-mesh asset --stride 1 \
        --label-map clip --out $OUT --no-glb || echo "[$id] ASSET FAIL"
    else echo "[$id] asset done"; fi
  fi
done
echo "########## VAL-10 STAGE DONE (asset=$DO_ASSET) ##########"
