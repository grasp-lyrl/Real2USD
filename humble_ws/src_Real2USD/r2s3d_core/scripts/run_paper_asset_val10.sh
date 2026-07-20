#!/usr/bin/env bash
# Asset val-10 campaign (docs/EXPERIMENT_MATRIX.md). Produces, per val id, the SAM3D asset under
# THREE registration variants that share ONE generation pass (mesh cache is registration-independent):
#   layout  = registration none  -> SAM 3D's OWN predicted scale/rotation/translation (make_scene-style)
#   icp     = +rigid ICP          -> our pose correction
#   scaleicp= +depth-extent scale-fit (reproj) + ICP -> our scale+pose correction
# => the C1 table "SAM3D-native localization vs our registration", across 10 scenes; and the asset
#    meshes needed to run gen_shape_completion.py at n=10.
#
# Stages (renders already cached from the cluster run; the SAM3D WORKER is the slow part):
#   queue   : render+track+enqueue SAM3D jobs for the 9 new ids into the shared queue (fast)
#   collect : after the worker drains, collect all 10 ids x {layout,icp,scaleicp} (free eval re-runs)
# Between them, run the worker (conda sam3d-objects):
#   conda activate sam3d-objects && python \
#     humble_ws/src_Real2USD/real2sam3d/scripts_sam3d_worker/run_sam3d_worker.py \
#     --queue-dir <Q> --sam3d-repo humble_ws/src_Real2USD/real2sam3d/sam-3d-objects --use-depth --once
set -u
cd /home/chris.hsu/repos/Real2USD/humble_ws/src_Real2USD/r2s3d_core
export DISPLAY=${R2S3D_DISPLAY:-:0}
export XAUTHORITY=${R2S3D_XAUTHORITY:-/run/user/1003/gdm/Xauthority}
EX="--extra procthor --extra mesh --extra registration --extra detector"
DET=results/detections/procthor/gt
Q=results/paper/sim/_assetq                 # shared SAM3D queue for the 9 NEW ids
S200Q=results/procthor_procthor_object_track_icp_s200_val/sam3d_queue  # s200 already drained
NEW_IDS="137 428 434 534 569 573 683 771 912"
ALL_IDS="137 200 428 434 534 569 573 683 771 912"
MODE=${1:-queue}

if [ "$MODE" = "queue" ]; then
  for id in $NEW_IDS; do
    echo "########## queue $id ##########"
    uv run $EX python -m r2s3d_core.eval.run --source procthor --scene $id --split val \
      --method object_track_icp --detections $DET --gt-mesh asset --stride 1 --label-map clip \
      --sam3d-queue $Q --out results/paper/sim/asset_icp_gt_s$id --no-glb || echo "[$id] QUEUE FAIL"
  done
  echo "########## QUEUE DONE — pending jobs: $(ls $Q/input 2>/dev/null | wc -l) done: $(ls $Q/output 2>/dev/null | wc -l) ##########"

elif [ "$MODE" = "collect" ]; then
  for id in $ALL_IDS; do
    q=$Q; [ "$id" = "200" ] && q=$S200Q
    for reg in none icp scale_icp; do
      case $reg in none) name=layout;; scale_icp) name=scaleicp;; *) name=$reg;; esac
      out=results/paper/sim/asset_${name}_gt_s$id
      [ -f "$out/run.json" ] && { echo "[$id/$name] done"; continue; }
      echo "########## collect $id / $name ##########"
      uv run $EX python -m r2s3d_core.eval.run --source procthor --scene $id --split val \
        --method object_track --registration $reg --scale-source reproj \
        --detections $DET --gt-mesh asset --stride 1 --label-map clip \
        --sam3d-queue $q --out $out --no-glb || echo "[$id/$name] COLLECT FAIL"
    done
  done
  echo "########## COLLECT DONE ##########"
fi
