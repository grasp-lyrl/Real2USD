#!/usr/bin/env bash
# Val-10 ORACLE campaign: sam3d_layout driven by GT (native ProcTHOR) instance masks
# -> the perception UPPER BOUND (perfect detection). Detector-free; mirrors
# run_paper_asset_val10.sh but GT-mask-driven.
#
# stride 1 (NOT the old s200 stride-20) so the oracle matches the object_track detector
# runs frame-for-frame -> a fair "swap only the masks" comparison, and fixes the
# stride confound in the s200 perception-ceiling table. Renders are cached from the
# detector runs, so the queue stage reuses them.
#
# Registration variants share ONE generation pass (mesh cache is registration-independent):
#   layout   = sam3d_layout           (SAM 3D-native pose)
#   icp      = sam3d_layout_icp        (+ our ICP)
#   scaleicp = sam3d_layout_scale_icp  (+ reprojection scale-fit + ICP)  <- best
#
# Two stages (run the SAM3D worker in between; conda sam3d-objects):
#   queue   : render GT-mask best views + enqueue SAM3D jobs into the shared queue
#   collect : after the worker drains, collect all 10 x {layout,icp,scaleicp} (free re-runs)
set -u
cd /home/chris.hsu/repos/Real2USD/humble_ws/src_Real2USD/r2s3d_core
export DISPLAY=${R2S3D_DISPLAY:-:0}
export XAUTHORITY=${R2S3D_XAUTHORITY:-/run/user/1003/gdm/Xauthority}
EX="--extra procthor --extra mesh --extra registration"
Q=results/paper/sim/_oracleq          # shared SAM3D queue for the oracle campaign
IDS="137 200 428 434 534 569 573 683 771 912"
MODE=${1:-queue}

if [ "$MODE" = "queue" ]; then
  for id in $IDS; do
    echo "########## queue $id ##########"
    # NB: queue-stage --out must NOT be the collect target (oracle_layout_gt_s$id) --
    # at queue time meshes aren't generated yet, so its run.json is n_pred=0 garbage and
    # the collect's "[ -f run.json ] && continue" would then SKIP the real layout collect.
    uv run $EX python -m r2s3d_core.eval.run --source procthor --scene $id --split val \
      --method sam3d_layout --gt-mesh asset --stride 1 --label-map clip \
      --sam3d-queue $Q --out results/paper/sim/_oracleq_stage_s$id --no-glb \
      || echo "[$id] QUEUE FAIL"
  done
  echo "########## QUEUE DONE — pending: $(ls $Q/input 2>/dev/null | wc -l) done: $(ls $Q/output 2>/dev/null | wc -l) ##########"

elif [ "$MODE" = "collect" ]; then
  for id in $IDS; do
    for m in sam3d_layout sam3d_layout_icp sam3d_layout_scale_icp; do
      case $m in
        sam3d_layout)           name=layout;;
        sam3d_layout_icp)       name=icp;;
        sam3d_layout_scale_icp) name=scaleicp;;
      esac
      out=results/paper/sim/oracle_${name}_gt_s$id
      [ -f "$out/run.json" ] && { echo "[$id/$name] done"; continue; }
      echo "########## collect $id / $name ##########"
      uv run $EX python -m r2s3d_core.eval.run --source procthor --scene $id --split val \
        --method $m --gt-mesh asset --stride 1 --label-map clip \
        --sam3d-queue $Q --out $out --no-glb || echo "[$id/$name] COLLECT FAIL"
    done
  done
  echo "########## COLLECT DONE ##########"
fi
