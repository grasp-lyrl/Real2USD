#!/usr/bin/env bash
# s200 FRONT-END-QUALITY end-to-end (stride 10, self-consistent 3-way):
#   yoloe  = object_track on YOLOE detections   (deployed detector)
#   sam3   = object_track on SAM 3 detections    (strong detector)
#   oracle = sam3d_layout on GT native masks     (perfect detection, ceiling)
# All feed the SAME back-end (best-view -> SAM3D mesh -> registration), so the only
# variable is the front-end. Answers: does a better detector improve placement?
#
# SEPARATE SAM3D queue per front-end: object_track job ids are procthor_200_t{track}_full,
# and YOLOE vs SAM 3 track numbering overlaps -> shared queue would collide/reuse meshes.
#
# queue-stage --out is a THROWAWAY dir (never the collect target) so its n_pred=0
# placeholder run.json can't make collect's "[ -f run.json ] && skip" skip the real run.
#
#   bash scripts/run_s200_e2e.sh queue     # enqueue SAM3D jobs (3 queues)
#   <run the worker on each of _feq_{yoloe,sam3,oracle} until drained>
#   bash scripts/run_s200_e2e.sh collect   # collect x {icp, scale_icp}
set -u
cd /home/chris.hsu/repos/Real2USD/humble_ws/src_Real2USD/r2s3d_core
export DISPLAY=${R2S3D_DISPLAY:-:0}
export XAUTHORITY=${R2S3D_XAUTHORITY:-/run/user/1003/gdm/Xauthority}
EX="--extra procthor --extra mesh --extra registration"
S=200; STR=10; SPLIT=val
DET_yoloe=results/detections/probe_yoloe/gt
DET_sam3=results/detections/probe_sam3/gt
MODE=${1:-queue}

if [ "$MODE" = "queue" ]; then
  for fe in yoloe sam3; do
    det=DET_$fe; q=results/paper/sim/_feq_$fe
    uv run $EX python -m r2s3d_core.eval.run --source procthor --scene $S --split $SPLIT \
      --method object_track --registration icp --scale-source reproj \
      --detections ${!det} --gt-mesh asset --stride $STR --label-map clip \
      --sam3d-queue $q --out results/paper/sim/_feq_stage_$fe --no-glb || echo "[$fe] QUEUE FAIL"
  done
  uv run $EX python -m r2s3d_core.eval.run --source procthor --scene $S --split $SPLIT \
    --method sam3d_layout --gt-mesh asset --stride $STR --label-map clip \
    --sam3d-queue results/paper/sim/_feq_oracle --out results/paper/sim/_feq_stage_oracle --no-glb \
    || echo "[oracle] QUEUE FAIL"
  for fe in yoloe sam3 oracle; do q=results/paper/sim/_feq_$fe; echo "$q input: $(ls $q/input 2>/dev/null|wc -l)"; done

elif [ "$MODE" = "collect" ]; then
  for fe in yoloe sam3; do
    det=DET_$fe; q=results/paper/sim/_feq_$fe
    for reg in icp scale_icp; do
      out=results/paper/sim/fe_${fe}_${reg}_s200
      [ -f "$out/run.json" ] && { echo "[$fe/$reg] done"; continue; }
      uv run $EX python -m r2s3d_core.eval.run --source procthor --scene $S --split $SPLIT \
        --method object_track --registration $reg --scale-source reproj \
        --detections ${!det} --gt-mesh asset --stride $STR --label-map clip \
        --sam3d-queue $q --out $out --no-glb || echo "[$fe/$reg] FAIL"
    done
  done
  for m in sam3d_layout_icp sam3d_layout_scale_icp; do
    case $m in sam3d_layout_icp) name=icp;; *) name=scaleicp;; esac
    out=results/paper/sim/fe_oracle_${name}_s200
    [ -f "$out/run.json" ] && { echo "[oracle/$name] done"; continue; }
    uv run $EX python -m r2s3d_core.eval.run --source procthor --scene $S --split $SPLIT \
      --method $m --gt-mesh asset --stride $STR --label-map clip \
      --sam3d-queue results/paper/sim/_feq_oracle --out $out --no-glb || echo "[oracle/$name] FAIL"
  done
fi
