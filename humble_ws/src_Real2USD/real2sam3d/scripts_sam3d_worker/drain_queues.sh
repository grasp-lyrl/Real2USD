#!/usr/bin/env bash
# Drain a list of SAM3D queues sequentially (one worker model-load per queue).
# Launch via conda run (CONDA_PREFIX) + --no-current-run, from this dir. See
# [[sam3d-worker-launch]]. Kills the worker between queues to free the GPU before
# the next model load. Args: queue dir basenames under results/paper/sim/.
set -u
cd /home/chris.hsu/repos/Real2USD/humble_ws/src_Real2USD/real2sam3d/scripts_sam3d_worker
R2S3D=/home/chris.hsu/repos/Real2USD/humble_ws/src_Real2USD/r2s3d_core
REPO=/home/chris.hsu/repos/Real2USD/humble_ws/src_Real2USD/real2sam3d/sam-3d-objects
for name in "$@"; do
  q="$R2S3D/results/paper/sim/$name"
  n=$(ls "$q/input" 2>/dev/null | wc -l)
  if [ "$n" -eq 0 ]; then echo ">>> $name: empty, skip"; continue; fi
  echo ">>> draining $name ($n jobs)"
  ~/miniforge3/bin/conda run -n sam3d-objects python -u run_sam3d_worker.py \
    --no-current-run --queue-dir "$q" --sam3d-repo "$REPO" --use-depth &
  until [ "$(ls "$q/input" 2>/dev/null | wc -l)" -eq 0 ]; do sleep 20; done
  # kill this queue's worker (+ conda wrapper) by matching its queue path; grep[.]trick avoids self-match
  for p in $(ps -eo pid,cmd | grep -F "$name" | grep "run_sam3d_worker" | grep -v grep | awk '{print $1}'); do kill -9 "$p" 2>/dev/null; done
  sleep 4
  echo ">>> $name drained: $(ls $q/input_processed 2>/dev/null|wc -l) ok, $(ls $q/input_failed 2>/dev/null|wc -l) failed"
done
echo ">>> ALL QUEUES DRAINED"
