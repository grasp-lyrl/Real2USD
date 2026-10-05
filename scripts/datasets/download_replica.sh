#!/usr/bin/env bash
# Download the NICE-SLAM posed RGB-D renders of Replica (community-standard eval
# sequences used by ConceptGraphs/HOV-SG: ~2000 frames/scene, GT poses + intrinsics).
# Usage: bash scripts/datasets/download_replica.sh [DEST_DIR]
set -euo pipefail

DEST="${1:-/data/datasets/replica}"
mkdir -p "$DEST"
cd "$DEST"

# ~12 GB. Same source ConceptGraphs' download script uses.
if [ ! -d "$DEST/Replica" ]; then
  wget -c https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip
  unzip -q Replica.zip && rm Replica.zip
else
  echo "Replica/ already present, skipping RGB-D download."
fi

# camera intrinsics (shared by all sequences) from NICE-SLAM's config
wget -c -O replica_intrinsics.yaml \
  https://raw.githubusercontent.com/cvg/nice-slam/master/configs/Replica/replica.yaml \
  || echo "NOTE: fetch intrinsics from nice-slam configs manually if this URL moved."

echo "Done. Sequences in $DEST/Replica/{room0,room1,room2,office0..office4}/"
echo "GT semantic meshes: see https://github.com/facebookresearch/Replica-Dataset"
echo "(needed for Chamfer/F-score metrics; download per its README)."
