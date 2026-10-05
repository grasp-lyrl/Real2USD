#!/usr/bin/env bash
# Download the original Replica-Dataset (facebookresearch/Replica-Dataset) semantic
# assets used as GT for Phase 0 metrics: per-scene habitat/mesh_semantic.ply (with
# per-face object_id) + info_semantic.json.
#
# The upstream release is a single ~100 GB tarball split into parts a..q. There is
# no official per-scene download, so we fetch and extract all of it into
# <DEST>/replica_semantic/ (resumable). r2s3d_core's Replica backend looks there.
#
# Usage: bash scripts/datasets/download_replica_semantic.sh [DEST_DIR]
#        DEST_DIR defaults to the replica data root (~/Data/datasets/replica).
set -euo pipefail

DEST="${1:-$HOME/Data/datasets/replica}"
SEM="$DEST/replica_semantic"
mkdir -p "$SEM"
cd "$SEM"

BASE="https://github.com/facebookresearch/Replica-Dataset/releases/download/v1.0/replica_v1_0.tar.gz.part"

echo "Downloading Replica-Dataset parts a..q into $SEM (resumable)..."
for p in {a..q}; do
  wget --continue "${BASE}a${p}"
done

echo "Extracting..."
if command -v unpigz >/dev/null 2>&1; then
  cat replica_v1_0.tar.gz.part?? | unpigz | tar -xvC "$SEM"
else
  cat replica_v1_0.tar.gz.part?? | tar -xzvC "$SEM"
fi

echo "Done. Semantic assets under $SEM/<scene>/habitat/{mesh_semantic.ply,info_semantic.json}"
echo "You may delete the .part?? files to reclaim ~100 GB once extraction succeeds."
