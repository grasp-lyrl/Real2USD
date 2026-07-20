#!/usr/bin/env bash
# Run the shape-completion experiment (gen_shape_completion.py) on all 10 val ids, then combine
# the per-scene CSVs into an n=10 coverage-binned aggregate (the generation-value figure at n=10).
# Asset meshes come from the shared queue via each scene's asset run scene_graph.json (mesh is
# registration-independent, so any variant's run works — we use scaleicp).
set +u
cd /home/chris.hsu/repos/Real2USD/humble_ws/src_Real2USD/r2s3d_core
export DISPLAY=${R2S3D_DISPLAY:-:0}
export XAUTHORITY=${R2S3D_XAUTHORITY:-/run/user/1003/gdm/Xauthority}
EX="--extra procthor --extra mesh --extra registration --extra detector"
IDS="137 200 428 434 534 569 573 683 771 912"

for id in $IDS; do
  run=results/paper/sim/asset_scaleicp_gt_s$id
  echo "########## shape-completion $id ##########"
  uv run $EX python scripts/gen_shape_completion.py --scene $id \
    --asset-run "$run" --tau 0.05 --min-extent 0.5 \
    2>&1 | grep -avE "Loading (train|val|test)|it/s\]|AI2-THOR WARNING|pip install|prior.load_dataset|revision=" | tail -6 \
    || echo "[$id] SHAPECOMP FAIL"
done

echo "########## COMBINING n=10 ##########"
uv run python - <<'PY'
import csv, glob, numpy as np
rows=[]
for f in sorted(glob.glob("results/paper/_tables/shape_completion_s*.csv")):
    for r in csv.DictReader(open(f)):
        try:
            rows.append({k:(float(r[k]) if k in ("coverage","asset_unobs_recon","cluster_unobs_recon","unobs_frac") and r[k] not in ("","nan") else r[k]) for k in r})
        except Exception: pass
rows=[r for r in rows if isinstance(r.get("asset_unobs_recon"),float)]
print(f"n_objects total = {len(rows)} across {len(glob.glob('results/paper/_tables/shape_completion_s*.csv'))} scenes")
bins=[(0,.3,"low <0.3"),(.3,.6,"mid .3-.6"),(.6,1.01,"high >0.6")]
print(f"{'bin':12s}{'n':>4}{'unobs%':>9}{'asset':>9}{'cluster':>9}{'asset_wins':>12}")
tot_win=0
for lo,hi,name in bins:
    sel=[r for r in rows if lo<=r['coverage']<hi]
    if not sel: print(f"{name:12s}{0:>4}"); continue
    uf=np.mean([r['unobs_frac'] for r in sel]); a=np.mean([r['asset_unobs_recon'] for r in sel])
    c=np.mean([r['cluster_unobs_recon'] for r in sel]); w=sum(r['asset_unobs_recon']<r['cluster_unobs_recon'] for r in sel)
    tot_win+=w
    print(f"{name:12s}{len(sel):>4}{uf*100:>8.0f}%{a:>9.3f}{c:>9.3f}{f'{w}/{len(sel)}':>12}")
print(f"asset reconstructs unobserved surface better in {tot_win}/{len(rows)} objects (n=10 scenes)")
PY
echo "########## SHAPECOMP VAL10 DONE ##########"
