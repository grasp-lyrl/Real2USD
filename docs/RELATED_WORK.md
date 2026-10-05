# RELATED WORK — techniques to borrow, positioning to sharpen

Reading notes on directly-comparable systems. Purpose is **not** a lit review for the
paper — it is (a) what each system actually does, (b) *why it got published* (what the
reviewers rewarded), and (c) concrete, liftable ideas to improve our system and out-score
theirs. Positioning strategy lives in `REWORK_PLAN.md`; interfaces in `PHASE_SPECS.md`;
this is the "what the neighbors did and what we steal" doc.

> We are in the same genre as these systems — an object-centric **metric-semantic map**
> built online from a moving open-vocab sensor (we emit `scene_graph.json`). Our
> differentiator is the **node payload**: each object is backed by a retrieved/generated
> **3D mesh** (Sim(3)-registered, exported USD/GLB), not a labelled point cluster/bbox.
> Treat these papers as in-family baselines, not a different field.

---

## SuperMap (RSS 2026) — CMU AirLab

- **Title:** *SuperMap: A Spatio-Temporal SLAM System for Visual-Language Navigation*
- **Authors:** Shibo Zhao, Guofei Chen, Honghao Zhu, Zhiheng Li, Changwei Yao, Nader
  Zantout, Seungchan Kim, Wenshan Wang, Ji Zhang, Sebastian Scherer
- **Venue:** Robotics: Science and Systems (RSS) 2026
- **Project page:** https://superodometry.com/supermap
- **Abstract / program:** https://roboticsconference.org/program/papers/52/
- **Paper PDF:** https://www.roboticsproceedings.org/rss22/p052.pdf
- **Code (linked, not yet public as of 2026-07-14):** https://github.com/gfchen01/semantic_mapping

### What it is
A **4D (spatio-temporal) semantic-SLAM + scene-graph** system for language-guided
navigation. High-frequency LiDAR-visual geometric SLAM fused with *asynchronous*
open-vocab perception; maintains persistent instance IDs and prunes stale/moved objects,
producing a queryable 4D scene graph that a VLM reasons over. Hardware: Livox Mid-360 +
360° panoramic camera on a mecanum base.

### What they do about detections + clustering (the core of the method)
Detection stack is off-the-shelf: **GroundingDINO** (open-vocab 2D detection) + **SAM2**
(masks), per frame. The contribution is everything *after* detection:

1. **Tracking-by-detection, explicitly NOT point-feature clustering.** Their headline
   argument. Baselines (HOV-SG, ConceptGraphs, Clio) do offline 3D reconstruction → run
   SAM+CLIP over the *full point cloud* → cluster/over-segment 3D points into instances.
   SuperMap rejects this; their instances come from **online 2D-detection tracking anchored
   to 3D**, never from clustering raw point features. Quote: the gain "demonstrates the
   efficacy of our 3D-aware tracking-by-detection approach over methods relying on
   point-feature clustering or over-segmented geometries."

2. **3D-to-2D motion-compensated association (the clever bit).** Standard 2D trackers
   (ByteTrack) break under fast robot ego-motion. Instead of a linear 2D motion model, the
   Kalman prior for a tracklet's 2D centroid is obtained by **projecting the instance's 3D
   centroid from the map through the current SLAM pose** (`ĉ = π(K · Pₜ⁻¹ · X)`). Data
   association then assigns detections to IDs via 3D spatial consistency. This is what keeps
   identities stable through occlusion and aggressive maneuvers. Tracklet state =
   `[2D centroid, bbox w/h, image-plane velocity]` (R⁶/R⁸).

3. **Probabilistic geometric consistency (log-odds occupancy).** Per map point, compare
   projected depth vs. sensor depth (residual Δd) → classify Observable / behind-surface /
   "Disappeared (in front of surface)". Moved/transient points are penalized and pruned.
   Doubles as the **change-detection** mechanism and a stale-geometry filter.

4. **Bayesian semantic fusion.** Each instance holds a categorical label distribution,
   updated per frame with the detector's **confusion matrix** `P(z | L=c)`. A wrong
   single-frame label can't corrupt the instance — "suppresses transient misclassifications."
   Plus an existence-and-label confidence update for stable IDs.

5. **Clustering they *do* keep is narrow:** only object-level **centroid-distance
   clustering** to build scene-graph spatial edges (on/beside/under via class-dependent
   geometric predicates). Instance identity itself never comes from clustering.

### What they evaluate (metrics + datasets)
Segmentation + temporal consistency only — **no mesh / 3D-reconstruction metrics**.

| Task | Metrics | Dataset |
|---|---|---|
| Class-level semantic seg | mIoU %, f-mIoU %, Accuracy % | ScanNet |
| Instance-level seg | mAP50 / mAP25 per class (Chair, Window, Refrigerator, Sofa, Door) | ScanNet |
| Spatio-temporal change detection | Recall (appearance / disappearance) | self-collected 10-min dynamic scene |
| System | ablations + runtime | — |

Instance mAP50 on ScanNet (their Table III) — the clustering baselines collapse:

| Method | Chair | Window | Refrigerator | Sofa | Door |
|---|---|---|---|---|---|
| HOV-SG | 4.6 | 0.0 | 0.0 | 30.0 | 9.7 |
| ConceptGraphs | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **SuperMap** | **63.8** | **42.2** | **62.5** | **33.4** | **10.0** |

Baselines: HOV-SG, ConceptGraphs (instance seg); OpenMask3D, SeeGround, Khronos discussed.

### Why this got into RSS (what the reviewers rewarded)
Reading it as "how do I clear the same bar," not "why is this better than us":

- **One crisp, defensible claim with a decisive table.** "Tracking-by-detection beats
  point-feature clustering for online instance mapping" — and Table III shows baselines at
  ~0 vs their 40–60. A single, legible, large effect size on a *public* benchmark. That is
  the whole paper's spine.
- **A capability axis nobody else reports: 4D / change detection.** They *created their own
  benchmark* (appearance/disappearance recall) for a thing prior work can't do. New axis +
  new benchmark = "novel," even if each component (GroundingDINO, SAM2, log-odds, Kalman) is
  standard. **Novelty was in the *composition and the evaluation axis*, not the parts.**
- **Real-time on a real robot**, with runtime profiling and ablations. RSS rewards deployed
  systems; the 10-minute real-world dynamic-scene demo is the money shot.
- **Open-source + released benchmark.** "Deployable baseline for the community" framing.
- **Honest, narrow scope.** They don't over-claim geometry/reconstruction — they own the
  identity-consistency lane and defend exactly it.

**Takeaway for our resubmission (IROS reject → rework):** our v1 was dinged for a strawman
baseline (SAM3D single-image at scene scale), no ConceptGraphs/HOV-SG, no ablations, single
runs. SuperMap is the template for the fix — *one* headline claim, decisive table on public
data, a capability axis we own, runtime honesty, code release. We already have the axis
nobody reports (**per-node reconstructed asset / mesh quality**); we need SuperMap's
discipline in *presenting* it.

### Ideas to lift (and how to beat them)
Concrete, mapped to our code/phases:

1. **3D-to-2D motion-compensated association → ObjectTrack.** Directly attacks our known
   failure mode (render jitter → unstable `track_ids` → misbound SAM3D meshes; see
   `[[procthor-render-cache]]`, `[[phase2-detector-in-sim]]`). Seed next-frame association by
   projecting the fused 3D centroid through the current pose instead of trusting 2D IoU. Cheap
   to add, citable, and squarely on the perception-robustness crux that's our NEXT direction
   (`STATUS.md §NEXT`, `[[perception-robustness-crux]]`).
2. **Bayesian label fusion with a confusion matrix → ObjectTrack labels.** Replace ad-hoc
   per-track label voting with a categorical posterior updated by the detector's confusion
   matrix. Makes labels robust to transient misclassification and gives us a principled
   confidence to stamp on outputs (matches the "make fallbacks loud / stamp code path"
   directive in `CLAUDE.md`).
3. **We are already on the winning side of their central argument.** Real2USD is
   tracking/object-centric, not global-point-cloud clustering — so SuperMap is *independent
   validation* that our architecture beats the clustering line on instance mAP50. Use their
   HOV-SG/ConceptGraphs numbers as the reference bar, and report **instance mAP50 on ScanNet**
   so we sit on the same leaderboard (see the ScanNet plan below).
4. **Match them on perception, then win on the axis they skip.** They report seg/ID metrics
   only. If we (a) match on instance mAP50 + open-set P/R and (b) add **Scan2CAD 3D-IoU /
   rotation-scale alignment** and mesh quality, we cover their table *and* a row they can't —
   the asset-centric payoff. That's the "just as good + adds something different" pitch made
   concrete.
5. **Do NOT chase their 4D/change-detection lane** for this resubmission — different thesis
   (dynamic/temporal), out of our static single-session scope. Cite it as complementary.

### Consequence for datasets — add ScanNet
SuperMap makes the case to add **ScanNet** concrete, with two distinct value-adds (keep them
separate in the write-up):
- **ScanNet(++) instance mAP50 + open-set P/R** → direct comparability with SuperMap and the
  scene-graph line (their perception table).
- **Scan2CAD 3D-IoU + rotation/scale error** → the asset/mesh-alignment row that is ours
  alone. Already the planned ScanNet-derived benchmark in `PHASE_SPECS.md` (after Replica).

---

<!-- Add future in-family systems below in the same shape:
     What it is · Method (detections/clustering) · Metrics+datasets · Why it published ·
     Ideas to lift & how to beat · Consequence for us. Keep all source links. -->
