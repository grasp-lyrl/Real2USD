# Evaluation metrics

## How false positives (FP) are measured

- **FP = any prediction that is not matched to any GT** in the one-to-one greedy IoU matching.
- **Matching** (`metrics_3d.greedy_match`): All (pred, gt) pairs with IoU ≥ `iou_threshold` (e.g. 0.1) are candidates; optionally `require_label_match` forces same `label_canonical`. Candidates are sorted by IoU descending; assignments are made greedily (each pred and each GT used at most once). Whatever predictions remain unassigned are **unmatched** → counted as FP.

So CLIO (or any method) can get many FPs even when predictions “look good” because:

1. **No GT for that object** — GT may not contain every object in the scene (e.g. only chairs/tables/doors). A correct detection with no corresponding GT is FP.
2. **IoU below threshold** — Box is on the right object but 3D IoU &lt; 0.1 (e.g. OBB rotation/size/position off) → no match → FP. With `iou_mode: "obb"`, orientation errors quickly reduce IoU.
3. **Duplicate detections** — Two predictions on the same GT: only one can match; the other is FP.
4. **Wrong or unresolved label** — If `require_label_match` is true, pred and GT must share the same canonical label; wrong or null label → no match → FP. Unresolved labels are also counted under **fp_breakdown.unresolved_label**.

**fp_breakdown** (in the run JSON) splits FPs into: `wrong_label`, `low_iou` (best IoU &gt; 0 but &lt; threshold), `no_overlap` (best IoU = 0), `unresolved_label` (no canonical label). Use this to see whether FPs are mostly geometry (low_iou / no_overlap) vs labels (wrong_label / unresolved_label).

## Classes

Evaluation classes are the **canonical labels** from `canonical_labels.json`: **seat**, **table**, **door**. Predictions and GT are resolved to these via aliases and (optionally) embedding. **"Unknown" and unlabeled objects are not included as a class**: they are excluded from per-class tables, macro-averages, and open-set tasks. So reported classes are only the real canonical ones.

## Matching and IoU

- **iou_mode** (in `eval_config.json` → `matching`): `"obb"` or `"aabb"`.
  - **obb**: Matching uses **3D oriented bounding box** IoU. Boxes are normalized so the axis most aligned with world Z is the third (so `obb_dimensions` = [dx, dy, dz] with dx, dy = horizontal footprint, dz = height). Overlays plot the XY footprint using these orientations.
  - **aabb**: Matching uses axis-aligned bounding box IoU (legacy).
- **iou_threshold**: Minimum IoU for a (pred, gt) pair to count as a match (default **0.05** in config; was 0.1). With **obb** mode, 3D IoU is often much lower than 2D top-down overlap: rotation or height errors shrink the intersection volume. If the overlay shows red predictions that clearly overlap orange GT (e.g. seats on the left), they are unmatched because their **3D OBB IoU** is below the threshold. Lowering `iou_threshold` (e.g. to 0.05) allows more of those to match; re-run the evaluator and overlay to refresh.

## Localization metrics (strict / relaxed)

These are **general** metrics: they are computed **over all matched (pred, gt) pairs** in a run, not per class.

| Metric | Definition |
|--------|------------|
| **Strict accuracy** | Among matched pairs: fraction where **both** centroids are contained (pred center inside GT box **and** GT center inside pred box). Measures tight localization. |
| **Relaxed accuracy** | Among matched pairs: fraction where **at least one** centroid is contained (pred in GT **or** GT in pred). |
| **Strict precision** | Fraction of **all predictions** that are matched and strict (strict matches / num preds). |
| **Relaxed precision** | Fraction of all predictions that are matched and relaxed. |
| **mean_iou_3d** | Mean 3D IoU over matched pairs. |
| **mean_centroid_error_m** | Mean Euclidean distance (m) between predicted and GT box centers over matched pairs. |
| **f1_relaxed** | Harmonic combination of relaxed_accuracy and mean_iou_3d: `2 * (relaxed_accuracy * mean_iou_3d) / (relaxed_accuracy + mean_iou_3d)`. |

Strict/relaxed are computed in `metrics_3d.strict_relaxed_from_aabb` using AABB containment of centroids. They are written into each run’s `benchmark_compat` and are included in the **comparison summary CSV** and the **localization plot** (`*_localization_metrics.png`) when you run `compare_three_methods.py`.

## Per-class localization and macro-average

- **Per class**: Each class (label) has its own **strict_accuracy**, **relaxed_accuracy**, **mean_iou_3d**, **mean_centroid_error_m** over that class’s matched pairs only. These appear in `per_class[<label>]` and in the per-class CSV.
- **Macro-average over classes**: `mean_strict_accuracy_per_class`, `mean_relaxed_accuracy_per_class`, `mean_iou_3d_per_class`, `mean_centroid_error_m_per_class` are the **average of those per-class values** over all classes that have at least one match. So each class counts once regardless of how many instances it has. They are in `benchmark_compat` and in the summary CSV; `compare_three_methods.py` also produces `*_localization_per_class.png` for these.

## Per-class vs general

| Scope | Metrics |
|-------|--------|
| **General (over all matches or all preds)** | strict_accuracy, relaxed_accuracy, strict_precision, relaxed_precision, f1_relaxed, mean_iou_3d, mean_centroid_error_m, precision, recall, f1 |
| **Macro over classes** | mean_strict_accuracy_per_class, mean_relaxed_accuracy_per_class, mean_iou_3d_per_class, mean_centroid_error_m_per_class |
| **Per class** | tp, fp, fn, precision, recall, f1, mean_iou_3d, **strict_accuracy**, **relaxed_accuracy**, **mean_centroid_error_m** (in `per_class` and in the per_class CSV/plot). |

## Open-set metrics (osR, osP, F1_os)

Standard P/R use one-to-one IoU matching and do not capture open-set behavior. **Open-set Recall (osR)** and **Open-set Precision (osP)** use **task = GT class** and a **binary “similarity”**: 1 if the prediction’s `label_canonical` equals that task label, else 0. There is **no continuous similarity score** (e.g. no embedding or cosine similarity).

- **osR**: For each GT class (task), take the **n best** predictions that **match that task label** (n = GT count for that class), ranked by **IoU** to the task’s GTs. osR = (correct detections in those top-n, over all tasks) / (total GT). Correct is **strict** (both centroids contained) or **relaxed** (at least one). **osR_strict**, **osR_relaxed**.
- **osP**: **“Relevant” predictions** = those whose `label_canonical` is **one of the GT class labels** (i.e. pred label equals some task). We do **not** use a continuous similarity; the code compares `_sim_pred_task(pred, task) >= similarity_threshold`, but `_sim_pred_task` returns only **1.0 or 0.0** (exact label match). So with default threshold 0.9, “relevant” = predictions that have the same label as at least one GT class. osP = (among those relevant preds, how many are correct—matched to a GT with IoU ≥ threshold and strict/relaxed centroid containment) / (number of relevant preds). **osP_strict**, **osP_relaxed**.
- **F1_os** = harmonic mean of osR_relaxed and osP_relaxed (same as F1_os_relaxed).
- **F1_os_strict** = harmonic mean of osR_strict and osP_strict.
- **F1_os_relaxed** = harmonic mean of osR_relaxed and osP_relaxed.

Config: `matching.compute_open_set_metrics`, `matching.os_similarity_threshold` (default 0.9; with binary 0/1 similarity, only preds with similarity 1.0 pass, i.e. exact label match to a task). Written to `benchmark_compat` when enabled.

### Paper-style formulation (for methods / appendix)

We define open-set recall and precision over a set of predictions \(\mathcal{P}\) and ground-truth objects \(\mathcal{G}\). Each prediction \(p \in \mathcal{P}\) and each GT \(g \in \mathcal{G}\) has a class label \(\ell(p), \ell(g)\) from a vocabulary \(\mathcal{L}\); we treat only labels that appear in \(\mathcal{G}\) (excluding unknown/unlabeled) as *tasks* \(\mathcal{T} \subseteq \mathcal{L}\). A prediction \(p\) is *relevant* for a task \(c \in \mathcal{T}\) iff \(\ell(p) = c\). We use 3D IoU (AABB or OBB) with threshold \(\tau_{\text{iou}}\) and two localization criteria: *strict* (both centroids lie inside the other box’s AABB) and *relaxed* (at least one centroid inside the other’s AABB). A pair \((p, g)\) is *correct* (strict or relaxed) if \(\text{IoU}(p, g) \geq \tau_{\text{iou}}\) and the corresponding centroid-containment condition holds.

**Open-set Recall (osR)** measures the fraction of ground-truth objects that receive a correct detection when we assign predictions *per task*. For each task \(c \in \mathcal{T}\), let \(\mathcal{G}_c = \{ g \in \mathcal{G} : \ell(g) = c \}\) and \(n_c = |\mathcal{G}_c|\). Consider the set of predictions with label \(c\), ranked by their maximum IoU to any \(g \in \mathcal{G}_c\); take the top \(n_c\) such predictions and greedily assign them to the \(n_c\) GTs in \(\mathcal{G}_c\) by decreasing IoU (each GT at most once). Count how many of these assigned pairs are correct (strict or relaxed). **Open-set recall** is the total number of GTs that receive a correct assignment in this procedure, summed over all tasks, divided by \(|\mathcal{G}|\):
\[
\text{osR} = \frac{1}{|\mathcal{G}|} \sum_{c \in \mathcal{T}} \bigl| \bigl\{ g \in \mathcal{G}_c : \text{assigned } (p,g) \text{ is correct} \bigr\} \bigr|.
\]
We report \(\text{osR}_{\text{strict}}\) and \(\text{osR}_{\text{relaxed}}\) depending on the correctness criterion.

**Open-set Precision (osP)** measures the fraction of *task-relevant* predictions that are correct. Let \(\mathcal{P}_{\text{rel}} = \{ p \in \mathcal{P} : \ell(p) \in \mathcal{T} \}\) be the set of predictions whose label is one of the GT tasks. For each \(p \in \mathcal{P}_{\text{rel}}\), find the GT \(g\) that maximizes \(\text{IoU}(p, g)\); if \(\text{IoU}(p, g) \geq \tau_{\text{iou}}\) and the (strict or relaxed) centroid condition holds, \(p\) is counted as correct. **Open-set precision** is the number of correct predictions in \(\mathcal{P}_{\text{rel}}\) divided by \(|\mathcal{P}_{\text{rel}}|\):
\[
\text{osP} = \frac{ \bigl| \bigl\{ p \in \mathcal{P}_{\text{rel}} : \exists g \in \mathcal{G},\, \text{IoU}(p,g) \geq \tau_{\text{iou}} \wedge \text{correct}(p,g) \bigr\} \bigr| }{ |\mathcal{P}_{\text{rel}}| }.
\]
We report \(\text{osP}_{\text{strict}}\) and \(\text{osP}_{\text{relaxed}}\). The open-set F1 is the harmonic mean of osR and osP (reported separately for strict and relaxed).

### Short formulation (methods / appendix)

**Setup.** Tasks = GT class labels (unknown/unlabeled excluded). A prediction is *relevant* if its label equals some task. A (pred, GT) pair is *correct* if IoU ≥ τ and (strict: both centroids inside the other’s AABB; relaxed: at least one).

- **osR (open-set recall):** For each task, take the top-n predictions with that label (n = number of GTs of that class), ranked by IoU to that task’s GTs; greedily assign them to those GTs. osR = (number of GTs that get a correct assignment) / (total GT). Reported as osR_strict and osR_relaxed.
- **osP (open-set precision):** Among predictions whose label is a task, count how many have IoU ≥ τ with some GT and satisfy the centroid condition. osP = (that count) / (number of such predictions). Reported as osP_strict and osP_relaxed.
- **F1_os** = harmonic mean of osR and osP (strict and relaxed variants).
