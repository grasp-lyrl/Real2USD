# Evaluation metrics

## Classes

Evaluation classes are the **canonical labels** from `canonical_labels.json`: **seat**, **table**, **door**. Predictions and GT are resolved to these via aliases and (optionally) embedding. **"Unknown" and unlabeled objects are not included as a class**: they are excluded from per-class tables, macro-averages, and open-set tasks. So reported classes are only the real canonical ones.

## Matching and IoU

- **iou_mode** (in `eval_config.json` → `matching`): `"obb"` or `"aabb"`.
  - **obb**: Matching uses **3D oriented bounding box** IoU. Boxes are normalized so the axis most aligned with world Z is the third (so `obb_dimensions` = [dx, dy, dz] with dx, dy = horizontal footprint, dz = height). Overlays plot the XY footprint using these orientations.
  - **aabb**: Matching uses axis-aligned bounding box IoU (legacy).
- **iou_threshold**: Minimum IoU for a (pred, gt) pair to count as a match (e.g. 0.1).

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

Standard P/R use one-to-one IoU matching and do not capture open-set behavior. **Open-set Recall (osR)** and **Open-set Precision (osP)**:

- **osR**: For each GT class (task), take the **n best** predictions for that task (n = GT count for that class), ranked by similarity to the task. osR = (correct detections in those top-n, over all tasks) / (total GT). Correct is **strict** (both centroids contained) or **relaxed** (at least one). **osR_strict**, **osR_relaxed**.
- **osP**: Only predictions with **similarity to some task ≥ threshold** (e.g. 0.9) count. osP = (correct among those) / (number of such predictions). **osP_strict**, **osP_relaxed**.
- **F1_os** = harmonic mean of osR_relaxed and osP_relaxed.

With label-only similarity (1 if pred label == task else 0), “relevant” for osP = predictions whose label is one of the GT classes. Config: `matching.compute_open_set_metrics`, `matching.os_similarity_threshold` (default 0.9). Written to `benchmark_compat` when enabled.
