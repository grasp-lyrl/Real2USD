# Evaluation Improvement Plan: Telling the Registration + Retrieval Story

This plan is informed by the evaluation design in **REASONINGGRAPH** (Relationship-Aware Hierarchical 3D Scene Graph for Task Reasoning, arxiv 2602.02456) and by your current `evaluations/` setup. The goal is to make your experiments clearly support two claims:

1. **Registration improves fine-grained localization of objects** (centroid/IoU alignment with GT).
2. **Retrieval helps discard bad object generations** by selecting better labels from the database (fewer wrong-label FPs, higher precision).

---

## Why REASONINGGRAPH’s Evaluations Work

- **One table per claim**: Retrieval quality → Table I (Acc@k, AUC); task reasoning → Table II (SR%, FP); robustness → Table III (accuracy, F1 over 100 runs).
- **Baselines on the same benchmarks**: Same datasets (HM3DSem, Replica), same metrics, filtered comparably.
- **Controlled ablations**: e.g. VL2 vs InstructBLIP; with vs without relations.
- **Clear metrics**: Each number maps to a claim (e.g. “retrieval is better”, “task success is high”).
- **Multiple angles**: Standard benchmarks + real-world tasks + runtime + robustness.

Your evaluations already have the right *ingredients* (registration_helped_rate, retrieval_swapped_tp_rate, pipeline vs sam3d_only vs CLIO). The gap is **structure**: one main table and a few focused plots that directly support the two stories above.

---

## Current State (Brief)

- **Metrics**: F1, precision, recall, mean IoU 3D, per-class, range (near/mid/far), FP breakdown (wrong_label, low_iou, no_overlap, unresolved).
- **Diagnostics** (in `collect_ablation_table` / `plot_eval`): `registration_helped_rate`, `registration_hurt_rate`, `registration_delta_centroid_m`, `retrieval_swapped_rate`, `retrieval_swapped_tp_rate` — read from `by_run` JSON `diagnostics.registration_summary` / `diagnostics.retrieval_summary`.
- **Issue**: `run_eval.py` is removed; the code that *produces* these diagnostics (and writes `by_run` JSON with `instances.predictions` / `instances.gts`) may be missing or elsewhere. That must be in place for the plan below to work.

---

## Plan

### 1. Ensure Diagnostics Are Produced and Stored

**Registration summary** (for “registration improves localization”):

- For each matched prediction, you need (conceptually) “pose with registration” vs “pose without registration”. If “without registration” is represented by **SAM3D-only** and “with” by **pipeline_full**, then:
  - Either compute registration diagnostics inside the eval that compares pipeline vs SAM3D-only (e.g. same GT, same matching; compare centroid error or IoU), or
  - Use existing pipeline outputs (e.g. `transform_odom_from_raw` or diagnostics from `registration_metrics.jsonl` in real2sam3d) to get per-object “before vs after registration” and compute:
    - **helped_rate**: fraction of objects where registration reduced centroid error to GT.
    - **hurt_rate**: fraction where it increased.
    - **mean_centroid_error_delta_m**: mean change in centroid error (negative = improvement).
- Write these into `by_run` JSON under `diagnostics.registration_summary` so `collect_ablation_table` and plots get them.

**Retrieval summary** (for “retrieval picks better objects from DB”):

- From matched pairs, identify predictions that used a **retrieved** label (e.g. `retrieved_label_raw` present and used as final label).
- **swapped_rate**: fraction of predictions that used retrieval.
- **swapped_tp_rate**: among those, fraction that are true positives. High value supports “retrieval picks better objects”.
- Write under `diagnostics.retrieval_summary`.

**Action**: Restore or reimplement the eval entrypoint (e.g. `run_eval.py`) that (a) runs matching, (b) computes registration_summary and retrieval_summary from predictions/GT and (optionally) from pipeline/registration outputs, (c) writes `by_run` JSON including `instances.predictions`, `instances.gts`, and `diagnostics`.

---

### 2. Registration Story: “Registration Improves Fine-Grained Localization”

**Main table (paper-ready):**

| Method        | Mean centroid error (m) ↓ | Mean IoU 3D ↑ | F1 ↑ | Recall ↑ |
|---------------|---------------------------|---------------|------|----------|
| SAM3D-only    | …                         | …             | …    | …        |
| Pipeline (full) | …                       | …             | …    | …        |

- Same scenes, same GT, same matching config (IoU threshold, label match, etc.).
- **Story**: Pipeline (with registration) has lower centroid error and higher IoU/F1 than SAM3D-only.

**Optional breakdown:**

- Same table by **range** (near / mid / far) to show registration helps especially where odometry drift is larger (e.g. mid/far).
- If you have per-object registration deltas: a small table or histogram of “mean centroid error delta (m)” (e.g. pipeline − SAM3D-only) or “registration helped rate” by scene.

**Plots:**

- Bar chart: “Mean centroid error (m)” or “Mean IoU 3D” for SAM3D-only vs Pipeline (and optionally CLIO).
- If you have range metrics: bar chart of centroid error or IoU by range bucket for the two methods.

**Implementation:**

- In `compare_three_methods.py` (or equivalent), ensure “Mean centroid error” is computed from match details (e.g. from `metrics_3d` / match output) and included in the summary CSV and in any script that generates the “main story” table.
- Add a small script or section in `plot_eval.py` that outputs the **registration story table** (and optional range breakdown) from the same `by_run` or comparison CSVs.

---

### 3. Retrieval Story: “Retrieval Discards Bad Generations by Picking Better Objects from DB”

**Main table (paper-ready):**

| Setting              | Precision ↑ | F1 ↑ | FP (total) ↓ | FP (wrong label) ↓ |
|----------------------|-------------|------|--------------|---------------------|
| Raw labels only      | …           | …    | …            | …                   |
| Prefer retrieved     | …           | …    | …            | …                   |

- Same pipeline, same scenes/GT; only change is label source (e.g. `label_source`: raw only vs `prefer_retrieved_else_label` in `eval_config.json`).
- **Story**: With retrieval, precision and F1 go up and FP (especially wrong_label) go down.

**Supporting metric:**

- **Swapped TP rate**: among predictions that used a retrieved label, fraction that are TP. Report in table or in text (e.g. “X% of retrieval swaps are true positives”), to show retrieval is not random but selects better labels.

**Optional:**

- Per-class: where does retrieval fix the most wrong_label FPs? (e.g. small table or bar chart of “FP wrong_label” with vs without retrieval by class.)

**Implementation:**

- Run eval twice per scene (or in sweep): once with `label_source` = raw only, once with `prefer_retrieved_else_label`; same `by_run` structure so you can aggregate.
- Add a script or `plot_eval` section that builds the **retrieval story table** from these runs (and optionally per-class wrong_label comparison).
- Ensure `retrieval_swapped_tp_rate` and `retrieval_swapped_rate` from diagnostics are in the CSV so you can report them in the table or caption.

---

### 4. One Unified “Main Story” Table (Optional but Strong)

Single table that shows the additive effect of each component:

| Method        | F1 ↑ | Precision ↑ | Recall ↑ | Mean IoU 3D ↑ | Mean centroid err (m) ↓ | FP ↓ |
|---------------|------|--------------|----------|---------------|--------------------------|------|
| SAM3D-only    | …    | …            | …        | …             | …                        | …    |
| + Registration (pipeline, raw labels) | … | …            | …        | …             | …                        | …    |
| + Retrieval (full pipeline) | …    | …            | …        | …             | …                        | …    |
| CLIO (baseline) | …   | …            | …        | …             | …                        | …    |

- **Story**: Adding registration improves localization (IoU, centroid error, recall); adding retrieval improves precision and reduces FP. CLIO is your external baseline.

**Implementation:**

- Ensure you have (or can run) variants: SAM3D-only, pipeline with raw-only labels, full pipeline (registration + retrieval), CLIO. Same GT and matching for all.
- Single script that reads the same comparison/ablation CSVs and outputs this table (e.g. Markdown/LaTeX) plus the same numbers for plots.

---

### 5. Plots That Match the Story

- **Registration**: Bar plot “Mean centroid error (m)” or “Mean IoU 3D” — SAM3D-only vs Pipeline (and optionally CLIO). Optionally by range.
- **Retrieval**: Bar plot “Precision” and “F1” — raw-only vs prefer-retrieved; or “FP (wrong label)” with vs without retrieval.
- **Existing**: Keep “Registration vs Retrieval diagnostics” scatter (registration_helped_rate vs retrieval_swapped_tp_rate) but add a short caption: e.g. “Registration helped rate = % of objects where registration reduced centroid error; Retrieval swap TP rate = % of retrieval swaps that are true positives.”

---

### 6. Baselines and Reproducibility

- Keep **CLIO** and **SAM3D-only** as baselines; same scenes, same GT, same matching (IoU threshold, frame normalization, etc.). Document in README or `eval_config.json` so reviewers see a fair comparison.
- If you have multiple runs per scene: report **mean ± std** for F1, centroid error, precision (like REASONINGGRAPH Table II) to show stability.

---

### 7. Suggested Order of Implementation

1. **Restore or reimplement** the eval that writes `by_run` JSON with `diagnostics.registration_summary` and `diagnostics.retrieval_summary` (and `instances`).
2. **Registration story**: Add centroid error (and optionally range) to comparison output; generate “Registration story” table and bar plot (SAM3D-only vs Pipeline).
3. **Retrieval story**: Add “raw only” vs “prefer retrieved” runs; generate “Retrieval story” table and bar plot; report swapped_tp_rate.
4. **Unified table**: Script that outputs the single main table (SAM3D-only → +Registration → +Retrieval → CLIO).
5. **Caption and README**: Short description of what each metric means and how it supports the two claims.

---

## Axis-aligned visualization (math note)

Applying the **same** rigid rotation (e.g. dominant GT yaw) to GT, point cloud, and predicted boxes is **mathematically correct**: all relative geometry (IoU, centroid distances, containment) is preserved. The evaluation pipeline already applies this rotation when loading preds and GT (see `eval_common._yaw_normalization_matrix` and `frame_normalization: gt_global_yaw` in config). So:

- **By_run instances** are already in the axis-aligned frame; no extra rotation is needed for bbox-vs-GT plots.
- **Point cloud** from the run (e.g. `run_dir/diagnostics/pointclouds/lidar/*.npy`) is in the original odom frame; the overlay script applies the same yaw rotation to the PC so it aligns with the boxes. You do **not** need the experiment to be axis-aligned at capture time; alignment is done at eval/visualization time.

---

## Summary

- **Registration**: Compare SAM3D-only vs Pipeline on **localization** (centroid error, IoU, F1); one clear table + plot.
- **Retrieval**: Compare raw-only vs prefer-retrieved on **precision/FP** (and swapped_tp_rate); one clear table + plot.
- **Unified**: One table showing step-wise improvement (SAM3D → +Reg → +Retrieval) and CLIO baseline.
- Ensure diagnostics are actually computed and written into `by_run` so existing `collect_ablation_table` and `plot_eval` pipelines can surface them, and add minimal new scripts/plots to turn these into the story tables and figures above.
