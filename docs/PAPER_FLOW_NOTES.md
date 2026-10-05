# Paper-flow notes — reshaping `main.tex` after the 2026-07-24 algorithm push

Working notes (not final prose) for how the paper's *narrative flow* should change given
this session's findings. Ordered by priority. Items marked **[PENDING]** depend on the
full-scene sim `--render-compare` IoU-F1 number (running); do not commit those until it lands.

See `STATUS.md` (2026-07-24 session) for the evidence behind each claim.

---

## 0. The one-line reframe of the whole story

Current spine: C1 registration supplies pose · C2 generation completes geometry ·
C3 real-robot · C4 front-end is the bottleneck. That still holds, but three things must
change so the paper is *honest and stronger*:

1. **Scale-fit is an OOD-only tool**, not a general win — stop implying it helps everywhere.
2. **Registration's value is placement, and it has a rotation caveat** — state it, and (if it
   wins) present **render-and-compare** as the method that fixes pose *and* scale from masks.
3. **The real-robot ceiling is calibration**, now *proven*, not asserted — lead the real
   section with that framing.

---

## 1. Scale — kill the "fit scale to depth" overclaim (do regardless of pending result)

Evidence: sim sweep shows SAM3D's own scale (err 0.217) beats every depth scale-fit; scale-fit
only helps OOD/real. The headline reproj method fits the in-plane extent from the *mask* at
median depth + the 3rd axis from SAM3D's aspect — it does **not** fit the depth *cloud*.

Edits:
- **C1 sim para (`~main.tex:407`)**: "so fitting it to depth is roughly neutral here" → the data
  says it's mildly-to-severely *negative* in-distribution. State: SAM3D's in-distribution scale
  is already accurate, so scale-fit is reserved for the OOD/real regime.
- **C1 setup (`~main.tex:369`)**: "additionally fitting scale to that [fused] depth" is wrong —
  scale is fit to the mask (frontal) at median depth. Reword to "recovering metric scale from
  the object's masked depth" (methods §258–268 already say this precisely).
- Everywhere else: keep the shorthand "depth scale-fit" only if defined once; the methods text
  already contains the correct definition. Ensure captions never imply cloud-fitting.
- **Consider demoting the sim `+scale+ICP` row** in `tab:sim` (it's strictly worse than `+ICP`);
  present `+ICP` as the asset representative and frame scale-fit as a real/OOD lever (C3).

## 2. Table 1 (`tab:sim`) — reframe why cluster is competitive (do regardless)

Evidence: all rows share the YOLOE(gt-vocab) front-end; cluster is *not* mask-privileged. Its
edge is a box-metric artifact — a box fit to observed points wins box-geometry metrics; the
asset wins the completeness-sensitive metric (IoU-F1) and everything in C2.

Edits:
- Add to the `tab:sim` caption: "All rows share the same YOLOE front-end (scene-vocabulary
  prompt), tracks, and depth; only the node payload / registration changes."
- Disambiguate **gt-vocab ≠ oracle GT masks** at first use (later tables use "oracle" for real
  GT masks — a reader must not blur them).
- 2–3 sentences: box metrics on observed points structurally favor "box the points"; the asset's
  value is completion (C2) and a sim-ready mesh, so we defer the payload verdict to C2/C3.

## 3. Front-end-is-the-bottleneck section — adopt the merged draft (do regardless)

The `[DRAFT]` block already in `main.tex` merges `tab:oracle`+`tab:detector` into one
single-setting table (s200) with the SAM3 single-scene footnote and a plainer 3-paragraph prose
(setup → result → detector comparison → registration aside). Decision needed: adopt it (replace
the two tables + prose, delete the draft banners, restore the section title) or keep two tables.
Recommendation: adopt — it's tighter and the footnote resolves the stride confound.

## 4. Rotation — state the caveat rigorously, using the paired test (do regardless)

Evidence: `paired_rotation_real.py` — +ICP worsens rotation (23/34 objects, p=0.046). The
per-scene rotation *median* in `tab:real` is additionally confounded by the IoU-matched set.

Edits:
- In C3, replace/augment the noisy per-scene rotation median with a **paired-rotation statement**
  (same objects, layout vs +ICP), mirroring the paired *scale* test. Honest and defensible.
- Frame it as motivation for the render-compare / gravity-aware back-end (below), not a defeat.

## 5. **[PENDING]** Render-and-compare as a new back-end contribution

IF the full-scene sim IoU-F1 beats ICP (0.545):
- **New methods subsection** "Placement by render-and-compare" (or fold into the back-end §):
  fit pose+per-axis scale so the mesh's rendered silhouette+depth matches the observed masks
  across kept views; motivated by masks(0.96) ≫ contaminated cloud(3×); L2 scale-reg toward the
  SAM3D prior so thin/occluded axes don't explode.
- **New result**: sim per-object 3D-IoU 0.37→0.62 median, scale_err halved (0.48→0.29), incl.
  fixing gross scale errors (dresser 2.5→0.16); + the scene-level IoU-F1 delta vs ICP.
- Position it as the method that delivers what C1 promised (metric pose+scale SAM3D can't) and
  supersedes the ICP+depth-scale back-end on clean poses.
- **Scope honestly**: fails on thin/complex objects (chairs, paintings); report the class split.

IF it does NOT win at scene level:
- Keep it as a rigorously-scoped negative/positive-per-class result + the calibration diagnosis
  (§6), which is still valuable. Do not overclaim.

## 6. Real-robot section — lead with the PROVEN calibration limit

Evidence: identical render-compare method improves silhouette IoU on real but *worsens* GT
3D-IoU (0.33→0.19) because camera poses are miscalibrated, while on sim (perfect poses) it
improves GT 3D-IoU (0.54→0.90). Only the poses differ.

Edits:
- Reframe the real-robot caveat from a hedge ("noisy, few matches, an extrinsic offset") into a
  **diagnosed, evidenced statement**: the back-end is calibration-limited; a mask-based method
  that is near-perfect in sim degrades on real *solely* due to pose error. Cite the sim/real
  overlay pair as a figure.
- This converts the weakest part of the paper into a controlled diagnosis — reviewers respect that.

## 7. Naming / hygiene (do regardless)

- `\texttt{object\_track}` is a code identifier never introduced to the reader (methods write
  "object track $k$"). Either define it once ("we call this pipeline object\_track") or replace
  with "our back-end / tracker" in reader-facing text and captions.
- Keep every results table regenerated from `run.json` (no hand-edits) — the render-compare and
  diagnostic numbers included.

---

## Sequencing suggestion
1. Land the do-regardless edits (§1–4, §7) — they're honesty fixes independent of any run.
2. When the sim scene number lands, resolve §5 (contribution vs scoped result).
3. Rewrite §6 real-robot around calibration with the overlay figure.
4. Only then write final prose from the `\todo`/POINTS-TO-HIT scaffolds.
