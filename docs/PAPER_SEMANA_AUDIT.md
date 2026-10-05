# SeMaNa paper audit — `paper/main.tex` (2026-07-27)

> **Resolution status (same day):** A1 RESOLVED — the val-10 silhouette runs were found
> in another session's /tmp scratchpad and transferred to
> `results/paper/sim/asset_silscaleicp_gt_s*`; 4 partial scenes re-running. A2–A7 and
> §B fixed in `paper/mainv2.tex`. Per Chris: SAM3 row = ungated; centroid-floor wording
> stays on odometry drift (quick mention only); nav figures stay stacked; Overleaf is
> the compile target (repo tex is the source that gets copied over).

Pre-rewrite analysis pass: wrong/unverifiable claims, mechanical bugs, unclear text,
and the shortening plan (4¾ pp → 4¼ pp before layout tuning; CFP hard cap is 4 pp
excl. refs, **deadline 2026-07-29**). Every number below was checked against
`results/paper/_tables/*.csv` and the per-run `run.json` files. Companion docs:
`PAPER_FLOW_NOTES.md` (narrative), `WORKSHOP_PAPER_PLAN.md` (claims + guardrails).

---

## A. BLOCKING — wrong or unverifiable claims

### A1. The `tab:sim` `+scale+ICP` row has no provenance (headline numbers, in the abstract)
The row reads **0.45 / 0.20 / 0.13 / 5.9° / 0.27**. The only val-10 scale+ICP run on
disk (`asset_scaleicp_gt_s*`, reproj scale source) gives **0.449 / 0.190 / 0.043 /
6.3° / 0.335** — S2C and scale flatly disagree. No `run.json` anywhere in the repo
(or `~/Data`) is newer than 2026-07-23, so the silhouette-scale val-10 numbers were
never written to disk; STATUS's own NEXT list still says "(1) confirm across val-10
with the ICP+silhouette-scale config". The unverified values are quoted in the
**abstract** (scale 0.32→0.27), **intro** (rot 9.7°→5.9°), and C1 prose (S2C
"0.09→0.13").

This is not a cosmetic swap: with the *verified* reproj row, scale-fit **hurts** sim
S2C (0.088→0.043) and scale (0.319→0.335), which inverts the C1 sub-claim.
**Decision needed (before Wed):**
- (a) re-run the val-10 campaign with `--render-compare --rc-scale-only
  --icp-denoise` (silhouette scale) and regenerate the row + `agg_paper.py` CSV, or
- (b) keep `+ICP` as the sim asset recipe (verified: 0.485/0.224/0.088/6.8/0.319),
  present silhouette scale-fit on the **s200 + oracle-mask evidence** only, and let
  the real-robot paired test carry the scale claim (the WORKSHOP_PAPER_PLAN
  guardrail was exactly this: `icp` = sim recipe, scale-fit = real/OOD lever).

### A2. `tab:frontend` SAM3 row mixes gated and ungated configs
From `s200_frontend_e2e.csv`: the printed f1 0.46, prec 0.49, cd-F1 0.82 are
**+icp+gate** (81 preds); the printed recall 0.51 is **ungated +icp** (116 preds;
gated recall is 0.43 — *worse* than YOLOE's 0.45, which would kill the "SAM3 finds
more objects" sentence). Prose compounds it: "116 predictions … dropping precision
0.62→0.49" — at 116 predictions precision is **0.41**, not 0.49. Same class of
mixed-config bug fixed in `tab:real` on 07-22. **Fix:** use the ungated row
consistently (0.45 / 0.75 / 0.41 / 0.51) — recall ↑, precision ↓, F1 ↓ is the
cleaner story anyway; YOLOE row is already ungated.

### A3. Oracle prose mixes val-10 and s200
"native placement reaches 0.62 IoU-F1 and registration lifts it to 0.75": 0.62 is
oracle-layout **val-10** (0.6229); 0.75 is oracle+icp **s200** (val-10 is 0.7435).
Say **0.62→0.74 (val-10)**. Bonus (currently only in comments): oracle **scale-fit**
on clean masks is the strongest pro-scale-fit evidence on disk — val-10 S2C
0.135→**0.328**, scale 0.320→**0.130** — worth one sentence, and it partially
substitutes for A1(b).

### A4. "~0.5 m centroid error from odometry drift" — wrong attribution
The 07-24 diagnosis (render-compare sim/real contrast: identical method improves
silhouette IoU 0.22→0.59 on real yet *worsens* GT 3D-IoU; near-perfect in sim)
pins the real floor on the **uncalibrated camera extrinsic** (~0.5 m, one-time
fixable), with drift secondary. Also in Methods: "odometry drift makes cross-frame
rendering inconsistent" → say "camera-pose error (extrinsic miscalibration and
odometry drift)". Reframe per PAPER_FLOW_NOTES §6: this is a *proven, controlled
diagnosis* (a strength), not a hedge.

### A5. Real rotation-worsening: use the paired test
"That same drift pushes rotation up (10.8→17°)" — the per-scene medians are
confounded by the IoU-matched set. The rigorous statement exists:
`paired_rotation_real.py`, +ICP worsens rotation on **23/34 objects, median +1.4°,
Wilcoxon p=0.046** — mirror the paired scale test. (And drop "drift" per A4.)

### A6. Small numeric overclaims
- "nearly doubles Scan2CAD (0.09→0.13)" — that's +44% (and 0.13 is A1-unverified).
- "(+22%)" for IoU-F1 0.40→0.49 — underlying 0.4043→0.4849 = **+20%**.
- Abstract pairs 0.49 (from +ICP) with 0.27 (from +scale+ICP) as if one config
  achieves both; word it as the registration progression.
- `fig:scenes` caption "(83 vs 72)": 83 = oracle n_pred (s200 e2e, stride 10);
  72 = the *stride-1* detector run. Pick counts from the runs the figure renders.

### A7. Related-work leftover v1 overclaim
"We extend SAM3D from single images to continuous streams … remains view-consistent
despite occlusions, and integrates metric and semantic data" — v1 language; the
occlusion/view-consistency claim is untested in v2. Rewrite to the v2 claim
(shape-only use; pose/scale from registration).

## B. Mechanical / LaTeX bugs (paper won't build from this repo as-is)

1. **`\cref`/`\Cref` used ~20× but `cleveref` is never loaded** → undefined control
   sequence. (If the 4¾-page PDF came from an Overleaf copy with a different
   preamble/bib, the repo and Overleaf have diverged — reconcile before editing.)
2. **`Fig.~\ref{fig:pipeline}` referenced; no such figure exists.** `figs/fig1.pdf`
   *is* the pipeline teaser (sensors → tracking → SAM3D → registration → map) and is
   never included. Add it as Fig. 1.
3. **Nav-section figure paths are wrong:** `IROS_workshop/figures/system-diagram.pdf`
   → files live at `figs/system-diagram.pdf`, `figs/scene_s200_hallway_icp.pdf`.
4. **`references.bib` keys don't match the cites.** Bib has short stub keys
   (`sam3d`, `clio`, `icp`, …); main.tex cites long keys
   (`sam3dteam2025sam3d3dfyimages`, `maggio2024clio…`, …) → ~15 undefined cites.
   Missing entries entirely: voxblox, NeRF, 3DGS, Kimera, Kanade 1997, RoboGSim,
   RialTo, ACDC, Proc4Gem, Nav2/marathon2. Several stubs say `Anonymous` /
   `\todo{fill}`. Appendix cites `\cite{clio}` while the body cites the long key —
   same paper, two keys.
5. `\label{fig:scene_oracle}` sits outside any caption (mislabels); subfig (c) has
   no label though the caption discusses it.
6. Duplicated `POINTS TO HIT` comment block in §C2 (~20 lines, twice).
7. Typos: "System diagram is show in"; "…and a visualization of the hallway
   metric-semantic map." (sentence trails off); experiments preamble run-on ("…for
   the AI2-THOR simulator and we render posed RGB-D frames…").
8. `\todo{limitations + conclusion prose}` — the conclusion does not exist yet.
9. Drafting scaffolding to strip: `% POINTS TO HIT` blocks, title-candidate comment,
   `\todo` macro, float-tuning block can stay.

## C. Unclear (not wrong, but a reviewer stumbles)

- **eq:rc's two regimes** (multi-view render-and-compare in sim vs single-view
  closed form on real) is the subtlest paragraph in the paper; the "for one view
  eq. (1) has a closed form" sentence is 60 words. Split it; state the principle
  first (pose from depth's bulk surface, size from the mask's clean boundary — the
  design principle in WORKSHOP_PAPER_PLAN is crisper than the current text).
- "usable views $\mathcal V_k$" never defined (kept/least-occluded views).
- gt-vocab prompting is disclosed ("prompted with the scene vocabulary") but its
  implication (near-oracle label prior) is not; one honesty clause suffices.
- If the appendix is cut (it is), §Metrics must stand alone — it mostly does; keep
  the Hungarian-on-IoU + greedy-XY-1m sentences, drop the equations.
- Cluster baseline: say once that it's *our own front-end's* fused cloud, denoised
  to the ConceptGraphs node convention — "made confound-free" is currently only in
  the appendix, and it's the fairness argument.

## D. Fit to SeMaNa CFP

CFP centers **semantic 3D explicit maps + MM-LLMs for spatial reasoning**. Our
weakest-positioned section (nav demo) is the most workshop-relevant — keep it, but
tighten: the gray transcript is ~½ column of tiny text; compress to a 5–8-line
excerpt + 2 sentences of analysis. Name the LLM hook earlier (intro P1/P4 — map as
text an LLM can act on; abstract already has it).

## E. Shortening plan (≈ −¾ col of text + −⅓ col of figure area, then add ¼-col conclusion)

| Cut | Est. saving |
|---|---|
| Appendix (both sections; fold one fairness sentence into §Experiments preamble) | already excluded from count |
| Nav transcript 55 lines → ~10 | ~⅓ col |
| Related work: merge Real2Sim para to 2 sentences; cut A7 leftovers | ~⅕ col |
| Intro P1 −2 sentences; P4 overlaps abstract — tighten | ~⅛ col |
| C1/C3: each restates its table once; cut the restatement, keep the reading | ~¼ col |
| Metrics: compress loose-metric sentence | lines |
| Figures: nav figs side-by-side subfigs instead of stacked 0.7-width; `fig_c3_real` 0.8→0.65\linewidth; `fig_completion` top plot 0.7→0.6 | ~⅓ col |
| Add: Limitations+Conclusion | +¼ col |

## F. Voice notes (from the v1 IROS paper)

Chris's register: plain assertive topic sentences ("We show that…", "We found
that…"), first-person plural, "In other words," transitions, concrete numbers
inline, honest engineering asides, at most one rhetorical question per section
("it is reasonable to ask why…"). The current draft is heavier on flourish ("earns
its keep", "the regime a moving robot actually lives in", "we stake the real claim
on scale") — keep one or two, flatten the rest. Avoid em-dash chains.

## G. Edit workflow (recommendation)

Write `paper/mainv2.tex` alongside `main.tex` (same `figs/`), one commit per audit
section so the diff is reviewable; `latexdiff main.tex mainv2.tex` for a visual
pass; replace `main.tex` once approved. Order of work given the Wed deadline:
1. Resolve **A1** (run the val-10 silhouette campaign now — it can run while
   editing — or adopt fallback (b)).
2. Mechanical fixes (B1–B5: bib, cleveref, figure paths) so the repo compiles.
3. Honesty edits (A2–A7) + write Limitations/Conclusion.
4. Shorten per §E, keeping §F voice.
