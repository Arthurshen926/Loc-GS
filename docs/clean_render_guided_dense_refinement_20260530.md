# Clean Render Guided Dense Refinement - 2026-05-30

This note resets the sparse-to-dense transition work around a narrower claim:
guided pose and Gaussian gating are clean render generation tools, not dense
pose acceptance tools.

## Scope

Given a STDLoc sparse pose `T_s` and sparse PnP inlier anchors, the module
should generate a cleaner dense render pose `T_r` and optional render-time
Gaussian gating metadata. `T_r` is used only to render dense RGB/depth/features
for later matching. It is not a final localization pose.

This keeps three stages separate:

1. Clean render generation: choose a sparse-anchor-consistent render candidate.
2. Dense candidate generation: global and patch-level rendered-query matches.
3. Pose refinement: robust dense residuals plus sparse-anchor residuals.

## Implemented In This Round

- `loc_gs.dense_support.clean_render_generator`
  - Scores render candidates from existing SLCDP preflight diagnostics.
  - Selects native render when it is already clean.
  - Selects guided/gated render only when anchor visibility, coverage, feature
    agreement, and artifact metrics improve enough after translation/gating
    penalties.
  - Explicitly marks `selected_role=render_pose_only` and
    `does_not_select_final_pose=True`.

- `loc_gs.scripts.build_clean_render_report`
  - Offline diagnostic report over existing `pgsh_summary.json` artifacts.
  - Does not run localization and does not use GT to select candidates.
  - Writes `report.md`, `metrics_summary.json`, `manifest.json`,
    `split_audit.json`, `command.txt`, and `git_status.txt`.

- `visualize_stdloc_hard_matches`
  - Adds `slcdp_repair_search.clean_render_generation` metadata for future
    diagnostics without changing the existing dense pose selection behavior.

## Diagnostic Recheck

Report:

`output/diagnostics/clean_render_generator_report_20260530/report.md`

Observed on existing diagnostic artifacts:

- KingsCollege #239: native render is not anchor-explanatory. The clean render
  scorer selects `gated_ray_side-5.000` as a render candidate with anchor-safe
  preflight, improved feature agreement, and low gating fraction.
- KingsCollege #33: native render is already anchor-safe. The scorer keeps
  `base`, which matches the intended "no action on clean normal render" behavior.

These cases remain diagnostic-only because they are official test examples.

## Next Required Work

Phase 1 should now be run on train/self-map hard and normal subsets using only
render-quality metrics:

- anchor visibility ratio
- anchor depth consistency
- anchor feature cosine
- local render feature variance
- coarse MNN count / query-render agreement
- gating removed fraction

Only after clean render quality improves on Type A cases without hurting Type D
cases should this output be consumed by patch dense candidates or final pose
refinement.
