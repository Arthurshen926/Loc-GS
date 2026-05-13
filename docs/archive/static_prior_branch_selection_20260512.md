# Archived Direction: Static Prior And Branch Selection

Date: 2026-05-12

The following artifacts are preserved for reproducibility, but should not guide
the active optimization direction unless a future experiment explicitly revisits
them as ablations:

- `docs/cambridge_branch_manifest_reliability_20260511.json`
- `docs/cambridge_branch_manifest_reliability_b8_e10_20260511.json`
- `docs/cambridge_branch_manifest_traincalib_20260512.json`
- `docs/cambridge_branch_manifest_querylike_20260512.json`
- `loc_gs.scripts.select_cambridge_branch`
- `loc_gs.scripts.eval_cambridge_branch_selected`
- `loc_gs.scripts.select_cambridge_query_pose`
- query-like calibrated matchability caches under
  `output/stdloc_hybrid/query_like_matchability_20260512_v64/`
- guarded selector outputs under
  `output/stdloc_hybrid/branch_selection_querylike_20260512/`

## Why Archived

The full-test evidence shows that static per-landmark reliability and
branch-level selection do not solve query-conditioned ambiguity:

- The query-like static prior improved q80 but dropped full-test macro recall
  and median pose quality.
- Guarded branch selection is defensible as a calibration diagnostic, but it
  does not establish the paper claim that localization feedback shapes the
  feature field.
- Multi-branch pose selection and `pnp_hypotheses > 1` make inference look like
  a search trick rather than a clean localization system.

## Replacement Mainline

Use `docs/loc_gs_lff_scenematch_mainline_20260512.md` as the current mainline:

```text
STDLoc/PLY descriptor backbone
+ localization-feedback detector labels
+ query-conditioned SceneMatchNet
+ single-path PROSAC priority
```

Static reliability can still appear in tables as a negative ablation showing
why query-conditioned pair scoring is necessary.
