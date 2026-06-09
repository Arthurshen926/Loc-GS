# SADC Phase-0 Diagnostic Status 2026-05-31

## Scope

This is a train/self-map diagnostic for Sparse-Anchored Dense Correspondence
Field (SADC). It is not a paper-facing Cambridge result because the current
`locgsctl list-scenes` audit reports 8192 sampled landmarks while native STDLoc
expects 16384.

SADC is implemented as a correspondence-level dense-stage candidate filter:

- It does not use GT.
- It does not select among final poses.
- It scores dense correspondences by the geometric mean of:
  `match_quality`, `anchor_flow_consistency`, and `geometry`.
- Optional patch candidates are merged as another correspondence source before
  filtering.

## Implementation

- `loc_gs/dense_support/sadc.py`
  - `SADCCorrespondencePolicy`
  - `score_sadc_correspondences`
  - `filter_sadc_correspondences`
  - `merge_sadc_candidate_sources`
- `loc_gs/scripts/visualize_stdloc_hard_matches.py`
  - Adds `--sadc_dense`, `--sadc_include_patch_candidates`,
    `--sadc_max_candidates`, and `--sadc_anchor_flow_scale_px`.
  - SADC filtering is applied before the existing STDLoc dense PnP solve.
- `loc_gs/scripts/eval_sparse_conditioned_dense_control.py`
  - Adds the same SADC CLI options.
  - Writes SADC candidate/kept/drop counts and score summaries into result rows.
- `tests/test_sadc.py`
  - Synthetic checks for sparse-anchor flow consistency and deterministic
    filtering.

## Experiments

All runs use `eval_split=train` and
`output/diagnostics/clean_render_phase0_benchmark_20260530/phase0_cases.csv`.

| Run | Rows | Median sparse/base/candidate TE | Macro candidate delta vs base | R10 base->cand | R5 base->cand | P95 base->cand |
| --- | ---: | --- | ---: | --- | --- | --- |
| `sadc_eval_phase0_A_all_native_filter_20260531` | 7 | 13.42 / 41.33 / 41.04 cm | +2.84 cm | 0.00 -> 0.00 | 0.00 -> 0.00 | 54.36 -> 57.84 cm |
| `sadc_eval_phase0_A_all_patch_20260531` | 7 | 13.42 / 41.33 / 37.74 cm | -6.31 cm | 0.00 -> 0.00 | 0.00 -> 0.00 | 54.36 -> 51.49 cm |
| `sadc_eval_phase0_D50_patch_20260531` | 50 | 7.61 / 5.66 / 5.73 cm | +0.07 cm | 0.94 -> 0.88 | 0.36 -> 0.32 | 10.60 -> 12.43 cm |
| `sadc_eval_phase0_A_all_patch8192_20260531` | 7 | 13.42 / 41.33 / 38.70 cm | -4.26 cm | 0.00 -> 0.00 | 0.00 -> 0.00 | 54.36 -> 52.64 cm |
| `sadc_eval_phase0_D50_patch8192_20260531` | 50 | 7.61 / 5.66 / 5.67 cm | +0.01 cm | 0.94 -> 0.88 | 0.36 -> 0.40 | 10.60 -> 10.83 cm |

For comparison with the previous APD risk-weighted diagnostic:

| Run | Rows | Median sparse/base/candidate TE | Macro candidate delta vs base | R10 base->cand | R5 base->cand | P95 base->cand |
| --- | ---: | --- | ---: | --- | --- | --- |
| `apd_dense_eval_phase0_A_all_risk_weighted_v2_20260531` | 7 | 13.42 / 41.33 / 31.72 cm | -17.23 cm | 0.00 -> 0.00 | 0.00 -> 0.00 | 54.36 -> 29.72 cm |
| `apd_dense_eval_phase0_D50_risk_weighted_v2_20260531` | 50 | 7.61 / 5.66 / 5.67 cm | +0.01 cm | 0.94 -> 0.94 | 0.36 -> 0.36 | 10.60 -> 10.56 cm |

## Interpretation

SADC is cleaner than APD structurally because it stays at the dense
correspondence level and reuses the existing STDLoc dense PnP backend. The
initial diagnostic does not support SADC as an APD upper replacement:

- Native dense correspondence filtering alone does not fix A-class damage.
- Adding patch candidates helps hard A cases but remains behind APD
  risk-weighted pose mitigation.
- On the D50 normal smoke set, SADC+patch slightly worsens median, R10/R5, and
  P95 at a 4096 candidate cap. Raising the cap to 8192 reduces the normal median
  and P95 regression but still drops R10, so it is not safe as a full dense
  replacement policy.

Current best role: SADC is a useful candidate-quality diagnostic and a possible
input to APD/risk-weighted dense mitigation, not a standalone replacement.
