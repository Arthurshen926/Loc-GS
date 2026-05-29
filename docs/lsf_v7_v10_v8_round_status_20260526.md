# LSF v7/v8/v10 Round Status - 2026-05-26

## Scope

This round extends the v7+ method-level validation path without using Cambridge
official test data for route selection or tuning.

Primary artifacts:

- v7 report: `output/reports/lsf_v7_failure_aware512_train_q80_20260525/report.json`
- v10 transition audit: `output/lsf_v10_transition_audit_20260526/v7_failure_aware512_q80_report_baseline`
- v8 observation proxies: `output/lsf_v8_support_banks_20260526/query_observations`

## v7 Gate Status

Using the same baseline as the v7 q80 report:
`output/stdloc_native/baseline_train_q80_mixed_20260525`

v7-512 shows a real median/strict-recall signal:

- macro dense median TE delta: `-0.505cm`
- macro dense R10 delta: `+0.75pp`
- macro dense R5 delta: `+1.75pp`

But it is rejected by v13 gates:

- macro dense P95 TE delta: `+436.402cm`
- total dense-worsened count: baseline `5`, candidate `6`
- candidate regression count: Reg20 `11`, Reg50 `7`

This confirms v7 is not a SOTA candidate despite the positive median signal.

## v10 Dense Transition Audit

The v10 audit tool now writes:

- `dense_transition_audit.json`
- `hard_dense_cases.csv`
- required audit bundle: `manifest.json`, `command.txt`, `metrics_summary.json`,
  `split_audit.json`, `git_status.txt`

Per-scene v7-512 transition results against the report baseline:

| Scene | Baseline DenseWorse | Candidate DenseWorse | Reg20 | Reg50 |
| --- | ---: | ---: | ---: | ---: |
| GreatCourt | 2 | 2 | 0 | 0 |
| KingsCollege | 1 | 0 | 0 | 0 |
| OldHospital | 1 | 2 | 2 | 0 |
| ShopFacade | 0 | 0 | 0 | 0 |
| StMarysChurch | 1 | 2 | 9 | 7 |

Conclusion: v7 failures are not only dense-refinement absorption. OldHospital
and StMarysChurch include candidate-induced pose/map regressions before or
through dense refinement.

## v8 Support Bank Status

Added query-observation proxy generation from STDLoc results. The proxy uses:

- query id from results or split-file order
- sparse inlier count
- pnp-confidence proxy from sparse inliers
- fixed detector centroid fallback

It deliberately excludes TE/RE/GT fields.

Important blocker found and fixed:

- Existing v7 failure profiles use query ids that do not overlap the q80 train
  eval query ids.
- The v8 support-bank CLI now rejects failure-profile support when support query
  ids have zero overlap with query observations.
- Previously generated native256 banks under
  `output/lsf_v8_support_banks_20260526/banks_top2_native256_filtered` should be
  treated as diagnostic only, not route-validation input.

Corrected v8 artifacts were rebuilt from feedback-bank `query_summaries.json`,
which matches the failure-profile query universe:

- observations: `output/lsf_v8_support_banks_20260526/query_observations_feedback`
- banks: `output/lsf_v8_support_banks_20260526/banks_top2_feedback_native256`

Support query overlap:

| Scene | Support Query Count | Overlap | Non-Empty Banks |
| --- | ---: | ---: | --- |
| GreatCourt | 64 | 64 | native_safe_core |
| KingsCollege | 64 | 64 | native_safe_core, hard_tail_recovery |
| OldHospital | 64 | 64 | native_safe_core, hard_tail_recovery |
| ShopFacade | 63 | 63 | native_safe_core, hard_tail_recovery |
| StMarysChurch | 61 | 61 | native_safe_core, ambiguity_safe, occlusion_robust, hard_tail_recovery |

Router was also fixed to skip empty banks when `non_empty_bank_ids` is present.

## Implementation Changes

- Added `loc_gs/scripts/build_dense_transition_audit.py`.
- Added `loc_gs/scripts/build_query_observations_from_stdloc_results.py`.
- Extended `loc_gs/scripts/build_query_support_banks.py` with failure-profile
  support loading and support-query overlap validation.
- Extended `loc_gs/stdloc_native/support_banks.py` to filter unmatched hard
  queries and report unmatched counts.
- Extended `loc_gs/stdloc_native/v13_gate.py` with dense-transition, latency,
  R2, and candidate-regression gates.

## Verification

Ran:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest \
  tests/test_method_artifact_clis.py \
  tests/test_support_banks.py \
  tests/test_v13_gate.py \
  tests/test_dense_match_verifier.py \
  tests/test_build_cambridge_eval_report.py -q
```

Result: `21 passed`.

## Next Round

1. Implement v8 sparse-only route validation using the corrected feedback-bank
   support-bank artifacts.
2. For v10, implement the Loc-GS parity/ablation dense verifier path outside
   `third_party/stdloc`; native STDLoc cannot accept a true correspondence
   verifier through config alone.
