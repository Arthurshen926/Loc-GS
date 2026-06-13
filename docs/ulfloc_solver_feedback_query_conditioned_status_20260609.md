# ULF-Loc Query-Conditioned Solver Feedback Status 20260609

## Method Summary

Current sparse-only mainline keeps the fixed high-quality ULF sampled landmark set and injects self-localization solver feedback into three offline/query-time consumers:

1. `solver_feedback_impact`: converts non-test sparse PnP traces into positive/negative landmark, view, and detector attribution.
2. `solver_feedback_feature_fusion_v2`: uses positive inlier views and excludes negative views to export a ULF-compatible fused `keypoints_features.pkl`, preserving fixed sampled ids.
3. `query_landmark_activation`: trains a query-conditioned pre-matching active-landmark selector from self-map feedback.

The current disjoint protocol is:

```text
self-map train feedback -> impact / observation cache / fused descriptor / activation
disjoint train-dev sparse-only eval
```

Official Cambridge test is not used for feedback, model selection, or hyperparameter tuning.

## Implementation Status

| Component | Status | Evidence |
| --- | --- | --- |
| Solver feedback impact attribution | implemented | `loc_gs/feedback/solver_feedback_impact.py`, tests pass |
| STDLoc-style visibility teacher | implemented | `loc_gs/training/ulfloc_visibility_teacher.py`, tests pass |
| Negative-aware detector targets | implemented | `loc_gs/training/ulfloc_solver_feedback_detector_targets.py`, tests pass |
| Solver-feedback feature fusion v2 | implemented | `loc_gs/stdloc_native/solver_feedback_feature_fusion_v2.py`, tests pass |
| Query-conditioned activation | implemented | `loc_gs/localization/query_landmark_activation.py`, tests pass |
| ULF sparse inference activation hook | implemented | `/root/ULF-Loc/ulfloc.py`, `eval_ulfloc_sparse_only.py`, tests pass |
| Launcher dry-run | implemented | `loc_gs/scripts/launch_ulfloc_solver_feedback_mainline.py`, tests pass |
| Five-scene current-mainline eval | not complete | blocked by ShopFacade gate failure |

## ShopFacade Disjoint Sparse Results

Feedback split: self-map train. Eval split: held-out train-dev.

| Config | Median TE cm | Median RE deg | R@10/5 | R@5/5 | R@2/2 | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| geomw baseline | 2.678167 | 0.162860 | 0.913043 | 0.717391 | 0.326087 | baseline |
| selfmap fused a=0.1 | 2.947690 | 0.177556 | 0.913043 | 0.695652 | 0.304348 | reject |
| selfmap fused + activation top16000 | 3.303425 | 0.173113 | 0.934783 | 0.673913 | 0.260870 | reject |
| selfmap fused a=0.05 | 2.791636 | 0.176453 | 0.913043 | 0.717391 | 0.326087 | reject |

Primary report:

```text
output/reports/ulfloc_qcsf_shopfacade_disjoint_desc_20260609_summary/summary_compare.md
```

## Key Negative Result

The plan's implementation path is now code-complete through ShopFacade smoke, but the current sparse recipe does not pass the main accuracy gate. Directly fusing self-map matched query descriptors into fixed ULF landmark descriptors hurts median TE on disjoint train-dev. Lowering `trust_alpha` from `0.1` to `0.05` reduces damage but still does not beat the geomw baseline.

This indicates that the current feedback signal is useful as attribution and filtering evidence, but direct descriptor mean-shift is not yet a valid mainline improvement.

## Previous Five-Scene Diagnostic Context

Older train-dev active/fused diagnostics exist for all five scenes, but they were not generated from the current disjoint per-match descriptor protocol and should not be used as the new mainline claim:

| Scene | Diagnostic Median TE cm | R@10/5 | R@5/5 |
| --- | ---: | ---: | ---: |
| GreatCourt | 13.410473 | 0.384365 | 0.143322 |
| KingsCollege | 17.294405 | 0.270492 | 0.081967 |
| OldHospital | 15.439202 | 0.346369 | 0.111732 |
| ShopFacade | 3.327472 | 0.913043 | 0.739130 |
| StMarysChurch | 6.616101 | 0.694631 | 0.399329 |

These are diagnostic only.

## Module Gate Status

| Module | Gate |
| --- | --- |
| Impact attribution | pass: produces non-test positive/negative attribution |
| Visibility teacher | pass: implemented and unit-tested |
| Detector target composition | pass at unit level; sparse accuracy gate not re-established |
| Feature fusion | fail as main result: worsens disjoint ShopFacade median |
| Query activation | fail as main result: improves R@10 but worsens median/R@5 |
| Five-scene expansion | blocked until ShopFacade sparse gate passes |

## Next Action

Do not expand the current descriptor-shift recipe to official test or claim it as paper-facing. The next sparse-only iteration should keep the implemented attribution stack but change the consumer:

1. Use solver feedback for view pruning / feature aggregation sample selection, not descriptor mean-shift.
2. Train descriptor/fusion with a contrastive objective against positive inlier views and negative PnP outliers.
3. Keep activation as diagnostic until it improves median TE or preserves R@5 on ShopFacade.
4. Re-run ShopFacade disjoint train-dev before any five-scene expansion.
