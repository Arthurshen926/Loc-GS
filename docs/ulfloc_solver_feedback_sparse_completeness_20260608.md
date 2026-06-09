# ULF-Loc Solver Feedback Sparse-Only Completeness, 2026-06-08

## Claim Scope

Current paper-safe claim should stay sparse-only and train-dev:

> On the ULF-Loc 3DGS localization pipeline, self-localization sparse PnP feedback can improve landmark descriptors and make sparse localization more PnP-solvability aware than keypoint-consensus / visibility-only descriptors.

Do not claim Cambridge official test SOTA from these runs.  These results use train-dev overlays and are suitable for method development and ablation evidence.

## Main Sparse-Only Evidence

Baseline is the current ULF-Loc sparse log with the geom-weighted scene-specific detector.  Candidate keeps the same sampled landmarks and detector, and only replaces selected landmark descriptors with solver-weighted fused descriptors.

| Scene | Baseline TE cm | Solver-fused TE cm | Delta cm | Baseline R@10 | Solver-fused R@10 | Baseline R@5 | Solver-fused R@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GreatCourt | 14.2596 | 13.0172 | -1.2424 | 0.3648 | 0.3909 | 0.1303 | 0.1498 |
| KingsCollege | 17.5299 | 15.9274 | -1.6024 | 0.2254 | 0.2541 | 0.0574 | 0.0615 |
| OldHospital | 15.2542 | 13.6315 | -1.6226 | 0.3184 | 0.3799 | 0.1341 | 0.1341 |
| ShopFacade | 2.6698 | 2.5233 | -0.1465 | 0.9348 | 0.9348 | 0.7174 | 0.7174 |
| StMarysChurch | 6.9275 | 6.2679 | -0.6596 | 0.6879 | 0.7349 | 0.3557 | 0.4128 |

Macro sparse median TE improves from 11.3282 cm to 10.2735 cm, delta -1.0547 cm.  Macro R@10 improves by +0.0326 and macro R@5 by +0.0161.

This is the strongest current support for the main story.  It shows solver feedback is more effective as descriptor fusion supervision than as a weak post-hoc landmark ranking signal.

## Do-No-Harm Guard Ablation

Two guard variants were tested on real paired sparse PnP validation profiles:

- `all_regressions`: revert descriptor updates from every >20 cm candidate-induced regression query.
- `protected_regressions`: revert only from regressions where the baseline query was already good/protected.

`all_regressions` is rejected.  On GreatCourt it worsens the regression count and loses improvements, because hard-tail query drift is not reliable negative supervision.

`protected_regressions` is kept as an ablation, not as the main result.  On GreatCourt it improves the median delta and reduces regression count compared with the unguarded candidate, but slightly lowers R@10/R@5:

| GreatCourt Variant | Median TE cm | R@10 | R@5 | Regression >20 cm vs baseline | Improvement >20 cm vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline geomw | 14.2596 | 0.3648 | 0.1303 | - | - |
| Solver-fused | 13.0172 | 0.3909 | 0.1498 | 12 | 17 |
| Guarded protected | 12.8935 | 0.3811 | 0.1466 | 11 | 18 |

On StMarysChurch there are no protected regressions, so protected guard is a no-op.  This supports the conclusion that hard-tail failures should not be blindly converted into landmark-level negative labels.

## Method Status

Implemented and validated:

- Solver-weighted descriptor fusion from ULF-Loc pair cache.
- Source split rejection for test split metadata.
- Strict descriptor trust region and native fallback.
- ULF-Loc log export with identical sampled index and sparse geometry.
- Paired sparse PnP validation loop.
- Sparse-PnP guarded descriptor ablation with audited artifacts.

Still incomplete for a submission-level full method:

- Landmark sampling: full-Gaussian set optimization is not yet stronger than ULF K.C. sampling.
- Scene-specific detector target: solver-feedback target quality audit exists, but not yet a proven detector training improvement across scenes.
- Pair/match scorer: current learned/rerank variants have not beaten descriptor fusion robustly.
- Dense stage: not yet paper-safe; sparse-only should remain the current main evidence.

## Recommended Next Step

Do not spend more cycles on hard-tail descriptor guards.  The next submission-relevant step is correspondence-level supervision:

1. Keep geomw detector + solver-fused descriptors as the sparse baseline.
2. Build pair/match scorer features with query-conditioned geometry: descriptor margin, 2D coverage, depth spread, local ambiguity, and PnP inlier membership.
3. Train scorer on train/self-map only and validate sparse-only.
4. In parallel, turn solver feedback into detector target weights, but report it only after it beats the current geomw detector on at least three scenes.

