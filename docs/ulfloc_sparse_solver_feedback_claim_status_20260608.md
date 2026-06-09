# ULF-Loc Sparse Solver-Feedback Claim Status 2026-06-08

## Scope

This note tracks sparse-only evidence for the current mainline claim:
self-localization solver feedback should improve ULF-Loc sparse localization by
guiding landmark sampling, landmark descriptor fusion, scene-specific detector
targets, and correspondence-level matching/ranking toward PnP solvability.

All results below are train-dev / self-map validation artifacts. They are not
official Cambridge test results.

## Supported By Localization Metrics

### Solver-Weighted Landmark Feature Fusion

Five-scene sparse-only train-dev comparison against the geomw detector baseline
supports this module as the current strongest positive claim.

- GreatCourt: 14.2596 -> 13.0172 cm, R10 0.3648 -> 0.3909, R5 0.1303 -> 0.1498.
- KingsCollege: 17.5299 -> 15.9274 cm, R10 0.2254 -> 0.2541, R5 0.0574 -> 0.0615.
- OldHospital: 15.2542 -> 13.6315 cm, R10 0.3184 -> 0.3799, R5 unchanged.
- ShopFacade: 2.6698 -> 2.5233 cm, recall unchanged.
- StMarysChurch: 6.9275 -> 6.2679 cm, R10 0.6879 -> 0.7349, R5 0.3557 -> 0.4128.

Macro sparse median TE improves by about 1.05 cm. This is the clearest
solver-feedback contribution so far.

ShopFacade alpha ablation:

- `trust_alpha=0.1`: median TE 2.5233 cm, R10 0.9348, R5 0.7174.
- `trust_alpha=0.2`: median TE 2.5549 cm, R10 0.9348, R5 0.7174, P90 7.8591 cm.

Conclusion: keep `trust_alpha=0.1` as the precision-oriented recipe. Alpha 0.2
is tail-oriented but not better for median precision.

## Implemented But Not Yet Localization-Positive

### Solver-Feedback Detector Target

The detector target path now supports solver-validity weighting end to end:

- correspondence target artifact stores `solver_validity_weights`;
- compact target rasterization consumes `solver_validity_weights`;
- detector training CLI records `solver_validity_power`;
- tests cover artifact, rasterization, and CLI behavior.

ShopFacade sparse-only results with solver-fused descriptors:

- Baseline geomw detector: median TE 2.5233 cm, R10 0.9348, R5 0.7174, R2 0.3261.
- Solver-valid detector `power=1.0`: median TE 2.8627 cm, R10 0.9348, R5 0.6957, R2 0.2826.
- Solver-valid detector `power=0.25`: median TE 2.9029 cm, R10 0.9130, R5 0.7174, R2 0.3261.

Conclusion: direct solver-valid detector reweighting is not a positive
localization result. It should not be a main claim yet. The likely failure mode
is over-suppressing keypoints that are still useful for sparse retrieval/matching
even when their correspondence-level solver validity is low.

### Correspondence-Level Negative Supervision

Implemented and audited:

- `query_regression_delta_cm` can upweight outlier correspondences from
  candidate-induced regression queries.
- hard-negative flags and weights are exported to match scorer targets and
  conflict graph edges.
- the CLI can read a non-test sparse PnP validation profile and write the
  hard-negative supervision metrics.

Current limitation: most available sparse PnP validation profiles were built
without feedback-bank support, so `landmark_regression_risk_count`,
`protected_landmark_support_count`, and `validated_landmark_support_count` are
often zero. The implementation is ready, but the available profiles are not yet
strong enough to prove the claim.

### Full-Gaussian SparseSet + Sparse PnP Validation Loop

Implemented:

- sparse PnP validation profiles are converted into selector-side tensors:
  `sparse_validation_risk`, `sparse_validation_hard_reject_exempt`,
  `validation_per_query_support`, and `per_query_observations`;
- support scope supports `all` and `protected_regressions`;
- test-split profiles are rejected.

Current limitation: the existing profiles mostly contain paired metric deltas
but no real per-landmark support/risk because candidate/baseline feedback banks
were not provided. Without those per-landmark observations, SparseSet still
falls back to weak unary or proxy evidence and cannot yet support a strong
sampling claim.

## Current Mainline Recommendation

For paper-facing sparse-only development, freeze the current positive path as:

1. ULF-Loc geometry/K.C. sampled map.
2. Geomw scene-specific detector.
3. Solver-weighted landmark feature fusion with `trust_alpha=0.1`.

Keep the following as active but not-yet-positive modules:

1. Full-Gaussian SparseSet with real feedback-bank-backed sparse PnP validation.
2. Correspondence-level hard-negative match scorer/conflict graph.
3. Detector target solver feedback as a diagnostic/auxiliary branch only.

The next meaningful improvement should target feedback-bank-backed
correspondence supervision and match ranking, not direct detector heatmap
reweighting.

## Implementation Correction: Profile-Backed SparseSet

`full_gaussian_sparse_set` no longer requires the legacy unary
`solver_feedback.pkl` when an attributed `sparse_pnp_validation_profile_v2` is
provided.  The intended SparseSet input is now:

1. real non-test ULF sparse PnP traces exported as `feedback_bank_v3`;
2. paired sparse validation profile with nonzero correspondence/landmark
   attribution;
3. full-Gaussian K.C./visibility/mask signals;
4. optional legacy unary solver/ray feedback only as extra prior.

The validation profile's per-query PnP support is also aggregated into a
full-Gaussian solver-support vector so profile-supported landmarks cannot be
dropped before set optimization.  This fixes the previous leak in the method
logic where Full-Gaussian SparseSet was still effectively anchored to a weak
unary `landmark_weights` artifact.
