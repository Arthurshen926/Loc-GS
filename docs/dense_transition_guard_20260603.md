# Dense Transition Guard Status - 2026-06-03

Route C now has a reusable dense transition guard in
`loc_gs.dense_support.dense_transition_guard`.

The guard is inference-observable only. It compares the sparse pose and proposed
dense pose with:

- camera-center and rotation trust-region deltas,
- sparse landmark reprojection preservation,
- dense valid-match ratio,
- artifact/depth-risk score.

The default policy is conservative: if sparse evidence is missing or weak, dense
is accepted while failures are recorded. Dense is rejected only when sparse
support is high-confidence and one or more observable checks fail, unless
`reject_weak_sparse_failures` is explicitly enabled for diagnostics.

This is not a GT oracle. It does not inspect TE/RE or test labels. The report
builder counts `dense_worsened_proxy_count` from observable guard failures under
high-confidence sparse support.

Integration caveat: current ULF/STDLoc eval paths may not expose every required
field. The minimal hook should pass, per query, the sparse PnP pose, dense pose,
sparse inlier `query_xy`, `p3d`, `inliers`, `intrinsic`, `image_size`, plus dense
stats such as `valid_dense_match_ratio` and `artifact_depth_risk_score`. Until
that hook exists, `build_dense_transition_report` can still consume synthetic or
summary-style records with pose-delta, retained-ratio, valid-match-ratio, and
risk fields.

Paper-safety caveat: a per-query accept/reject guard is diagnostic or ablation
infrastructure unless a task explicitly validates it as a fixed, paper-safe
single-path dense-stage policy.
