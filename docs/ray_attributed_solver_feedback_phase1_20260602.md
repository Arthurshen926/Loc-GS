# Ray-Attributed Solver Feedback Phase 1

## Purpose

Phase 1 implements the feedback aggregation core for the next LSF/ULF direction:

```text
cross-view or rendered-view feature match
-> ray/pixel observation with top-K Gaussian contributors
-> PnP / local geometry evidence
-> soft Gaussian-level solver feedback
```

This avoids assigning a rendered pixel to only one rendered depth or top-1
Gaussian. Instead, solver evidence is distributed to all top-K contributors on
the ray according to contribution and reliability.

## Observation Schema

The CLI consumes JSON or JSONL. JSONL supports:

```json
{"type": "manifest", "manifest": {"scene": "ShopFacade", "split_name": "selfmap_train"}}
{"type": "observation", "observation": {
  "query_id": "foldA/frame00001.png",
  "source_view_id": "foldB/frame00017.png",
  "split_name": "selfmap_train",
  "pnp_inlier": true,
  "reprojection_error_px": 1.2,
  "descriptor_score": 0.83,
  "local_geometry_score": 0.91,
  "expected_depth": 12.0,
  "rendered_depth": 3.0,
  "contributors": [
    {"gaussian_id": 123, "contribution": 0.72, "depth": 3.1, "xyz": [1.0, 2.0, 3.0]},
    {"gaussian_id": 987, "contribution": 0.28, "depth": 11.8, "xyz": [1.2, 2.1, 11.8]}
  ]
}}
```

Required fields:

- `split_name`: must not be `test`.
- `contributors[].gaussian_id`: index in the target Gaussian map.
- `contributors[].contribution`: alpha/composition or ray contribution weight.

Optional fields:

- `contributors[].reliability`: multiplies the contribution before normalization.
- `contributors[].xyz`: required only when calling `soft_point_from_contributors`.
- `expected_depth` and `rendered_depth`: used to estimate near-occlusion artifact risk.
- `descriptor_score`, `local_geometry_score`, `reprojection_error_px`: used to score positive PnP evidence.

## Output Artifact

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.build_ray_attributed_solver_feedback \
  --observations path/to/observations.jsonl \
  --output_dir output/ray_feedback/ShopFacade \
  --scene ShopFacade \
  --split_name selfmap_train \
  --num_gaussians 1868039
```

Output:

- `ray_solver_feedback.pt`
- `manifest.json`
- `metrics_summary.json`
- `split_audit.json`
- `command.txt`
- `git_status.txt`

Tensor keys:

- `support_score`: positive solver evidence softly assigned to Gaussians.
- `weighted_support`: alias for the positive support mass.
- `contribution_mass`: all normalized ray contribution mass.
- `positive_contribution_mass`: contribution mass from PnP-inlier observations.
- `observed_count`: number of observations touching each Gaussian.
- `positive_observed_count`: number of positive observations touching each Gaussian.
- `hard_negative_risk`: weak outlier evidence. Non-inliers are not treated as strong negatives by default.
- `artifact_risk`: near-occlusion/render artifact risk from ray depth disagreement.

## Artifact Handling

Phase 1 does not extract image matches. It only defines the stable contract for
later extractors:

1. Real-image cross-view extractor: train-fold pseudo-query image matched to a
   different train-fold source image with shared 3DGS visibility.
2. Rendered-view extractor: virtual/rendered source view matched to train-fold
   pseudo-query, with guided-pose and Gaussian-gating repair.
3. ULF integration: use `support_score`, `artifact_risk`, and
   `positive_observed_count` for overcomplete landmark selection and
   solver-weighted feature fusion.

## Research Safety

- Cambridge official test queries must not be used to create observations.
- This artifact is feedback/training-side evidence, not a per-query branch selector.
- Rendered-view observations must be gated or rejected when artifact risk is high.
- Negative evidence is weak unless repeated across multiple independent views.
