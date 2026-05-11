# Loc-GS-FT Mainline

This note records the paper-facing mainline after the SP-GS4 review.

## Claim

Loc-GS-FT is **matchability-guided fine-tuning of Gaussian feature fields** for
2D-3D relocalization. It does not replace the STDLoc/SuperPoint descriptor
backbone. Instead, map construction rehearses downstream localization with
known geometry, then uses the resulting feedback to constrain feature-field
fine-tuning and landmark reliability.

## Method

1. Build a base SuperPoint/STDLoc feature Gaussian field.
2. Rehearse localization during training:
   - render from the GT training pose;
   - render from perturbed, neighbor, or interpolated map-side poses;
   - match query SuperPoint keypoints to rendered 3D-backed descriptors;
   - use GT pose/depth/alpha to label valid matches and localization quality.
3. Fine-tune conservatively:
   - keep descriptor reconstruction and same-view geometric matching;
   - use low-weight PnP/matchability feedback;
   - regularize decoded descriptors toward the PLY/STDLoc descriptor;
   - train locability/reliability from localization support plus
     detector/visibility/depth priors, without large geometry drift.
4. Test with a single baseline-preserving 2D-3D localization path.

## Rehearsal Pose Coverage

The original training path rendered the localization target from only a small
perturbation of the current training pose. That under-covers test-time viewpoint
variation. The training CLI now exposes `--rehearsal_pose_mode`:

- `perturb`: original current-pose perturbation.
- `pair`: render from a nearby overlapping training pose.
- `interpolate`: render from a novel pose sampled on the line between current
  and nearby overlapping training poses; the sampler also accepts mild
  extrapolation beyond the two endpoints.
- `mixed`: fixed mixture of current-pose perturbation and interpolated neighbor
  poses.

The paper-facing recipe uses `mixed`, stronger pose noise, mild neighbor-pose
extrapolation, and a small neighbor jitter. This makes self-localization
rehearsal less tied to the exact training views while retaining overlap for
stable geometric supervision.

The matchability calibration script also supports rendered rehearsal views via
`--rendered_rehearsal_views`. Those views are sampled with the same mixed
neighbor/extrapolation policy and labeled with known poses, so the retained
calibrated matchability can cover map-side novel viewpoints instead of only
memorizing training images.

## Default Recipe Intent

The launcher recipe is constrained rather than replacement-heavy:

- low PnP feedback weight;
- no pose/reprojection PnP loss by default;
- same-view geometric matching remains active;
- mixed rehearsal poses cover a wider camera neighborhood;
- PLY residual regularization limits descriptor drift.

The expected ablation is:

| Variant | Purpose |
|---|---|
| STDLoc baseline | strong descriptor/sampling backbone |
| base reconstruction | feature reconstruction without localization rehearsal |
| Loc-GS-FT | constrained rehearsal fine-tuning |
| hard filtering / full replacement | negative-result ablation |

The claim is supported only if Loc-GS-FT improves matchability/PnP input quality
and final pose while preserving the baseline on repetitive scenes.
