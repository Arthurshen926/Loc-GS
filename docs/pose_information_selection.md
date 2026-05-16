# Pose-Information-Aware Selector Objective

`loc_gs/losses/pose_information_selection.py` provides a unit-testable selector
objective for per-Gaussian or per-landmark scores. It is intentionally not wired
into long Cambridge training by default.

## Difference From Existing Locability Losses

Existing locability-related losses in `loc_gs/losses/landmark_selection.py`,
`loc_gs/losses/geometric_match.py`, and `loc_gs/losses/differentiable_pnp.py`
mainly teach whether a point is locally matchable, visible, distinctive, or
aligned with a locability prior. Those losses are useful but can still preserve
points that are easy to detect while being weak for downstream pose estimation.

The pose-information selector adds four explicit training signals:

- `pnp_information_proxy_loss`: favors selected points that have both high
  inlier probability and high pose-information proxy. The information input may
  be a scalar, vector, or square matrix such as a `J^T J` proxy; matrices are
  reduced by trace.
- `hard_negative_suppression_loss`: suppresses high-scoring false matches from
  feedback-bank hard-negative labels.
- `selection_budget_loss`: keeps the expected selected fraction or count near a
  requested budget.
- `coverage_regularization_loss`: discourages selected points from collapsing
  into one small image region.

`combined_pose_info_selector_loss` returns each component plus the weighted
total loss, so training logs can show which signal is active.

## Intended Inputs

Inputs are all `torch.Tensor` values and support batched `[B, N]` candidate
layouts:

- `selection_logits`: selector logits before sigmoid.
- `inlier_probability`: feedback-bank inlier probability or label.
- `hard_negative_risk`: hard-negative score or binary label.
- `pose_information`: scalar/vector/matrix pose-information proxy.
- `positions_xy`: candidate image or normalized spatial positions.
- `visibility_score`: optional visibility weight.
- `mask`: optional valid-candidate mask.

Empty masks return finite zero-valued component losses. This makes synthetic
unit tests and sparse batches safe without hiding missing data at the caller
level.

## Example

```python
out = combined_pose_info_selector_loss(
    selection_logits,
    inlier_probability=inlier_prob,
    hard_negative_risk=hard_negative,
    pose_information=jtj_proxy,
    positions_xy=positions_xy,
    visibility_score=visibility,
    mask=valid,
    budget_target_fraction=0.25,
)
loss = out["loss"]
```

Use this objective as training-time feedback only. It does not change the
Cambridge evaluator, test split, or query-time PROSAC/dense-refinement path.
