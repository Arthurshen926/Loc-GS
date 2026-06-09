# Rendered Feedback Augmentation Status

Date: 2026-06-03

This pass adds a paper-safe Route B artifact builder for rendered-RGB/depth-style
solver feedback augmentation. It does not render 3DGS directly and does not touch
`third_party/stdloc`; it consumes train/self-map observation JSONL records that a
future renderer/exporter can populate.

## Contract

- Input records may use existing `contributors` or rendered-style
  `ray_contributors`.
- `observed_only` mode accumulates existing source-ray observations into
  per-Gaussian support/count vectors without requiring rendered fields.
- `rendered_depth_augmented` mode accepts `rendered_depth`, `expected_depth`,
  `artifact_score`, and per-ray contributor weights. Near contributors can receive
  artifact-risk contribution when rendered depth is in front of expected depth.
- `split_name=test` is rejected at both artifact and CLI entry points.

## Outputs

`loc_gs.scripts.build_rendered_feedback_observations` writes:

- `rendered_feedback_support.pt`
- `manifest.json`
- `command.txt`
- `metrics_summary.json`
- `split_audit.json`
- `git_status.txt`

The `.pt` payload includes `support_score`, `hard_negative_risk`,
`artifact_risk`, `observed_count`, and `positive_observed_count`, so it is shaped
for `export_ulfloc_solver_feedback`.

## Caveat

This is a data-contract and attribution-math builder only. Renderer integration
still needs a separate exporter that writes full rendered RGB/depth observations
from train/self-map views.
