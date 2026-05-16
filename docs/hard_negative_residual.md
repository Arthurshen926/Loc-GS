# Hard-Negative Descriptor Residual

This module is a first training-time hook for feedback-bank hard negatives. It
is not an inference-time reranker and does not change the Cambridge evaluator.

## What It Adds

`loc_gs/losses/hard_negative_descriptor.py` provides:

- `hard_negative_margin_loss(query_desc, pos_desc, neg_desc, ...)`
- `residual_trust_region_loss(base_desc, residual_desc, alpha_max, ...)`
- `feedback_weighted_descriptor_loss(...)`

`loc_gs/scripts/train_cambridge_hybrid.py` exposes disabled-by-default
arguments:

- `--feedback_bank_path`
- `--enable_hard_negative_residual_loss`
- `--hard_negative_margin`
- `--hard_negative_loss_weight`
- `--residual_trust_region_weight`

Defaults keep existing behavior unchanged:

```text
feedback_bank_path = ""
enable_hard_negative_residual_loss = false
hard_negative_loss_weight = 0
residual_trust_region_weight = 0
```

## Training Hook

When explicitly enabled with a non-test feedback bank, the training entry loads
hard-negative `matched_gaussian_id` values from the bank and samples those
Gaussians for the protected residual trust-region loss. The active cap is the
existing `--hybrid_residual_alpha_max`; this keeps the residual inside the same
STDLoc/PLY descriptor trust region used by the protected LFF path.

The training metrics include:

- `hard_negative_residual`
- `hard_negative_residual_samples`

If no feedback bank is provided, or if the enable flag/weights are inactive,
these metrics remain zero and no feedback-bank path is loaded.

## Safety Rules

- Feedback banks with `split_name=test` are rejected.
- This is training-time feedback only.
- Query-time inference remains one path: descriptor matching, OpenCV
  PROSAC/RANSAC PnP, and STDLoc-style dense refinement.
- `rho=0`, `alpha=0`, or disabled feedback settings must remain
  baseline-compatible.

## Minimal Check

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest -q tests/test_hard_negative_descriptor.py
/root/miniconda3/envs/cybersim_agent/bin/python -m compileall loc_gs
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.locgsctl smoke --scene ShopFacade --dry-run
```
