# ULF-Loc Solver-Feedback Migration Status 2026-06-02

## Goal

Move LSF-Loc from a Loc-GS/STDLoc-local sparse selector into a stronger ULF-Loc
backbone. The first migration step is intentionally narrow:

```text
train/self-map solver traces
-> per-Gaussian solver-feedback weights
-> ULF-Loc keypoint-consensus sampling and geometry-weighted feature fusion
```

This keeps official ULF-Loc inference unchanged unless `solver_feedback.enabled`
is explicitly enabled in the ULF config.

## Implemented

- Added `/root/ULF-Loc/utils/solver_feedback.py`.
- Added optional solver-feedback hooks in:
  - `/root/ULF-Loc/utils/keypoints_sample.py`
  - `/root/ULF-Loc/utils/gsfeature_fusion.py`
- Added a default disabled config block in `/root/ULF-Loc/configs/ulfloc_cambridge.yaml`.
- Added `loc_gs.scripts.export_ulfloc_solver_feedback` to convert Loc-GS
  `solver_consensus_support.pt` artifacts into ULF-readable
  `solver_feedback.pkl`.
- Added focused tests:
  - `tests/test_ulfloc_solver_feedback.py`
  - `tests/test_export_ulfloc_solver_feedback.py`

## Safety Rules

- `solver_feedback.enabled: False` preserves native ULF-Loc behavior.
- `sampling_alpha: 0.0` preserves native ULF keypoint-consensus sampling scores.
- `fusion_alpha: 0.0` preserves native ULF geometry-weighted feature fusion.
- `split_name == "test"` is rejected by both the Loc-GS exporter and the
  ULF-Loc loader.
- The feedback artifact is a train/self-map sidecar and must not be built from
  Cambridge official test queries, descriptors, features, poses, or result
  labels.

## Artifact Pipeline

Build solver-consensus support from an audited `feedback_bank_v2`:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.build_solver_consensus_support \
  --feedback_bank /path/to/feedback_bank.jsonl \
  --output_dir output/ulfloc_solver_feedback/<Scene>/support \
  --num_gaussians <gaussian_count>
```

Export the ULF-compatible artifact:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.export_ulfloc_solver_feedback \
  --scene <Scene> \
  --solver_support output/ulfloc_solver_feedback/<Scene>/support/solver_consensus_support.pt \
  --output_dir output/ulfloc_solver_feedback/<Scene>/ulf_feedback \
  --alpha 0.1 \
  --risk_penalty 1.0 \
  --min_weight 0.25 \
  --max_weight 1.75
```

Copy or symlink `solver_feedback.pkl` into the ULF-Loc scene output directory,
then enable:

```yaml
solver_feedback:
  enabled: True
  artifact_path: solver_feedback.pkl
  sampling_alpha: 0.1
  fusion_alpha: 0.1
```

## First Experiment Gate

Run only train/self-map or train-dev first. Do not use official test for tuning.

Compare:

```text
ULF native
ULF + solver-feedback sampling only
ULF + solver-feedback fusion only
ULF + solver-feedback sampling + fusion
```

Required report:

```text
median TE/RE
R@50/15/10/5/2
P90/P95 TE
CVaR10 TE
TE>1m / TE>5m
candidate regression counts
sparse inliers / inlier ratio
runtime
map size
split audit
```

Promotion gate:

```text
sparse median TE improves by >= 0.5 cm, or R@10/R@5 improves by >= 0.5 pp
P90/P95 do not increase
OldHospital and StMarysChurch do not regress
no test split in any feedback artifact
```

## Current Blocker

The previous Loc-GS/STDLoc solver-consensus support artifacts are not directly
usable as ULF-Loc solver feedback. They are indexed by the Gaussian ids of the
Loc-GS/STDLoc map, while ULF-Loc creates its own Gaussian model and therefore a
different Gaussian id space. The ULF loader correctly rejects mismatched
`landmark_weights` lengths.

Required next step:

```text
train official ULF-Loc map/checkpoint
-> run ULF-native train/self-map sparse localization capture
-> build feedback_bank_v2 in ULF Gaussian id space
-> build solver_consensus_support.pt in ULF Gaussian id space
-> export solver_feedback.pkl
-> rerun ULF K.C. sampling / GWFF with solver feedback
```

The synthetic `selfmap_train_smoke` artifact used during integration only proves
that hooks are wired and does not support any accuracy claim.

## Environment Compatibility Notes

- The active env has `gsplat 1.2.0`, which lacks `rasterization_2dgs`. The ULF
  renderer now imports 2DGS rasterization optionally and raises only when
  `gaussian_type=2dgs` is actually used.
- `gsplat 1.2.0` returns 1D `radii`/visibility tensors for 3DGS. The ULF
  renderer and training loop now handle both 1D and `[N, 1]` shapes.
- The local Cambridge `processed` images and `masks.pkl` can have different
  spatial sizes. ULF sampling/fusion now resizes masks with nearest-neighbor
  interpolation before applying them.
- Full-resolution feature fusion can OOM with the original batch size of 10.
  `sample.fusion_batch_size` is now configurable; the solver-feedback config
  uses `1`.
