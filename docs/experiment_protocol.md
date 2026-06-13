# Cambridge Experiment Protocol

## Data Splits

Cambridge train and test image lists are read from the scene directories:

- Train split: `dataset_train.txt`
- Test split: `dataset_test.txt`

`loc_gs/data/cambridge_dataset.py` uses these files when loading
`CambridgeHybridDataset`. When it loads `cameras.json`, it filters the records
by the corresponding split file if the file exists. This split behavior is the
contract for Loc-GS training, self-map generation, calibration, and evaluation.

## Allowed Split Usage

### Full Train Split

Allowed uses:

- STDLoc or Loc-GS reconstruction and training.
- SuperPoint teacher extraction only for explicitly labeled diagnostic
  ablations. It is not part of the current ULF sparse solver-feedback mainline.
- Training-time localization rehearsal.
- Hard-negative mining when labels come from train/self-map episodes only.
- Feedback bank construction with `split_name` set to `train`,
  `selfmap_train`, `calibration`, or another non-test label.

Not allowed:

- Selecting hyperparameters from test query performance.
- Rewriting test poses or test image ids.

### Self-Localization Split

Self-localization episodes may use rendered, perturbed, interpolated, neighbor,
or held-out train-derived views with known geometry. These episodes are allowed
for:

- self-map reliability summaries;
- feedback bank records;
- per-Gaussian/per-landmark LSF support labels;
- solver-tuple admissibility diagnostics;
- dense residual support labels;
- hard-negative mining;
- failure taxonomy diagnostics.

They must write a manifest recording source scene, source split, view sampling
policy, command, checkpoint, map, and git commit. The split name must not be
`test`.

### Calibration Split

Calibration uses train-derived or self-map views only. It can tune fixed
thresholds or reliability scalars before deployment. It cannot consume test
query images, poses, features, or result files.

Every calibration artifact must record:

- `scene`
- `split_name`
- source image ids or a reproducible source list
- command
- checkpoint/map paths
- parameters
- git commit

### Test Split

The Cambridge test split is evaluation-only. Allowed operations:

- Run one fixed localization path on each query.
- Compute final pose metrics.
- Produce per-scene and aggregate summaries.
- Run audits that read test ids only to prove disjointness.

Forbidden operations:

- Training, calibration, model selection, self-map reliability, selector labels,
  feedback bank generation, or hard-negative mining.
- Per-query branch selection.
- Any GT-assisted candidate ordering except explicitly labeled oracle
  diagnostics that are excluded from main claims.

## Main Metrics

Report full Cambridge macro summaries and per-scene summaries:

- Median translation error in centimeters.
- Median rotation error in degrees.
- Recall at 10 cm / 5 deg (`R@10`).
- Recall at 5 cm / 5 deg (`R@5`).
- Recall at 2 cm / 2 deg (`R@2`).

The active starting point is native STDLoc parity plus the LSF audit gates.
Older residual, locability-prior, quality-gate, SceneMatchNet, LoFTR, oracle,
and selector-sweep rows are retained only as ablations or diagnostics unless
they are rerun as one fixed LSF recipe with full audit material.

Current reference rows:

| Variant | Status | Median | R@10 | R@5 | R@2 |
| --- | --- | ---: | ---: | ---: | ---: |
| Native STDLoc parity | baseline | 9.127 cm / 0.156 deg | 0.5761 | 0.3712 | 0.1331 |
| Self-map quality gate | diagnostic only | 8.682 cm / 0.151 deg | 0.5947 | 0.4144 | 0.1459 |
| Native-backed soft locability, R5-tempered | historical ablation | 9.124 cm / 0.157 deg | not recorded | 0.3731 | 0.1300 |
| LFF descriptor export, alpha 0.10 | historical ablation | 9.097 cm / 0.157 deg | not recorded | 0.3704 | 0.1323 |
| Unified gated LFF, reliability boost, alpha 0.10 | historical ablation | 9.087 cm / 0.157 deg | not recorded | 0.3704 | 0.1323 |

If a source summary lacks `R@10`, mark it missing rather than inferring it.

## Result Categories

Every experiment summary should use one of these labels:

- `main_candidate`: one fixed LSF recipe with feedback-bank-v2 audit, split
  audit, native parity control, train-dev gates, and single-path evaluation.
- `ablation`: audited residual descriptor, dense mask, solver metric, sampling,
  or selector variant that is not the main fixed recipe.
- `diagnostic`: quality gate, oracle ordering, SceneMatchNet, LoFTR/DIM,
  scalar sweep, support-count guard, or failure-mechanism report.
- `rejected`: any method that uses test-query signal for training/selection,
  modifies evaluator behavior, changes metrics, or fails paper-safety audits.

## Auxiliary Diagnostics

Use these only to explain behavior, not to replace the main metrics:

- Sparse PnP median and recalls.
- Dense refinement delta from sparse PnP.
- Inlier ratio and PnP success.
- Reprojection error distribution.
- Landmark visibility, depth consistency, detector score, descriptor score.
- Hard-negative rate by scene and by landmark.
- Self-map reliability `rho` and self-map median/R@5.
- Feedback-bank-v2 audit status and image-group coverage.
- Viable solver-tuple mass, pose-information logdet/min-eigenvalue proxy, and
  ambiguity risk.
- Dense improved/worsened query counts and dense residual keep ratio.
- Failure taxonomy by no-match, repeated-structure, low-visibility,
  pose-refinement divergence, and map/checkpoint mismatch.

## Required Commands

Quick non-GPU static check:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m compileall loc_gs
```

Native STDLoc parity wrapper help:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.eval_stdloc_native --help
```

Hybrid Cambridge evaluator help:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.eval_cambridge_hybrid --help
```

Agent-friendly status and smoke checks, when `locgsctl` is available:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.locgsctl status
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.locgsctl smoke --scene ShopFacade --dry-run
```

Expected behavior for missing data or checkpoints: fail clearly with the missing
path. Do not download, install, or start a long Cambridge run automatically.
