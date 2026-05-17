# Loc-GS Research Contract

## Scope

Loc-GS studies localization-oriented Gaussian feature fields for Cambridge
camera relocalization. The active paper line is now a localization-aware
sampling field on top of native STDLoc Feature Gaussians:

```text
native STDLoc descriptor / geometry / radiance field
  -> train/rendered self-localization rehearsal
  -> per-Gaussian localization utility / selector field
  -> STDLoc-compatible sampling, ranking, and locability payload
  -> single-path feature matching, OpenCV PROSAC PnP, and dense refinement
```

For Cambridge, the source-of-truth baseline is native STDLoc executed through
the vendored `third_party/stdloc` code and the wrappers in
`loc_gs/stdloc_native/commands.py`. The current README records native STDLoc
parity at `9.127 cm / 0.156 deg, R5 0.3712, R2 0.1331`. The strongest current
self-map quality-gate diagnostic is `8.682 cm / 0.151 deg, R5 0.4144,
R2 0.1459`, but it must be described as reconstruction-time/self-map validation
of a selected feature field, not per-query inference branching.

## Core Claim

The defensible claim is:

Loc-GS turns reconstruction-time virtual self-localization feedback into a
paper-auditable localization utility / sampling field. The main method keeps
native STDLoc descriptors and the standard geometric backend fixed, then exports
the learned utility as STDLoc-compatible landmark sampling, ranking, and
locability payloads.

This claim requires all of the following:

- `rho=0` or feedback disabled reproduces native STDLoc parity.
- Feedback is produced only from training, self-map, calibration, or rendered
  rehearsal views that are disjoint from Cambridge test queries.
- Real test queries use one selected map/checkpoint and one evaluator path:
  matching, OpenCV PROSAC/RANSAC PnP, and STDLoc-style dense refinement.
- The main result does not require descriptor residuals; residual descriptor
  reconstruction is an ablation unless future full-split evidence makes it
  strictly better on median and recall.
- Metrics are reported as median translation/rotation plus recall at
  10 cm / 5 deg, 5 cm / 5 deg, and 2 cm / 2 deg, with per-scene breakdowns.

## Non-Claims

The project does not claim:

- A new official Cambridge benchmark split.
- A modified STDLoc metric, evaluator, or ground truth.
- A replacement of the STDLoc/SuperPoint descriptor backbone.
- A learned descriptor field that is the source of the current main gain.
- Test-time per-query model selection, branch selection, or oracle pose
  selection.
- That SceneMatchNet, LoFTR, static calibrated priors, or oracle ordering are
  the primary method. They are diagnostics unless a future full-split audit
  promotes them.

## Forbidden Claims

Do not write any paper, report, README, or result table that implies:

- Test query images, poses, descriptors, or results were used for training,
  calibration, self-map reliability, feedback bank construction, or model
  selection.
- The self-map quality gate is an inference-time query branch selector.
- Diagnostic oracle ordering or GT reprojection sorting is a deployable method.
- A result is paper-safe when its manifest, split audit, checkpoint, map, or
  command cannot be reconstructed.
- Improvements from changed thresholds, altered evaluators, deleted failures,
  or cherry-picked scenes are method gains.

## Reviewer Risks

The main reviewer risks are:

- Leakage: self-map, calibration, or feedback bank artifacts accidentally include
  test query ids.
- Evaluator drift: `eval_cambridge_hybrid` or a native export path diverges from
  the vendored STDLoc evaluator while being compared as if it were parity.
- Branch-selection ambiguity: scene-level quality gates are described too much
  like test-time policy selection.
- Cherry-picking: full Cambridge means improve while per-scene failures or
  strict recall regressions are hidden.
- Overclaiming diagnostics: SceneMatchNet, LoFTR, oracle ordering, and static
  matchability results are presented as the main method before they beat the
  native baseline under one fixed recipe.

## Evidence Standard

A paper-facing result must provide:

- `manifest.json`, `command.txt`, `metrics_summary.json`, and split audit.
- Native STDLoc control and `rho=0` or feedback-disabled parity control.
- Full Cambridge aggregate and per-scene metrics.
- Explicit label: `main_candidate`, `ablation`, `diagnostic`, or `rejected`.
- Explanation of any regression in median, R@10, R@5, R@2, or failure mode.
