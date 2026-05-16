# Loc-GS Paper Evidence Audit

Date: 2026-05-16

## Audit Scope

This audit checks whether the current repository can support three submission
claims using `docs/experiment_board.md`, available manifests, available
summaries, and the research/protocol documents.

Machine-readable artifacts currently visible in this checkout:

- `docs/cambridge_branch_manifest_*.json`: historical branch-selection
  manifests.
- `results/smoke_trainprobe-output_stdloc_map_cambridge_spgs_GreatCourt_stream_stable2-20260512_051306/summary.json`:
  one smoke summary, diagnostic only.
- No checked-in `split_audit.json` or `metrics_summary.json` for the full
  Cambridge tables.

The generated board from checked-in `results/` marks the smoke run as
`diagnostic` and not paper-safe because manifest, split audit, and audit-bundle
files are missing. The stronger full-Cambridge numbers are documented in README
and mainline notes, but their machine-readable paper-safety bundle is not
present in this checkout.

## Claim 1

Claim: feature-field reconstruction objective and geometric localization
objective are mismatched.

Current evidence:

- `docs/loc_gs_lff_scenematch_mainline_20260512.md` records that protected LFF
  residuals and self-map quality gates can improve localization while simple
  descriptor export or eval-time priors yield only small gains.
- The same note records negative eval-time probes: LoFTR replacement,
  q-score filtering, tighter reprojection thresholds, and SceneMatchNet variants
  do not consistently improve full Cambridge under one recipe.
- README states that the strongest evidence comes from
  reconstruction-time/self-localization feedback rather than inference-time
  score nudges.

Missing evidence:

- A paper-safe table linking reconstruction metrics to pose metrics per scene.
- Failure taxonomy showing cases where descriptor reconstruction looks good but
  PnP/dense localization fails.
- Checked-in `metrics_summary.json`, `manifest.json`, and `split_audit.json`
  for the claim-level full Cambridge runs.

Status: plausible but not fully paper-audited in this checkout.

## Claim 2

Claim: virtual self-localization feedback can produce useful per-Gaussian or
per-landmark feature-selection and hard-negative residual reconstruction
signals.

Current evidence:

- The repository now contains `loc_gs/feedback/*`,
  `loc_gs/scripts/export_feedback_bank_from_cambridge.py`,
  `loc_gs/losses/pose_information_selection.py`, and
  `loc_gs/losses/hard_negative_descriptor.py`.
- Unit tests validate feedback-bank save/load, hard-negative labels, landmark
  reliability, selector losses, and disabled-by-default hard-negative residual
  training hooks.
- Existing mainline docs report that self-map quality selects useful
  scene-level candidates for OldHospital, ShopFacade, and StMarysChurch.

Missing evidence:

- Full feedback-bank exports from real self-map/calibration artifacts with
  manifests and split audits.
- A full Cambridge training run using the new pose-information selector or
  hard-negative residual loss.
- Per-Gaussian/per-landmark analysis showing selected or suppressed points,
  hard-negative rates, and downstream pose deltas.

Status: implemented and testable as infrastructure, but not yet empirically
proven as a paper claim.

## Claim 3

Claim: feedback-guided reconstruction improves pose accuracy and recall without
changing inference-time PnP pipeline or using per-query branch selection.

Current evidence:

- README records native STDLoc parity at
  `9.127 cm / 0.156 deg, R5 0.3712, R2 0.1331`.
- README and mainline docs record self-map quality gate diagnostic at
  `8.682 cm / 0.151 deg, R10 0.5947, R5 0.4144, R2 0.1459`.
- Native-backed soft locability and LFF descriptor export ablations show
  smaller single-backend gains, e.g. unified gated LFF at
  `9.087 cm / 0.157 deg, R5 0.3704, R2 0.1323`.
- The docs explicitly state that real-query localization remains a single path:
  matching, OpenCV PROSAC/RANSAC PnP, and STDLoc-style dense refinement.

Missing evidence:

- Checked-in split audits proving self-map/calibration ids are disjoint from
  test ids for the selected full-Cambridge runs.
- Checked-in command, git diff/status, manifest, and metrics summaries for the
  self-map quality selector and native-backed ablations.
- Per-scene failure-mode table for GreatCourt and KingsCollege, where feedback
  candidates are weak or rejected.

Status: strongest documented claim, but current checkout still needs
paper-safety bundles before it should be treated as final submission evidence.

## Excluded Or Non-Main Experiments

These should not be used as main claims without new paper-safe evidence:

- Historical guarded train-calibrated branch selection: diagnostic fallback.
- Query-like static calibrated matchability: ablation; full-test generalization
  is not clean.
- SceneMatchNet pair scoring: useful diagnostic; current fixed weights trail
  residual/default full-split results.
- Feedback detector alone: implemented but not yet a mainline gain.
- Oracle PROSAC ordering or GT reprojection sorting: upper-bound diagnostic only.
- LoFTR/rendered-image replacement: negative replacement route.
- The checked-in GreatCourt smoke summary under `results/`: diagnostic only,
  missing manifest and split audit.

## Risk Assessment

Test leakage risk:

- The protocol forbids using test query images, poses, descriptors, or results
  for calibration, self-map reliability, feedback banks, and model selection.
- Current full-claim docs describe train/self-map selection, but the checked-in
  machine-readable split audits are missing. Risk remains `unknown` until those
  audits are generated.

Evaluator modification risk:

- Native STDLoc parity is documented and wrappers exist under
  `loc_gs.stdloc_native`.
- Any result from `eval_cambridge_hybrid` must be clearly separated from native
  STDLoc parity unless `rho=0` or disabled-feedback parity is shown.

Cherry-picking risk:

- Self-map quality gate is scene-level candidate validation. It is defensible
  only if fixed before test deployment and reported with rejected scenes.
- Per-scene tables are required; aggregate-only reporting is insufficient.

Overclaiming risk:

- New feedback-bank, selector, and hard-negative residual code is unit-tested
  infrastructure. It is not yet full-Cambridge empirical evidence.

## Next Minimal Experiment Set

1. Generate audit bundles for native STDLoc parity:
   `manifest.json`, `command.txt`, `metrics_summary.json`, `split_audit.json`,
   and git status/diff for each scene and aggregate.
2. Generate feedback-bank exports from existing self-map/calibration pair caches
   with `split_name` not equal to `test`.
3. Re-run or re-materialize the self-map quality selector with the new manifest
   and split-audit helpers.
4. Run one full Cambridge native-backed LFF export ablation with the same audit
   bundle.
5. Run a minimal full-split or q80-to-full staged experiment using the new
   hard-negative residual/selector infrastructure only after feedback banks are
   audited.
6. Build a failure taxonomy from per-scene results and feedback-bank summaries.

## Suggested Tables

Main table:

- Native STDLoc parity.
- Self-map quality-gated reconstruction-time feedback, if audit passes.
- Best native-backed single-backend compatibility ablation.

Ablation table:

- Soft locability prior variants.
- LFF descriptor export variants.
- Unified gated LFF selector modes.
- Hard-negative residual/pose-info selector once real feedback-bank training
  results exist.

Failure taxonomy table:

- Scene.
- Failure class.
- Native metric.
- Feedback candidate metric.
- Self-map median/R@5.
- Hard-negative rate.
- Reason for accepted/rejected paper role.

Diagnostic-only table:

- SceneMatchNet variants.
- Feedback detector-only route.
- Oracle ordering.
- LoFTR/rendered replacement.
- Checked-in smoke runs.
