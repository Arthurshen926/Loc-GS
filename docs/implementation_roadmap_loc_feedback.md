# Loc-GS Localization-Feedback Roadmap

This roadmap defines the next development steps. It is a plan only; it does not
authorize evaluator changes or new method claims without paper-safe evidence.

## Starting Point

The current README mainline is:

```text
STDLoc/PLY descriptor backbone
  -> virtual self-localization feedback
  -> protected tiny descriptor residual, capped inside the PLY trust region
  -> optional diagnostics: feedback detector / SceneMatchNet pair scoring
  -> one OpenCV PROSAC PnP path
  -> STDLoc-style dense refinement
```

Native STDLoc parity is the baseline:

```text
9.127 cm / 0.156 deg, R5 0.3712, R2 0.1331
```

The strongest current self-map quality-gate diagnostic is:

```text
8.682 cm / 0.151 deg, R5 0.4144, R2 0.1459
```

The immediate goal is to turn this diagnostic signal into auditable training
and export artifacts without using test data and without adding per-query
branch selection.

## 1. Self-Localization Feedback Bank

Create `loc_gs/feedback/` with:

- `schema.py`: typed records for episodes, matches, poses, and summaries.
- `io.py`: save/load/summarize feedback banks with manifest metadata.
- `labels.py`: derive inlier, hard-negative, landmark reliability, and
  baseline-relative scene reliability labels.

Inputs:

- self-map or train-derived virtual localization episodes;
- scene name, query/source ids, split name, command, checkpoint, map, git
  commit;
- match-level fields such as keypoint xy, landmark id, Gaussian id, descriptor
  score, detector score, match rank, PnP inlier, reprojection error, depth
  consistency, visibility, and pose error.

Outputs:

- `feedback_bank.jsonl` or `feedback_bank.pt`;
- `feedback_summary.json`;
- `manifest.json`.

Acceptance:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest -q tests/test_feedback_bank.py
/root/miniconda3/envs/cybersim_agent/bin/python -m compileall loc_gs
```

## 2. Export Existing Cambridge Signals

Add `loc_gs/scripts/export_feedback_bank_from_cambridge.py` to convert existing
query-like calibration, SceneMatchNet pair caches, and self-map reliability
summaries into the feedback bank format.

Inputs:

- `--scene`
- `--pair_cache`
- `--selfmap_summary`
- `--output_path`
- `--split_name`
- optional `--baseline_summary`
- `--dry_run`

Outputs:

- feedback bank file;
- `feedback_summary.json`;
- `manifest.json`.

The exporter must not change existing experiment paths or metric meanings. A
dry run must show the expected schema even when real Cambridge files are
missing.

Documentation:

- `docs/feedback_bank_export.md` with source paths, command examples, expected
  files, and leakage rules.

## 3. Pose-Information-Aware Feature Selector

Add `loc_gs/losses/pose_information_selection.py` with tensor-only,
batch-friendly objectives:

- `pnp_information_proxy_loss`
- `hard_negative_suppression_loss`
- `selection_budget_loss`
- `coverage_regularization_loss`
- `combined_pose_info_selector_loss`

The selector should reward high inlier probability and high pose-information
proxy, suppress hard negatives, enforce a selection budget, and maintain
spatial/visibility coverage. It should use stable proxies such as trace or
logdet of a `J^T J`-style information matrix; it must not require a
differentiable real PnP solver.

Acceptance:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest -q tests/test_pose_information_selection.py
/root/miniconda3/envs/cybersim_agent/bin/python -m compileall loc_gs
```

## 4. Protected Descriptor Residual From Hard Negatives

Add `loc_gs/losses/hard_negative_descriptor.py` with:

- `hard_negative_margin_loss(query_desc, pos_desc, neg_desc, ...)`
- `residual_trust_region_loss(base_desc, residual_desc, alpha_max, ...)`
- `feedback_weighted_descriptor_loss(...)`

Integrate disabled-by-default training arguments into the appropriate Cambridge
training entry, currently `loc_gs/scripts/train_cambridge_hybrid.py` unless a
more focused helper exists:

- `--feedback_bank_path`
- `--enable_hard_negative_residual_loss`
- `--hard_negative_margin`
- `--hard_negative_loss_weight`
- `--residual_trust_region_weight`

Default behavior must remain unchanged. If no feedback bank is provided, the
loss is disabled. The residual alpha cap must not exceed the protected residual
trust region, and `rho=0` or `alpha=0` must recover baseline-compatible
behavior.

Documentation:

- `docs/hard_negative_residual.md`, explicitly labeling this as training-time
  feedback rather than inference-time reranking.

## 5. Native STDLoc-Compatible Export And Eval Audit

Add or enhance `loc_gs/export/manifest.py` so every candidate field export or
evaluation can write:

- `manifest.json`
- `command.txt`
- `git_diff.patch` or `git_status.txt`
- `metrics_summary.json`
- `split_audit.json`

`split_audit.json` must check:

- self-map/calibration image ids are disjoint from test image ids;
- feedback bank `split_name` is not `test`;
- quality gates are not per-query branch selectors.

Missing split information must produce `audit_status = "unknown"`, not
`"passed"`.

Acceptance:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest -q tests/test_manifest_split_audit.py
/root/miniconda3/envs/cybersim_agent/bin/python -m compileall loc_gs
```

## 6. Failure Taxonomy And Evidence Report

Use the experiment board and manifests to produce:

- `docs/experiment_board.md`: table of runs marked `main_candidate`,
  `ablation`, `diagnostic`, or `rejected`;
- compact JSON board output from `loc_gs/scripts/update_experiment_board.py`;
- `docs/paper_evidence_audit.md`: claim-by-claim evidence, missing evidence,
  excluded experiments, leakage/evaluator/cherry-picking risks, next minimal
  experiments, and proposed main/ablation/failure tables.

The evidence report must check per-scene metrics and failure modes. It must not
promote diagnostic results to main claims.

## Development Order

1. Land the feedback schema and IO first.
2. Export existing Cambridge signals into that schema.
3. Add selector and hard-negative losses with synthetic unit tests.
4. Wire descriptor residual training disabled by default.
5. Add manifest and split audits around export/eval.
6. Build the experiment board and evidence audit from the resulting artifacts.

This order keeps each step testable without full Cambridge runs and prevents
new training code from appearing before the split/audit contract exists.
