# Feedback Bank Export

`loc_gs.scripts.export_feedback_bank_from_cambridge` converts existing
Cambridge self-localization artifacts into the `loc_gs.feedback` feedback-bank
format. It does not change existing experiment paths, summary metrics, SceneMatchNet
evaluation logic, or STDLoc evaluator behavior.

## Source Artifacts

The exporter is designed around artifacts produced by:

- `loc_gs.scripts.calibrate_landmark_matchability`
  - pair cache: `scene_match_pairs.pt`
  - sidecar calibration summary: `stdloc_bank.json`
- `loc_gs.localization.scene_matcher`
  - pairwise caches with `query_desc`, `landmark_desc`, `cosine`, `label`
  - listwise caches with `cosine`, `label`, `query_yx`, `landmark_id`,
    `candidate_mask`, and optional `reprojection_error`
- self-map reliability summaries from Cambridge eval directories:
  `summary.json` with `dense.median_te` / `dense.median_ae` or aliases

These artifacts must come from train, calibration, rendered rehearsal, or
self-map splits. They must not come from Cambridge test queries.

## Command

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.export_feedback_bank_from_cambridge \
  --scene ShopFacade \
  --pair_cache output/scenematch_pairs/lff/ShopFacade/scene_match_pairs.pt \
  --selfmap_summary output/stdloc_hybrid/ShopFacade_lff_refined_20260513/eval_selfmap/summary.json \
  --baseline_summary output/stdloc_native/results/parity/ShopFacade/summary.json \
  --output_path output/feedback_banks/ShopFacade \
  --split_name selfmap_train
```

## Outputs

When `--output_path` is a directory, the exporter writes:

- `feedback_bank.jsonl`
- `feedback_summary.json`
- `manifest.json`

When `--output_path` has `.json` or `.jsonl` suffix, that path is used for the
bank and the sidecars are written next to it.

The feedback bank embeds manifest metadata in the bank file. The standalone
`manifest.json` repeats the export context for audit tools.

## Dry Run

Use `--dry_run` to inspect the expected schema without requiring real files:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.export_feedback_bank_from_cambridge \
  --scene ShopFacade \
  --pair_cache missing_pairs.pt \
  --selfmap_summary missing_summary.json \
  --output_path output/feedback_banks/ShopFacade \
  --split_name selfmap_train \
  --dry_run
```

Dry run writes nothing and prints compact JSON describing inputs, output paths,
and feedback record fields.

## Mapping Rules

Listwise caches produce one feedback record per valid top-k candidate:

- `query_yx` becomes `keypoint_xy` as `[x, y]`.
- `landmark_id` becomes `matched_landmark_id`.
- `base_gaussian_id` or `gaussian_id`, when present, becomes
  `matched_gaussian_id`.
- `cosine` becomes `descriptor_score`.
- `query_score` becomes `detector_score`.
- `label` values in `[0, topk)` mark the matching candidate as
  `pnp_inlier = true` and `pnp_success = true`; `-1` or dustbin labels mark all
  candidates as outliers and the query as no-success.
- `reprojection_error` becomes `reprojection_error_px`.
- Dense self-map median translation/rotation are copied into pose-error fields.

Pairwise caches produce one feedback record per flat pair. Without keypoint or
landmark-id fields, synthetic ids are generated from the pair index.

## Safety Rules

- `--split_name test` is rejected.
- Missing real files fail clearly unless `--dry_run` is used.
- The exporter never launches Cambridge training or evaluation.
- It does not rewrite source pair caches, summaries, or metrics.
