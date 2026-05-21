# LSF Implementation Status 2026-05-18

This note records the current implementation state for the Loc-GS
Localization Support Field mainline. It is not a paper-facing result table.

## Implemented

1. LSF evaluation protocol and report utilities
   - `loc_gs/eval/locgs_metrics.py`
   - `loc_gs/eval/timing.py`
   - `loc_gs/scripts/summarize_lsf_eval.py`
   - `docs/lsf_evaluation_protocol_202605.md`

2. Alpha-composition reliability proxy diagnostics
   - `loc_gs/diagnostics/alpha_composition.py`
   - `loc_gs/diagnostics/rendered_reliability.py`
   - `loc_gs/scripts/build_alpha_reliability_cache.py`
   - exact raster weights are not assumed; current caches must be marked
     `composition_proxy=true`.

3. Solver-level tuple diagnostics
   - `loc_gs/diagnostics/pose_information.py`
   - `loc_gs/diagnostics/ambiguity_metrics.py`
   - `loc_gs/diagnostics/solver_tuple_bank.py`
   - `loc_gs/scripts/build_solver_tuple_bank.py`
   - `loc_gs/scripts/compare_solver_tuple_banks.py`

4. Evidence-gated solver-aware sparse LSF export
   - `loc_gs/stdloc_native/evidence_gate.py`
   - `loc_gs/stdloc_native/solver_admissibility.py`
   - `loc_gs/stdloc_native/solver_aware_resampling.py`
   - `loc_gs/stdloc_native/query_conditioned_support.py`
   - `loc_gs/scripts/export_lsf_solver_aware_map.py`
   - `loc_gs/scripts/build_query_conditioned_solver_constraints.py`
   - The exporter now supports `--solver_admissibility_path`, so
     query-conditioned replacement constraints can reject edits before export.

5. Dense residual support mask/filter
   - `loc_gs/dense_support/rendered_support_mask.py`
   - `loc_gs/dense_support/dense_match_filter.py`
   - `loc_gs/scripts/eval_dense_support_mask.py`

## Smoke Diagnostics

All diagnostics below use self-map/train feedback caches, keep a single native
STDLoc-compatible map path, and do not use test query data for training or
selection. Current tuple banks use `synthetic_chunks_512` because the old
episode cache does not store real per-row `query_id`.

| Scene | Candidate | Edits | Support Delta | Viable Mass Delta | Mean Logdet Delta | Report |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| OldHospital | evidence gate, pose >= 0.00 | 29 | +77 | +4.7701 | -1.2988 | `docs/solver_aware_oldhospital_quick_20260518.md` |
| OldHospital | evidence gate, pose >= 0.05 | 12 | +53 | +5.3021 | -1.1979 | `docs/solver_aware_oldhospital_pose005_quick_20260518.md` |
| OldHospital | evidence gate, pose >= 0.20 | 5 | +28 | +6.9819 | -0.3551 | `docs/solver_aware_oldhospital_pose020_quick_20260518.md` |
| OldHospital | evidence gate, pose >= 0.50 | 2 | +14 | +1.5510 | -0.0656 | `docs/solver_aware_oldhospital_pose050_quick_20260518.md` |
| StMarysChurch | evidence gate, pose >= 0.50 | 2 | +27 | +3.4953 | -1.3899 | `docs/solver_aware_stmaryschurch_pose050_quick_20260518.md` |
| StMarysChurch | evidence gate, pose >= 0.50, max edit 1 | 1 | +15 | +1.8332 | -0.5732 | `docs/solver_aware_stmaryschurch_pose050_edit1_quick_20260518.md` |

## Train-Q80 Pose Smoke

These runs use `eval_split=train --max_test_cameras=80` and are for
development validation only. They do not use Cambridge test queries.

| Scene | Method | Sparse Median cm/deg | Sparse R5/R2 | Dense Median cm/deg | Dense R5/R2 |
| --- | --- | ---: | ---: | ---: | ---: |
| OldHospital | native12288 | 12.749 / 0.257 | 0.0875 / 0.0000 | 8.276 / 0.182 | 0.2875 / 0.0000 |
| OldHospital | LSF pose>=0.50, 2 edits | 12.750 / 0.257 | 0.0875 / 0.0000 | 8.276 / 0.182 | 0.2875 / 0.0000 |
| StMarysChurch | native12288 | 9.742 / 0.356 | 0.2625 / 0.0500 | 5.407 / 0.191 | 0.4750 / 0.1375 |
| StMarysChurch | LSF pose>=0.50, 1 edit | 9.641 / 0.356 | 0.2625 / 0.0500 | 5.407 / 0.191 | 0.4750 / 0.1375 |

Observed deltas:

- OldHospital: sparse median translation changes by +0.001cm; dense metrics
  are unchanged.
- StMarysChurch: sparse median translation improves by 0.101cm, but recall and
  dense metrics are unchanged.
- This is a parity/safety result, not a positive paper claim.

## Interpretation

The current smoke evidence supports the revised claim direction:

- Support count and positive self-map evidence are insufficient selection
  objectives.
- Evidence-gated local edits can increase support and viable tuple mass, but
  they may introduce additional low-condition tuples, reducing mean tuple
  logdet.
- Therefore the next paper-safe LSF export must use query-conditioned
  `solver_admissibility` constraints generated from feedback banks with real
  `query_id`, not just global scalar thresholds.

## Not Yet Paper-Facing

- No full Cambridge test evaluation has been run for these LSF maps.
- The old feedback cache lacks real `query_id`, so solver tuple grouping is a
  smoke diagnostic only.
- Pose accuracy improvement is not claimed here.
- The current positive evidence should be treated as mechanism evidence:
  viable tuple mass can be increased under a single-path native-compatible map,
  while the negative logdet signal explains why naive support selection is not
  enough.

## Next Required Step

Regenerate self-map feedback banks with real `query_id`, candidate landmark ids,
candidate cosine, reprojection error, depth/visibility flags, PnP inlier flags,
and dense refinement outcome. Then derive `solver_admissibility_path` files
before export and rerun OldHospital/StMarysChurch train-q80 plus shifted
train-s3/q75 smoke evaluations.

The intended sparse LSF export path is now:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python \
  -m loc_gs.scripts.build_query_conditioned_solver_constraints \
  --episode_cache output/.../selfmap_episode_v1.pt \
  --source_idx output/.../native12288/SCENE/detector/sampled_idx.pkl \
  --output_json output/.../SCENE_solver_admissibility.json \
  --num_gaussians NUM_GAUSSIANS \
  --hard_query_ids HARD_QUERY_IDS_FROM_TRAIN_SELF_MAP \
  --reprojection_threshold_px 3.0 \
  --score_threshold 0.7 \
  --min_candidate_positive 0.5 \
  --min_logdet_delta 0.0

/root/miniconda3/envs/cybersim_agent/bin/python \
  -m loc_gs.scripts.export_lsf_solver_aware_map \
  --source_map output/.../native12288/SCENE \
  --selector_path output/.../selector.pt \
  --positive_support_path output/.../positive_support.pt \
  --hard_negative_risk_path output/.../hard_negative_risk.pt \
  --pose_utility_path output/.../pose_utility.pt \
  --safe_core_path output/.../safe_core.pt \
  --solver_admissibility_path output/.../SCENE_solver_admissibility.json \
  --output_map output/.../SCENE_lsf_solver_admissible \
  --keep_source
```

This path keeps query-time inference single-path. The admissibility JSON is a
mapping-stage constraint generated only from self-map/train feedback; it must
not be generated from Cambridge test query results.
