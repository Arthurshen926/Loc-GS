# Loc-GS Mainline LSF, 2026-05-21

## One-Line Status

Loc-GS is being reset from a collection of selector, residual, prior, and
quality-gate diagnostics into one paper-facing direction:

> **Localization Support Field (LSF): use self-localization feedback during
> reconstruction/mapping to learn which sparse landmarks, solver tuples, and
> dense residuals are safe for the STDLoc geometric localization backend.**

The project is not ready for a strong SOTA claim yet. Current evidence is useful
because it exposes failure mechanisms and defines the next supervision target.

## Main Claim Candidate

The paper-facing claim should be:

> Feature reconstruction quality, support count, and unary matchability are not
> sufficient objectives for 3DGS relocalization. Loc-GS estimates localization
> support from self-localization episodes and feeds it back into sparse landmark
> selection and dense residual support, while keeping a single STDLoc-compatible
> inference path.

This claim is only valid if the final method uses one fixed recipe and one
single-path evaluator:

```text
query image
-> native STDLoc feature extraction / matching
-> OpenCV PROSAC/RANSAC PnP
-> STDLoc-style dense refinement with LSF support
-> final pose
```

Any poselib, multi-hypothesis, or alternative solver path is diagnostic unless a
task explicitly labels it as an ablation.  It must not replace the real-query
OpenCV path for paper-facing claims.

## Current Non-Claims

Do not present any of these as the main method:

- learned descriptor replacement;
- per-query branch selection;
- scene-level quality gate as deployed method;
- raw fixed dense locability prior;
- support-count preservation as a safety guarantee;
- qcov / churn / ret / hard-negative scalar sweeps;
- oracle ordering, SceneMatchNet, LoFTR, or LoFTR-style replacement unless a new
  full split proves them paper-safe.

## Architecture To Keep

### Stage 1: Native STDLoc Base

Keep native STDLoc descriptors, detector, sampled landmark ids, PnP, and dense
refinement as the parity anchor. Any disabled feedback setting must return to
native STDLoc behavior.

### Stage 2: Feedback Bank v2

Regenerate self-map feedback with real query grouping. A paper-facing feedback
bank must pass `audit_feedback_bank_v2` and include:

```text
scene
split_name != test
split_audit.audit_status = passed
schema_version = feedback_bank_v2
query_id_source = image_id or camera_id
query_id
image_id
keypoint_id
keypoint_xy
matched_landmark_id
matched_gaussian_id
descriptor_score / candidate cosine
match_rank
pnp_inlier
reprojection_error_px
depth_consistency / visibility when available
dense_transition or dense_delta_te_cm
```

Older banks with synthetic `query_000000` / `pair_000000` ids remain diagnostic
only.

Current implementation:

- `calibrate_landmark_matchability.py` now writes row-level `query_id`,
  `image_id`, `keypoint_id`, and `source_phase` into newly generated pair
  caches.
- `summarize_stage_transitions.py` now preserves `image_name` as the real
  query id when writing dense-transition labels.
- `export_feedback_bank_from_cambridge.py --schema_version feedback_bank_v2`
  refuses synthetic identity, requires a passed split audit, and can merge
  dense-stage supervision from `dense_transition_labels.json`.

Smoke artifacts generated on 2026-05-21:

```text
output/feedback_bank_v2_20260521/feedback_banks_smoke/OldHospital/feedback_bank.jsonl
output/feedback_bank_v2_20260521/feedback_banks_smoke/StMarysChurch/feedback_bank.jsonl
output/feedback_bank_v2_20260521/feedback_banks_smoke/ShopFacade/feedback_bank.jsonl
```

All three smoke banks pass `audit_feedback_bank_v2`. They are small data-chain
checks, not final feedback-bank training evidence.

Real-grouped artifacts generated on 2026-05-21:

```text
output/feedback_bank_v2_20260521/pair_caches_real_grouped/{OldHospital,StMarysChurch,ShopFacade}/scene_match_pairs.pt
output/feedback_bank_v2_20260521/feedback_banks_real_grouped/{OldHospital,StMarysChurch,ShopFacade}/feedback_bank.jsonl
output/feedback_bank_v2_20260521/feedback_banks_real_grouped/{OldHospital,StMarysChurch,ShopFacade}/feedback_bank_v2_audit.json
```

These caches were regenerated with 64 train views, 64 rendered rehearsal views,
`sample_limit=60000`, `topk=8`, and row-level `query_id/image_id/keypoint_id`.
The pair caches contain real image grouping rather than synthetic chunks:

```text
OldHospital:     60000 query-keypoint rows, 30 image groups, split audit passed
ShopFacade:      60000 query-keypoint rows, 30 image groups, split audit passed
StMarysChurch:   60000 query-keypoint rows, 32 image groups, split audit passed
```

The exported feedback banks pass v2 audit. The audit reports keypoint-level
`query_count` separately from real `image_group_count`; the current valid
candidate records cover 15 image groups per scene:

```text
OldHospital:     35803 records, 18849 keypoint queries, 15 image groups, audit passed
ShopFacade:     157335 records, 29897 keypoint queries, 15 image groups, audit passed
StMarysChurch:   11525 records,  8089 keypoint queries, 15 image groups, audit passed
```

Export command sketch:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python \
  -m loc_gs.scripts.export_feedback_bank_from_cambridge \
  --scene StMarysChurch \
  --pair_cache PATH/scene_match_pairs.pt \
  --selfmap_summary PATH/selfmap_summary.json \
  --dense_transition_labels PATH/dense_transition_labels.json \
  --output_path PATH/feedback_bank_v2 \
  --split_name selfmap_train \
  --schema_version feedback_bank_v2
```

### Stage 3: Solver-Aware Sparse LSF

Sparse LSF should not optimize support count alone. It must protect:

- hard-query viable tuple mass;
- pose-information logdet or min-eigenvalue proxy;
- spatial coverage;
- ambiguity / hard-negative risk;
- native STDLoc parity.

### Stage 4: Dense Residual Support LSF

Dense LSF should explain and reduce cases where dense refinement worsens a
reasonable sparse pose. Raw dense locability prior is only a teacher/diagnostic.
It should be distilled into a support readout that can suppress unreliable
locability where self-map risk says it hurts.

## Current Evidence

### Negative Evidence

Local sampled-id edits and support-count guards do not generalize strongly.
Recent StMarys train-dev splits showed sparse improvements can be erased by
dense refinement.

Supportguardall-ret90 mechanism reports have now been generated from existing
q80 diagnostics:

```text
output/lsf_reports/20260521_support_failure/oldhospital_supportguardall_ret90_mechanism.json
output/lsf_reports/20260521_support_failure/stmaryschurch_supportguardall_ret90_mechanism.json
```

Both are diagnostic-only because the old tuple banks used
`synthetic_chunks_512`, but both support the current problem statement: support
count increases while mean logdet(H) drops and dense median translation
regresses.

The same mechanism path has now been rebuilt from real-grouped feedback bank
v2, using `query_group_mode=feedback_bank_v2:image_id`:

```text
output/feedback_bank_v2_20260521/solver_tuple_v2_real_grouped/{OldHospital,StMarysChurch,ShopFacade}/solver_tuple_report.json
output/feedback_bank_v2_20260521/mechanism_real_grouped/{OldHospital,StMarysChurch,ShopFacade}/mechanism_summary.json
```

Dense-LSF high-confidence filtering is intentionally treated as a teacher /
candidate diagnostic, not as the final sparse selector. It reduces observed
landmark support and can also reduce solver tuple mass:

```text
OldHospital:
  support_count_delta      -67
  viable_tuple_mass_delta  -9.5766
  mean_logdet_H_delta      -0.1625

ShopFacade:
  support_count_delta       0
  viable_tuple_mass_delta  -18.5809
  mean_logdet_H_delta      -0.0295
  mechanism verdict: support-count preservation is insufficient

StMarysChurch:
  support_count_delta      -211
  viable_tuple_mass_delta  -12.7561
  mean_logdet_H_delta      -0.3087
```

This is not a positive accuracy claim. It is a mechanism result: dense-stage
support evidence alone is not enough for sparse map editing. The next LSF
candidate must combine dense support with solver admissibility or it can
discard correspondences that still matter for PnP tuple quality.

### Dense Prior Diagnostic

`dense_locability_prior_weight=0.05` helps StMarys hard-tail queries, but fixed
global use across q80 train-dev scenes is not a universal improvement:

```text
base macro median:     6.6393 cm
fixed prior_de005:     6.6902 cm  (+0.0509)
fixed prior_de005 R5:  unchanged
fixed prior_de005 R2:  unchanged
```

A train-dev self-map gate that only accepts the prior when strict recall is not
hurt and hard10 improves gives:

```text
gated diagnostic macro median: 6.6238 cm (-0.0155)
gated diagnostic macro R5:     +0.0025
gated diagnostic macro R2:     unchanged
```

This supports using self-map dense risk as a critic/teacher, not as a deployed
branch selector.

Dense LSF distillation v1 has been implemented as a feedback-bank target
generator. It aggregates per-Gaussian dense transition labels and
`dense_delta_te_cm` into:

```text
dense_lsf_target
dense_lsf_confidence
positive_weight / negative_weight
observed_count
selected_idx at a chosen keep threshold
```

Current real-grouped artifacts:

```text
output/feedback_bank_v2_20260521/dense_lsf_real_grouped/{OldHospital,StMarysChurch,ShopFacade}/dense_lsf_targets.pt
output/feedback_bank_v2_20260521/dense_lsf_highconf_real_grouped/{OldHospital,StMarysChurch,ShopFacade}/dense_lsf_targets.pt
```

High-confidence `keep_threshold=0.85` keeps:

```text
OldHospital:    1113 / 1697 observed Gaussians
ShopFacade:     3660 / 6040 observed Gaussians
StMarysChurch:    76 / 119 observed Gaussians
```

These targets are suitable as dense-support distillation labels or a dense
residual mask teacher. They should not be used alone as a sparse landmark
selector.

Round 2 dense-only guard: dense LSF artifacts must carry
`usage_scope=dense_residual_teacher_only` and `sparse_selector_safe=false`.
Using dense high-confidence `selected_idx` directly as a sparse sampled-map edit
is rejected unless it is combined with solver admissibility and passes the
fixed train-dev gates.

### Solver-Consensus Support Validation

Round 3 generated compact solver-consensus support artifacts from the
real-grouped `feedback_bank_v2` banks:

```text
output/feedback_bank_v2_20260521/solver_consensus_support_real_grouped/{OldHospital,ShopFacade,StMarysChurch}/solver_consensus_support.pt
output/feedback_bank_v2_20260521/solver_consensus_support_real_grouped/summary.json
output/feedback_bank_v2_20260521/solver_consensus_support_real_grouped/summary.md
```

This is a positive data-chain and mechanism validation, not a pose-accuracy
claim. It shows that audited real image groups can produce nonempty
per-Gaussian support candidates with the expected risk separation:

| Scene | Records | Image groups | Observed Gaussians | Selected @0.5 | Mean support observed | Mean support selected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OldHospital | 35803 | 15 | 1697 | 17 | 0.0911 | 0.7241 |
| ShopFacade | 157335 | 15 | 6040 | 75 | 0.0926 | 0.7031 |
| StMarysChurch | 11525 | 15 | 119 | 1 | 0.1490 | 0.5167 |

`export_lsf_solver_aware_map --solver_consensus_support_path ... --dry_run`
also accepts these artifacts while preserving `same_budget=true` and
`single_path_deployment=true`. On ShopFacade, the support gate accepted 75
candidates and rejected dense-worsened/high-risk candidates without launching a
Cambridge evaluation run.

### LSF v2 Native Proposal Train-Dev Validation

Round 4/5 implemented a paper-safe native proposal path that combines native
STDLoc reconstruction-time scores with audited v2 solver-consensus support and
risk. The proposal edits only the map payload and keeps one evaluator path:
feature matching, OpenCV PROSAC/RANSAC PnP, then STDLoc-style dense refinement.

Artifacts:

```text
output/feedback_bank_v2_20260521/solver_consensus_support_fullsize_20260521/{OldHospital,ShopFacade,StMarysChurch}/solver_consensus_support.pt
output/lsf_v2_native_proposal_20260521/proposal_fullsize/{OldHospital,ShopFacade,StMarysChurch}/
output/lsf_v2_native_proposal_20260521/maps_edits32/{OldHospital,ShopFacade,StMarysChurch}/
output/lsf_v2_native_proposal_20260521/maps_edits32_pointcloud/{OldHospital,StMarysChurch}/
```

The strongest positive train-dev evidence is ShopFacade with q80 train queries:

| Scene | Split | Stage | Baseline median | LSF median | Recall delta |
| --- | --- | --- | ---: | ---: | ---: |
| ShopFacade | train q20 | dense TE | 2.9435 cm | 2.4388 cm | R2 +0.20, R5 +0.05 |
| ShopFacade | train q80 | dense TE | 12.2225 cm | 5.7390 cm | R2 +0.025, R5 +0.025 |
| ShopFacade | train q80 | sparse TE | 323.891 cm | 251.784 cm | R5 +0.0125 |
| ShopFacade | train q160 | dense TE | 4.7782 cm | 3.7927 cm | R2 +0.025, R5 +0.04375 |
| ShopFacade | train q160 | sparse TE | 99.9789 cm | 80.8615 cm | R2 -0.00625, R5 -0.00625 |

This is a real positive validation of the LSF native proposal recipe on
ShopFacade train-dev. It is not a Cambridge test claim and not yet a SOTA claim.

OldHospital and StMarysChurch remain boundary/failure evidence rather than main
positive claims:

| Scene | Split | Stage | Result |
| --- | --- | --- | --- |
| OldHospital | train q20 | dense | TE improves from 3283.78 cm to 2050.73 cm, but absolute error remains very poor. |
| OldHospital | train q80 | sparse | TE improves from 608.05 cm to 455.03 cm. |
| OldHospital | train q80 | dense | Regresses from 51.13 cm to 75.94 cm; this blocks a global claim. |
| StMarysChurch | train q20 | sparse/dense | Rotation can improve slightly, but translation and recall do not meaningfully improve. |

The current paper-safe claim should therefore be limited to:

```text
Audited v2 LSF support can produce same-budget native STDLoc map edits that
improve single-path OpenCV STDLoc train-dev localization on ShopFacade; hard
scenes expose dense-refinement and solver-admissibility gaps that define the
next method step.
```

The q160 train-dev confirmation keeps the ShopFacade dense result positive. It
also shows that sparse median improvements can trade off strict sparse recall,
so the current accuracy claim should stay tied to the full sparse-to-dense
STDLoc path rather than to sparse PnP alone. OldHospital q160 native baseline
was also generated for reference, but q160 LSF OldHospital was not used for
claim selection because q80 already exposed the dense-regression boundary.

## Go / No-Go Gates

### Gate A: Feedback Validity

No solver-aware or dense LSF training result can be paper-facing unless its
feedback bank passes v2 audit.

Command:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python \
  -m loc_gs.scripts.audit_feedback_bank_v2 \
  --feedback_bank PATH/feedback_bank.jsonl \
  --output_json PATH/feedback_bank_v2_audit.json \
  --fail_on_error
```

### Gate B: Mechanism Figure Before New Candidate

Before proposing another selector candidate, produce a mechanism table/figure:

```text
native vs failed support-count method vs solver-aware candidate
support_count_delta
viable_tuple_mass_delta
logdet_H_delta
ambiguity_delta
dense_worsened_delta
pose_delta
```

If support count increases but logdet or dense reliability drops, this is
evidence for the LSF problem statement.

Command:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python \
  -m loc_gs.scripts.summarize_support_failure_mechanism \
  --solver_report PATH/solver_tuple_report.json \
  --baseline_metrics PATH/native/metrics_summary.json \
  --candidate_metrics PATH/candidate/metrics_summary.json \
  --baseline_transition PATH/native/stage_transition.json \
  --candidate_transition PATH/candidate/stage_transition.json \
  --output_json PATH/support_failure_mechanism.json \
  --output_md PATH/support_failure_mechanism.md
```

### Gate C: Fixed Recipe Before Test

No Cambridge test/full paper-facing evaluation until:

- feedback bank v2 audit passes;
- train q80 and shifted train-dev are positive or neutral on hard scenes;
- no fixed-recipe macro R5/R2 regression;
- dense worsened query count decreases or stays neutral;
- all outputs include manifest, metrics summary, command, split audit, and git
  status/diff material.

## Next Implementation Sprint

Round 1 reset status:

- Repository entry points now describe LSF-Loc rather than residual descriptor,
  selector, or quality-gate mainlines.
- `feedback_bank_v2` records with real image grouping are the required
  supervision input for paper-facing LSF work.
- Solver admissibility must accept real string `image_id` / `query_id` groups,
  not only synthetic integer query ids.

Round 2 implementation status, 2026-05-22:

- `is_replacement_admissible` now supports LSF v3 hard-query guards for
  `min_eigenvalue`, `dense_worsen_risk`, and hard-query CVaR utility.
- `export_lsf_solver_aware_map` passes these thresholds from
  `solver_admissibility_path` into the same-budget map edit callback and records
  rejected examples in the map manifest.
- Dense LSF remains marked as `dense_residual_teacher_only`; it is not a sparse
  selector.
- `loc_gs.reporting.fixed_recipe_gate` implements the train-dev fixed-recipe
  gate for ShopFacade/OldHospital/StMarysChurch style promotion checks.
- `loc_gs.scripts.write_fixed_recipe_gate_report` materializes a scene-level
  map acceptance report. It can keep native maps for scenes whose LSF candidate
  fails the predeclared train-dev gate; this remains a reconstruction-time
  acceptance policy, not a per-query branch selector.
- `update_experiment_board` now attaches `submission_alignment` rows with
  missing ULF/STDLoc-style dense metrics and runtime/memory/map-size/edit-budget
  reporting fields.
- ULF-Loc is reported through `write_ulfloc_alignment_reference
  --paper-reported-ok` as an external paper-reported reference rather than a
  local reproduction blocker.

1. Use the dense LSF targets as a dense residual support teacher, not as a
   sparse selector.
2. Use solver-admissible map editing: a candidate landmark may enter only if
   hard-query tuple mass/logdet/min-eigen do not regress and dense-worsen risk
   does not rise under the fixed map-construction thresholds.
3. Use CVaR-style hard-query aggregation over the real `image_id` groups.
4. Re-run train-dev fixed-recipe validation before any full split.
