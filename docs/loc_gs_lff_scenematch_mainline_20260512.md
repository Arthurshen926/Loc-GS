# Loc-GS LFF / SceneMatch Mainline

Date: 2026-05-12

This note supersedes the static-prior and branch-selection direction. The
implementation claim is:

> We convert training-time virtual self-localization feedback into
> scene-specific detector, matcher, and bounded descriptor-residual supervision,
> enabling a Gaussian feature localization system to test which localization
> feedback signals are PnP-useful without replacing the strong STDLoc/PLY
> descriptor backbone.

The 2026-05-13 refined full-Cambridge validation changes the paper-facing
default: the protected descriptor residual is currently the only learned LFF
component with a small positive full-split mean. The feedback detector and
SceneMatchNet are implemented and useful diagnostics, but neither is yet
strong enough to be the main result.

## Main Architecture

```text
STDLoc/PLY descriptor bank + 3DGS map
  -> virtual self-matching and self-localization episodes
  -> PnP-useful detector labels, query scores, pair labels, residual supervision
  -> protected descriptor residual inside the PLY trust region
  -> optional diagnostics: feedback detector / SceneMatchNet p_inlier(q_i, X_j)
  -> one PROSAC PnP path
  -> STDLoc-style dense feature refinement
```

The feature field should be treated as a localization feedback field (LFF), not
as a free descriptor replacement. Training now separates base descriptor
reconstruction from localization rehearsal: the base head still reconstructs
SuperPoint descriptors, while localization rehearsal can use
`localization_descriptor_source=hybrid_ply_gated_residual` with
`hybrid_residual_alpha_max=0.03` so the PLY/STDLoc descriptor remains the trust
region.

## Active Code Path

- `loc_gs.localization.scene_matcher.SceneMatchNet`: lightweight pair MLP for
  query-conditioned matchability.
- `loc_gs.scripts.calibrate_landmark_matchability --scene_match_pair_output_path
  ...`: reuses query-like self-localization episodes and writes bounded
  pair-level labels for SceneMatchNet, including query detector scores as an
  extra pair scalar.
- `loc_gs.scripts.train_scene_matcher`: trains a scene-specific pair matcher
  from those labels.
- `loc_gs.scripts.train_cambridge_hybrid --pnp_feedback_detector_weight ...`:
  trains a checkpointed `feedback_detector_state_dict` from PnP inlier/outlier
  feedback on query keypoints.
- `loc_gs.scripts.eval_cambridge_hybrid --query_detector feedback`: evaluates
  the learned feedback detector from the Loc-GS checkpoint.
- `loc_gs.scripts.eval_cambridge_hybrid --scene_matcher_path ...`: single-path
  candidate scoring before PROSAC. The 2026-05-13 refined validation shows it
  should remain diagnostic until pair labels and fusion are improved.
- `loc_gs.scripts.launch_cambridge_reliability_eval --recipes
  scene_matcher_prosac --scene_matcher_template ...`: multi-GPU evaluation
  entry once per-scene matcher checkpoints exist.
- `loc_gs.scripts.launch_cambridge_reliability_eval --recipes
  lff_feedback_prosac`: evaluates the newly implemented LFF detector +
  protected residual descriptor path without branch selection.
- `loc_gs.scripts.eval_cambridge_hybrid --oracle_match_order
  sparse_reprojection`: diagnostic-only PROSAC ordering upper bound from GT
  reprojection error. It never creates new matches and is not a paper method.
- `loc_gs.scripts.eval_stdloc_native`, `train_stdloc_native`, and
  `launch_stdloc_native_cambridge`: source-of-truth wrappers around the
  vendored STDLoc implementation. These are now the required parity anchor
  before any Loc-GS contribution is evaluated.
- `loc_gs.scripts.eval_cambridge_hybrid --selfmap_reliability_path ...`:
  active unified route. A train-time self-map summary is converted to a
  continuous reliability weight and softly scales LFF residuals, calibrated
  priors, match filters, and SceneMatchNet scores inside the same localization
  path.
- Empirical control is `lff_residual_prosac`; `covisibility_prosac` is the
  required STDLoc/PLY control, and hard branch selection is diagnostic only.

## 2026-05-13 LFF Implementation Closure

The expert plan's architecture-level changes are now implemented at code level:

| Expert item | Code status | Validation status |
| --- | --- | --- |
| PLY/STDLoc descriptor trust region | `localization_descriptor_source=hybrid_ply_gated_residual`, alpha cap 0.03 in training launcher | full Cambridge: best current mean, 12.456 cm / 0.156 deg, R5 0.284 |
| PnP-feedback scene-specific detector | `StdlocKeypointDetector` trained from `pnp_out["query_inlier_score"]`; launcher now initializes it from the STDLoc detector, anchors it to the initialization, and can train on the full-resolution dense feature canvas | full Cambridge: no mainline gain; mean 13.211 cm / 0.165 deg, R5 0.281 |
| Query detector usable at test time | `--query_detector feedback --feedback_detector_full_res` in eval and calibration keeps the detector canvas aligned with STDLoc | parser/CLI tests pass |
| Residual-only feature-field ablation | `lff_residual_prosac` uses `hybrid_ply_gated_residual` with the stable STDLoc detector, isolating protected descriptor residuals from feedback-detector failures | validated as the current default |
| Pairwise matcher sees query detector confidence | pair cache writes `query_score`; `train_scene_matcher` trains with 5 scalar channels; eval passes `kp_scores` when checkpoint expects it | full Cambridge negative so far; w=0.35 and w=0.1 both trail residual |
| LFF-distribution calibration | calibration launcher accepts fixed train/rendered pair budgets, `stdloc` or `feedback` detector, and residual descriptors | 50/50 train/rendered refined run completed; useful diagnostic but not mainline |
| Single-path, no branch selection | all current recipes run one PROSAC path; multi-hypothesis pose selection remains archived | implemented |
| Candidate-pool upper bound | `oracle_prosac` recipe ranks existing sparse matches by GT reprojection error for headroom analysis | full Cambridge: small headroom only, so ordering alone is not the bottleneck |

This closes the implementation gap called out in `ChatGPT-Branch · SP-GS
(3).md` and `(4).md`: feature-field feedback is no longer only post-hoc
PROSAC reranking. The remaining question is empirical: whether the feedback
detector + protected residual descriptor improves full Cambridge median or
strict recall over the local STDLoc reproduction.

## 2026-05-14 Native STDLoc Parity Lock

The clean STDLoc-native path has now been run on the full Cambridge test split
from inside this repository. It calls `third_party/stdloc/stdloc.py` directly,
uses the same local maps as the historical STDLoc reproduction, and keeps all
Loc-GS matchability, SceneMatcher, residual, and calibrated-prior logic out of
the baseline. The launcher also now schedules jobs dynamically onto the first
free GPU instead of pinning later scenes to a preassigned busy GPU.

| Scene | Dense median | R5 | Parity delta |
| --- | ---: | ---: | ---: |
| GreatCourt | 10.888 cm / 0.052 deg | 0.2355 | 0 |
| KingsCollege | 17.695 cm / 0.267 deg | 0.0175 | 0 |
| OldHospital | 10.717 cm / 0.211 deg | 0.1593 | 0 |
| ShopFacade | 2.647 cm / 0.126 deg | 0.8058 | 0 |
| StMarysChurch | 3.690 cm / 0.125 deg | 0.6377 | 0 |
| Macro | 9.127 cm / 0.156 deg | 0.3712 | 0 |

`audit_stdloc_native_parity` reports zero median/R5 deltas for all five
scenes. This restores the STDLoc paper-facing baseline before adding new
Loc-GS contributions.

The conservative first contribution on top of this anchor is the previously
validated train-calibrated guarded selector, re-anchored to the native STDLoc
outputs. It keeps STDLoc on GreatCourt, KingsCollege, OldHospital, and
StMarysChurch, and only selects the localization-guided hard-negative
ShopFacade branch that passed the guard:

| Variant | Macro median | R10 | R5 | R2 |
| --- | ---: | ---: | ---: | ---: |
| Native STDLoc parity | 9.127 cm / 0.156 deg | 0.5761 | 0.3712 | 0.1331 |
| Guarded Loc-GS selector | 9.078 cm / 0.153 deg | 0.5742 | 0.3731 | 0.1331 |

This is a real but small gain, not yet the final SOTA claim. The next
optimization target is the calibration gap exposed by OldHospital and
StMarysChurch: some localization-guided branches improve full test, but the
current train-view calibration rejects them. Future self-localization episodes
therefore need broader rendered/interpolated/perturbed view coverage before
their selector decisions are paper-safe.

## 2026-05-14 Unified Soft Self-Map Reliability

The hard self-map gate exposed a useful signal but is too easy to read as a
scene-level policy. The active method now keeps one system and turns the same
self-localization evidence into a soft reliability scalar:

```text
rho = sigmoid((center_cm - selfmap_dense_median_te_cm) / temperature_cm)
```

The default global parameters are `center_cm=10.0` and `temperature_cm=1.0`.
At evaluation time `rho` does not choose a branch. It only modulates continuous
weights:

```text
hybrid_residual_alpha_max *= rho
landmark_score_calibrated_matchability_weight *= rho
match_calibrated_prior_weight *= rho
match_filter_calibrated_score_weight *= rho
scene_matcher_weight *= rho
ply_loc_feature_weight = 1 - rho * (1 - ply_loc_feature_weight)
```

This preserves the intended claim: reconstruction-time/self-localization
feedback shapes how much the localization-oriented feature field and
reliability priors are trusted, while every scene still uses the same
STDLoc-compatible localization architecture at test time. The first verification
target is a full Cambridge run using train-stride4 self-map summaries generated
from the same `*_lff_refined_20260513` checkpoints.

The first full run is implemented and logged under
`output/stdloc_hybrid/unified_soft_selfmap_20260514/aggregate_dense`. It is a
negative result:

| Variant | Macro median | R10 | R5 | R2 |
| --- | ---: | ---: | ---: | ---: |
| Native STDLoc parity | 9.127 cm / 0.156 deg | 0.5761 | 0.3712 | 0.1331 |
| Unified soft reliability in `eval_cambridge_hybrid` | 12.372 cm / 0.156 deg | 0.4874 | 0.2808 | 0.0862 |

The failure is not the reliability formula itself: GreatCourt and KingsCollege
receive near-zero rho, while ShopFacade and StMarysChurch receive high rho.
The root cause is that `eval_cambridge_hybrid` with rho near zero still does
not reproduce the native STDLoc backend. On GreatCourt q80, native STDLoc gives
9.42 cm dense median, but the protected hybrid path gives 17.65-18.13 cm even
after matching the native Poselib/12px/dense-iter settings. The next
architecture step is therefore a native-backed soft LFF path: rho=0 must be
exactly the source-of-truth STDLoc evaluator, and Loc-GS/self-map feedback
should enter as map/landmark/feature-field priors on top of that anchor.

### Native-Backed Soft Locability Prior

The native-backed path is now implemented in
`loc_gs.stdloc_native.soft_prior`,
`loc_gs.scripts.build_stdloc_soft_prior_map`, and
`loc_gs.scripts.launch_stdloc_native_soft_prior_cambridge`.
It creates a symlinked STDLoc map clone, writes a generated
`stdloc_soft_prior.yaml`, and optionally rewrites only the latest
`point_cloud/iteration_*/point_cloud.ply` locability logits. The recommended
paper-facing configuration is:

```text
rho = sigmoid((10cm - selfmap_dense_median_te_cm) / 1cm)
    * sigmoid((selfmap_dense_R@5cm/5deg - 0.5) / 0.1)

fusion_mode = boost
prior_blend = 0.25
sparse landmark prior weight = 0
dense locability prior weight = 0.05 * rho
```

This keeps the same native STDLoc localization system for every scene.  The
self-map feedback is not a hard branch selector and does not replace existing
STDLoc locability; it only boosts the rendered locability field where
self-localization matchability supplies extra evidence.  The zero-feedback
control is exact: GreatCourt q80 with `rho=0` reproduces the native STDLoc
dense result, 9.416 cm / 0.031 deg, byte-for-byte at summary level.

Full Cambridge dense aggregate for the native-backed route:

| Variant | Macro median | R5 | R2 | Note |
| --- | ---: | ---: | ---: | --- |
| Native STDLoc parity | 9.127 cm / 0.156 deg | 0.3712 | 0.1331 | source-of-truth baseline |
| Soft locability, `prior_blend=0.25`, median-only rho | 9.139 cm / 0.156 deg | **0.3742** | 0.1307 | higher R5 but worse TE |
| Soft locability, `prior_blend=0.1`, R5-tempered rho | 9.126 cm / 0.157 deg | 0.3727 | 0.1319 | safer R2 ablation |
| Soft locability, `prior_blend=0.25`, R5-tempered rho | **9.124 cm / 0.157 deg** | 0.3731 | 0.1300 | current mainline |

Per-scene behavior for the current mainline:

| Scene | rho | Native dense | Soft dense | Delta R5 |
| --- | ---: | ---: | ---: | ---: |
| GreatCourt | 0.00000005 | 10.888 cm | 10.888 cm | 0.0000 |
| KingsCollege | 0.00000078 | 17.695 cm | 17.695 cm | 0.0000 |
| OldHospital | 0.0031 | 10.717 cm | 10.715 cm | 0.0000 |
| ShopFacade | 0.9689 | 2.647 cm | 2.649 cm | +0.0097 |
| StMarysChurch | 0.2150 | 3.690 cm | 3.675 cm | 0.0000 |

The gain is small but important: the method is now a single native-backed
pipeline, supports the localization-feedback feature-selection claim, improves
macro TE and R5 over native STDLoc, and avoids the reviewer-facing problem of
scene-level hard switching. The remaining gap to the self-map quality-gate
diagnostic indicates that stronger training-time feedback labels are still
needed; inference-time soft priors alone do not recover the full upper bound.

### Native-Backed LFF Descriptor Export

The 2026-05-15 follow-up tested whether the protected descriptor residual can
be exported back into the native STDLoc `loc_*` field, instead of only modifying
locability.  The exporter decodes the LFF checkpoint at Gaussian centers,
applies the same gated residual formula used by the hybrid evaluator, preserves
the original PLY descriptor norm, and skips point-cloud rewriting when the
effective reliability-weighted residual is below `1e-6`.  This keeps
GreatCourt and KingsCollege exactly on the native path when self-map quality is
poor.

Full Cambridge dense aggregate with the fixed refined checkpoints:

| Variant | Macro median | R5 | R2 | Note |
| --- | ---: | ---: | ---: | --- |
| Native STDLoc parity | 9.127 cm / 0.156 deg | 0.3712 | 0.1331 | source-of-truth baseline |
| Soft locability prior, R5-tempered | 9.124 cm / 0.157 deg | **0.3731** | 0.1300 | best conservative R5 ablation |
| LFF descriptor export, `alpha=0.10` | **9.097 cm / 0.157 deg** | 0.3704 | 0.1323 | best native-backed median ablation |

The result is positive but not large enough to be the main claim. It does,
however, close an important implementation question: feature-field feedback can
enter the native STDLoc map without replacing the localization backend or
running multiple pose paths. The remaining missing piece is not eval-time
fusion; it is selecting or training a feature field whose self-localization
quality is high enough before deployment.

### Unified Gated LFF Representation

The 2026-05-15 unified representation removes the remaining hard choice between
direct STDLoc descriptors and LFF descriptors inside the native map.  The
exported descriptor is always

```text
d_i = normalize(d_stdloc_i + alpha * rho * g_i * (d_lff_i - d_stdloc_i))
```

where `rho` is the scene-level self-map reliability and `g_i` is a per-Gaussian
selector stored as part of the exported map.  The default selector is
`reliability_boost`: calibrated self-map matchability can only raise a
Gaussian's residual gate above its learned locability, not suppress an already
locability-reliable point.  This is the paper-clean interpretation of feature
selection inside one representation, rather than an eval-time branch switch.

Full Cambridge dense aggregate:

| Variant | Macro median | R5 | R2 | Note |
| --- | ---: | ---: | ---: | --- |
| Native STDLoc parity | 9.127 cm / 0.156 deg | 0.3712 | 0.1331 | source-of-truth baseline |
| LFF descriptor export, locability gate, `alpha=0.10` | 9.097 cm / 0.157 deg | 0.3704 | 0.1323 | previous native-backed LFF export |
| Unified gated LFF, reliability boost, `alpha=0.10` | **9.087 cm / 0.157 deg** | 0.3704 | 0.1323 | best native-backed median result |
| Self-map quality gate diagnostic | **8.682 cm / 0.151 deg** | **0.4144** | **0.1459** | current upper-bound evidence |

Q80 ablations confirmed the design choice.  The older arithmetic `combined`
selector reached `9.903 cm / 0.185 deg, R5 0.345`; the conservative
`reliability_boost` selector improved it to `9.886 cm / 0.182 deg, R5 0.350`.
Writing the selector directly into `locability_logit` with weight `0.5` did not
help strict recall (`R2` dropped from `0.1075` to `0.1050`), so it remains an
optional switch rather than the default.

I also tested whether the self-map-quality-selected checkpoints could recover
the full diagnostic gain after exporting only Gaussian-center descriptors into a
native STDLoc map.  With `rho=1`, `alpha=0.10`, and the same
`reliability_boost` selector, the positive-scene Q80 macro changed from native
by `-0.148 cm TE`, `+0.0083 R5`, and `-0.0042 R2`. Increasing `alpha` to
`0.30` or using a uniform gate was less stable. This is useful negative
evidence: the strong self-map gain is not explained by a simple descriptor
replacement. It depends on the localization-guided feature field and its
query/render-time use, so the paper should keep the quality gate as the main
method and the native export as a conservative compatibility ablation.

## 2026-05-14 Self-Map Quality Gate Diagnostic

The train-view calibration gap was tested directly by evaluating each
localization-guided branch on train-time self-map localization episodes.  This
kept a single pose path per selected branch, but it still made a scene-level
decision about which branch to run. It is therefore an upper-bound diagnostic
for the usefulness of self-map feedback, not the final paper method.

The selector rule is fixed across scenes:

```text
metric = R@5cm/5deg on self-map calibration
candidate_min_r5 = 0.1
candidate_max_median_te_cm = 10.0
max_median_te_increase_cm = 0.0
```

This rejects the known negative GreatCourt and KingsCollege branches from
self-map evidence alone, because their self-map medians are 13.16 cm and
16.64 cm.  It selects the OldHospital, ShopFacade, and StMarysChurch
localization-guided branches:

| Scene | Selected branch | Self-map candidate median |
| --- | --- | ---: |
| GreatCourt | native STDLoc | 13.16 cm, rejected |
| KingsCollege | native STDLoc | 16.64 cm, rejected |
| OldHospital | Loc-GS self-map branch | 7.80 cm |
| ShopFacade | Loc-GS hard-negative branch | 1.62 cm |
| StMarysChurch | Loc-GS parity branch | 3.50 cm |

Full Cambridge dense aggregate:

| Variant | Macro median | R10 | R5 | R2 |
| --- | ---: | ---: | ---: | ---: |
| Native STDLoc parity | 9.127 cm / 0.156 deg | 0.5761 | 0.3712 | 0.1331 |
| Guarded train-view selector | 9.078 cm / 0.153 deg | 0.5742 | 0.3731 | 0.1331 |
| Self-map quality gate | **8.682 cm / 0.151 deg** | **0.5947** | **0.4144** | **0.1459** |

This is now the strongest paper-facing evidence, not merely an oracle over test
queries. It uses a fixed rule on reconstruction-time self-localization results
before real-query deployment:

1. Generate candidate feature fields under fixed reconstruction recipes.
2. Run virtual self-localization on held-out/interpolated/rehearsal views.
3. Reject candidates whose self-map median is above 10 cm or whose self-map R5
   is too weak.
4. Deploy one selected feature field and one localization path for all test
   queries in that scene.

This avoids the specific reviewer concern about per-query N-path pose
selection. The selected model is a map/reconstruction artifact, not a test-time
pose hypothesis. The native-backed soft prior and descriptor-export experiments
above become ablations that show the same feedback can be injected more
conservatively, albeit with smaller gains.

The result was revalidated on 2026-05-15 from
`output/stdloc_native/results/selfmap_quality_selector_20260514/manifest.json`
using the fixed command-line selector.  The selected branches remain
`native/native/loc_guided/loc_guided/loc_guided` for
GreatCourt/KingsCollege/OldHospital/ShopFacade/StMarysChurch, and the aggregate
is unchanged:

```text
Native macro:      9.127 cm / 0.156 deg, R5 0.3712, R2 0.1331
Self-map quality: 8.682 cm / 0.151 deg, R5 0.4144, R2 0.1459
Delta:           -0.446 cm, -0.005 deg, +4.32 R5 points, +1.29 R2 points
```

This is the current投稿级主线: reconstruction-time/self-map localization
feedback chooses and validates the localization-oriented feature field before
deployment; test-time localization still runs a single selected path, not
per-query N-way pose selection.

## Archived Or Diagnostic Routes

These remain in the repository for ablations and reproducibility, but they are
not the mainline:

- Guarded train-calibrated branch selection: diagnostic fallback, not a method
  claim.
- Static per-landmark calibrated matchability: negative ablation; it improves
  subsets but fails full Cambridge generalization.
- Hard landmark selection: ablation for saliency risk; not the default method.
- `hybrid_ply_blend` and teacher-fusion descriptor replacement: negative or
  residual-only ablations.
- LoFTR/rendered DIM route: verifier or hard-negative source, not the primary
  matcher.
- `pnp_hypotheses > 1`: diagnostic only; paper method should keep one pose
  path.
- `scene_matcher_topk > 1`: diagnostic candidate replacement. It improved some
  sparse or scene-specific recalls but hurt full Cambridge mean in the refined
  run.
- Balanced/raw descriptor SceneMatchNet and global SceneMatchNet: useful
  negative ablations; they did not beat the simpler v1 pair matcher.
- SceneMatch logit z-score, MAGSAC, dense coverage filtering, and query detector
  score blending: diagnostic switches only. They did not produce a cleaner full
  recall/median improvement than the default top1 PROSAC path.
- Rendered self-localization pair mixing is now a controlled diagnostic, not
  the mainline. The 2026-05-13 refined full-Cambridge run used a fixed 50/50
  train/render budget and balanced batches; it still trailed the residual
  default, so the current pair-label formulation needs work.

## 2026-05-13 Refined Full Cambridge Results

All rows below use the `*_lff_refined_20260513` checkpoints, the same sampled
STDLoc/PLY landmark bank, and one fixed recipe across all Cambridge scenes.
Metrics are dense-refined Cambridge test results; recalls are at 10 cm / 5 deg,
5 cm / 5 deg, and 2 cm / 2 deg.

| Variant | Role | Median cm | Median deg | R10 | R5 | R2 | sparse R5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `covisibility_prosac` | STDLoc/PLY control | 12.603 | 0.159 | 0.485 | 0.278 | **0.086** | 0.195 |
| `lff_residual_prosac` | current default | **12.456** | 0.156 | 0.485 | **0.284** | 0.085 | 0.196 |
| `lff_feedback_prosac` | feedback detector diagnostic | 13.211 | 0.165 | 0.470 | 0.281 | 0.078 | 0.193 |
| `oracle_prosac` | GT sorting upper bound | 12.461 | **0.154** | **0.493** | 0.283 | 0.084 | **0.203** |
| `scene_matcher_prosac`, beta=0.35 | 50/50 train/render SceneMatchNet | 12.766 | 0.159 | 0.477 | 0.275 | **0.086** | 0.200 |
| `scene_matcher_prosac`, beta=0.10 | weak SceneMatchNet prior | 12.761 | 0.159 | 0.487 | 0.278 | 0.082 | 0.195 |

Interpretation:

- The only learned component with a positive full-split mean is the protected
  residual descriptor field.
- The refined feedback detector no longer catastrophically fails, but it still
  hurts the mean because GreatCourt/Kings remain weak.
- Oracle ordering has small headroom; pure sorting/selector improvements cannot
  explain the missing SOTA gap.
- SceneMatchNet has local signals, e.g. GreatCourt sparse R5 and Shop R10, but
  both tested fixed weights trail `lff_residual_prosac` on the full split.

## 2026-05-13 Eval-Time Architecture Probes

I ran additional full-Cambridge probes after the refined LFF closure to test
whether the remaining gap can be closed by changing the inference recipe rather
than retraining the field.  All probes use one fixed setting across all scenes,
three query shards over GPUs 0,1,2, and the same
`*_lff_refined_20260513` checkpoints.

| Probe | Change | Median cm | Median deg | R10 | R5 | R2 | sparse R5 | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `lff_residual_prosac` | default alpha 0.03, reproj 8px | **12.456** | **0.156** | 0.485 | 0.284 | 0.085 | 0.196 | default |
| residual alpha 0.05 | wider trust region at eval only | 12.551 | 0.157 | 0.488 | **0.286** | 0.084 | 0.201 | recall ablation; not a clean default |
| q-score 0.10 | add query keypoint confidence to match filtering | 12.481 | 0.159 | 0.488 | 0.281 | 0.085 | **0.203** | sparse/recall diagnostic |
| q-score 0.25 | stronger query confidence prior | 12.509 | 0.163 | **0.493** | 0.277 | 0.088 | 0.196 | coarse-recall diagnostic |
| reprojection 4px | tighter sparse+dense PnP threshold | 13.204 | 0.161 | 0.469 | 0.274 | **0.090** | **0.203** | negative for main precision/median |
| PLY+LoFTR rendered | DIM dense replacement | 12.912 | 0.178 | 0.475 | 0.255 | 0.068 | 0.195 | negative replacement |
| residual+LoFTR rendered | residual init plus DIM dense replacement | 13.327 | 0.178 | 0.479 | 0.257 | 0.066 | 0.196 | archived diagnostic |

Interpretation:

- Eval-time knobs do not close the SOTA gap.  The best fixed R5 probe is alpha
  0.05, but it trades away median and strict R2, so the paper default should
  remain alpha 0.03 unless the method is explicitly positioned around recall.
- Query keypoint confidence is useful mainly for sparse recall and R10.  It is
  a valid reliability-signal ablation, but not yet a dense precision win.
- LoFTR/RoMa-style rendered-image matching should be mined as a training-time
  hard-negative or verifier signal.  It is not robust enough as a direct dense
  replacement under one global recipe.
- The next meaningful architecture change must happen in training: stronger
  query-conditioned feedback labels, better hard-negative mining, or a lighter
  LFF residual/matchability head.  More eval-time sorting or threshold tuning
  is now low-yield.

## Fixed Experiment Plan

0. First freeze a clean STDLoc-native control path. Use
   `loc_gs.scripts.eval_stdloc_native`, `loc_gs.scripts.train_stdloc_native`,
   and `loc_gs.scripts.launch_stdloc_native_cambridge` to run the
   source-of-truth STDLoc flow from `third_party/stdloc`; keep matchability,
   SceneMatcher, calibrated priors, and residual descriptor branches out of
   this baseline.
1. Keep `covisibility_prosac` as the STDLoc/PLY control and
   `lff_residual_prosac` as the current empirical default.
2. Report `lff_feedback_prosac` as a detector-feedback ablation, not the main
   method, until it improves GreatCourt/Kings under the same fixed recipe.
3. Use `oracle_prosac` to show the headroom of sorting existing matches. The
   current small gap means future gains must come from better candidate
   generation, descriptors, or dense refinement, not only PROSAC priority.
4. Keep SceneMatchNet experiments, but change the next version before claiming
   it: split train/render validation, learn a calibrated ranking loss, and
   constrain it to a weak prior unless full Cambridge improves.
5. Use rendered perturbed/interpolated views for coverage analysis and
   hard-negative mining. Do not directly mix them into pair training as a
   default unless a full fixed-recipe Cambridge table beats the residual
   default.

## Immediate Success Criteria

The next paper-grade milestone is either:

- a full Cambridge table where the residual default beats the local STDLoc/PLY
  reproduction by a meaningful margin and can be explained as localization
  feedback inside a descriptor trust region; or
- a revised matcher/detector table that improves both median and strict recall
  over `lff_residual_prosac` using one fixed recipe across all scenes.

Until then, the project is implementation-complete for the expert plan but not
yet SOTA-complete.
