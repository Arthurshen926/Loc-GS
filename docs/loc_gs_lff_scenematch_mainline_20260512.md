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
- Empirical default is currently `lff_residual_prosac`; `covisibility_prosac`
  is the required STDLoc/PLY control, and SceneMatchNet is a diagnostic branch.

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
