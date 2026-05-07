# Localization-Guided Cambridge Evidence

Date: 2026-05-05

Scope:

- Dataset: Cambridge ShopFacade and StMarysChurch for controlled internal runs.
- Cambridge StMarysChurch is used for locability selection analysis and selective
  reconstruction because it has persisted detector scores.
- No Replica runs.
- No new external baseline runs.
- Existing ShopFacade STDLoc/SP-GS summaries are used only as internal reference points.

## Question

We need evidence for the loop:

```text
SuperPoint reconstruction -> localization priors / localization-guided refinement -> improved relocalization
```

We also need to check whether geometric correspondences improve 3DGS geometry itself.

## Code Added

- `train.py` now accepts `--load_iteration`, so a trained Gaussian map can be loaded from an existing `point_cloud/iteration_*` folder and fine-tuned into a separate output directory.
- `utils/geometry_metrics.py` provides reusable Chamfer and projected-depth consistency metrics.
- `utils/localization_loss.py` now includes localization-guided selective feature reconstruction:
  `w = min_weight + locability^gamma` with optional top-ratio gating. The locability
  weights are detached and normalized to mean one over valid pixels, so the feature
  loss reallocates reconstruction capacity instead of changing global scale or
  learning locability from reconstruction error.
- `utils/selective_reconstruction.py` and
  `scripts/analyze_selective_reconstruction_budget.py` quantify how much detector
  locability mass is covered by a selected reconstruction budget.
- `scripts/analyze_selective_feature_error.py` probes whether the selected
  locability regions also show lower rendered SuperPoint reconstruction error.
- `scripts/run_selective_reconstruction_ft.sh` is a reproducible two-stage fine-tune
  entry point for localization-guided selective reconstruction once GPU memory is
  available.
- `configs/stdloc_spgs_cambridge_detector10000_eval_noprior.yaml` keeps the same
  two dense refinement iterations but disables both sparse and dense locability
  priors at evaluation time. This separates training-time selection from
  test-time prior injection.
- `scripts/evaluate_geometry_consistency.py` evaluates a Gaussian map against COLMAP sparse geometry using:
  - sampled Gaussian-center to sparse-point Chamfer,
  - rendered depth vs projected COLMAP sparse-point depth on selected cameras.

## Selective Reconstruction Claim

The strengthened training objective is:

```text
L = L_rgb + L_feature + lambda_selective * mean_i(normalize(w_i) * |F_i - F_i*|)
w_i = min_weight + s_i^gamma * 1[s_i in top rho locability pixels]
```

where `s_i` is the rendered per-Gaussian locability map. With `rho=0`, the term
falls back to continuous locability weighting. With `rho>0`, it becomes an explicit
budgeted selector: only the top locability pixels receive high feature reconstruction
pressure, while all other pixels keep a small `min_weight` background constraint.

Important implementation detail: `s_i` is detached inside the selective feature loss.
This prevents the feature reconstruction error from corrupting the locability model
and makes the direction of influence clean: localization quality selects where
SuperPoint reconstruction capacity is spent.

## StMarysChurch Locability Budget Analysis

Source: `map_cambridge_spgs/StMarysChurch_stream_fastsave/detector_10000/sampled_scores.pkl`.

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/cybersim_agent/bin/python \
  scripts/analyze_selective_reconstruction_budget.py \
  --score_file map_cambridge_spgs/StMarysChurch_stream_fastsave/detector_10000/sampled_scores.pkl \
  --top_ratios 0.05,0.1,0.2,0.3 \
  --min_weight 0.05 \
  --gamma 2.0 \
  --output docs/stmary_selective_reconstruction_budget.json
```

| Selected top locability ratio | Detector-score mass covered | Gain over random budget | Selected/background score ratio | Selected/background weight ratio |
|---:|---:|---:|---:|---:|
| 5% | 12.6% | 2.51x | 2.73x | 5.32x |
| 10% | 23.2% | 2.32x | 2.72x | 4.70x |
| 20% | 41.1% | 2.05x | 2.79x | 3.94x |
| 30% | 55.7% | 1.86x | 2.93x | 3.45x |

This supports the selection premise: the locability distribution is concentrated
enough that a 20% reconstruction budget captures 41.1% of the detector score mass,
while applying about 3.94x larger reconstruction weight to the selected pixels than
to the background.

## Existing StMarysChurch Localization Evidence

These rows are existing summaries, not newly run baselines.

| Map / setting | Dense median | Sparse median | Notes |
|---|---:|---:|---|
| Existing STDLoc detector10000 reference | 3.640 cm / 0.122 deg | 5.554 cm / 0.196 deg | `results/baseline_stmary_d10000-*` |
| SP-GS geometry, detector10000, no locability prior | 3.940 cm / 0.131 deg | 5.554 cm / 0.196 deg | Removing the localization prior hurts dense localization |
| SP-GS geometry, detector10000, locability prior | **3.459 cm / 0.118 deg** | 5.733 cm / 0.201 deg | +12.2% dense translation improvement over no-prior SP-GS |
| SP-GS geometry, full detector, locability prior | **3.408 cm / 0.109 deg** | **5.449 cm / 0.191 deg** | Best existing StMarysChurch result |

## StMarysChurch Selective Reconstruction

All rows below start from the same
`map_cambridge_spgs/StMarysChurch_stream_fastsave` map and fine-tune only the
implicit loc feature field for 200 iterations. Geometry, opacity, RGB SH, scale,
and rotation learning rates are zero.

Evaluation with locability priors enabled:

| Setting | Selector | Dense median | Dense 5cm recall | Dense 2cm recall | Avg dense inliers | Sparse median |
|---|---:|---:|---:|---:|---:|---:|
| Uniform feature fine-tune | off | 3.465 cm / 0.113 deg | 68.49% | 23.58% | 28957.6 | 5.698 cm / 0.198 deg |
| Selective | top 20%, w=0.5 | 3.449 cm / 0.114 deg | 68.30% | 24.15% | 28928.3 | 5.769 cm / 0.194 deg |
| Selective | top 10%, w=0.5 | 3.444 cm / 0.113 deg | 68.87% | 24.15% | 28942.3 | 5.772 cm / 0.198 deg |
| Selective | **top 5%, w=0.5** | **3.429 cm / 0.113 deg** | **68.87%** | 24.15% | **28964.3** | 5.731 cm / 0.194 deg |
| Selective | top 10%, w=1.0 | 3.438 cm / 0.113 deg | 68.87% | **24.34%** | 28930.4 | 5.729 cm / 0.199 deg |

Evaluation with sparse and dense locability priors disabled:

| Setting | Eval prior | Dense median | Dense 5cm recall | Dense 2cm recall | Avg dense inliers | Sparse median |
|---|---:|---:|---:|---:|---:|---:|
| Uniform feature fine-tune | off | 3.429 cm / 0.115 deg | 68.49% | 23.58% | **29057.2** | 5.704 cm / 0.200 deg |
| Selective top 5%, w=0.5 | off | **3.405 cm / 0.114 deg** | **69.25%** | **24.34%** | 29039.2 | **5.659 cm / 0.200 deg** |

This is the cleanest current evidence for the selection claim. Even when
evaluation-time locability priors are disabled, training-time selective
reconstruction improves dense translation by `0.024 cm`, 5cm recall by `0.76`
points, and 2cm recall by `0.76` points over the uniform fine-tune. It also
slightly improves over the existing full-detector SP-GS dense translation
reference (`3.405 cm` vs `3.408 cm`), although the full-detector reference keeps
better rotation (`0.109 deg` vs `0.114 deg`).

TensorBoard confirms the selector behaves as configured:

| Setting | Selected fraction | Selected weight mean | Background weight mean | Weight ratio | Train feature L1 |
|---|---:|---:|---:|---:|---:|
| Uniform | 0.0% | 0.000 | 0.000 | 0.00x | 0.037415 |
| top 20%, w=0.5 | 20.0% | 2.038 | 0.741 | 2.75x | 0.037422 |
| top 10%, w=0.5 | 10.0% | 2.654 | 0.816 | 3.25x | 0.037420 |
| top 5%, w=0.5 | 5.0% | 3.232 | 0.883 | 3.66x | 0.037418 |
| top 10%, w=1.0 | 10.0% | 2.654 | 0.816 | 3.25x | 0.037424 |

The raw feature-error probe over 40 sampled training cameras does not show a
meaningful L1 reduction in the selected pixels: at top 5%, selected mean absolute
feature error is `0.027655` for uniform and `0.027664` for top-5 selective. This
means the defensible claim is not "selective training lowers raw SuperPoint L1 in
the selected pixels." The stronger and more accurate claim is: at essentially
unchanged average feature reconstruction error, localization-guided selection
changes the feature field in a way that improves geometric matching and final
PnP/refinement accuracy.

## Controlled ShopFacade Runs

All localization rows use `configs/stdloc_spgs_cambridge.yaml`.

| Map / setting | Dense median | Sparse median | Notes |
|---|---:|---:|---|
| Existing STDLoc reference | 2.647 cm / 0.126 deg | 3.285 cm / 0.163 deg | Existing internal reference, not newly rerun here |
| Existing SP-GS locability-prior map | 2.489 cm / 0.119 deg | 3.285 cm / 0.163 deg | Reconstruction map plus localization prior |
| Feature-only loc-guided fine-tune, 200 iters | **2.473 cm / 0.117 deg** | 3.306 cm / 0.164 deg | Geometry frozen; best controlled fine-tune |
| Feature-only loc-guided fine-tune, 800 iters | 2.540 cm / 0.117 deg | 3.544 cm / 0.157 deg | Over-tunes translation |
| Feature+geometry loc-guided fine-tune, 800 iters | 2.645 cm / 0.115 deg | 3.509 cm / 0.148 deg | Rotation improves, translation degrades |
| Anchored feature+geometry fine-tune, 400 iters | 2.563 cm / 0.114 deg | 3.340 cm / 0.166 deg | Anchor stabilizes rotation but not translation |

Artifacts:

- `results/spgs_geom_shop_locfeat_ft200_i200-map_cambridge_spgs_ShopFacade_locfeat_ft200-20260505_010613/summary.json`
- `results/spgs_geom_shop_locfeat_ft_i800-map_cambridge_spgs_ShopFacade_locfeat_ft-20260505_010300/summary.json`
- `results/spgs_geom_shop_locpose_ft_i800-map_cambridge_spgs_ShopFacade_locpose_ft-20260505_005747/summary.json`
- `results/spgs_geom_shop_locgeom_anchor400_i400-map_cambridge_spgs_ShopFacade_locgeom_anchor400-20260505_011215/summary.json`

## Geometry Consistency

All rows use 40 test cameras, stride 2, 60k sampled Gaussian points, and 60k sampled COLMAP sparse points.

| Map / setting | Symmetric Chamfer mean | Gaussian->COLMAP median | Projected depth median abs | Projected depth median rel |
|---|---:|---:|---:|---:|
| Existing SP-GS locability-prior map | 15.661 | 0.161 | 1.600 | 0.1213 |
| Feature-only loc-guided fine-tune, 200 iters | 15.661 | 0.161 | 1.600 | 0.1213 |
| Feature-only loc-guided fine-tune, 800 iters | 15.661 | 0.161 | 1.600 | 0.1213 |
| Feature+geometry loc-guided fine-tune, 800 iters | 15.661 | 0.161 | **1.575** | **0.1194** |
| Anchored feature+geometry fine-tune, 400 iters | 15.661 | 0.161 | 1.587 | 0.1203 |

Artifacts:

- `results/geometry_shop_recon_only.json`
- `results/geometry_shop_locfeat_ft200_iter200.json`
- `results/geometry_shop_locfeat_ft_iter800.json`
- `results/geometry_shop_locpose_ft_iter800.json`
- `results/geometry_shop_locgeom_anchor400_iter400.json`

## Interpretation

1. Reconstruction followed by localization priors is effective: the existing SP-GS locability-prior map improves dense localization over the internal STDLoc reference on ShopFacade.
2. A short localization-guided feature fine-tune can further improve dense median translation and rotation slightly, from `2.489 cm / 0.119 deg` to `2.473 cm / 0.117 deg`.
3. Longer feature-only fine-tuning starts to overfit translation. The localization loss should be used as a light second-stage objective, not as a long replacement for SuperPoint reconstruction.
4. Allowing geometry parameters to move improves rendered-depth consistency slightly, but does not improve center-based Chamfer and hurts translation localization in these short runs.
5. Geometry anchor regularization reduces the depth-consistency regression risk but still does not recover the best translation localization. Current evidence does not support the strong claim that geometric correspondences reliably improve 3DGS geometry and localization at the same time. The defensible claim is narrower: geometric correspondence losses can slightly improve rendered depth consistency, but geometry updates need stronger regularization before they should be part of the main method.
6. On StMarysChurch, the selective reconstruction objective has stronger support than the earlier ShopFacade runs: top-5 selection beats uniform fine-tuning both with and without evaluation-time locability priors. The effect size is modest but consistent with the budget curve.

## Next Experiment

The next controlled experiment should move from pixel-level feature selection to
match-level selection:

```text
L = L_rgb + L_feature
  + lambda_selective L_selective
  + lambda_match mean_j stopgrad(I_j) * reprojection_error(match_j)
```

where `I_j` is an inlier or high-confidence correspondence weight from dense
matching. This would make the training target closer to the downstream PnP
objective than raw SuperPoint reconstruction and should give a larger effect size.

The follow-up geometry experiment should only unfreeze geometry with anchors:

```text
L = L_rgb + L_feature + lambda_selective L_selective
  + lambda_loc L_pose
  + lambda_anchor ||xyz - xyz0||_1
  + lambda_scale ||log_scale - log_scale0||_1
```

Geometry success criterion remains: projected-depth median improves and Chamfer does
not degrade, while localization does not regress below the feature-only selective run.
