# Loc-GS

Localization-Oriented Gaussian Feature Fields for Accurate Camera Relocalization.

Loc-GS is an independent split of the SuperPoint feature-field and localization-guided reconstruction work. The repository is scoped to one main line:

```text
SuperPoint teacher features
  -> low-dimensional per-Gaussian latent feature field
  -> decoded descriptor and detector maps at novel views
  -> geometry-aware matching and PnP camera relocalization
```

The earlier RADIO feature reconstruction, open-vocabulary scene understanding, segmentation, depth-head, and grounding experiment entry points have been removed from the public workflow.

## What Is Included

- SuperPoint descriptor and detector extraction.
- Hybrid Gaussian feature-field training for SuperPoint reconstruction.
- Localization-guided training losses with differentiable matching, reprojection proxy, observability, and per-Gaussian locability.
- SuperPoint reconstruction evaluation, localization evaluation, and qualitative visualization.
- Minimal 3DGS asset preparation helpers.
- A vendored STDLoc copy under `third_party/stdloc` for Cambridge localization experiments and reproducibility.

## Repository Layout

```text
Loc-GS/
├── configs/                  # SuperPoint and localization-oriented run configs
├── docs/                     # Loc-GS technical notes
├── loc_gs/                   # Main Python package
│   ├── losses/
│   ├── models/
│   ├── rendering/
│   └── scripts/
├── tests/                    # Loc-GS unit tests
├── third_party/stdloc/       # Vendored STDLoc code
└── output -> /mnt/pool/sqy/results/Loc-GS/output
```

The `output` symlink is intentionally preserved for compatibility with existing runs.

## Environment

The currently verified CPython/CUDA environment is:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python
```

Install the package in editable mode from the repository root:

```bash
pip install -e .
```

STDLoc dependencies are vendored with the third-party project. Its SuperPoint checkpoint is expected at:

```text
third_party/stdloc/encoders/sp_encoder/weights/superpoint_v1.pth
```

## Main Commands

Extract SuperPoint teacher features:

```bash
CUDA_VISIBLE_DEVICES=0 python -m loc_gs.scripts.extract_superpoint_features \
  --scene room_0 \
  --image_dir /mnt/pool/sqy/dataset/room_0/Sequence_1/rgb \
  --output_dir output/superpoint_features/room_0/Sequence_1 \
  --weights third_party/stdloc/encoders/sp_encoder/weights/superpoint_v1.pth \
  --batch_size 8
```

Train a localization-oriented SuperPoint Gaussian feature field:

```bash
CUDA_VISIBLE_DEVICES=0 python -m loc_gs.scripts.train_feature_field \
  --config configs/superpoint_localization_room_0_v1.yaml
```

Evaluate feature reconstruction:

```bash
CUDA_VISIBLE_DEVICES=0 python -m loc_gs.scripts.eval_superpoint \
  --config configs/superpoint_hybrid_room_0_v3.yaml \
  --checkpoint output/sp_gs/room0_hybrid_v3/checkpoints/best.pth
```

Evaluate relocalization:

```bash
CUDA_VISIBLE_DEVICES=0 python -m loc_gs.scripts.eval_localization \
  --config configs/superpoint_hybrid_room_0_v3.yaml \
  --checkpoint output/sp_gs/room0_hybrid_v3/checkpoints/best.pth \
  --output_dir output/sp_gs/room0_hybrid_v3/localization \
  --num_samples 100
```

Generate qualitative results:

```bash
CUDA_VISIBLE_DEVICES=0 python -m loc_gs.scripts.visualize_superpoint_results \
  --config configs/superpoint_hybrid_room_0_v3.yaml \
  --checkpoint output/sp_gs/room0_hybrid_v3/checkpoints/best.pth \
  --output_dir output/sp_gs/room0_hybrid_v3/qualitative_superpoint
```

## STDLoc Cambridge Experiments

STDLoc has been copied into `third_party/stdloc` so Loc-GS does not depend on `/root/STDLoc` at runtime. Run STDLoc commands from that directory:

```bash
cd third_party/stdloc
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest -q tests
```

Large STDLoc maps, logs, and results are kept under `output/stdloc/` for reproducibility. The legacy paths under `third_party/stdloc/` are relative symlinks back to `output/stdloc/` so existing STDLoc commands keep working.

## Cambridge Hybrid Localization

The stronger Cambridge research path now lives in native `loc_gs` code:

```bash
CUDA_VISIBLE_DEVICES=0 python -m loc_gs.scripts.train_cambridge_hybrid \
  --scene ShopFacade \
  --image_width 640 --image_height 360 \
  --epochs 20 \
  --output_dir output/stdloc_hybrid/ShopFacade

CUDA_VISIBLE_DEVICES=0 python -m loc_gs.scripts.eval_cambridge_hybrid \
  --checkpoint output/stdloc_hybrid/ShopFacade/latest.pth \
  --landmark_source rendered \
  --output_dir output/stdloc_hybrid/ShopFacade/eval
```

See `docs/cambridge_hybrid_localization.md` for implementation status and
`docs/loc_gs_lff_scenematch_mainline_20260512.md` for the active paper-facing
mainline. Branch selection, static calibrated matchability, hard landmark
selection, and LoFTR replacement routes are archived in
`docs/archive/static_prior_branch_selection_20260512.md`; keep them as
diagnostic ablations, not as the primary method.

The current clean mainline is baseline-preserving and single-path, but it has
shifted from descriptor residuals to a localization-aware sampling field:

```text
STDLoc/PLY descriptor backbone
  -> virtual self-localization feedback
  -> unified localization utility / selector / sampling field
  -> STDLoc-compatible native-descriptor map export
  -> one OpenCV PROSAC PnP path
  -> STDLoc-style dense refinement
```

The active 2026-05-16 paper story is reconstruction-time/self-localization
feedback distilled into a localization utility field, not inference-time
multi-path pose selection and not descriptor replacement. The current clean
candidate is `selector005_native_desc`: native STDLoc descriptors are kept
unchanged, while the learned selector is exported into locability and detector
score payloads. Full Cambridge dense metrics move from
`12.5725cm / 0.1578deg, R5 0.2776, R2 0.0848` to
`12.4238cm / 0.1593deg, R5 0.2815, R2 0.0879`. This is positive but still
small; it supports the selector/sampling-field direction, not a strong SOTA
claim.

The next structural step is selector-guided landmark resampling under the same
landmark budget, so the learned field changes `sampled_idx.pkl` rather than only
nudging existing `sampled_scores.pkl`. Residual descriptor reconstruction,
SceneMatchNet, quality gates, and LoFTR replacements are diagnostics or
ablations unless future full-split, paper-audited results promote them.

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.launch_stdloc_native_soft_prior_cambridge \
  --scenes GreatCourt KingsCollege OldHospital ShopFacade StMarysChurch \
  --gpus 0 1 2 \
  --source_map_root output/stdloc/map_cambridge_spgs \
  --map_name_overrides GreatCourt=GreatCourt_stream_stable2 StMarysChurch=StMarysChurch_stream_fastsave \
  --calibrated_matchability_template 'output/stdloc_hybrid/listwise_v4_verifier_20260514/calibration/{scene}/stdloc_bank_query_like.pt' \
  --selfmap_reliability_template 'output/stdloc_hybrid/{scene}_lff_refined_20260513/eval_unified_soft_selfmap_calib_train_stride4_20260514_scene_matcher_residual_prosac/summary.json'
```

The quality gate is not per-query branching: real query localization still runs
one selected model/path. Its role is closer to validation-time reconstruction
model selection, except the validation signal is generated by 3DGS self-rendered
and self-localized views, which is exactly the localization-guided
reconstruction claim.

Follow-up full-split probes on the same checkpoints found only small
eval-time gains: raising the residual cap to `alpha=0.05` improved dense R@5
from 0.284 to 0.286 but worsened median/R@2, query-score filtering improved
R@10 or sparse R@5 while hurting dense R@5, and LoFTR rendered matching trailed
the default.  Treat these as ablations; the next expected improvement has to
come from training-time feedback labels and hard-negative mining, not more
inference-time branching.

For batched multi-GPU Cambridge training, launch one scene per idle GPU with the
constrained Loc-GS-FT rehearsal recipe. The recipe keeps the STDLoc/PLY
descriptor backbone as the trust region, caps the localization descriptor
residual at `alpha=0.03`, fine-tunes a PnP-feedback query detector from the
STDLoc detector initialization with an anchor regularizer, and rehearses
localization from mixed perturbed/interpolated poses:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.launch_cambridge_reliability_recipe \
  --scenes GreatCourt,KingsCollege,OldHospital,ShopFacade,StMarysChurch \
  --batch_size 16 \
  --localization_batch_size 8 \
  --feedback_detector_anchor_weight 0.1 \
  --gpus 0,1,2

# Current empirical default: isolate the protected residual feature field from
# feedback-detector and SceneMatchNet effects.
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.launch_cambridge_reliability_eval \
  --scenes GreatCourt,KingsCollege,OldHospital,ShopFacade,StMarysChurch \
  --tag reliability_recipe \
  --recipes lff_residual_prosac \
  --gpus 0,1,2

# Evaluate the implemented LFF feedback detector + protected residual path.
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.launch_cambridge_reliability_eval \
  --scenes GreatCourt,KingsCollege,OldHospital,ShopFacade,StMarysChurch \
  --tag reliability_recipe \
  --recipes lff_feedback_prosac \
  --gpus 0,1,2

# Baseline control without the residual descriptor.
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.launch_cambridge_reliability_eval \
  --scenes GreatCourt,KingsCollege,OldHospital,ShopFacade,StMarysChurch \
  --tag reliability_recipe \
  --recipes covisibility_prosac \
  --gpus 0,1,2

# Generate query-like self-localization labels for SceneMatchNet diagnostics.
# The 2026-05-13 refined run used a fixed 50/50 train/rendered split to cover
# perturbed and interpolated views, but the resulting matcher did not beat the
# residual default on full Cambridge.
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.launch_cambridge_matchability_calibration \
  --scenes GreatCourt,KingsCollege,OldHospital,ShopFacade,StMarysChurch \
  --checkpoint_tag reliability_recipe \
  --output_root output/stdloc_hybrid/query_like_matchability_lff \
  --scene_match_pair_output_root output/scenematch_pairs/lff \
  --scene_match_pair_sample_limit 400000 \
  --scene_match_pair_train_fraction 0.5 \
  --query_detector stdloc \
  --descriptor_source hybrid_ply_gated_residual \
  --hybrid_residual_alpha_max 0.03 \
  --rendered_query_source rendered_rgb_teacher \
  --gpus 0,1,2

# Train one scene-specific pair matcher from the self-localization labels.
# The pair cache includes query detector score as an extra scalar feature.
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.train_scene_matcher \
  --pair_files output/scenematch_pairs/lff/ShopFacade/scene_match_pairs.pt \
  --output_path output/scenematch/ShopFacade/best.pt \
  --batch_size 32768 \
  --epochs 8 \
  --samples_per_epoch 300000 \
  --balanced_batches \
  --balanced_positive_fraction 0.5 \
  --device cuda:0

# Once per-scene SceneMatchNet checkpoints exist, evaluate it as a diagnostic
# weak prior. The full split should decide whether it graduates into the method.
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.launch_cambridge_reliability_eval \
  --scenes GreatCourt,KingsCollege,OldHospital,ShopFacade,StMarysChurch \
  --tag reliability_recipe \
  --recipes scene_matcher_prosac \
  --scene_matcher_template output/scenematch/{scene}/best.pt \
  --scene_matcher_topk 4 \
  --scene_matcher_weight 0.1 \
  --gpus 0,1,2

# Optional for long single-scene evals: split each recipe by query index and
# merge shard summaries back into the recipe eval directory.
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.launch_cambridge_reliability_eval \
  --scenes StMarysChurch \
  --tag reliability_recipe \
  --recipes covisibility_prosac \
  --query_shards 3 \
  --gpus 0,1,2
```

For headroom diagnostics, `oracle_prosac` ranks the already generated sparse
matches by GT reprojection error before PROSAC. Use it only as an upper-bound
analysis, not as a method result.

## Verification

From `/root/Loc-GS`:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest -q tests
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest -q third_party/stdloc/tests
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.config configs/superpoint_localization_room_0_v1.yaml
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.extract_superpoint_features --help
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.train_feature_field --help
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.eval_localization --help
```
