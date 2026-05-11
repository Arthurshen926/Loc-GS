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

See `docs/cambridge_hybrid_localization.md` for the current implementation status and pilot evidence.

The current paper-facing Cambridge recipe is baseline-preserving: use the
STDLoc sampled/PLY descriptor backbone by default, then select learned branches
from a train/calibration-val split instead of hand-picking per-scene test
results.

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.select_cambridge_branch \
  --manifest docs/cambridge_branch_manifest_reliability_20260511.json \
  --output output/paper_figures/reliability_selection_20260511/selected_branch.json \
  --metric combined

/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.eval_cambridge_branch_selected \
  --selected_branch output/paper_figures/reliability_selection_20260511/selected_branch.json \
  --manifest docs/cambridge_branch_manifest_reliability_20260511.json \
  --output_dir output/paper_figures/reliability_selection_20260511

/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.select_cambridge_query_pose \
  --manifest docs/cambridge_branch_manifest_reliability_20260511.json \
  --output_dir output/paper_figures/reliability_query_selection_20260511/split_even_cal_odd_test \
  --mode calibrated_confidence \
  --calibration_stride 2 --calibration_offset 0 \
  --test_stride 2 --test_offset 1
```

For batched multi-GPU Cambridge training, launch one scene per idle GPU with the
constrained Loc-GS-FT rehearsal recipe. The recipe keeps the STDLoc/PLY
descriptor backbone as the trust region, rehearses localization from mixed
perturbed/interpolated poses, and avoids query-time multi-branch ensembles:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.launch_cambridge_reliability_recipe \
  --scenes GreatCourt,KingsCollege,OldHospital,ShopFacade,StMarysChurch \
  --batch_size 8 \
  --gpus 0,1,2

/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.launch_cambridge_reliability_eval \
  --scenes GreatCourt,KingsCollege,OldHospital,ShopFacade,StMarysChurch \
  --tag reliability_recipe \
  --recipes protected,learned_blend \
  --gpus 0,1,2

# Optional for long single-scene evals: split each recipe by query index and
# merge shard summaries back into the recipe eval directory.
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.launch_cambridge_reliability_eval \
  --scenes StMarysChurch \
  --tag reliability_recipe \
  --recipes protected,learned_blend \
  --query_shards 3 \
  --gpus 0,1,2

/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.select_cambridge_query_pose \
  --manifest docs/cambridge_branch_manifest_reliability_b8_e10_20260511.json \
  --output_dir output/paper_figures/reliability_query_selection_b8_e10_20260511/split_even_cal_odd_test \
  --mode calibrated_confidence \
  --calibration_stride 2 --calibration_offset 0 \
  --test_stride 2 --test_offset 1
```

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
