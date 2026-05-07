# Cambridge Hybrid Localization Pipeline

Date: 2026-05-06

This document tracks the new Cambridge path for the stronger Loc-GS claim:

```text
hybrid per-Gaussian latent + spatial hash feature field
  -> weak SuperPoint reconstruction auxiliary
  -> soft descriptor matching
  -> unrolled differentiable PnP
  -> pose/match-quality gradients update feature field, locability, and optional geometry
```

## What Changed

- `loc_gs/data/cambridge_dataset.py`
  loads Cambridge RGB, `cameras.json` train poses, and `dataset_test.txt` query poses.
- `loc_gs/losses/differentiable_pnp.py`
  implements soft matching plus unrolled Gauss-Newton PnP. Gradients flow through
  descriptor probabilities, rendered depth-derived 3D points, PnP pose updates,
  observability, and locability.
- `loc_gs/scripts/train_cambridge_hybrid.py`
  trains the hybrid hash+latent Gaussian field on Cambridge maps initialized from
  existing 3DGS PLYs under `output/stdloc/map_cambridge_spgs`.
- `loc_gs/localization/hybrid_localizer.py`
  provides self-contained keypoint extraction, descriptor sampling, descriptor
  matching, OpenCV PnP/RANSAC evaluation, and pose error metrics.
- `loc_gs/scripts/eval_cambridge_hybrid.py`
  evaluates absolute Cambridge localization using a Loc-GS rendered landmark bank
  or a Gaussian-center fallback. It does not call STDLoc localization code.

## Training Command

Small pilot:

```bash
CUDA_VISIBLE_DEVICES=3 /root/miniconda3/envs/cybersim_agent/bin/python \
  -m loc_gs.scripts.train_cambridge_hybrid \
  --scene ShopFacade \
  --image_width 320 --image_height 176 \
  --epochs 3 \
  --batch_size 1 \
  --num_workers 0 \
  --max_frames 16 \
  --localization_keypoints 64 \
  --latent_dim 16 \
  --hash_output_dim 32 \
  --fine_dim 64 \
  --coarse_dim 64 \
  --hybrid_output_dim 128 \
  --output_dir output/stdloc_hybrid/ShopFacade_pilot \
  --device cuda:0
```

Full-scale run should remove `--max_frames`, increase resolution to at least
`640x360`, and train long enough for the sparse landmark descriptors to become
discriminative.

## Evaluation Command

```bash
CUDA_VISIBLE_DEVICES=3 /root/miniconda3/envs/cybersim_agent/bin/python \
  -m loc_gs.scripts.eval_cambridge_hybrid \
  --checkpoint output/stdloc_hybrid/ShopFacade_pilot/latest.pth \
  --max_queries 5 \
  --landmark_source rendered \
  --landmark_ref_views 8 \
  --landmark_stride 2 \
  --max_landmarks 5000 \
  --query_keypoints 256 \
  --dense_iters 2 \
  --device cuda:0 \
  --output_dir output/stdloc_hybrid/ShopFacade_pilot/eval_q5_rendered
```

## Current Pilot Evidence

The 16-frame, 3-epoch low-resolution pilot is only a plumbing and optimization
sanity check. It is not a baseline comparison.

Training loss decreased:

| Epoch | Total | PnP | Pose | Match quality |
|---:|---:|---:|---:|---:|
| 1 | 12.254 | 5.480 | 1.295 | 1.346 |
| 2 | 9.449 | 5.107 | 1.131 | 0.812 |
| 3 | 8.569 | 4.926 | 1.142 | 0.430 |

Absolute localization is still poor on the 5-query pilot evaluation:

| Stage | Median translation | Median rotation | 5cm/5deg recall |
|---|---:|---:|---:|
| Sparse | 2215.9 cm | 160.0 deg | 0.0% |
| Dense | 25801.5 cm | 144.5 deg | 0.0% |

## Claim Status

The implementation now matches the intended logic: PnP and matching quality can
backpropagate into the hybrid feature field and locability weights. The empirical
claim that this beats STDLoc on Cambridge is not established yet. The next
required experiment is a full ShopFacade run followed by the same summary metrics
used by `output/stdloc/results/baseline_shop-*`.
