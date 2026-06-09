# Full Raw Gaussian Projection Phase 5

This phase corrects the feedback source definition:

```text
main ray-feedback source = full raw Gaussian map projection
bootstrap sparse teacher = ULF sampled-landmark sparse PnP traces
```

The projection artifact must be built from the original 3DGS Gaussian arrays,
not from `sampled_idx.pkl` or ULF's sampled landmark subset.

## Data Path

```text
full raw Gaussian xyz / opacity / radius
  + train/self-map source view camera matrices
  + ULF train/self-map sparse matches.jsonl
  -> streaming per-view full-raw projection
  -> source_rays.jsonl
  -> observations.jsonl
  -> ray_solver_feedback.pt
  -> ULF-Loc compatible solver_feedback.pkl
```

`projected_gaussians.jsonl` remains available for single-view smoke/debug, but
it is not the full-train path because materializing all projected raw Gaussians
would create a very large intermediate artifact.

## CLI

Generic tensor/view input:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.build_full_raw_gaussian_projections \
  --gaussians full_raw_gaussians.pt \
  --views train_selfmap_views.jsonl \
  --output_dir output/full_raw_projection/ShopFacade/selfmap_train \
  --scene ShopFacade \
  --split_name selfmap_train
```

Native ULF-Loc scene/map input:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.export_ulfloc_full_raw_projections \
  --scene ShopFacade \
  --source_path /mnt/pool/sqy/Cambridge_stdloc/ShopFacade \
  --model_path /root/ULF-Loc/outputs/cambridge_ulfloc_full_20260602/native/ShopFacade \
  --config /root/ULF-Loc/configs/ulfloc_cambridge.yaml \
  --output_dir output/ulfloc_full_raw_projection/ShopFacade/selfmap_train \
  --split_name selfmap_train \
  --images processed
```

Preferred full-train source-ray path:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.build_ulfloc_full_raw_source_rays \
  --matches output/ulfloc_sparse_feedback/ShopFacade/selfmap_train/matches.jsonl \
  --scene ShopFacade \
  --source_path /mnt/pool/sqy/Cambridge_stdloc/ShopFacade \
  --model_path /root/ULF-Loc/outputs/cambridge_ulfloc_full_20260602/native/ShopFacade \
  --config /root/ULF-Loc/configs/ulfloc_cambridge.yaml \
  --output_dir output/ulfloc_full_raw_source_rays/ShopFacade/selfmap_train \
  --split_name selfmap_train \
  --images processed
```

This streaming builder passes only the source view ids present in `matches.jsonl`
to ULF's `Scene(..., images_to_read=...)`, so self-map feedback does not read
official test images.

`full_raw_gaussians.pt` must contain:

```text
xyz: [N, 3]
opacity: optional [N]
radius_px or radius: optional [N]
```

It must not contain `sampled_idx` or `sampled_indices`.

`train_selfmap_views.jsonl` contains one or more train/self-map source views:

```json
{"type":"view","view":{
  "source_view_id":"seq1/frame001.png",
  "width":640,
  "height":480,
  "world_to_camera":[[...],[...],[...],[...]],
  "intrinsic":[[...],[...],[...]]
}}
```

## Audit Fields

The projection bundle writes:

- `projection_source: full_raw_gaussians`
- `source_gaussian_count`
- `projected_gaussian_count`
- `dropped_behind_count`
- `dropped_out_of_frame_count`
- `sampled_idx_used: false`

Downstream projected-ray builders now reject projection rows whose
`projection_source` is not `full_raw_gaussians` by default.

The ULF exporter uses `Scene.getTrainCameras()` and `gaussians.get_xyz`. It does
not read `keypoints_sampled_idx.pkl` or ULF sampled landmark features.

The streaming ULF source-ray builder writes:

- `projection_source: full_raw_gaussians`
- `sampled_idx_used: false`
- `materialized_projection_jsonl: false`
- `images_to_read_count`
- `source_view_count_projected`
- `source_gaussian_count`

## Scope

This phase still does not claim final Cambridge accuracy. It fixes the artifact
source so later experiments can test whether solver feedback over all raw
Gaussians improves selection and feature reconstruction.
