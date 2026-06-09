# Projected Gaussian Rays Phase 4

Phase 4 adds a sparse-pixel proxy contributor extractor. It is designed for the
case where ULF/gsplat can provide projected Gaussian metadata for a source view
but not exact per-pixel alpha-composition contributors.

The data path is:

```text
ULF sparse matches.jsonl
  + projected_gaussians.jsonl
  -> source_rays.jsonl
  -> observations.jsonl
  -> ray_solver_feedback.pt
```

## Input Contract

`matches.jsonl` is produced by `export_ulfloc_sparse_feedback.py`.

`projected_gaussians.jsonl` must come from full raw Gaussian projection and
accepts:

```json
{"type":"manifest","manifest":{"scene":"ToyScene","split_name":"selfmap_train"}}
{"type":"projection","projection":{
  "projection_source":"full_raw_gaussians",
  "source_view_id":"seq1/frame001.png",
  "gaussian_id":17,
  "xy":[152.3,88.1],
  "radius":4.5,
  "opacity":0.72,
  "depth":8.4,
  "xyz":[1.0,2.0,8.4]
}}
```

`xy` can also be named `means2d`, `pixel_xy`, or `source_xy`.

## Proxy Weight

For each sparse source pixel, the extractor selects nearby projected Gaussians:

```text
distance <= min(max_radius_px, radius_scale * projected_radius)
```

The proxy contribution is:

```text
opacity * exp(-0.5 * (distance / max(radius, 1px))^2)
```

This is not exact alpha composition. The output manifest and metrics mark the
source as `projected_gaussian_radius_opacity_proxy`.

## CLI

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.build_projected_gaussian_rays \
  --matches matches.jsonl \
  --projections projected_gaussians.jsonl \
  --output_dir output/projected_rays/ShopFacade/selfmap_train \
  --scene ShopFacade \
  --split_name selfmap_train \
  --top_k 8 \
  --max_radius_px 8
```

The output `source_rays.jsonl` can be consumed by
`loc_gs.scripts.build_cross_view_ray_observations`.

## Scope

This is a fast first implementation for train/self-map feedback. It should be
used before exact packed raster contributors are available. It must not be used
with official Cambridge test queries for tuning or feedback construction. By
default, the builder rejects sampled-subset projection sources.
