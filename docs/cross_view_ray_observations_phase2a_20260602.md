# Cross-View Ray Observations Phase 2A

Phase 2A adds an audited bridge from precomputed cross-view matches to the
ray-attributed solver feedback accumulator.

The intended data path is:

```text
train/self-map cross-view matches
  + source-view ray Gaussian contributors
  -> observations.jsonl
  -> ray_solver_feedback.pt
```

This phase does not run Cambridge test mining, does not change the STDLoc
evaluator, and does not introduce a per-query branch selector.

## Input Contract

`matches.jsonl` accepts:

```json
{"type":"manifest","manifest":{"scene":"ToyScene","split_name":"selfmap_train"}}
{"type":"match","match":{
  "query_id":"q1",
  "source_view_id":"s1",
  "query_xy":[5.0,6.0],
  "source_xy":[10.0,20.0],
  "descriptor_score":0.9,
  "local_geometry_score":0.8,
  "pnp_inlier":true,
  "reprojection_error_px":1.0,
  "expected_depth":8.0
}}
```

`source_rays.jsonl` accepts:

```json
{"type":"manifest","manifest":{"scene":"ToyScene","split_name":"selfmap_train"}}
{"type":"ray","ray":{
  "source_view_id":"s1",
  "pixel_xy":[10.0,20.0],
  "rendered_depth":3.0,
  "contributors":[
    {"gaussian_id":0,"contribution":0.75,"depth":3.0,"xyz":[0.0,0.0,3.0]},
    {"gaussian_id":1,"contribution":0.25,"depth":8.0,"xyz":[1.0,0.0,8.0]}
  ]
}}
```

All inputs must be train/self-map style splits. `test` is rejected by the core
and CLI.

## Output Bundle

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.build_cross_view_ray_observations \
  --matches matches.jsonl \
  --source_rays source_rays.jsonl \
  --output_dir output/ray_observations/ToyScene/selfmap_train \
  --scene ToyScene \
  --split_name selfmap_train \
  --radius_px 1.0
```

The bundle writes:

- `observations.jsonl`
- `manifest.json`
- `metrics_summary.json`
- `split_audit.json`
- `command.txt`
- `git_status.txt`

The resulting `observations.jsonl` can be consumed by:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.build_ray_attributed_solver_feedback \
  --observations output/ray_observations/ToyScene/selfmap_train/observations.jsonl \
  --output_dir output/ray_feedback/ToyScene/selfmap_train \
  --scene ToyScene \
  --split_name selfmap_train \
  --num_gaussians 200000
```

## Current Scope

This module consumes precomputed matches and source ray contributors. It does
not yet extract features, render source rays, or run cross-view PnP itself. The
next phase should add a data extractor that populates these records from ULF-Loc
or STDLoc train/self-map artifacts while preserving split audit.
