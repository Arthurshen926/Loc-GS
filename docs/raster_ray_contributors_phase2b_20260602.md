# Raster Ray Contributors Phase 2B

Phase 2B adds a renderer-agnostic adapter that turns packed raster
intersections into source-view ray contributor records.

The intended data path is now:

```text
packed raster intersections
  -> source_rays.jsonl
  + train/self-map cross-view matches
  -> observations.jsonl
  -> ray_solver_feedback.pt
  -> ULF-Loc solver_feedback.pkl
```

This phase does not modify ULF-Loc or STDLoc evaluator behavior. It defines the
artifact contract needed by a later ULF/gsplat wrapper.

## Input Contract

`intersections.jsonl` accepts rows such as:

```json
{"type":"manifest","manifest":{"scene":"ToyScene","split_name":"selfmap_train"}}
{"type":"intersection","intersection":{
  "pixel_id":0,
  "gaussian_id":1,
  "opacity":0.8,
  "depth":3.0,
  "xyz":[0.0,0.0,3.0]
}}
```

or:

```json
{"type":"intersection","intersection":{
  "pixel_xy":[10.0,20.0],
  "gaussian_id":7,
  "contribution":0.42,
  "depth":6.5
}}
```

The adapter groups intersections by pixel, keeps top-k contributors by
`contribution`, and falls back to `opacity` when no explicit contribution is
available. This makes the later solver feedback attribution soft over multiple
Gaussians on the ray rather than assigning each match to a single rendered
depth or top contributor.

## Output Bundle

Run:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.build_raster_ray_contributors \
  --intersections intersections.jsonl \
  --output_dir output/raster_rays/ToyScene/s1 \
  --scene ToyScene \
  --source_view_id s1 \
  --split_name selfmap_train \
  --width 640 \
  --top_k 8
```

The bundle writes:

- `source_rays.jsonl`
- `manifest.json`
- `metrics_summary.json`
- `split_audit.json`
- `command.txt`
- `git_status.txt`

The resulting `source_rays.jsonl` is accepted by
`loc_gs.scripts.build_cross_view_ray_observations`.

## Current Scope

This module expects a renderer wrapper to provide packed intersections. The
next integration step should patch or wrap ULF-Loc rendering to emit these
records for train/self-map source views only. If exact alpha-composition weights
are unavailable, the artifact must mark proxy weights in its manifest.
