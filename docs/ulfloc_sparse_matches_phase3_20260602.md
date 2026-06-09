# ULF Sparse Matches Phase 3

Phase 3 connects native ULF-Loc sparse self-map localization traces to the
ray-attributed feedback pipeline.

These traces are bootstrap/diagnostic teacher evidence because ULF sparse
matching uses ULF's sampled landmark subset and fused landmark features. They
are not the final full-raw-Gaussian feedback source.

The pipeline is now:

```text
ULF loc_coarse(return_matches=True) on train/self-map views
  -> matches.jsonl
  + source_rays.jsonl
  -> observations.jsonl
  -> ray_solver_feedback.pt
  -> ULF-Loc solver_feedback.pkl
```

## What Changed

`loc_gs.scripts.export_ulfloc_sparse_feedback` now writes both:

- `feedback_bank.jsonl`: legacy feedback-bank-v2 records.
- `matches.jsonl`: Phase 2A-compatible cross-view match rows.

For train/self-map feedback, each row uses:

```text
query_id = train image id
source_view_id = same train image id
query_xy = sparse keypoint xy
source_xy = same sparse keypoint xy
```

This is intentionally limited to train/self-map views. Official Cambridge test
queries must not be used to produce these records.

For Cambridge, the exporter now reads `dataset_train.txt` by default and passes
those image ids into ULF's `Scene(..., images_to_read=...)`. The manifest records
`train_list` and `images_to_read_count`, so sparse feedback export does not
passively load official test images.

## Why This Is Needed

The earlier ULF solver feedback path could only weight ULF's already sampled
landmark set. The new path can use these sparse matches as bootstrap solver
evidence, then attribute that evidence to full raw Gaussian projections on the
same source ray.

## Next Integration Step

Build source rays from full raw Gaussian projections for train/self-map source
views:

```text
projection_source = full_raw_gaussians
source_view_id
gaussian_id
xy / means2d
depth
xyz
opacity / radius
```

For full-train runs, use `loc_gs.scripts.build_ulfloc_full_raw_source_rays`.
It streams one source view at a time and writes `source_rays.jsonl` directly,
instead of materializing a huge all-view `projected_gaussians.jsonl`.

`loc_gs.scripts.build_projected_gaussian_rays` remains useful for small
single-view smoke/debug artifacts. Exact packed raster intersections remain a
stronger future replacement when available.
