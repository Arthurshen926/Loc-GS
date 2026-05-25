# STDLoc Precision Bug Diagnosis 2026-05-23

## Root Cause

The meter-level OldHospital/StMarysChurch numbers mixed three different issues:

1. The local `configs/stdloc_cambridge_opencv.yaml` path used a plain OpenCV
   `solvePnPRansac` branch. It was not PROSAC and did not sort correspondences
   by descriptor match score. This made the OpenCV path much weaker than the
   native poselib STDLoc baseline.
2. The current StMarysChurch source map was not a clean native baseline: its
   `detector/sampled_idx.pkl` had been overwritten on 2026-05-16 with an 8192
   landmark payload. The old centimeter-level native runs were produced before
   this overwrite, using a 16384 landmark payload.
3. Several contiguous StMarysChurch `train` prefix windows still contain a small
   hard tail, but the median explosion disappears once the clean 16384 detector
   payload is restored.

## Fix

- Added `opencv_prosac` and `opencv_prosac_magsac` support in the vendored
  STDLoc pose utility.
- Sparse and dense STDLoc matching now pass match scores into PnP.
- OpenCV PROSAC sorts correspondences by match score before RANSAC.
- The local OpenCV Cambridge config now uses `opencv_prosac_magsac` for sparse
  and dense stages.
- Added a focused regression test for PROSAC ordering and MAGSAC parameter use.
- Built a clean StMarysChurch native source map at
  `output/stdloc/map_cambridge_spgs_clean/StMarysChurch_native16384_20260523`
  by replacing the polluted `detector/` sampling payload with
  `detector_rebuilt_16384_20260516` while preserving the native detector
  weights, point cloud, and camera metadata.

## Verification

All runs below are train-split diagnostics, not paper-facing Cambridge test
claims.

| Run | Old behavior | Fixed behavior | Interpretation |
| --- | ---: | ---: | --- |
| OldHospital train q20 dense median | 3283.78 cm | 52.75 cm | OpenCV solver bug fixed; same-sample poselib is 44.80 cm |
| OldHospital train q80 dense median | 51.13 cm | 13.50 cm | Back to tens-of-cm baseline range |
| StMarysChurch train q20 dense median | 4974.68 cm | 4.31 cm | OpenCV solver bug fixed |
| StMarysChurch train q80 dense median, polluted 8192 map | n/a | 2772.38 cm | Not a clean native baseline; same polluted map fails with poselib at 3518.42 cm |
| StMarysChurch train q80 dense median, clean 16384 map | 3518.42 cm polluted poselib | 5.34 cm poselib / 6.92 cm OpenCV PROSAC | Detector sampling contamination fixed |
| StMarysChurch native official test dense median | 3.69 cm | unchanged | Baseline/test precision is normal |

## Protocol Consequence

Do not use source maps whose `detector/sampled_idx.pkl` has been overwritten by
selector or LSF experiments as native baselines. StMarysChurch train q80 should
be evaluated against the clean 16384 map above, or against another audited clean
native map with the expected landmark payload.

Future gates must first require a native baseline sanity check on the exact query
list and map artifact. If native/poselib fails on a clean map, the
candidate-vs-baseline delta may be diagnostic only.

Paper-facing claims still require the official native STDLoc parity path and
audited fixed query sets. Test-set results must not be used for tuning or
selection.
