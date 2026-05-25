# LSF clean-16384 train-dev evaluation protocol

Date: 2026-05-23

Purpose: re-evaluate precision after fixing native map artifact contamination.

## Scope

- Scenes: ShopFacade, OldHospital, StMarysChurch.
- Split: Cambridge train split only, used as train-dev sanity evaluation.
- Query subset: first 80 selected train cameras through STDLoc `max_test_cameras=80`, `test_stride=1`.
- No Cambridge test split is used.

## Maps

- Baseline source maps: `output/stdloc/map_cambridge_spgs_clean_20260523`.
- Required native detector budget: `detector/sampled_idx.pkl` length must be `16384`.
- Candidate LSF maps must be rebuilt from the same clean source maps and must keep the same sampled count.
- Existing 8192 LSF/native maps are excluded from this protocol.

## Evaluator

- Entry point: `loc_gs.scripts.eval_stdloc_native`.
- PnP path: OpenCV PROSAC/MAGSAC config at `/root/Loc-GS/configs/stdloc_cambridge_opencv.yaml`.
- Real query path remains matching -> OpenCV PnP -> STDLoc dense refinement.
- `--expected_sampled_count 16384` is mandatory for all baseline and LSF evals.

## Reporting

Primary metrics:

- dense median translation error in cm.
- dense median rotation error in degrees.
- dense recall at 5cm/5deg and 2cm/2deg as secondary context.

Every eval output must contain:

- `summary.json`
- `results.json`
- `manifest.json`
- `command.txt`
- `metrics_summary.json`
- `split_audit.json`
- `git_status.txt`

## Decision rule

- Precision is considered normal if clean native baseline is centimeter or low-decimeter, not meter-scale explosion.
- LSF is considered effective only where the clean same-budget LSF map improves dense median translation without unacceptable recall regression.
- Negative or neutral hard-scene results remain reported as method boundaries.
