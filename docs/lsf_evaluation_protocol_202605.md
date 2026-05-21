# LSF Evaluation Protocol, 2026-05

## Scope

Loc-GS is evaluated as a Localization Support Field (LSF) on top of the native
STDLoc descriptor and localization backend. The protocol does not modify pose
metrics, evaluator behavior, or test splits.

## Required Pose Metrics

Every candidate report must separate:

```text
sparse initial pose
dense final pose
```

For each stage report:

```text
median translation / rotation
R@50cm,5deg
R@15cm,5deg
R@10cm,5deg
R@5cm,5deg
R@2cm,2deg
average inliers
```

`loc_gs.eval.locgs_metrics` wraps the existing pose metrics and adds the
ULF-style `R@15cm,5deg` key. This is a reporting layer only.

## Dense Transition Metrics

Dense refinement must report:

```text
improved_count
worsened_count
unchanged_count
recovered_r5_count
lost_r5_count
```

This makes the paper claim more precise: an LSF candidate should either improve
the sparse stage, make dense refinement safer, or both.

## Timing Metrics

Offline cost:

```text
STDLoc/base map time
feedback cache time
selector/LSF training time
map export time
peak GPU memory
final map size
```

Online cost:

```text
feature extraction
sparse matching
PnP/PROSAC
rendering/dense matching
dense pose refinement
total latency p50/p95
FPS
```

Timing is reported as accuracy-latency Pareto evidence, not as a standalone
real-time claim.

## Current Smoke Reports

Generated from existing train-q80 supportguardall-ret90 profiles:

```text
output/lsf_reports/20260518/oldhospital_supportguardall_ret90_lsf_eval.json
output/lsf_reports/20260518/stmaryschurch_supportguardall_ret90_lsf_eval.json
```

Observed smoke values:

| Scene | Sparse median cm | Dense median cm | Dense improved | Dense worsened | Dense R5 | Total mean ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OldHospital | 14.622 | 8.896 | 73 | 7 | 0.2875 | 748.23 |
| StMarysChurch | 11.711 | 6.549 | 63 | 17 | 0.4625 | 523.42 |

These are not new method wins; they are protocol smoke checks on an already
rejected diagnostic candidate.

## CLI

```bash
python -m loc_gs.scripts.summarize_lsf_eval \
  --run_dir <profile_dir> \
  --scene OldHospital \
  --method supportguardall_ret90 \
  --landmark_count 12288 \
  --output_json output/lsf_reports/<name>.json
```

