# LSF v2 Train-Dev Validation, 2026-05-21

Scope: train-dev only. No Cambridge test query image, pose, feature,
descriptor, or result was used for training, selection, or tuning.

Evaluator path:

```text
native STDLoc feature extraction / matching
-> OpenCV PROSAC/RANSAC PnP
-> STDLoc-style dense refinement
```

Config: `configs/stdloc_cambridge_opencv.yaml`.

## Main Positive Evidence

ShopFacade dense localization improves under the same-budget LSF v2 native
proposal map:

| Split | Baseline TE | LSF TE | Delta TE | Baseline RE | LSF RE | R@5 delta | R@2 delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train q20 | 2.9435 cm | 2.4388 cm | -0.5047 cm | 0.0873 deg | 0.0781 deg | +0.0500 | +0.2000 |
| train q80 | 12.2224 cm | 5.7390 cm | -6.4835 cm | 0.7891 deg | 0.4662 deg | +0.0250 | +0.0250 |
| train q160 | 4.7782 cm | 3.7927 cm | -0.9854 cm | 0.2496 deg | 0.2480 deg | +0.0438 | +0.0250 |

ShopFacade sparse PnP also improves median TE/RE at q80/q160, but q160 sparse
strict recall drops slightly:

| Split | Sparse TE delta | Sparse RE delta | R@5 delta | R@2 delta |
| --- | ---: | ---: | ---: | ---: |
| train q80 | -72.1068 cm | -6.3828 deg | +0.0125 | +0.0000 |
| train q160 | -19.1173 cm | -0.6561 deg | -0.0063 | -0.0063 |

The current positive claim should therefore be phrased around the fixed
sparse-to-dense STDLoc path, not sparse-only recall.

## Boundary Evidence

OldHospital is mixed:

| Split | Stage | Result |
| --- | --- | --- |
| train q20 | dense | TE improves from 3283.78 cm to 2050.73 cm, but absolute error remains poor. |
| train q80 | sparse | TE improves from 608.05 cm to 455.03 cm. |
| train q80 | dense | TE regresses from 51.13 cm to 75.94 cm and R@5 drops by 0.0125. |

StMarysChurch is a failure boundary:

| Split | Stage | Result |
| --- | --- | --- |
| train q20 | dense | RE improves by 7.93 deg, but TE and recall are unchanged. |
| train q20 | sparse | RE improves by 2.12 deg, but TE and recall are unchanged. |

## Audit Material

Completed eval leaves under these roots have `manifest.json`, `command.txt`,
`metrics_summary.json`, `split_audit.json`, and `git_status.txt`:

```text
output/stdloc_native/train_dev_q20_20260521/
output/stdloc_native/train_dev_q80_20260521/
output/stdloc_native/train_dev_q160_20260521/
```

The eval split is `train`. The split audit intentionally remains `unknown` for
image-id disjointness where the eval wrapper does not materialize image-id sets;
these runs are train-dev validation, not paper-facing test results.
