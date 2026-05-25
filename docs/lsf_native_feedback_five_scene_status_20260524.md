# LSF-Loc Native-Feedback Five-Scene Status, 2026-05-24

## Protocol

This round replaces the invalid old map-index-bound LSF chain with a native
STDLoc feedback chain:

1. clean/train-only STDLoc map
2. train split native STDLoc sparse matching and OpenCV PnP feedback
3. audited `feedback_bank_v2`
4. solver-consensus support and localization support field
5. same-budget 32-edit STDLoc-compatible map export
6. one fixed native evaluator path: matching -> OpenCV PnP -> STDLoc dense

No Cambridge test query is used for feedback, support construction, recipe
selection, or hard-query mining.

## Valid Map Sources

| scene | source map | reason |
| --- | --- | --- |
| GreatCourt | `output/stdloc/map_cambridge_spgs_clean_20260523/GreatCourt_stream_stable2` | clean map audit passed |
| KingsCollege | `output/stdloc/map_cambridge_spgs_trainonly_20260524/KingsCollege` | old clean map had test-camera overlap |
| OldHospital | `output/stdloc/map_cambridge_spgs_trainonly_20260524/OldHospital` | old clean map had test-camera overlap |
| ShopFacade | `output/stdloc/map_cambridge_spgs_trainonly_20260524/ShopFacade` | old clean map had test-camera overlap |
| StMarysChurch | `output/stdloc/map_cambridge_spgs_clean_20260523/StMarysChurch_stream_fastsave` | clean map audit passed |

## Main Result Snapshot

Dense median translation delta is LSF minus baseline, in cm. Negative is better.

| split | macro dense TE delta | dense TE wins | macro dense RE delta | macro R5cm delta |
| --- | ---: | ---: | ---: | ---: |
| q80 train-dev | -0.2949 | 3/5 | -0.0018 | +0.0025 |
| q160 train-dev | -0.0857 | 4/5 | -0.0012 | +0.0013 |
| official test, frozen recipe | -0.2330 | 4/5 | -0.0015 | +0.0004 |

Official test per-scene dense TE deltas:

| scene | dense TE delta |
| --- | ---: |
| GreatCourt | -0.2805 |
| KingsCollege | -0.0326 |
| OldHospital | -0.8485 |
| ShopFacade | -0.0202 |
| StMarysChurch | +0.0165 |

## Artifact Roots

- Baseline, clean valid scenes: `output/stdloc_native/clean_valid_20260524_cfgfix`
- Baseline, train-only scenes: `output/stdloc_native/trainonly_20260524_cfgfix`
- LSF, clean valid scenes: `output/stdloc_native/clean_valid_lsf_native_20260524`
- LSF, train-only scenes: `output/stdloc_native/trainonly_lsf_native_20260524`
- LSF selected maps, clean valid scenes: `output/stdloc/map_cambridge_spgs_clean_valid_lsf_native_20260524`
- LSF selected maps, train-only scenes: `output/stdloc/map_cambridge_spgs_trainonly_lsf_native_20260524`
- Combined summary JSON: `output/lsf_trainonly_native_20260524/five_scene_native_lsf_summary_20260524.json`

## Current Claim

The defensible current claim is:

> Native STDLoc self-localization feedback can build a solver-level localization
> support field that improves same-budget 3DGS localization support selection,
> especially pose precision, without modifying the STDLoc evaluator or using
> per-query branch selection.

This is now stronger than the previous ShopFacade-only claim, but StMarysChurch
test remains a neutral/slightly negative boundary and should not be hidden.
