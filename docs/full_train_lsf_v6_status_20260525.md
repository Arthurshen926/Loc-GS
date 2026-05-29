# LSF v6 Full-Train Status 2026-05-25

## Protocol Update

- `q80` / `q160` means `eval_split=train` with `--max_test_cameras 80/160`.
- These subsets are now treated as smoke diagnostics only.
- Main validation should use full train split first, then fixed official test only after the recipe is frozen.
- Full-train feedback export now defaults to the vendored poselib STDLoc config:
  `third_party/stdloc/configs/stdloc_spgs_cambridge_dense1.yaml`.
- The exporter rejects `opencv_prosac*` dense configs because the vendored dense evaluator does not support them.

## Implementation Fixes

- Fixed `export_native_stdloc_feedback_bank` sparse pose API compatibility:
  vendored `solve_pose` no longer receives unsupported `match_scores`.
- Added a local OpenCV PROSAC route for sparse feedback capture, while keeping dense feedback on the vendored path.
- Fixed a v6 performance bug in `negative_support_memory.conflict_delta`:
  it no longer copies the full 16k selected set for every candidate score.

## Full-Train Feedback Artifacts

### ShopFacade

- Feedback bank: `output/feedback_bank_native_fulltrain_20260525/ShopFacade/feedback_bank.jsonl`
- Query count: 231
- Record count: 473,088
- Audit: passed
- Split: `selfmap_train_native_stdloc`

Derived artifacts:

- Solver-consensus support selected count: 1,055
- Full-train feedback admissibility: `candidate_gain_count=0`, `source_loss_count=11722`
- Candidate-probe admissibility is still required for non-trivial candidate gains:
  `candidate_gain_count=462`
- Full-train negative graph: `edge_count=0`, `unary_count=10315`

## Full-Train Evaluation Results

### ShopFacade

| Map | Dense median TE | Dense R@10cm/5 | Dense R@5cm/5 | Dense R@2cm/2 |
| --- | ---: | ---: | ---: | ---: |
| baseline | 2.0116 cm | 0.9654 | 0.8831 | 0.4978 |
| q80-derived guarded512 | 1.9343 cm | 0.9740 | 0.8831 | 0.5152 |
| fulltrain-support guarded128 | 1.9756 cm | 0.9740 | 0.8874 | 0.5152 |

Delta vs baseline:

- q80-derived guarded512: dense TE `-0.0773 cm`, R@2 `+1.73 pp`
- fulltrain-support guarded128: dense TE `-0.0360 cm`, R@5 `+0.43 pp`, R@2 `+1.73 pp`

Interpretation:

- Full-train feedback improves strict recall but does not improve median TE as much as the earlier q80-derived guarded512 map.
- A large ShopFacade outlier exists in both baseline and candidate, so it is not introduced by v6.
- Current v6 still behaves like support selection with small accuracy gain, not SOTA-scale feature reconstruction.

### OldHospital

| Map | Dense median TE | Dense R@10cm/5 | Dense R@5cm/5 | Dense R@2cm/2 |
| --- | ---: | ---: | ---: | ---: |
| baseline | 9.2422 cm | 0.5240 | 0.2514 | 0.0536 |
| q80-derived guarded512 | 9.3368 cm | 0.5218 | 0.2525 | 0.0581 |
| fulltrain-support guarded128 | 9.2403 cm | 0.5251 | 0.2559 | 0.0648 |

Delta vs baseline:

- q80-derived guarded512: dense TE `+0.0946 cm` worse, R@5 `+0.11 pp`, R@2 `+0.45 pp`
- fulltrain-support guarded128: dense TE `-0.0020 cm`, R@10 `+0.11 pp`, R@5 `+0.45 pp`, R@2 `+1.12 pp`

Delta vs q80-derived guarded512:

- fulltrain-support guarded128: dense TE `-0.0966 cm`, R@10 `+0.34 pp`, R@5 `+0.34 pp`, R@2 `+0.67 pp`

Interpretation:

- Hard-scene full-train median precision is still not solved.
- Replacing q80-derived support with full-train support fixes the guarded512 median regression on OldHospital.
- Current support editing can improve strict recall, especially R@2, but the median TE gain is still too small for a SOTA-level claim.

## Current Bottlenecks

1. Full-train feedback-bank-v2 admissibility is too conservative.
   It produces `candidate_gain_count=0`, so candidate-probe constraints are still needed.

2. Selection-only edits are not enough.
   The best full-train ShopFacade delta is still sub-millimeter-to-sub-centimeter scale, not ULF-Loc scale.

3. Dense stage remains the main limiter.
   Sparse support gains do not reliably convert into dense median TE improvement on OldHospital.

4. GPU utilization is inherently low for current tasks.
   Current exports are CPU set optimization and map editing; native STDLoc eval is a per-query localization loop with CPU PnP/RANSAC and intermittent GPU rendering/feature extraction.

## Next Required Step

Move from map selection to method-level reconstruction:

- Implement solver-weighted landmark descriptor fusion for selected landmarks.
- Keep native descriptors as fallback and constrain descriptor shift.
- Validate first on sparse pose before dense.

## Verification

- `locgsctl status` passed and reports the current repo/root paths.
- `locgsctl list-scenes` passed; the legacy default maps are 8,192-sampled and should not be used for the current 16,384-native paper-safe path.
- OldHospital full-train comparisons were generated with `locgsctl compare` for dense and sparse stages.
- Unit tests:
  `pytest tests/test_native_stdloc_feedback.py tests/test_negative_support_memory.py tests/test_solver_coverage_coreset.py tests/test_export_lsf_solver_aware_map.py -q`
  passed with `33 passed, 14 warnings`.
