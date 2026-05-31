# APD-Dense Phase0 Hard-Case Evaluation - 2026-05-30

## Scope

This is a train/self-map diagnostic evaluation on Phase0 hard cases only.
It does not use official Cambridge test cases for tuning.

Case source:

```text
output/diagnostics/clean_render_phase0_benchmark_20260530/phase0_cases.csv
case_type: A_sparse_good_dense_bad
source_split: train
```

This type is the dense-transition target case:

```text
sparse pose is already reasonable,
but native dense refinement worsens it by >= 20 cm.
```

## Commands

APD with patch candidates:

```bash
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/cybersim_agent/bin/python \
  -m loc_gs.scripts.eval_sparse_conditioned_dense_control \
  --candidate_root output/stdloc_native/cambridge_test_v6_guarded512_20260525/selected \
  --output_dir output/diagnostics/apd_dense_eval_phase0_A_all_20260530 \
  --eval_split train \
  --case_csv output/diagnostics/clean_render_phase0_benchmark_20260530/phase0_cases.csv \
  --case_type A_sparse_good_dense_bad \
  --phase0_split all \
  --progress_interval 1 \
  --candidate_render_control none \
  --apd_dense \
  --apd_include_patch_candidates
```

APD without patch candidates:

```bash
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/cybersim_agent/bin/python \
  -m loc_gs.scripts.eval_sparse_conditioned_dense_control \
  --candidate_root output/stdloc_native/cambridge_test_v6_guarded512_20260525/selected \
  --output_dir output/diagnostics/apd_dense_eval_phase0_A_all_no_patch_20260530 \
  --eval_split train \
  --case_csv output/diagnostics/clean_render_phase0_benchmark_20260530/phase0_cases.csv \
  --case_type A_sparse_good_dense_bad \
  --phase0_split all \
  --progress_interval 1 \
  --candidate_render_control none \
  --apd_dense
```

## Summary

| Method | Median TE cm | P90 TE cm | P95 TE cm | R@50cm/5deg | Dense-worsened count |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native dense | 40.5384 | 53.2902 | 54.3606 | 0.8333 | 7 |
| APD + patch | 21.2964 | 24.0983 | 24.3879 | 1.0000 | 5 |
| APD no patch | 24.1078 | 27.6164 | 27.7169 | 1.0000 | 6 |

Compared with native dense, APD + patch improves median TE by `-19.2420 cm`
and reduces the tail substantially.

## Per-Case APD + Patch

| Scene | Image | Sparse | Native dense | APD | APD-native | APD-sparse |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GreatCourt | seq3/frame00135.png | 13.42 | 36.64 | 32.30 | -4.33 | +18.88 |
| GreatCourt | seq3/frame00228.png | 12.33 | 37.13 | 23.66 | -13.47 | +11.33 |
| GreatCourt | seq3/frame00315.png | 4.71 | 27.66 | 15.88 | -11.78 | +11.17 |
| GreatCourt | seq5/frame00313.png | 29.15 | 69.53 | 32.43 | -37.10 | +3.27 |
| GreatCourt | seq5/frame00393.png | 29.56 | 60.97 | 34.74 | -26.22 | +5.18 |
| GreatCourt | seq5/frame00571.png | 11.70 | 42.37 | 19.69 | -22.68 | +7.99 |
| ShopFacade | seq2/frame00218.png | 13.72 | 41.33 | 14.61 | -26.72 | +0.89 |

## Interpretation

This partially achieves the original hard-case goal:

- APD improves all 7 A-type sparse-good/dense-bad cases relative to native dense.
- There are no APD-vs-native regressions over 20 cm or 50 cm in this subset.
- Patch candidates help: median TE improves from `24.1078 cm` without patch to
  `21.2964 cm` with patch.

However, APD still does not fully solve the transition problem:

- APD remains worse than sparse-only on every A-type case.
- `clean_render_selected_label` is `base` on these cases, so the gain is not
  from guided/gated clean render. It comes from APD dense/anchor robust
  refinement and patch candidates.
- Normal-case smoke still regresses, so this cannot be promoted to a full
  method yet.

The next minimal change should be a sparse-anchor step acceptance or line search:
if APD improves over native dense but is still worse than sparse-anchor
consistency, the final update should be bounded between native dense and sparse,
without adding another case-heavy controller.
