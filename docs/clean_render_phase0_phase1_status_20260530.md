# Clean Render Phase0/Phase1 Status 20260530

This note records the first two validation phases for Sparse-Anchored Clean
Dense Rendering. These runs are diagnostic-only and use train/self-map artifacts.
They are not paper-facing Cambridge test results.

## Phase0: Hard/Normal Case Stratification

Command:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.build_clean_render_phase0_benchmark \
  --results output/diagnostics/slcdp_train_selfmap_full_transition_basefallback_penalty075_20260529/GreatCourt_job/GreatCourt/results.json \
  --results output/diagnostics/slcdp_train_selfmap_full_transition_basefallback_penalty075_20260529/KingsCollege_job/KingsCollege/results.json \
  --results output/diagnostics/slcdp_train_selfmap_full_transition_basefallback_penalty075_20260529/OldHospital_job/OldHospital/results.json \
  --results output/diagnostics/slcdp_train_selfmap_full_transition_basefallback_penalty075_20260529/ShopFacade_job/ShopFacade/results.json \
  --results output/diagnostics/slcdp_train_selfmap_full_transition_basefallback_penalty075_20260529/StMarysChurch_job/StMarysChurch/results.json \
  --output_dir output/diagnostics/clean_render_phase0_benchmark_20260530 \
  --max_per_type 50 \
  --val_fraction 0.25
```

Output:

```text
output/diagnostics/clean_render_phase0_benchmark_20260530/
```

Case counts:

| Type | Count |
| --- | ---: |
| A_sparse_good_dense_bad | 7 |
| B_sparse_marginal_dense_recoverable | 50 |
| C_sparse_catastrophic | 50 |
| D_normal_dense_good | 50 |

Split counts:

| Phase0 Split | Count |
| --- | ---: |
| train | 119 |
| val | 38 |

Audit status:

- `official_test_used=false`
- `paper_safe_for_tuning=true`
- `source_splits=["train"]`
- source split is train/self-map, not official test

## Phase1: Clean Render Action Safety And Render Health

Command:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.build_clean_render_phase1_report \
  --phase0_cases output/diagnostics/clean_render_phase0_benchmark_20260530/phase0_cases.csv \
  --case_analysis output/diagnostics/slcdp_train_selfmap_sparse_conditioned_failure_analysis_20260529/case_analysis.csv \
  --output_dir output/diagnostics/clean_render_phase1_validation_20260530
```

Output:

```text
output/diagnostics/clean_render_phase1_validation_20260530/
```

Action safety summary:

| Type | Count | Non-Base Actions | Reg20 | Imp20 | Missing Delta | Median Delta cm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A_sparse_good_dense_bad | 7 | 0 | 0 | 0 | 0 | 0.0 |
| B_sparse_marginal_dense_recoverable | 50 | 2 | 0 | 0 | 0 | 0.0 |
| C_sparse_catastrophic | 50 | 4 | 0 | 4 | 0 | 0.0 |
| D_normal_dense_good | 50 | 0 | 0 | 0 | 0 | 0.0 |

Selected render health summary:

| Category | Count | Non-Base Selected | Median Visible | Median Near-Occluder | Median Feature Cos | Reg20 | Imp20 | Missing Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| improvement | 5 | 5 | 0.9423 | 0.0444 | 0.5448 | 0 | 4 | 0 |
| regression | 4 | 4 | 0.9486 | 0.0335 | 0.6213 | 4 | 0 | 0 |
| small_regression | 1 | 1 | 0.7603 | 0.1565 | 0.5932 | 0 | 0 | 0 |

## Interpretation

Phase0 is complete: the repository now has a fixed train/self-map diagnostic
benchmark that separates sparse-good/dense-bad cases, marginal recoverable
cases, sparse-catastrophic cases, and normal dense-good cases.

Phase1 is complete as a diagnostic validation layer, but it does not validate
clean render generation as a main method yet. The selected-render health signals
currently available in the existing case-analysis artifact do not separate
positive and negative outcomes: regression cases can have high anchor visibility
and high feature cosine. Therefore these metrics are insufficient for accepting
a clean render into final dense refinement.

Safety changes from review:

- Phase0 benchmark generation now rejects non-train source splits by default.
- Phase1 reports derive `official_test_used`, `paper_safe_for_tuning`, and
  `source_splits` from the Phase0 CSV instead of hardcoding them.
- Missing numeric deltas are counted explicitly instead of being treated as
  neutral zero-change outcomes.
- The standalone clean-render candidate report writes `split_audit.json`.

The next phase should add global geometric consistency, off-anchor consistency,
patch/global agreement, and sparse-anchor residual diagnostics before consuming
clean renders in the final pose path.

## Files

- `loc_gs/diagnostics/clean_render_phase0_benchmark.py`
- `loc_gs/diagnostics/clean_render_phase1_validation.py`
- `loc_gs/scripts/build_clean_render_phase0_benchmark.py`
- `loc_gs/scripts/build_clean_render_phase1_report.py`
- `tests/test_clean_render_phase0_benchmark.py`
- `tests/test_clean_render_phase1_validation.py`
- `output/diagnostics/clean_render_phase0_benchmark_20260530/report.md`
- `output/diagnostics/clean_render_phase1_validation_20260530/report.md`
