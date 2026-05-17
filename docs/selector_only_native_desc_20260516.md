# Selector-Only Native Descriptor Evidence, 2026-05-16

## Mainline Decision

The current paper-facing mainline is **Loc-GS Sampling Field**:

```text
native STDLoc descriptor / geometry / radiance field
  -> train/rendered self-localization rehearsal
  -> per-Gaussian localization utility selector
  -> STDLoc-compatible locability / detector / sampling payload
  -> one descriptor matching + OpenCV PROSAC PnP + dense refinement path
```

The method does **not** claim descriptor replacement. Protected descriptor
residuals are retained as ablations because they improve some medians but are
less stable on strict recall. The clean candidate keeps `descriptor_mode=native`
and only distills self-localization feedback into selection / locability.

## Current Results

Full Cambridge, dense stage:

| Method | Descriptor | Selector | Median cm/deg | R@10 | R@5 | R@2 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Native STDLoc | native | no | 12.5725 / 0.1578 | 0.4860 | 0.2776 | 0.0848 |
| Selector-only 0.05 | native | sampled-score + locability | 12.4238 / 0.1593 | 0.4857 | 0.2815 | 0.0879 |
| Delta | native | selector | -0.1487 / +0.0015 | -0.0003 | +0.0039 | +0.0031 |

q80 Cambridge, dense stage:

| Method | Descriptor | Selector | Median cm/deg | R@10 | R@5 | R@2 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Native STDLoc | native | no | 9.1045 / 0.1128 | 0.6000 | 0.3675 | 0.1075 |
| Selector-only 0.05 | native | sampled-score + locability | 9.0268 / 0.1144 | 0.6050 | 0.3725 | 0.1175 |
| Selector-only 0.10 | native | sampled-score + locability | 9.0687 / 0.1114 | 0.6050 | 0.3675 | 0.1150 |

`selector0.10` was checked as a fixed global ablation, but it did not dominate
`selector0.05`; further blend tuning on test results is not paper-safe. The
next paper-safe route is self-map calibration or budgeted resampling, not
additional test-set blend search.

## Claim Boundary

Supported:

- Self-localization feedback can be distilled into a single STDLoc-compatible
  localization utility / selector field.
- The selector improves full Cambridge dense median, R@5, and R@2 while keeping
  native descriptors and a single query-time path.
- The result is evidence for feature selection / sampling-field reconstruction,
  not for descriptor replacement.

Not supported yet:

- Strong SOTA accuracy.
- A learned Gaussian descriptor that beats native STDLoc descriptors.
- Inference-time branch selection.
- A paper-safe main-candidate label, because the current feedback cache audit
  lacks recoverable per-image split id lists.

## Audit Bundle

Current audit bundle:

```text
output/unified_lff_v2/audit/selector005_native_desc_full_20260516/
  command.txt
  git_status.txt
  manifest.json
  metrics_summary.json
  split_audit.json
```

The bundle is intentionally marked `audit_status=unknown`, not passed. The
cache manifests record train/rendered rehearsal phase counts, but they do not
store enough source image ids to prove disjointness mechanically. This result
should be treated as an ablation / clean candidate under investigation until a
new feedback cache writes full split manifests.

## Required Next Evidence

1. Causality ablations: native, selector, uniform, permuted, inverted,
   detector-only, locability-only, and both.
2. Selector-guided landmark resampling under the same landmark budget, so the
   field changes `sampled_idx.pkl` rather than only nudging existing scores.
3. Landmark-budget and latency sweep: `16384 / 8192 / 4096 / 2048 / 1024`
   landmarks, with native STDLoc and Loc-GS Sampling Field under the same
   evaluator and GPU.
4. A new paper-safe feedback cache manifest containing train/rendered source ids
   and explicit disjointness from Cambridge test query ids.
