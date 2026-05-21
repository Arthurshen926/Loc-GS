# Dense LSF Prior Findings, 2026-05-21

## Scope

This note records train-dev diagnostics only. No Cambridge test image, pose, feature,
descriptor, or result was used for method selection.

The evaluated intervention keeps the native STDLoc map, sampled landmark ids,
descriptors, sparse matching, PnP, and dense refinement path unchanged. It only
sets `dense.locability_prior_weight` in the STDLoc dense matching score.

## Claim Guard

The current evidence does **not** support a paper-facing claim that a fixed raw
locability prior is a universal improvement. It supports a narrower diagnostic:

> Raw dense locability/support can expose scenes where dense refinement has
> hard-tail failures, but it must be mediated by self-map risk feedback before it
> becomes a unified LSF component.

This keeps the project aligned with the Loc-GS claim: localization support should
come from reconstruction/self-localization feedback, not from a manually selected
test-time branch or per-query path.

## StMarysChurch Train-Dev Evidence

On StMarysChurch, `dense_locability_prior_weight=0.05` is consistently useful
for dense hard-tail robustness across three train-dev splits.

| split | median delta | R5 delta | R2 delta | hard10 mean TE delta |
|---|---:|---:|---:|---:|
| q80 | -0.0776 cm | +0.0125 | +0.0000 | -9240.48 cm |
| s3/q75 | -0.0269 cm | +0.0000 | +0.0000 | -2597.32 cm |
| s5/q100 | +0.0389 cm | +0.0000 | +0.0000 | -1080.16 cm |

This is positive evidence that dense-stage support can reduce hard-query tail
failures without changing the sparse map.

## Cross-Scene Q80 Fixed-Recipe Check

Fixed global dense priors are not yet strong enough:

| method | macro median delta | macro R5 delta | macro R2 delta | macro hard10 delta |
|---|---:|---:|---:|---:|
| `prior_de002` | +0.0572 cm | +0.0025 | -0.0025 | -412.54 cm |
| `prior_de005` | +0.0509 cm | +0.0000 | +0.0000 | -1848.02 cm |

The hard-tail mean improves mostly because of StMarysChurch, but median degrades
on GreatCourt and OldHospital. Therefore this is not yet a unified fixed recipe.

## Self-Map Gate Diagnostic

A train-dev diagnostic gate that accepts `prior_de005` only when strict recall is
not reduced and hard10 mean TE improves by at least 100 cm selects StMarysChurch
only. This gives:

| setting | macro median delta | macro R5 delta | macro R2 delta |
|---|---:|---:|---:|
| fixed `prior_de005` | +0.0509 cm | +0.0000 | +0.0000 |
| gated diagnostic | -0.0155 cm | +0.0025 | +0.0000 |

This gate is a critic/validation signal, not a final method. It should be used
to generate supervision for a unified dense support field, not as a test-time
branch selector.

## Next Direction

1. Keep `dense_prior=0.05` as a positive teacher on StMarysChurch hard-tail
   cases, not as a fixed global recipe.
2. Add dense self-map labels: dense improved/worsened, hard-tail query groups,
   and rendered locability reliability.
3. Distill those labels into a dense LSF mask/readout that can suppress raw
   locability when it is not predictive.
4. Re-evaluate with one fixed recipe across all train-dev scenes before any
   paper-safe test/full run.

## Artifact Pointers

- `output/lsf_query_support_20260520/dense_prior_train_dev_summary.md`
- `output/lsf_query_support_20260520/dense_prior_q80_cross_scene_summary.md`
- `output/lsf_query_support_20260520/dense_prior_q80_fixed_recipe_summary.md`
- `output/lsf_query_support_20260520/dense_prior_q80_selfmap_gate_diagnostic.md`
