# ULF Sparse Solver-Feedback Reset Status 2026-06-11

## Current Mainline Contract

The intended sparse-only mainline is:

```text
Stage 1: ULF K.C./visibility/mask landmark sampling
Stage 2: ULF geometry-weighted descriptor aggregation
Stage 3: initial STDLoc-style online scene detector
Stage 4: self-map sparse PnP with the complete initial system
Stage 5: solver-feedback alternating updates
Stage 6: frozen disjoint train-dev sparse eval
```

Offline descriptor fusion / feature-log rewriting and SuperPoint-teacher
distillation are diagnostic only. They must not be called by the main launcher.
ShopFacade is no longer the first gate for this reset path.

## Implemented Corrections

- `launch_ulfloc_solver_feedback_mainline.py` no longer emits offline fusion or
  feature-log rewriting commands.
- `train_ulfloc_scene_detector_online.py` disables SuperPoint teacher by
  default; it is only enabled by an explicit diagnostic flag.
- `launch_ulfloc_solver_feedback_mainline.py` now trains an initial online
  scene detector before the self-map trace and uses that detector to generate
  Feedback v4.
- The old query-conditioned plan is marked deprecated and points to the reset
  spec/plan.
- Experiment protocol now treats SuperPoint teacher extraction as diagnostic
  only for this ULF sparse mainline.
- Non-rerank scene-detector modes no longer report `native_keep_fraction` as an
  effective parameter, because only `rerank_superpoint` consumes it.

## GreatCourt Large-Scene Sparse Gate

All numbers below are GreatCourt train-dev sparse-only and use the same ULF
native map/log source.

| config | median TE cm | median RE deg | R@10cm/5deg | R@5cm/5deg | mean inliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| native ULF sparse | 14.4342 | 0.0938 | 0.3648 | 0.1531 | 477.20 |
| online detector residual, 200 iter, stdloc_fullres | 20.5866 | 0.1527 | 0.1922 | 0.0326 | 537.89 |

The detector residual smoke does not pass the large-scene gate. It increases
inlier count while worsening pose accuracy, which indicates many extra matches
are geometrically misleading under GreatCourt repeated structure.

## Feedback v4 Audit

GreatCourt self-map sparse trace:

- split: `train_selfmap`
- query rows: 1224
- correspondence rows: 2,506,752

Feedback v4:

- query_count: 1224
- protected_support_count: 56,355
- positive_inlier_count: 33,336
- harmful_negative_count: 63,258
- risky_competitor_negative_count: 57,154

`query_tokens.pt` is not a valid activation source for this run because the
trace was exported without per-query match descriptors. This does not invalidate
Feedback v4 correspondence/query outcome labels, but it blocks activation from
being a main claim.

## Current Claim Status

The reset implementation is partially corrected, but the method is not yet
validated as a positive sparse result:

- mainline leakage from fusion/SP-teacher defaults: fixed;
- launcher order for initial detector -> feedback -> residual detector: fixed;
- GreatCourt native baseline: established;
- Feedback v4 solver residual labels: available;
- STDLoc-style direct detector residual: implemented but negative on GreatCourt
  smoke;
- online aggregation/view-selection loop: not implemented yet;
- online activation loop: not implemented yet.

Next technical focus should be detector failure analysis on GreatCourt:
heatmap/target visualization, descriptor alignment at detected points, and
whether direct heatmap keypoints preserve the descriptor-quality distribution
needed by ULF sparse matching.
