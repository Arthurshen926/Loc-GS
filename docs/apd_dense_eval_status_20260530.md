# APD-Dense Diagnostic Evaluation - 2026-05-30

## Scope

This is a diagnostic train-split smoke evaluation of the APD-Dense integration.
It is not paper-facing. Current `locgsctl list-scenes` reports default sampled
maps as 8192 while the expected native count is 16384.

The evaluated query is:

```text
scene: ShopFacade
split: train
image: seq2/frame00001.png
candidate_root: output/stdloc_native/cambridge_test_v6_guarded512_20260525/selected
```

## Result

| Variant | Base Dense TE cm | APD TE cm | Delta cm | Base RE deg | APD RE deg |
| --- | ---: | ---: | ---: | ---: | ---: |
| APD patch, anchor=1, sparse reference | 2.3768 | 3.8448 | +1.4680 | 0.0524 | 0.1102 |
| APD no patch, anchor=1, sparse reference | 2.3768 | 3.9520 | +1.5751 | 0.0524 | 0.1102 |
| APD patch, anchor=0, sparse reference | 2.3768 | 3.1202 | +0.7434 | 0.0524 | 0.0822 |
| APD no patch, anchor=0, sparse reference | 2.3768 | 3.2880 | +0.9112 | 0.0524 | 0.0758 |
| APD patch, anchor=1, dense reference | 2.3768 | 3.8439 | +1.4671 | 0.0524 | 0.1094 |
| APD patch, anchor=0, dense reference | 2.3768 | 3.1186 | +0.7418 | 0.0524 | 0.0812 |
| APD no patch, anchor=0, dense reference | 2.3768 | 3.2867 | +0.9099 | 0.0524 | 0.0778 |

## Diagnosis

This result rejects the current APD robust-refinement path as a main method.
The regression remains even when patch candidates and sparse-anchor residuals
are disabled. Therefore the primary issue is not patch noise or anchor weight:
it is that APD replaces native STDLoc dense PnP with a custom local robust
refinement over dense correspondences.

Native dense already improves this normal query:

```text
sparse TE: 3.7085 cm
native dense TE: 2.3768 cm
```

Current APD moves the estimate away from this stronger native dense pose.

## Code Adjustment Made During Evaluation

APD now accepts an optional `dense_reference_pose` and the visualizer/evaluator
pass the native dense pose as that reference. This is the correct reference for
a dense-stage add-on. However, the diagnostic above shows that changing the
reference alone is not sufficient.

## Decision

Do not run full Cambridge APD evaluation with the current final-pose optimizer.
It would waste compute and likely increase normal-case regressions.

The next implementation change should be stricter:

```text
APD should preserve native STDLoc dense PnP as the default final pose.
Clean render and patch outputs should only provide additional dense
correspondence candidates to the same OpenCV/STDLoc-style dense PnP path,
or remain diagnostic until they prove no-regression on train/self-map.
```

In other words, APD should be candidate construction plus anchor diagnostics,
not a replacement robust optimizer.
