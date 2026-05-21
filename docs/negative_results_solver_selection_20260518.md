# Negative Results: Solver Selection, 2026-05-18

These runs are retained as diagnostic evidence only. They should not be promoted
to paper-facing results or used for full/test recipe selection.

## supportguardcount_ret90

Hypothesis:

```text
Preserving hard-query native support counts while allowing limited selector
edits should keep the map safe and recover selector gains.
```

Result:

```text
Stage A failed.
Macro dense median_te regressed by about +0.519 cm.
R5 improved only about +0.005.
```

Why rejected:

```text
The accuracy regression is too large for a paper-facing candidate, and the
strict recall change is within small-run noise.
```

What it teaches:

```text
Hard-query support count preservation alone is not a sufficient invariant.
```

## supportguardall_ret90

Hypothesis:

```text
If every query preserves native positive support count, the map edit should be
safe enough to evaluate on shifted/full splits.
```

Result:

```text
All-query support audit passed with query_loss_count=0 and worst_support_delta=0.
Stage A still failed.
Macro dense median_te regressed by about +0.513 cm.
R5 improved only about +0.010 and R2 by about +0.005.
```

Why rejected:

```text
The candidate changes sampled_idx but does not pass the train-q80 gate.
No shifted/full/test promotion is allowed.
```

What it teaches:

```text
Support count can be non-decreasing while pose accuracy degrades. The new
solver diagnostic shows supportguardall-ret90 increases cached support/tuple
mass on OldHospital and StMarysChurch, but lowers mean logdet(H), which points
to weaker PnP geometry rather than missing matches.
```

## OldHospital ret95 / ret975

Hypothesis:

```text
Smaller edit distance from native should reduce supportguardall_ret90
regression while keeping any selector benefit.
```

Result:

```text
OldHospital ret95 and ret975 still regressed median translation in Stage A.
Hard-negative weight increases, including hn100 diagnostics, did not recover
the scene.
```

Why rejected:

```text
The failure survives conservative retention, so the issue is not simply too
many native landmarks dropped.
```

What it teaches:

```text
Replacement admissibility must be checked at the solver/query-set level. A
candidate can be high-score and low hard-negative under unary metrics yet still
harm hard-query geometric conditioning.
```

## StMarysChurch ret95 / hn100

Hypothesis:

```text
Retaining more native support or heavily suppressing hard negatives should fix
the StMarysChurch regression.
```

Result:

```text
Stage A still failed or remained below promotion thresholds.
```

Why rejected:

```text
The recipe does not produce a robust hard-scene improvement and remains an
unprincipled scalar-tuning variant.
```

What it teaches:

```text
StMarysChurch is a hard-query tail-risk case. Future edits should optimize
CVaR / worst-query logdet and ambiguity, not just global count or scalar risk.
```

## Paused Lines

The following are paused as mainline candidates:

```text
qcov/churn/retention scalar sweeps
inlier-only early exit
native subset low-budget speed probe
residual descriptor main claim
test-time branch or per-query multi-path selection
full/test candidate selection
```

They may be reused only as diagnostics or baselines after the solver-centric
Stage A gate is established.

