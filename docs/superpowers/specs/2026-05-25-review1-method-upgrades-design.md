# Review1 Method Upgrades Design

## Goal

Implement the seven method-level directions from `review1.md` as tested,
paper-safe building blocks without changing Cambridge splits, metric
definitions, or the vendored STDLoc evaluator.

## Scope

This sprint is not a direct full-test tuning pass. It adds method components
that can be evaluated on train-dev diagnostics first:

1. query-conditioned support banks;
2. solver-weighted landmark feature fusion;
3. dense match verification;
4. saturated query-tail coverage controller;
5. LSF detector target refinement;
6. pairwise negative support memory;
7. bank-level large-edit resampling policy.

Direction 4 is already implemented as LSF v6. The new work focuses on the
remaining six directions and a shared status document that explains how they
connect.

## Paper-Safety Contract

- No module may read Cambridge test poses, test feedback, or test metrics for
  training, model selection, hard-negative mining, routing, or calibration.
- New modules produce offline artifacts or fixed reconstruction-time decisions.
  They are not per-query result branch selectors.
- Real inference remains the STDLoc-compatible path: matching, single-path
  poselib/OpenCV PnP, then STDLoc dense refinement.
- Full Cambridge test evaluation is allowed only after a recipe is frozen from
  train-dev or self-map evidence.

## Components

### Query-Conditioned Support Banks

Create support-bank utilities that group training/self-map queries by observable
feedback features and build per-bank landmark activation weights. The first
version is deterministic and offline: it emits bank membership, support weights,
and fixed router metadata. It does not use test results.

### Solver-Weighted Landmark Feature Fusion

Create a tensor-level landmark descriptor fusion module. It fuses train/self-map
observed descriptors using solver, geometry, and visibility weights, then
applies a trust-region blend against the native descriptor. Native descriptors
remain fallback for landmarks without enough evidence.

### Dense Match Verifier

Move dense LSF from a scalar prior toward a correspondence verifier. The first
version scores matches using local geometry, LSF support, alpha dominance,
ambiguity, depth uncertainty, and pose leverage, then returns keep masks and
reweighted scores.

### LSF Detector Target Refinement

Add an offline detector-target builder from projected landmarks, solver support,
hard-negative risk, and dense-worsen risk. It emits heatmap and weight maps for
training-time refinement only.

### Negative Support Memory

Create a pairwise conflict graph from query/keypoint grouped hard negatives.
This converts hard-negative risk from unary penalties into group-level conflict
penalties usable by sampling and bank construction.

### Bank-Level Resampling Policy

Create a policy object that computes safe large-edit budgets and rejects
large-edit recipes unless saturation and conflict controls are present. This
prevents repeating v5/v6 uncontrolled 512-edit behavior.

## Validation

Each component gets focused synthetic unit tests that run without Cambridge data
or a GPU. After unit tests pass, the next stage is train-dev diagnostics:

- sparse-only descriptor-fusion smoke before dense evaluation;
- dense transition audit for verifier before full dense replacement;
- support-bank train-dev q80/q160 smoke before test/full evaluation.

## Acceptance

The sprint is successful when the six new components plus v6 are implemented,
tested, and documented. Experimental promotion requires separate evidence:
train-dev macro dense TE must improve beyond v5-128 and hard scenes must not
regress.

## Self-Review

- Placeholder scan: no TODO/TBD placeholders.
- Scope check: this is a method-component sprint, not a frozen Cambridge test
  recipe.
- Ambiguity check: per-query branch selection remains disallowed; all routing is
  fixed from observable pre-PnP inputs or offline reconstruction artifacts.
