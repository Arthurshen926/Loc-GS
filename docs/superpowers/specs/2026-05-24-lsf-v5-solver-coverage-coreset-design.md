# LSF v5 Solver-Coverage Coreset Design

## Goal

Move LSF-Loc from small per-landmark utility edits to a solver-coverage
selection method that can produce larger, defensible precision gains while
preserving the paper-safe STDLoc evaluation contract.

## Problem

The current LSF v4 path changes too little of the map and optimizes the wrong
unit. It uses per-landmark utility and local add/drop replacement, while PnP
solvability depends on query-level correspondence sets and tuple geometry. The
latest diagnostics show:

- strict official poselib sparse edits are too weak;
- fixed SPGS-prior consumption helps only after scene-level acceptance;
- dense-only LSF is not a standalone positive teacher;
- adaptive candidate exports may silently use the full Gaussian set when no
  candidate pool is provided;
- map manifests do not reliably expose the candidate pool path used by export.

## Method

LSF v5 introduces a solver-coverage coreset selector. It reads the existing
paper-safe artifacts:

- source sampled landmarks;
- native proposal candidate pool;
- LSF selector / risk tensors;
- feedback-bank-derived solver admissibility constraints.

Instead of scoring each replacement independently, the selector maintains hard
query coverage. A candidate receives higher priority when it adds support,
viable tuple mass, logdet, or min-eigen information to queries that are still
under-covered. A source landmark becomes harder to drop when it is important to
hard queries with low current coverage. Dense worsen risk and ambiguity remain
penalties.

The output is still a single fixed same-budget map. There is no per-query
branch selector, no oracle ordering, and no Cambridge test feedback.

## Architecture

1. `loc_gs.stdloc_native.solver_coverage_coreset`
   - Pure functions for parsing solver constraints and selecting same-budget
     coverage-aware local edits.
   - Unit tested with synthetic hard-query/candidate/source maps.

2. `loc_gs.scripts.export_lsf_solver_aware_map`
   - Adds `--selection_policy local|coverage`.
   - Adds `--require_candidate_pool` for mainline exports.
   - Records `candidate_pool_path`, `candidate_pool_required`, and policy
     metadata in the map manifest.
   - Uses the existing local replacement by default for backward compatibility.

3. v5 experiment artifacts
   - Export maps with candidate pool required and coverage policy enabled.
   - Evaluate only train-dev q80/q160 before any full/test use.
   - Report raw all-selected results plus precision-primary scene-level
     acceptance.

## Expected Impact

This is a method-level change because it changes the optimization objective from
unary matchability/support to query-level solver coverage. It should be able to
use larger edit budgets than v4 without destroying hard-query support, which is
the necessary condition for gains larger than the current sub-millimeter to
sub-centimeter changes.

## Safety

- `third_party/stdloc` evaluator remains unmodified.
- Test split is rejected for feedback/constraints.
- Mainline v5 exports require an explicit candidate pool.
- Evaluation remains one fixed path: matching, fixed PnP solver, STDLoc dense
  refinement.
- Dense-only LSF remains diagnostic unless a fixed train-dev recipe proves it.

## First Acceptance Target

Run q80/q160 train-dev on five Cambridge scenes.

Promotion criteria:

- raw macro dense median TE improves by at least `0.25 cm`, or
- precision-primary scene-level gate improves q80 and q160 by at least `0.25 cm`
  with no selected-scene dense median TE regression;
- split audits pass;
- selected runs are fixed poselib method candidates or strict official parity
  candidates;
- no per-query branching.

