# LSF v3 Hard-Scene Protocol, 2026-05-22

## Scope

LSF v3 keeps the paper claim fixed:

```text
self-localization feedback -> solver-level support selection -> one STDLoc path
```

The deployed query path remains:

```text
feature matching -> OpenCV PROSAC/RANSAC PnP -> STDLoc dense refinement
```

The new code affects only reconstruction-time or self-map map export decisions.
It is not a per-query branch selector and does not change the vendored STDLoc
evaluator.

## Implemented Hard-Scene Gate

`loc_gs.stdloc_native.solver_admissibility.is_replacement_admissible` now checks
these replacement deltas for every hard query id:

```text
support
viable_tuple_mass
logdet_H
min_eigenvalue
dense_worsen_risk
ambiguity
```

A same-budget replacement can be rejected by:

```text
support_drop
tuple_mass_drop
logdet_drop
min_eigen_drop
dense_worsen_rise
ambiguity_rise
hard_query_cvar
missing_candidate_gain
```

`loc_gs.scripts.export_lsf_solver_aware_map` reads these thresholds from the
`solver_admissibility_path` JSON:

```json
{
  "hard_query_ids": ["seq/frame00001.png"],
  "candidate_gain": {
    "123": {
      "seq/frame00001.png": {
        "support": 1.0,
        "viable_tuple_mass": 1.0,
        "logdet_H": 0.2,
        "min_eigenvalue": 0.1,
        "dense_worsen_risk": 0.0,
        "ambiguity": 0.0
      }
    }
  },
  "source_loss": {
    "45": {
      "seq/frame00001.png": {
        "support": 1.0,
        "viable_tuple_mass": 1.0,
        "logdet_H": 0.2,
        "min_eigenvalue": 0.1,
        "dense_worsen_risk": 0.0,
        "ambiguity": 0.0
      }
    }
  },
  "thresholds": {
    "min_support_delta": 0.0,
    "min_viable_tuple_delta": 0.0,
    "min_logdet_delta": 0.0,
    "min_min_eigen_delta": 0.0,
    "max_dense_worsen_delta": 0.0,
    "max_ambiguity_delta": 0.0,
    "cvar_alpha": 0.2,
    "min_cvar_score": 0.0,
    "cvar_weights": {
      "support": 1.0,
      "viable_tuple_mass": 1.0,
      "logdet_H": 1.0,
      "min_eigenvalue": 1.0,
      "dense_worsen_risk": -1.0,
      "ambiguity": -1.0
    }
  }
}
```

Aliases accepted by the implementation include `min_eigen`,
`min_eigen_delta`, `max_dense_worsen_risk_delta`, and
`hard_query_cvar_alpha`.

Rejected examples are written into `manifest.json` under
`solver_admissibility.rejected_examples`, including the delta and CVaR payload.

## Fixed-Recipe Promotion Gate

Use `loc_gs.reporting.fixed_recipe_gate.evaluate_fixed_recipe_gate` before any
Cambridge test/full paper-facing run. The required train-dev gate is:

```text
ShopFacade: must stay positive
OldHospital: dense metrics must not regress
StMarysChurch: dense metrics must not regress
macro dense median/R@5/R@2: must not regress
```

The gate operates on baseline and candidate dictionaries keyed by scene, each
with a dense metrics block containing:

```text
median_te_cm
recall_5cm_5deg
recall_2cm_2deg
```

Passing this gate is necessary but not sufficient for a submission result. A run
still needs the audit bundle and split audit required by `AGENTS.md`.

## Submission Alignment

`loc_gs.reporting.submission_alignment.build_submission_alignment_report` and
`loc_gs.scripts.update_experiment_board` now flag missing submission-table
requirements:

```text
dense median_te_cm
dense median_re_deg
dense recall_10cm_5deg
dense recall_5cm_5deg
dense recall_2cm_2deg
runtime
memory
map_size
edit_budget
```

The board integration is reporting-only. It does not change metrics or evaluator
behavior.

ULF-Loc is tracked as an external paper-reported reference, not a local
reproduction blocker. Reports must label its Cambridge numbers as
paper-reported and not locally reproduced. LSF-Loc claims still stay at
train-dev/fixed-recipe level until the fixed recipe passes paper-safe split
audits and a frozen full evaluation.

`loc_gs.scripts.write_ulfloc_alignment_reference --paper-reported-ok` writes
this reporting mode explicitly.

## Next Runs

Run the fixed recipe on q80 and q160 train-dev for five Cambridge scenes. The
recipe may advance only if:

```text
ShopFacade remains positive
OldHospital dense does not regress
StMarysChurch translation and recall are at least neutral
macro dense median/R@5/R@2 do not regress
all runs include manifest, command, metrics_summary, split_audit, and git status/diff
feedback bank split_name is not test
```

Only after this gate passes should the recipe be frozen for full split
paper-facing evaluation.

A scene-level reconstruction-time acceptance report may select native for a
scene whose candidate map fails the predeclared train-dev gate. This is a
fixed map acceptance policy, not a per-query branch selector, and must be
frozen before any full/test evaluation.
