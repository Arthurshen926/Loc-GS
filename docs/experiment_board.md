# Experiment Board

This document defines the Loc-GS experiment board format. Generate a live board
from result roots with:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.update_experiment_board \
  --result_roots output/stdloc_hybrid output/stdloc_native/results \
  --output_markdown docs/experiment_board.md \
  --output_json output/experiment_board.json
```

The script scans each root for `metrics_summary.json` or `summary.json`, reads
sidecar audit files from the same run directory, and writes:

- a Markdown table;
- compact JSON;
- one row per discovered run.

When both metric files exist in the same run directory, `metrics_summary.json`
is used as the canonical audit-bundle source, including when `--result_roots`
points directly at that run's `summary.json`.

## Roles

Rows are marked as one of:

- `main_candidate`: paper-facing candidate with a complete audit bundle and
  passed split audit.
- `ablation`: controlled comparison, usually marked by `manifest.run_role`.
- `diagnostic`: useful for debugging or failure analysis but not paper-safe.
- `rejected`: failed split audit or invalid run.

Missing manifest, required manifest fields, split audit,
`metrics_summary.json`, `command.txt`, or git diff/status evidence makes
`paper_safe = false`; it does not pass by default. Required manifest fields are
`git_commit`, timestamp, command, scene, split, checkpoint path, map path, data
root(s), hyperparameters, `rho`, and flags for feedback, residual, and selector
enablement.

## Columns

The Markdown board includes:

- run directory name;
- scene;
- role;
- paper-safe flag;
- dense median translation/rotation;
- dense R@10, R@5, R@2;
- paper-safety reason.

The script only aggregates existing outputs. It does not modify experiment
results or run Cambridge evaluation.
