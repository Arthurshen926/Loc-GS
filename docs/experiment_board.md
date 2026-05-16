# Experiment Board

This document defines the Loc-GS experiment board format. Generate a live board
from result roots with:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.update_experiment_board \
  --result_roots output/stdloc_hybrid output/stdloc_native/results \
  --output_markdown docs/experiment_board.md \
  --output_json output/experiment_board.json
```

The script scans each root for `summary.json` or `metrics_summary.json`, reads
sidecar `manifest.json` and `split_audit.json` from the same run directory, and
writes:

- a Markdown table;
- compact JSON;
- one row per discovered run.

## Roles

Rows are marked as one of:

- `main_candidate`: paper-facing candidate with manifest and passed split audit.
- `ablation`: controlled comparison, usually marked by `manifest.run_role`.
- `diagnostic`: useful for debugging or failure analysis but not paper-safe.
- `rejected`: failed split audit or invalid run.

Missing manifest or split audit makes `paper_safe = false`; it does not pass by
default.

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
