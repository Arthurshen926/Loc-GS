# Loc-GS Agent Rules

These rules apply to all future Codex and CodexPotter work in this repository.
They protect the Cambridge/STDLoc research claim from leakage, evaluator drift,
and irreproducible experiment handling.

## Non-Negotiable Research Constraints

- Do not modify Cambridge test splits, ground-truth poses, official STDLoc
  evaluator behavior, or metric definitions to improve numbers.
- Do not use test query pose, image, feature, descriptor, or evaluation result
  for training, self-map reliability, calibration, model selection, branch
  selection, selector labels, hard-negative mining, or hyperparameter tuning.
- `rho=0`, `alpha=0`, and disabled feedback options must preserve native
  STDLoc evaluator parity. Validate this with the native STDLoc wrappers before
  claiming any Loc-GS contribution.
- Real query inference must stay on one path: feature matching, OpenCV
  PROSAC/RANSAC PnP, then STDLoc-style dense refinement. Multi-hypothesis,
  branch-selected, or oracle-ordered paths are diagnostics unless a task
  explicitly labels them as ablations.
- Quality gates may be used only for reconstruction-time or self-map validation
  of a candidate map/checkpoint, or as a diagnostic. They must not become
  per-query branch selectors.
- SceneMatchNet, LoFTR, static matchability priors, and oracle ordering are not
  the main method unless a new full-split paper-safe experiment proves otherwise.

## Experiment And Artifact Rules

- Every new experiment must write a `manifest.json` containing at least:
  `git_commit`, command, scene, split, checkpoint path, map path, data roots,
  hyperparameters, timestamp, and whether residual/selector/rho feedback was
  enabled.
- Paper-facing export/eval outputs must also include enough audit material to
  reconstruct the run: `command.txt`, `metrics_summary.json`, split audit, and
  either `git_diff.patch` or `git_status.txt`.
- Missing split information is not a pass. Mark it as `unknown` and keep the run
  out of paper-safe tables until the split can be audited.
- Keep output compact by default. Write large logs to files.
- Never auto-download Cambridge data, checkpoints, or large dependencies.
  Missing data/checkpoints should produce clear errors.

## Implementation Rules

- Do not change files under `third_party/stdloc` evaluator paths unless the task
  is explicitly about vendored reproducibility and includes a parity test.
- New modules must include a focused unit test or smoke test. Prefer synthetic
  inputs for tests that should run without Cambridge data or a GPU.
- Use the verified Python from the README when running repository checks:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python
```

- Prefer existing entry points and helpers:
  `loc_gs.stdloc_native.commands`, `loc_gs.scripts.eval_stdloc_native`,
  `loc_gs.scripts.train_stdloc_native`,
  `loc_gs.scripts.launch_stdloc_native_cambridge`,
  `loc_gs.scripts.train_cambridge_hybrid`,
  `loc_gs.scripts.eval_cambridge_hybrid`, and `loc_gs.scripts.locgsctl` when
  present.
- For Cambridge/STDLoc experiment tasks, use `locgsctl` first for status,
  smoke checks, summaries, manifests, and comparisons rather than hand-writing
  long commands.
- Do not start full Cambridge training/evaluation runs unless the user
  explicitly asks for long experiments.

## Paper-Safety Checklist

Before describing a result as paper-facing, verify:

- The baseline is native STDLoc parity from the vendored path, not a weakened
  local reimplementation.
- The split audit shows self-map/calibration image ids are disjoint from test
  image ids.
- Feedback bank `split_name` is not `test`.
- The result uses one fixed evaluator path for all test queries.
- The claim is stated at the right level: diagnostic ablations are not main
  results, negative results stay negative, and per-scene failures are reported.
