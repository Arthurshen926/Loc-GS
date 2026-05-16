# locgsctl

`locgsctl` is a compact, agent-friendly helper for Loc-GS experiment checks and
result aggregation. It does not start full Cambridge experiments.

Run it from the repository root with the verified Python:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.locgsctl --help
```

All command output is compact JSON by default.

## status

Prints environment and key path status.

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.locgsctl status
```

Output fields:

- `git_commit`
- `python_executable`
- `cuda_visible_devices`
- `paths.repo_root`
- `paths.loc_gs`
- `paths.docs`
- `paths.tests`
- `paths.third_party_stdloc`
- `paths.output`
- `paths.stdloc_superpoint_weights`

Missing paths are reported with `"exists": false`; the command does not download
or install anything.

## list-scenes

Lists Cambridge scenes plus default data, STDLoc map, and Loc-GS checkpoint
paths.

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.locgsctl list-scenes
```

Optional roots:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.locgsctl list-scenes \
  --data-root /mnt/pool/sqy/dataset/Cambridge \
  --map-root output/stdloc/map_cambridge_spgs \
  --checkpoint-root output/stdloc_hybrid
```

## smoke

Runs minimal path checks for one scene and explicitly avoids long experiments.

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.locgsctl smoke \
  --scene ShopFacade \
  --dry-run
```

Checks:

- scene data root
- STDLoc map path
- checkpoint path
- `loc_gs` package path
- vendored `third_party/stdloc` path

Without `--dry-run`, missing paths make the command return non-zero. With
`--dry-run`, the command returns JSON schema and missing-path diagnostics
without failing.

## summarize

Reads a run directory or a direct JSON file and emits compact stage metrics.
Directories are searched for `summary.json`, `metrics_summary.json`, then
`metrics.json`.

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.locgsctl summarize \
  output/stdloc_hybrid/ShopFacade/eval
```

Recognized aliases include:

- translation: `median_te_cm`, `median_te`, `median_translation_cm`
- rotation: `median_re_deg`, `median_ae`, `median_re`
- recall: `recall_10cm_5deg`, `recall_10cm_5d`, `r10`;
  `recall_5cm_5deg`, `recall_5cm_5d`, `r5`;
  `recall_2cm_2deg`, `recall_2cm_2d`, `r2`

## compare

Compares baseline and candidate summaries. Deltas are candidate minus baseline.
Negative median deltas are improvements; positive recall deltas are
improvements.

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.locgsctl compare \
  output/stdloc_native/results/parity/ShopFacade \
  output/stdloc_hybrid/ShopFacade/eval \
  --stage dense
```

Output fields:

- `baseline`
- `candidate`
- `delta`
- `stage`
- `baseline_source`
- `candidate_source`

## manifest

Generates a manifest template and optionally writes it to disk.

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.locgsctl manifest \
  --scene ShopFacade \
  --split selfmap_train \
  --checkpoint output/stdloc_hybrid/ShopFacade/latest.pth \
  --map output/stdloc/map_cambridge_spgs/ShopFacade \
  --feedback-enabled \
  --rho 0.5 \
  --residual-enabled \
  --output output/stdloc_hybrid/ShopFacade/manifest.json \
  --command -- python -m loc_gs.scripts.eval_cambridge_hybrid --scene ShopFacade
```

Manifest fields include git commit, UTC timestamp, scene, split, command,
checkpoint, map, feedback flags, `rho`, and notes.
