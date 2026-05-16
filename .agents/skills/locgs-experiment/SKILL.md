---
name: locgs-experiment
description: Use for Loc-GS Cambridge/STDLoc experiment setup, smoke checks, result summaries, comparisons, and manifests.
---

# Loc-GS Experiment Skill

Use `locgsctl` before writing long commands by hand.

## Required First Checks

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.locgsctl status
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.locgsctl list-scenes
```

## Safe Smoke Check

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.locgsctl smoke --scene ShopFacade --dry-run
```

Do not start full Cambridge experiments unless explicitly requested.

## Summaries And Comparisons

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.locgsctl summarize output/stdloc_hybrid/ShopFacade/eval
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.locgsctl compare BASELINE_DIR CANDIDATE_DIR --stage dense
```

## Research Safety

- Do not modify evaluator behavior or metric definitions.
- Do not use Cambridge test queries for training, calibration, self-map
  reliability, selector labels, feedback banks, or model selection.
- Keep real query inference on one path: matching, OpenCV PROSAC/RANSAC PnP,
  then STDLoc-style dense refinement.
- Generate a manifest for every new experiment artifact.
