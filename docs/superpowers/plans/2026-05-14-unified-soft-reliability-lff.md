# Unified Soft Reliability LFF Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace scene-level hard branch selection with one evaluation/training-facing localization pipeline whose LFF and self-map signals are continuously modulated by reconstruction-time self-localization reliability.

**Architecture:** A small reliability helper converts self-map validation summaries into a scalar `rho` in `[0, 1]`. `eval_cambridge_hybrid.py` applies `rho` to existing Loc-GS knobs: descriptor residual strength, PLY/LFF blend strength, calibrated landmark prior, match-filter prior, and SceneMatch re-rank strength. The same code path runs for every scene; low-quality self-map feedback fades toward protected STDLoc-compatible behavior instead of switching branches.

**Tech Stack:** Python, PyTorch, argparse CLI, existing Cambridge eval launchers, pytest.

---

### Task 1: Reliability Unit Tests

**Files:**
- Test: `tests/test_eval_cambridge_hybrid_cli.py`
- Modify later: `loc_gs/scripts/eval_cambridge_hybrid.py`

- [ ] **Step 1: Write failing tests**

Add tests that assert:

```python
def test_selfmap_reliability_weight_is_continuous_and_quality_ordered():
    good = eval_cambridge_hybrid.selfmap_reliability_weight(3.0, center_cm=10.0, temperature_cm=1.0)
    border = eval_cambridge_hybrid.selfmap_reliability_weight(10.0, center_cm=10.0, temperature_cm=1.0)
    bad = eval_cambridge_hybrid.selfmap_reliability_weight(13.0, center_cm=10.0, temperature_cm=1.0)
    assert 0.95 < good < 1.0
    assert border == 0.5
    assert 0.0 < bad < 0.1


def test_apply_selfmap_reliability_softens_lff_knobs_without_branch_switching():
    args = build_argparser().parse_args([
        "--checkpoint", "output/model/latest.pth",
        "--descriptor_source", "hybrid_ply_blend",
        "--ply_loc_feature_weight", "0.9",
        "--hybrid_residual_alpha_max", "0.03",
        "--landmark_score_calibrated_matchability_weight", "0.4",
        "--match_calibrated_prior_weight", "0.2",
        "--match_filter_calibrated_score_weight", "0.25",
        "--scene_matcher_weight", "0.3",
    ])
    meta = eval_cambridge_hybrid.apply_selfmap_reliability(args, 0.25)
    assert args.descriptor_source == "hybrid_ply_blend"
    assert args.ply_loc_feature_weight == 0.975
    assert args.hybrid_residual_alpha_max == 0.0075
    assert args.landmark_score_calibrated_matchability_weight == 0.1
    assert args.match_calibrated_prior_weight == 0.05
    assert args.match_filter_calibrated_score_weight == 0.0625
    assert args.scene_matcher_weight == 0.075
    assert meta["rho"] == 0.25
```

- [ ] **Step 2: Run red test**

Run: `/root/miniconda3/envs/cybersim_agent/bin/python -m pytest tests/test_eval_cambridge_hybrid_cli.py -q`

Expected: fail because the helper functions do not exist.

### Task 2: Eval Helper Implementation

**Files:**
- Modify: `loc_gs/scripts/eval_cambridge_hybrid.py`

- [ ] **Step 1: Implement pure helpers**

Add:

```python
def selfmap_reliability_weight(median_te_cm, center_cm=10.0, temperature_cm=1.0):
    return sigmoid((center_cm - median_te_cm) / temperature_cm)
```

Also add summary loading from `summary.json` so `--selfmap_reliability_path` can point to a self-map eval directory or file.

- [ ] **Step 2: Implement argument scaling**

Add `apply_selfmap_reliability(args, rho)` that records original values and rewrites only continuous weights:

```python
args.hybrid_residual_alpha_max *= rho
args.landmark_score_calibrated_matchability_weight *= rho
args.match_calibrated_prior_weight *= rho
args.match_filter_calibrated_score_weight *= rho
args.scene_matcher_weight *= rho
args.ply_loc_feature_weight = 1.0 - rho * (1.0 - args.ply_loc_feature_weight)
```

- [ ] **Step 3: Run green test**

Run: `/root/miniconda3/envs/cybersim_agent/bin/python -m pytest tests/test_eval_cambridge_hybrid_cli.py -q`

Expected: pass.

### Task 3: CLI and Summary Integration

**Files:**
- Modify: `loc_gs/scripts/eval_cambridge_hybrid.py`
- Test: `tests/test_eval_cambridge_hybrid_cli.py`

- [ ] **Step 1: Add parser tests**

Assert parser accepts:

```text
--selfmap_reliability_path output/selfmap/ShopFacade/summary.json
--selfmap_reliability_center_cm 10
--selfmap_reliability_temperature_cm 1
```

- [ ] **Step 2: Add parser args**

Add CLI arguments and apply reliability immediately after argument validation in `main()`, before model/landmark construction.

- [ ] **Step 3: Add summary fields**

Write both raw and effective reliability state into `eval_config` and top-level `summary.json`.

### Task 4: Launcher Support

**Files:**
- Modify: `loc_gs/scripts/launch_cambridge_reliability_eval.py`
- Test: `tests/test_dim_experiment_launcher.py`

- [ ] **Step 1: Add failing launcher test**

Assert `build_eval_command(..., selfmap_reliability_path=".../summary.json")` appends `--selfmap_reliability_path`.

- [ ] **Step 2: Extend launcher arguments**

Add `--selfmap_reliability_template`, `--selfmap_reliability_center_cm`, and `--selfmap_reliability_temperature_cm`.

### Task 5: Verification Experiment

**Files:**
- Output only under `output/stdloc_hybrid/unified_soft_selfmap_20260514/`

- [ ] **Step 1: Smoke run on one query**

Run ShopFacade and GreatCourt with `--max_queries 1` and reliability paths to prove CLI and summary writing.

- [ ] **Step 2: Full Cambridge eval**

Launch five scenes with three GPUs and query shards if useful:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.launch_cambridge_reliability_eval \
  --scenes GreatCourt,KingsCollege,OldHospital,ShopFacade,StMarysChurch \
  --tag lff_refined_20260513 \
  --checkpoint_template 'output/stdloc_hybrid/{scene}_lff_refined_20260513/latest.pth' \
  --output_suffix eval_unified_soft_selfmap_20260514 \
  --recipes scene_matcher_residual_prosac \
  --calibrated_matchability_template 'output/stdloc_hybrid/listwise_v4_verifier_20260514/calibration/{scene}/stdloc_bank_query_like.pt' \
  --scene_matcher_template 'output/stdloc_hybrid/listwise_v4_verifier_20260514/scenematch/{scene}/best.pt' \
  --selfmap_reliability_template 'output/stdloc_hybrid/{scene}_lff_refined_20260513/eval_selfmap_stride4/summary.json' \
  --gpus 0,1,2
```

- [ ] **Step 3: Aggregate and compare**

Compare native STDLoc parity, previous hard selector, and unified soft reliability macro metrics. If soft reliability loses materially to the hard selector, inspect per-scene `rho` and tune only global center/temperature, not scene-specific knobs.

---

**Self-review:** This plan covers the reviewer concern about branch switching, keeps the original localization-oriented reconstruction claim, and uses fixed global reliability parameters rather than per-scene hyperparameter tuning.
