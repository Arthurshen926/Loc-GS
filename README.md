# Loc-GS

Solvability-Guided Localization Support Fields for 3D Gaussian Relocalization.

Loc-GS is an independent Cambridge/STDLoc research workspace.  The repository
is being reset to one paper-facing line:

```text
STDLoc native Feature Gaussian map
  -> audited train/rendered self-localization feedback
  -> Localization Support Field for sparse landmarks, solver tuples, and dense residuals
  -> STDLoc-compatible sampled landmarks and support masks
  -> one OpenCV PROSAC/RANSAC PnP + STDLoc-style dense refinement path
```

The core claim under development is **from matchability to solvability**:
point-wise matchability and support count are insufficient proxies for PnP
success, so landmark and dense residual selection should be guided by solver
conditioning, hard-query risk, ambiguity, and dense-refinement reliability.

The earlier RADIO feature reconstruction, open-vocabulary scene understanding, segmentation, depth-head, and grounding experiment entry points have been removed from the public workflow.

## What Is Included

- SuperPoint descriptor and detector extraction.
- Hybrid Gaussian feature-field training for SuperPoint reconstruction.
- Localization-guided training losses with differentiable matching, reprojection proxy, observability, and support-field diagnostics.
- SuperPoint reconstruction evaluation, localization evaluation, and qualitative visualization.
- Minimal 3DGS asset preparation helpers.
- A vendored STDLoc copy under `third_party/stdloc` for Cambridge localization experiments and reproducibility.
- Feedback-bank-v2, solver-tuple, and dense-support diagnostics for LSF-Loc.

## Repository Layout

```text
Loc-GS/
├── configs/                  # SuperPoint and localization-oriented run configs
├── docs/                     # Loc-GS technical notes
├── loc_gs/                   # Main Python package
│   ├── losses/
│   ├── models/
│   ├── rendering/
│   └── scripts/
├── tests/                    # Loc-GS unit tests
├── third_party/stdloc/       # Vendored STDLoc code
└── output -> /mnt/pool/sqy/results/Loc-GS/output
```

The `output` symlink is intentionally preserved for compatibility with existing runs.

## Environment

The currently verified CPython/CUDA environment is:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python
```

Install the package in editable mode from the repository root:

```bash
pip install -e .
```

STDLoc dependencies are vendored with the third-party project. Its SuperPoint checkpoint is expected at:

```text
third_party/stdloc/encoders/sp_encoder/weights/superpoint_v1.pth
```

## Main Commands

Extract SuperPoint teacher features:

```bash
CUDA_VISIBLE_DEVICES=0 python -m loc_gs.scripts.extract_superpoint_features \
  --scene room_0 \
  --image_dir /mnt/pool/sqy/dataset/room_0/Sequence_1/rgb \
  --output_dir output/superpoint_features/room_0/Sequence_1 \
  --weights third_party/stdloc/encoders/sp_encoder/weights/superpoint_v1.pth \
  --batch_size 8
```

Train a localization-oriented SuperPoint Gaussian feature field:

```bash
CUDA_VISIBLE_DEVICES=0 python -m loc_gs.scripts.train_feature_field \
  --config configs/superpoint_localization_room_0_v1.yaml
```

Evaluate feature reconstruction:

```bash
CUDA_VISIBLE_DEVICES=0 python -m loc_gs.scripts.eval_superpoint \
  --config configs/superpoint_hybrid_room_0_v3.yaml \
  --checkpoint output/sp_gs/room0_hybrid_v3/checkpoints/best.pth
```

Evaluate relocalization:

```bash
CUDA_VISIBLE_DEVICES=0 python -m loc_gs.scripts.eval_localization \
  --config configs/superpoint_hybrid_room_0_v3.yaml \
  --checkpoint output/sp_gs/room0_hybrid_v3/checkpoints/best.pth \
  --output_dir output/sp_gs/room0_hybrid_v3/localization \
  --num_samples 100
```

Generate qualitative results:

```bash
CUDA_VISIBLE_DEVICES=0 python -m loc_gs.scripts.visualize_superpoint_results \
  --config configs/superpoint_hybrid_room_0_v3.yaml \
  --checkpoint output/sp_gs/room0_hybrid_v3/checkpoints/best.pth \
  --output_dir output/sp_gs/room0_hybrid_v3/qualitative_superpoint
```

## STDLoc Cambridge Experiments

STDLoc has been copied into `third_party/stdloc` so Loc-GS does not depend on `/root/STDLoc` at runtime. Run STDLoc commands from that directory:

```bash
cd third_party/stdloc
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest -q tests
```

Large STDLoc maps, logs, and results are kept under `output/stdloc/` for reproducibility. The legacy paths under `third_party/stdloc/` are relative symlinks back to `output/stdloc/` so existing STDLoc commands keep working.

## LSF-Loc Cambridge Mainline

The active Cambridge research path is **LSF-Loc: solvability-guided
localization support fields for 3D Gaussian relocalization**.  The claim is not
descriptor replacement.  The claim is that 3DGS localization feature selection
must be optimized for the geometric solver, not only for point-wise
matchability or support count.

```text
STDLoc native Feature Gaussian map
  -> train/rendered self-localization feedback_bank_v2
  -> solver-consensus support estimation
  -> solvability-aware landmark sampling
  -> support-consistent dense residual verification
  -> one STDLoc-compatible sparse-to-dense inference path
```

The deployed query path must remain fixed:

```text
query image
  -> native STDLoc feature extraction and matching
  -> OpenCV PROSAC/RANSAC PnP
  -> STDLoc-style dense refinement
  -> final pose
```

The active implementation note is `docs/mainline_lsf_20260521.md`.  The
paper-safety contract is `docs/research_contract.md`, and split/evaluation
rules are in `docs/experiment_protocol.md`.

### Method Modules

LSF-Loc is organized into three modules:

- **Solver-consensus support estimation:** build `feedback_bank_v2` from
  training, calibration, or rendered self-localization episodes with real
  `query_id`, `image_id`, `keypoint_id`, PnP inliers, reprojection errors, and
  dense transition labels.
- **Solvability-aware landmark sampling:** generate STDLoc-compatible
  `sampled_idx.pkl`, `sampled_scores.pkl`, and locability payloads using
  hard-query tuple quality, pose-information proxies, ambiguity risk, and
  same-budget replacement constraints.
- **Support-consistent dense verification:** use LSF support as a dense
  residual reliability signal to reduce cases where dense refinement worsens a
  valid sparse pose.

### Baselines And Diagnostics

Native STDLoc through `third_party/stdloc` and
`loc_gs.stdloc_native.commands` is the Cambridge baseline.  Feedback-disabled
settings must reproduce native STDLoc parity before any Loc-GS contribution is
claimed.

The following lines are not the main method unless a future full-split,
paper-audited result promotes them: residual descriptor reconstruction,
SceneMatchNet, LoFTR or DIM replacement, quality gates, oracle ordering,
per-query or scene-level branch selection, qcov/churn/retention scalar sweeps,
support-count guards, and raw dense locability priors.  They may be kept only
as diagnostics, ablations, or negative evidence.

### Safe Entry Points

Use `locgsctl` before writing long Cambridge commands by hand:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.locgsctl status
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.locgsctl list-scenes
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.locgsctl smoke --scene ShopFacade --dry-run
```

Do not start full Cambridge training or evaluation unless the run is explicitly
requested and the candidate has passed the audit gates in
`docs/mainline_lsf_20260521.md`.

## Verification

From `/root/Loc-GS`:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest -q tests
/root/miniconda3/envs/cybersim_agent/bin/python -m pytest -q third_party/stdloc/tests
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.config configs/superpoint_localization_room_0_v1.yaml
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.extract_superpoint_features --help
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.train_feature_field --help
/root/miniconda3/envs/cybersim_agent/bin/python -m loc_gs.scripts.eval_localization --help
```
