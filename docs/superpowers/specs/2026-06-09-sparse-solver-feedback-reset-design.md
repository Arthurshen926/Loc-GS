# Sparse Solver-Feedback Reset Design

## Purpose

This design resets the sparse-only ULF-Loc improvement path around one strict claim:

> First build a complete sparse localization system, then use self-localization
> sparse PnP feedback to iteratively improve its trainable sparse components
> toward final PnP solvability.

The previous routes are explicitly out of scope for the main method:

- post-hoc descriptor mean-shift after ULF aggregation;
- offline solver-feedback descriptor fusion as a main branch;
- SuperPoint teacher distillation as a main detector target;
- SuperPoint candidate reranking as the scene-specific detector main path;
- mean-query-descriptor MLP activation;
- correspondence reranking as an isolated final-stage fix;
- Full-Gaussian SparseSet replacement as the sparse main path.

These modules may remain as diagnostics, but they cannot enter the main recipe unless a new gate explicitly requalifies them.

## Correct Sparse Training Flow

The sparse mainline must follow this order:

```text
Stage 0: Base 3DGS / masks / train split
  Cambridge processed images + semantic/object/sky masks
  no Cambridge test image, pose, feature, result, or query id for training

Stage 1: Initial Landmark Sampling
  ULF K.C. sampling + visibility + mask validity

Stage 2: Initial Landmark Descriptor Aggregation
  ULF multi-view geometry-weighted landmark descriptor aggregation

Stage 3: Initial Scene-Specific Detector
  current sampled landmarks
  -> render-visible projection
  -> semantic/mask filtering
  -> detector heatmap target
  -> online scene-specific detector training

Stage 4: Initial Sparse Localization / Self-Map Feedback
  current landmark set + current landmark descriptors + initial detector
  -> train/self-map sparse PnP
  -> per-query / per-correspondence / per-landmark Feedback v4

Stage 5: Solver-Feedback Training Loop
  outer loop: sparse PnP feedback
  inner alternating updates:
    a. landmark feature aggregation / view selection
    b. scene-specific detector residual targets
    c. query-conditioned landmark activation

Stage 6: Freeze Sparse Recipe
  fixed landmark set / descriptors / detector / activation
  -> disjoint train-dev sparse-only evaluation
```

Official Cambridge test queries are never used for feedback, hard-negative mining, model selection, detector calibration, activation training, or hyperparameter selection.

The main launcher must not generate feedback from a half system. In particular,
the first self-map trace must already use the initial scene-specific detector.
Native ULF-only traces are allowed only as diagnostics or baselines.

## Feedback v4

Feedback v4 is the shared supervision source for all three trainable consumers. It stores query-level outcome and per-correspondence attribution in one schema.

Required query-level fields:

- `scene`, `split_name`, `query_id`, `image_id`;
- `pose_success`, `sparse_te_cm`, `sparse_re_deg`;
- `candidate_minus_baseline_te_cm` when comparing a candidate against a protected baseline;
- `catastrophic_failure`, `protected_good_query`, `hard_query`;
- `inlier_count`, `match_count`, `inlier_ratio`;
- `inlier_image_cell_count`, `inlier_depth_bin_count`, `bearing_spread`, `depth_spread`;
- `pnp_logdet`, `pnp_min_eigenvalue` when available.

Required correspondence-level fields:

- `gaussian_id`, `sampled_row`, `query_keypoint_index`;
- `keypoint_xy`, `query_xy_norm`, `image_cell`;
- `query_descriptor`, `landmark_descriptor`;
- `descriptor_score`, `descriptor_margin`, `detector_score`;
- `pnp_inlier`, `reprojection_error_px`;
- `camera_xyz`, `depth_m`, `bearing`;
- `label_role`, one of `positive_inlier`, `harmful_negative`, `neutral_outlier`, `protected_support`;
- `source_role`, one of `baseline_trace`, `candidate_trace`, `render_aug_trace`.

Effective feedback quantities:

- positive support: successful-query PnP inliers with low reprojection error and sufficient descriptor margin;
- harmful negative: high-score PnP outliers from failed/regressed queries, repeated-structure competitors, and non-reference support stealers;
- protected support: inliers from baseline-good queries that must not be suppressed;
- query utility: image/depth/bearing coverage and PnP conditioning, not only inlier count.

`query_global_feature` from the old pipeline is not sufficient supervision. It may be logged as a diagnostic descriptor summary, but it is not the main activation input.

## Landmark Descriptor Aggregation

Stage 2 uses the original ULF multi-view geometry-weighted aggregation to build
the initial landmark descriptors. Solver feedback must not be applied as a
post-hoc descriptor mean shift after localization.

The intended Stage 5 descriptor update is an aggregation/view-selection update:
feedback labels select or weight raw landmark-view observations before
aggregation, then the sparse system is rerun to validate the new descriptors.
This is solver-in-the-loop alternating training, not one static offline cache.

Allowed diagnostics:

- audit native ULF multi-view descriptor observations;
- measure whether feedback labels identify harmful or protected observations;
- run isolated descriptor-fusion ablations outside the main launcher.

These diagnostics must not be reported as the main sparse method unless they are
converted into the Stage 5 aggregation/view-selection loop and pass the
large-scene gate.

## Direct Scene-Specific Detector

The scene-specific detector is a direct heatmap detector. In main inference it must produce keypoints by heatmap NMS and descriptor sampling. It must not be implemented as SuperPoint candidate reranking.

Training supervision combines:

- sampled-landmark visibility heatmap, matching the STDLoc/ULF scene-specific detector idea;
- solver-positive residual boosts from Feedback v4 positive inlier correspondences;
- solver-negative suppression from harmful-negative correspondences.

The loss is:

```text
landmark visibility target loss
+ solver-positive residual boost
+ solver-negative suppression
```

The initial target is the STDLoc-style sampled-landmark projection heatmap after render visibility and semantic/stability filtering. SuperPoint teacher heatmaps are diagnostic only and are disabled by default. Solver feedback never multiplies the entire visibility target down to zero. Low-validity positives are handled through bounded residual suppression, not by erasing the projection target.

The initial detector must be trained before the first self-map feedback run.
Feedback generated without the initial detector is a native-baseline diagnostic,
not mainline solver feedback.

## Query-Conditioned Landmark Activation v2

Landmark activation v2 predicts a per-query overlap/activation mask over fixed ULF landmarks. It is a query-conditioned map-side detector.

Inputs:

- query tokens: keypoint descriptors, xy position, detector score, local grid/patch embedding;
- landmark tokens: aggregated descriptor, normalized 3D xyz, visibility/K.C. statistics, solver support/risk, view reliability;
- optional coarse global query token for context only.

Architecture:

```text
query token encoder
+ landmark token encoder
+ cross-attention or top-k query-to-landmark aggregation
+ per-landmark MLP head
-> activation logit
```

Supervision:

- geometry visibility mask from train/self-map pose;
- solver-positive inlier labels;
- harmful-negative suppression labels;
- protected-support retention;
- coverage regularizers for image cells, depth bins, and bearing spread.

The old mean-query-descriptor MLP is diagnostic only.

## Evaluation Gates

Main modules are evaluated first on a large Cambridge scene, currently GreatCourt train-dev sparse-only, before any five-scene expansion. ShopFacade is useful for smoke tests only and cannot be used as the main gate for this reset.

Primary metrics:

- median translation error;
- median rotation error;
- R@10cm/5deg;
- R@5cm/5deg.

Secondary metrics:

- inlier count and inlier ratio;
- active landmark count;
- protected-support loss;
- harmful-negative retention;
- query utility coverage.

GreatCourt large-scene gate:

- median TE must improve over or at minimum not regress against the current geomw baseline;
- R@10 or R@5 must improve, and the other must not drop;
- direct heatmap detector must not be weaker than the current geomw/native detector;
- activation must not pass by selecting nearly all landmarks.

Only after this gate can the method expand to KingsCollege, OldHospital, ShopFacade, and StMarysChurch train-dev. Official test is reserved for frozen recipes only.

## Non-Negotiable Implementation Guardrails

- Do not tune on official test.
- Do not use test query images/features/poses/results in feedback.
- Do not modify `third_party/stdloc`.
- Do not reintroduce descriptor mean-shift as a main method.
- Do not reintroduce offline fusion/feature-log rewriting as a main method.
- Do not use SuperPoint teacher distillation in the main detector.
- Do not use SuperPoint reranking as the main scene detector.
- Do not use mean query descriptor activation as the main activation model.
- Do not claim correspondence supervision unless it trains detector residuals or query-conditioned activation on non-test traces.
- Do not expand to five scenes until the GreatCourt disjoint sparse gate passes.
