# Sparse-Dense Distilled Internal Mainline Design

## Goal

Move the active Loc-GS development path away from runtime coupling to
`/root/ULF-Loc` and vendored STDLoc execution scripts, and into an internal
Loc-GS pipeline for:

```text
training-time sparse-dense teacher + online 3DGS simulation
-> student sparse modules
-> inference-time sparse-only localization
```

The first implementation slice is intentionally narrow: internal sparse
pipeline boundaries, teacher label schemas, correspondence scoring/reranking,
and sparse-only evaluation. Detector, descriptor fusion, landmark selection,
and conflict graph training will consume the same teacher labels after this
slice proves sparse reranking headroom.

## Research Contract

This design follows the repository research constraints:

- Official Cambridge test queries must not be used for training, calibration,
  model selection, hard-negative mining, selector labels, feedback banks, or
  hyperparameter tuning.
- Native STDLoc and ULF-Loc behavior remains a reference/parity baseline, not a
  hidden runtime dependency for the new mainline.
- Disabled feedback settings must preserve baseline sparse behavior for the
  corresponding internal pipeline.
- Inference-time evaluation for the new mainline is sparse-only:

```text
query detector
-> query descriptors
-> top-K landmark candidates
-> correspondence scorer / reranker
-> OpenCV PROSAC/RANSAC PnP
-> sparse local geometry consistency validation
-> second sparse PnP
-> final pose
```

Dense refinement is only a training-time teacher and diagnostic. A result that
uses dense at inference must not be reported as the new sparse-only method.

## External Dependency Boundary

The project may continue to keep these assets as references:

- `third_party/stdloc`: vendored parity/reference implementation and SuperPoint
  checkpoint source.
- `/root/ULF-Loc`: reproduction/reference checkout while migration is underway.

The new mainline runtime must not:

- import `ulfloc`, `scene`, or ULF utility modules by adding `/root/ULF-Loc` to
  `sys.path`;
- execute `/root/ULF-Loc` scripts through subprocess calls;
- depend on `third_party/stdloc/stdloc.py`, `third_party/stdloc/train.py`, or
  evaluator internals for its own sparse inference loop.

The new mainline may use standard library and third-party libraries such as
PyTorch, OpenCV, gsplat, NumPy, and existing Loc-GS modules.

## First Slice Scope

The first slice builds a stable internal surface, not a full final method.

In scope:

- Internal camera, projection, PnP, pose-metric, and candidate schemas.
- Internal sparse episode schema that can represent ULF/STDLoc-like matches
  without importing ULF/STDLoc code.
- Training-time teacher label schema for sparse and dense outcomes.
- Correspondence scorer feature extraction and deterministic reranking.
- Sparse-only evaluation wrapper that refuses dense inference for the new
  mainline.
- Audit helpers that detect accidental `/root/ULF-Loc` or vendored STDLoc
  runtime coupling in new-mainline entry points.

Out of scope for this slice:

- Full rewrite of every historical script.
- Full detector retraining.
- Full descriptor-fusion training.
- Full landmark selector and conflict graph optimization.
- Official Cambridge test evaluation.
- Moving or modifying vendored STDLoc evaluator code.

Historical scripts may remain for reproducibility, but new mainline entry
points must call the internal APIs.

## Proposed Package Layout

```text
loc_gs/core/
  camera.py
  geometry.py
  pnp.py
  metrics.py

loc_gs/gaussian/
  map.py
  renderer.py
  visibility.py

loc_gs/sparse/
  candidates.py
  correspondences.py
  scorer.py
  rerank.py
  pipeline.py
  audit.py

loc_gs/teacher/
  episodes.py
  labels.py
  dense_teacher.py
  simulation.py

loc_gs/students/
  correspondence_scorer.py

loc_gs/scripts/
  eval_sparse_distilled_cambridge.py
  train_sparse_distilled_correspondence_scorer.py
  audit_internal_mainline.py
```

Existing modules can be reused behind these interfaces when they are already
internal to Loc-GS. The point of the layout is to make runtime ownership clear:
new mainline code imports from `loc_gs`, not from external research checkouts.

## Data Model

### Sparse Candidates

A sparse candidate batch contains:

- query id and split name;
- keypoint xy coordinates and descriptor vectors;
- candidate landmark ids, xyz coordinates, descriptors, and candidate scores;
- optional detector score, landmark prior, visibility score, and mask validity;
- camera intrinsics and optional known training pose for label generation.

The schema must support top-K candidates per query keypoint. It must not assume
that a single candidate has already been selected.

### Teacher Labels

Training-time labels are attached per correspondence candidate:

- geometric correctness by reprojection error under the known training pose;
- sparse PnP inlier membership;
- sparse residual after the first PnP;
- pose-information proxy such as logdet contribution or min-eigen proxy;
- dense consistency label from training-time dense teacher;
- hard-negative flags for descriptor-strong but geometrically wrong matches;
- split audit metadata.

Dense labels are allowed only when the source split is non-test. Dense teacher
outputs are not consumed by inference-time code.

## First Model

The first student is a correspondence scorer. It takes fixed candidate features
and produces a scalar reranking score.

Initial features:

- descriptor cosine;
- candidate rank;
- top-1/top-2 margin for the keypoint;
- detector score;
- landmark prior score;
- keypoint normalized xy;
- candidate depth and depth spread proxy;
- local descriptor ambiguity;
- training-only sparse/dense teacher labels for loss construction.

Initial behavior:

- deterministic linear/logistic scorer for smoke tests;
- Torch module for trainable scorer;
- reranker that preserves a configurable high-score prefix for PROSAC stability
  and reranks the remaining candidate budget using solver-aware score.

## Evaluation

The first slice evaluates sparse-only behavior with these metrics:

- median translation error and rotation error;
- R@10cm/5deg;
- R@5cm/5deg;
- candidate availability;
- oracle top-K gap;
- native top-1 versus reranked top-K;
- per-scene Cambridge train-dev summaries.

The immediate development gate is GreatCourt train-dev sparse median moving
from the current 15 cm class toward the current dense 10 cm class without
official test usage. This is a development gate, not a paper-facing claim.

## Safety And Failure Handling

- New mainline CLI commands reject `split_name` values equal to `test`,
  `official_test`, or values ending in `_test` when training labels, teacher
  labels, feedback, or model selection are requested.
- Manifests include git commit, command, scene, split, map/checkpoint paths,
  data roots, hyperparameters, timestamp, teacher settings, and whether dense
  teacher labels were enabled.
- Missing split information is recorded as `unknown`, and the run is excluded
  from paper-safe summaries.
- If dense teacher generation is requested without a known pose or non-test
  split audit, the command fails before producing labels.
- If a new-mainline entry point attempts to import `/root/ULF-Loc` or execute
  vendored STDLoc runtime scripts, the audit fails.

## Testing Strategy

Unit tests use synthetic inputs by default:

- camera and pose metric tests;
- PnP wrapper tests with known synthetic 2D-3D correspondences;
- candidate schema validation tests;
- teacher-label construction tests;
- reranker order and PROSAC-prefix preservation tests;
- split rejection tests;
- audit tests that scan new-mainline modules for forbidden external runtime
  imports.

Integration tests are dry-run or tiny synthetic smoke tests. Cambridge data,
GPU rendering, and long training runs are not required for the default test
suite.

## Migration Plan

1. Create internal schemas and audit guardrails.
2. Port the sparse correspondence loop into internal Loc-GS APIs using
   synthetic tests first.
3. Add teacher label generation from existing Loc-GS feedback artifacts.
4. Add correspondence scorer and reranker.
5. Add sparse-only CLI and manifest output.
6. Validate with train-dev runs only.
7. After scorer/reranker proves headroom, connect detector, descriptor fusion,
   landmark selector, and conflict graph training to the same teacher labels.

## Non-Goals

- This slice does not delete historical scripts.
- This slice does not modify vendored STDLoc evaluator paths.
- This slice does not claim official Cambridge test gains.
- This slice does not make dense a deployed inference stage.
