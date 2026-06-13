# Internal Sparse-Dense Student Mainline

This branch now separates the sparse-dense teacher from the sparse-only student
with an explicit three-stage training and inference flow.

## Stage 1: Bootstrap Teacher Observations

The training-only teacher may still use the STDLoc-style sparse+dense flow. Its
output is not used directly at inference time. Teacher output is converted into
internal listwise sparse candidates with per-candidate labels:

- `dense_consistent`
- `sparse_inlier`
- `reprojection_error`
- `solver_weight`
- `label_roles`

The legacy cached-source path is still supported for parity and smoke tests, but
it is marked as `candidate_binding_mode=cached_source_candidate_reuse`.

## Stage 2: Direct Online 3DGS Observation Stream

The direct online path uses `OnlineTeacherObservationRow` records. These records
bind candidate rows to `synthetic_query_id`, not to the original source image.
The resulting artifact is marked:

- `training_source=online_3dgs_student_teacher_observation_stream`
- `candidate_binding_mode=direct_online_teacher_observations`
- `source_candidate_reuse_enabled=false`

The in-memory loop contract is:

```text
SimulatedQuerySpec
  -> render_query
  -> run_sparse_student
  -> run_sparse_dense_teacher
  -> OnlineTeacherObservationRow
  -> internal listwise candidate payload
  -> student training bundle
```

The loop lives in `loc_gs.teacher.online_training_loop`. The real renderer and
teacher can be plugged in through callables without changing the student
training code.

## Stage 3: Sparse-Only Student Runtime

`train_internal_sparse_students` can now train or export these student modules:

- linear sparse candidate scorer
- candidate MLP scorer
- landmark selector
- conflict graph
- descriptor fusion
- detector grid prior
- optional query-conditioned landmark activation v2

`eval_internal_sparse_cached` can consume the sparse-only student bundle through
the standard sparse path. The runtime combines enabled student score rows before
OpenCV PnP. It does not run dense refinement.

## Safety

Training entry points reject test splits. Evaluation manifests mark
`dense_inference_enabled=false` for sparse-only student inference. Direct online
training does not require or reuse a cached candidate artifact.
