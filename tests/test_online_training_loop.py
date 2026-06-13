from loc_gs.simulation.query_sampler import SimulatedQuerySpec
from loc_gs.teacher.online_training_loop import Online3DGSTrainingLoop, run_online_3dgs_training_loop


def _spec(idx: int, split_name: str = "train") -> SimulatedQuerySpec:
    return SimulatedQuerySpec(
        scene="GreatCourt",
        split_name=split_name,
        synthetic_query_id=f"sim/GreatCourt/{split_name}/{idx:06d}",
        source_image_id=f"source_{idx}.png",
        translation_delta_m=(0.0, 0.0, 0.0),
        rotation_delta_deg=(0.0, 0.0, 0.0),
    )


def test_online_3dgs_training_loop_builds_observations_without_candidate_cache():
    calls = []

    def render_query(spec):
        calls.append(("render", spec.synthetic_query_id))
        return {
            "render_rgb_path": f"/renders/{spec.synthetic_query_id}.png",
            "render_ready": True,
            "render_engine": "3dgs",
        }

    def run_sparse_student(spec, rendered):
        calls.append(("student", spec.synthetic_query_id, rendered["render_engine"]))
        return [
            {
                "keypoint_id": "kp0",
                "query_yx": [10.0, 20.0],
                "query_desc": [1.0, 0.0],
                "landmark_ids": [11, 12],
                "landmark_desc": [[0.0, 1.0], [1.0, 0.0]],
                "candidate_scores": [0.8, 0.2],
                "candidate_mask": [True, True],
            }
        ]

    def run_sparse_dense_teacher(spec, rendered, candidates):
        calls.append(("teacher", spec.synthetic_query_id, len(candidates)))
        row = dict(candidates[0])
        row.update(
            {
                "geometric_correct": [False, True],
                "dense_consistent": [False, True],
                "sparse_inlier": [False, True],
                "reprojection_error_px": [18.0, 1.0],
                "solver_weight": [0.25, 3.0],
                "label_roles": ["hard_negative", "protected_support"],
                "dense_helped": True,
                "distill_weight": 0.75,
            }
        )
        return [row]

    observations, summary = run_online_3dgs_training_loop(
        [_spec(0), _spec(1)],
        Online3DGSTrainingLoop(
            render_query=render_query,
            run_sparse_student=run_sparse_student,
            run_sparse_dense_teacher=run_sparse_dense_teacher,
        ),
        scene="GreatCourt",
        split_name="train",
    )

    assert [call[0] for call in calls] == ["render", "student", "teacher", "render", "student", "teacher"]
    assert len(observations) == 2
    assert observations[0].synthetic_query_id == "sim/GreatCourt/train/000000"
    assert observations[0].landmark_ids == (11, 12)
    assert observations[0].dense_consistent == (False, True)
    assert summary["online_observation_count"] == 2
    assert summary["candidate_binding_mode"] == "direct_online_teacher_observations"
    assert summary["source_candidate_reuse_enabled"] is False
