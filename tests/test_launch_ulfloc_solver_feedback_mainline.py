from loc_gs.scripts.launch_ulfloc_solver_feedback_mainline import build_commands


def test_launcher_uses_non_test_feedback_for_training():
    commands = build_commands(scenes=["GreatCourt"], split_name="train_selfmap", dry_run=True)
    text = "\n".join(" ".join(cmd) for cmd in commands)
    assert "split_name test" not in text
    assert "--split_name train_selfmap" in text


def test_launcher_rejects_test_split():
    try:
        build_commands(scenes=["GreatCourt"], split_name="test", dry_run=True)
    except ValueError as exc:
        assert "test split" in str(exc)
    else:
        raise AssertionError("launcher must reject test split for feedback/model selection")


def test_launcher_keeps_fusion_branches_out_of_mainline_commands():
    commands = build_commands(scenes=["GreatCourt"], split_name="train_selfmap", dry_run=True)
    text = "\n".join(" ".join(cmd) for cmd in commands)

    assert "loc_gs.scripts.build_ulfloc_native_fusion_feedback" not in text
    assert "loc_gs.scripts.build_ulfloc_native_feature_fusion_log" not in text
    assert not any("loc_gs.scripts.build_ulfloc_solver_feedback_feature_log" in cmd for cmd in commands)


def test_launcher_emits_schema_complete_mainline_commands():
    commands = build_commands(
        scenes=["GreatCourt"],
        split_name="train_selfmap",
        output_root="/tmp/out",
        feedback_root="/tmp/feedback",
        data_root="/tmp/data",
        best_log_root="/tmp/best",
        dry_run=True,
    )
    joined = [" ".join(cmd) for cmd in commands]

    assert not any("loc_gs.scripts.build_ulfloc_trace_observation_cache" in item for item in joined)
    assert not any("loc_gs.scripts.build_ulfloc_visibility_teacher" in item for item in joined)
    assert not any("loc_gs.scripts.build_ulfloc_solver_feedback_detector_targets" in item for item in joined)
    assert not any("loc_gs.scripts.train_ulfloc_scene_detector " in item for item in joined)

    assert not any("fusion" in item for item in joined)


def test_launcher_runs_initial_detector_before_feedback_trace_and_residual_detector_after_feedback():
    commands = build_commands(scenes=["GreatCourt"], split_name="train_selfmap", dry_run=True)
    joined = [" ".join(cmd) for cmd in commands]

    initial_idx = next(i for i, item in enumerate(joined) if "initial_detector" in item and "train_ulfloc_scene_detector_online" in item)
    trace_idx = next(i for i, item in enumerate(joined) if "sparse_trace" in item and "eval_ulfloc_sparse_only" in item)
    feedback_idx = next(i for i, item in enumerate(joined) if "build_sparse_feedback_v4" in item)
    impact_idx = next(i for i, item in enumerate(joined) if "build_solver_feedback_impact" in item)
    residual_idx = next(i for i, item in enumerate(joined) if "residual_detector" in item and "train_ulfloc_scene_detector_online" in item)
    eval_idx = next(i for i, item in enumerate(joined) if "eval_train_dev_residual_detector" in item and "eval_ulfloc_sparse_only" in item)

    assert initial_idx < trace_idx < feedback_idx < impact_idx < residual_idx < eval_idx

    initial_cmd = commands[initial_idx]
    assert "--solver_impact" not in initial_cmd
    assert "--use_superpoint_teacher" not in initial_cmd

    trace_cmd = commands[trace_idx]
    assert "--scene_detector_checkpoint" in trace_cmd
    assert trace_cmd[trace_cmd.index("--scene_detector_checkpoint") + 1].endswith("initial_detector/scene_detector.pth")

    residual_cmd = commands[residual_idx]
    assert "--solver_impact" in residual_cmd
    assert "--init_checkpoint" in residual_cmd
    assert residual_cmd[residual_cmd.index("--init_checkpoint") + 1].endswith("initial_detector/scene_detector.pth")
    assert residual_cmd[residual_cmd.index("--solver_feedback_residual_alpha") + 1] != "0"


def test_launcher_does_not_claim_activation_until_online_activation_exists():
    commands = build_commands(scenes=["GreatCourt"], split_name="train_selfmap", dry_run=True)
    text = "\n".join(" ".join(cmd) for cmd in commands)

    assert "loc_gs.scripts.build_ulfloc_landmark_activation_cache" not in text
    assert "loc_gs.scripts.train_query_landmark_activation_v2" not in text
