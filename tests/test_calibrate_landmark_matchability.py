import torch

from loc_gs.scripts.launch_cambridge_matchability_calibration import build_calibration_command
from loc_gs.scripts.calibrate_landmark_matchability import (
    accumulate_matchability_counts,
    build_argparser,
    calibration_query_canvas_hw,
    matchability_from_counts,
    write_matchability_calibration,
)


def test_matchability_from_counts_uses_beta_smoothing():
    tp = torch.tensor([10.0, 0.0, 2.0])
    fp = torch.tensor([0.0, 10.0, 2.0])

    score = matchability_from_counts(tp, fp, alpha=1.0)

    assert score[0] > score[2] > score[1]
    assert torch.all((score > 0.0) & (score < 1.0))


def test_calibration_query_canvas_uses_resized_dataset_frame():
    resized_rgb = torch.empty(1, 3, 360, 640)
    original_teacher_rgb = torch.empty(1, 3, 768, 1024)

    assert calibration_query_canvas_hw(resized_rgb, original_teacher_rgb) == (360, 640)


def test_calibration_parser_defaults_to_baseline_preserving_descriptor_and_rendered_rehearsal_knobs():
    args = build_argparser().parse_args(
        [
            "--checkpoint",
            "output/stdloc_hybrid/KingsCollege/latest.pth",
            "--rendered_rehearsal_views",
            "8",
        ]
    )

    assert args.descriptor_source == "ply_loc"
    assert args.rendered_rehearsal_views == 8
    assert args.rendered_query_source == "rendered_rgb_teacher"
    assert args.rendered_rehearsal_pose_mode == "mixed"
    assert args.rendered_rehearsal_interpolation_min == -0.15
    assert args.rendered_rehearsal_interpolation_max == 1.15


def test_query_like_calibration_launcher_uses_rendered_rgb_teacher():
    cmd, env = build_calibration_command(
        gpu_id=2,
        scene="OldHospital",
        checkpoint="output/stdloc_hybrid/OldHospital/latest.pth",
        output_path="output/calib/OldHospital/stdloc_bank.pt",
        max_views=32,
        rendered_rehearsal_views=64,
    )

    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert cmd[cmd.index("--rendered_query_source") + 1] == "rendered_rgb_teacher"
    assert cmd[cmd.index("--rendered_rehearsal_views") + 1] == "64"
    assert cmd[cmd.index("--max_views") + 1] == "32"
    assert cmd[cmd.index("--rendered_rehearsal_pose_mode") + 1] == "mixed"


def test_accumulate_matchability_counts_marks_topk_projection_inliers():
    landmark_xyz = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [5.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    query_yx = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    q_ids = torch.tensor([0, 0, 1], dtype=torch.long)
    lm_ids = torch.tensor([0, 2, 1], dtype=torch.long)
    pose = torch.eye(4)
    K = torch.eye(3)
    tp = torch.zeros(3)
    fp = torch.zeros(3)

    accumulate_matchability_counts(
        tp,
        fp,
        query_yx=query_yx,
        landmark_xyz=landmark_xyz,
        q_ids=q_ids,
        lm_ids=lm_ids,
        pose_w2c=pose,
        K=K,
        reprojection_threshold_px=0.25,
    )

    assert tp.tolist() == [1.0, 1.0, 0.0]
    assert fp.tolist() == [0.0, 0.0, 1.0]


def test_accumulate_matchability_counts_uses_depth_visibility_when_available():
    landmark_xyz = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    query_yx = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    q_ids = torch.tensor([0, 1], dtype=torch.long)
    lm_ids = torch.tensor([0, 1], dtype=torch.long)
    depth = torch.tensor([[2.0, 1.0], [0.0, 0.0]], dtype=torch.float32)
    alpha = torch.ones_like(depth)
    tp = torch.zeros(2)
    fp = torch.zeros(2)

    accumulate_matchability_counts(
        tp,
        fp,
        query_yx=query_yx,
        landmark_xyz=landmark_xyz,
        q_ids=q_ids,
        lm_ids=lm_ids,
        pose_w2c=torch.eye(4),
        K=torch.eye(3),
        reprojection_threshold_px=0.25,
        depth_map=depth,
        alpha_map=alpha,
        depth_abs_tolerance=0.1,
        depth_rel_tolerance=0.0,
        alpha_threshold=0.5,
    )

    assert tp.tolist() == [0.0, 1.0]
    assert fp.tolist() == [1.0, 0.0]


def test_write_matchability_calibration_writes_sidecar_tensors(tmp_path):
    payload = {
        "landmark_matchability": torch.tensor([0.25, 0.75]),
        "landmark_tp_count": torch.tensor([1.0, 3.0]),
        "landmark_fp_count": torch.tensor([3.0, 1.0]),
        "landmark_fp_rate": torch.tensor([0.75, 0.25]),
        "matchability_calibrator": {"type": "beta_smoothed_tp_fp"},
        "metadata": {"scene": "ShopFacade"},
    }
    output_path = tmp_path / "stdloc_bank.pt"

    write_matchability_calibration(payload, output_path)

    assert output_path.exists()
    assert (tmp_path / "landmark_matchability.pt").exists()
    assert (tmp_path / "landmark_tp_count.pt").exists()
    assert (tmp_path / "landmark_fp_count.pt").exists()
    assert (tmp_path / "landmark_fp_rate.pt").exists()
    assert (tmp_path / "matchability_calibrator.pt").exists()
    assert (tmp_path / "stdloc_bank.json").exists()
