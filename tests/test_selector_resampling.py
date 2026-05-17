import pickle

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement

from loc_gs.scripts.export_selector_resampled_map import build_argparser, main
from loc_gs.stdloc_native.selector_resampling import (
    build_selector_sampling_scores,
    hard_negative_risk_from_episode_cache,
    hard_query_support_from_episode_cache,
    coverage_constrained_topk,
    positive_support_from_episode_cache,
    pose_information_from_episode_cache,
    query_coverage_reservation_from_episode_cache,
    resample_detector_landmarks,
)


def _write_ply(path):
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("locability_logit", "f4"),
        ("loc_0", "f4"),
        ("loc_1", "f4"),
    ]
    data = np.empty(5, dtype=dtype)
    data["x"] = [0.0, 1.0, 2.0, 3.0, 4.0]
    data["y"] = [0.0, 0.0, 0.0, 0.0, 0.0]
    data["z"] = [1.0, 1.0, 1.0, 1.0, 1.0]
    data["locability_logit"] = [0.0, 0.0, 0.0, 0.0, 0.0]
    data["loc_0"] = [1.0, 0.0, 1.0, 2.0, 1.0]
    data["loc_1"] = [0.0, 2.0, 1.0, 0.0, 3.0]
    PlyData([PlyElement.describe(data, "vertex")], text=True).write(path)


def test_build_selector_sampling_scores_combines_source_and_selector():
    source_scores = torch.tensor([0.1, 0.8, 0.2, 0.4])
    selector = torch.tensor([0.9, 0.1, 0.7, 0.2])

    scores = build_selector_sampling_scores(
        selector=selector,
        source_scores=source_scores,
        selector_weight=2.0,
        source_score_weight=1.0,
    )

    assert int(scores.argmax().item()) == 0
    assert scores.shape == selector.shape


def test_build_selector_sampling_scores_can_rank_normalize_narrow_selector():
    selector = torch.tensor([0.499, 0.501, 0.500, 0.502], dtype=torch.float32)

    scores = build_selector_sampling_scores(
        selector=selector,
        source_scores=torch.zeros_like(selector),
        selector_weight=1.0,
        source_score_weight=0.0,
        selector_transform="rank",
    )

    assert torch.allclose(scores, torch.tensor([0.0, 2.0 / 3.0, 1.0 / 3.0, 1.0]))


def test_build_selector_sampling_scores_can_suppress_hard_negative_risk():
    selector = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32)
    risk = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

    scores = build_selector_sampling_scores(
        selector=selector,
        source_scores=torch.zeros_like(selector),
        selector_weight=1.0,
        source_score_weight=0.0,
        hard_negative_risk=risk,
        hard_negative_weight=0.5,
    )

    assert int(scores.argmax().item()) == 1
    assert torch.allclose(scores, torch.tensor([0.4, 0.8, 0.7]))


def test_build_selector_sampling_scores_can_reward_positive_support():
    selector = torch.tensor([0.7, 0.6, 0.5], dtype=torch.float32)
    support = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)

    scores = build_selector_sampling_scores(
        selector=selector,
        source_scores=torch.zeros_like(selector),
        selector_weight=1.0,
        source_score_weight=0.0,
        positive_support=support,
        positive_support_weight=0.35,
    )

    assert int(scores.argmax().item()) == 1
    assert torch.allclose(scores, torch.tensor([0.7, 0.95, 0.5]))


def test_build_selector_sampling_scores_can_reward_pose_information():
    selector = torch.tensor([0.7, 0.6, 0.5], dtype=torch.float32)
    pose_information = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)

    scores = build_selector_sampling_scores(
        selector=selector,
        source_scores=torch.zeros_like(selector),
        selector_weight=1.0,
        source_score_weight=0.0,
        pose_information=pose_information,
        pose_information_weight=0.35,
    )

    assert int(scores.argmax().item()) == 1
    assert torch.allclose(scores, torch.tensor([0.7, 0.95, 0.5]))


def test_build_selector_sampling_scores_can_reward_hard_query_support():
    selector = torch.tensor([0.7, 0.6, 0.5], dtype=torch.float32)
    hard_query_support = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)

    scores = build_selector_sampling_scores(
        selector=selector,
        source_scores=torch.zeros_like(selector),
        selector_weight=1.0,
        source_score_weight=0.0,
        hard_query_support=hard_query_support,
        hard_query_support_weight=0.35,
    )

    assert int(scores.argmax().item()) == 1
    assert torch.allclose(scores, torch.tensor([0.7, 0.95, 0.5]))


def test_hard_negative_risk_from_episode_cache_maps_base_ids_to_gaussians():
    payload = {
        "landmark_id": torch.tensor([[0, 1], [0, 2]], dtype=torch.long),
        "cosine": torch.tensor([[0.9, 0.8], [0.95, 0.7]], dtype=torch.float32),
        "candidate_mask": torch.ones((2, 2), dtype=torch.bool),
        "reprojection_error": torch.tensor([[20.0, 1.0], [18.0, 2.0]], dtype=torch.float32),
        "metadata": {
            "reprojection_threshold_px": 8.0,
            "split_audit": {
                "audit_status": "passed",
            },
        },
    }

    risk, metadata = hard_negative_risk_from_episode_cache(
        payload,
        num_gaussians=5,
        base_gaussian_id=torch.tensor([3, 1, 4], dtype=torch.long),
        score_threshold=0.5,
    )

    assert risk.tolist() == [0.0, 0.0, 0.0, 1.0, 0.0]
    assert metadata["observed_landmarks"] == 3
    assert metadata["hard_negative_pairs"] == 2
    assert metadata["split_audit"]["audit_status"] == "passed"


def test_positive_support_from_episode_cache_maps_base_ids_to_gaussians():
    payload = {
        "landmark_id": torch.tensor([[0, 1], [0, 2]], dtype=torch.long),
        "cosine": torch.tensor([[0.9, 0.8], [0.95, 0.7]], dtype=torch.float32),
        "candidate_mask": torch.ones((2, 2), dtype=torch.bool),
        "reprojection_error": torch.tensor([[1.0, 20.0], [2.0, 3.0]], dtype=torch.float32),
        "metadata": {
            "reprojection_threshold_px": 8.0,
            "split_audit": {
                "audit_status": "passed",
            },
        },
    }

    support, metadata = positive_support_from_episode_cache(
        payload,
        num_gaussians=5,
        base_gaussian_id=torch.tensor([3, 1, 4], dtype=torch.long),
        score_threshold=0.5,
    )

    assert torch.allclose(support, torch.tensor([0.0, 0.0, 0.0, 1.0, 0.5]))
    assert metadata["positive_pairs"] == 3
    assert metadata["positive_landmarks"] == 2
    assert metadata["split_audit"]["audit_status"] == "passed"


def test_pose_information_from_episode_cache_weights_low_error_confident_inliers():
    payload = {
        "landmark_id": torch.tensor([[0, 1], [0, 2], [2, 1]], dtype=torch.long),
        "cosine": torch.tensor([[0.9, 0.8], [0.95, 0.7], [0.4, 0.8]], dtype=torch.float32),
        "candidate_mask": torch.ones((3, 2), dtype=torch.bool),
        "reprojection_error": torch.tensor([[1.0, 7.0], [4.0, 2.0], [1.0, 20.0]], dtype=torch.float32),
        "query_score": torch.tensor([1.0, 0.5, 1.0], dtype=torch.float32),
        "margin": torch.tensor([2.0, 1.0, 0.5], dtype=torch.float32),
        "metadata": {
            "reprojection_threshold_px": 8.0,
            "split_audit": {
                "audit_status": "passed",
            },
        },
    }

    pose_information, metadata = pose_information_from_episode_cache(
        payload,
        num_gaussians=5,
        base_gaussian_id=torch.tensor([3, 1, 4], dtype=torch.long),
        score_threshold=0.5,
    )

    assert int(pose_information.argmax().item()) == 3
    assert torch.isclose(pose_information[3], torch.tensor(1.0))
    assert 0.0 < float(pose_information[4].item()) < 1.0
    assert 0.0 < float(pose_information[1].item()) < float(pose_information[4].item())
    assert metadata["positive_pairs"] == 4
    assert metadata["pose_landmarks"] == 3
    assert metadata["split_audit"]["audit_status"] == "passed"


def test_hard_query_support_from_episode_cache_weights_low_margin_inliers():
    payload = {
        "landmark_id": torch.tensor([[0, 1], [0, 2]], dtype=torch.long),
        "cosine": torch.tensor([[0.9, 0.8], [0.95, 0.7]], dtype=torch.float32),
        "candidate_mask": torch.ones((2, 2), dtype=torch.bool),
        "reprojection_error": torch.tensor([[1.0, 20.0], [20.0, 1.0]], dtype=torch.float32),
        "margin": torch.tensor([1.0, 0.1], dtype=torch.float32),
        "metadata": {
            "reprojection_threshold_px": 8.0,
            "split_audit": {
                "audit_status": "passed",
            },
        },
    }

    support, metadata = hard_query_support_from_episode_cache(
        payload,
        num_gaussians=5,
        base_gaussian_id=torch.tensor([3, 1, 4], dtype=torch.long),
        score_threshold=0.5,
    )

    assert int(support.argmax().item()) == 4
    assert 0.0 < float(support[3].item()) < 1.0
    assert torch.isclose(support[4], torch.tensor(1.0))
    assert metadata["positive_pairs"] == 2
    assert metadata["hard_query_landmarks"] == 2
    assert metadata["split_audit"]["audit_status"] == "passed"


def test_query_coverage_reservation_from_episode_cache_covers_low_margin_queries():
    payload = {
        "landmark_id": torch.tensor([[0, 1], [1, 2], [3, 4]], dtype=torch.long),
        "cosine": torch.tensor([[0.9, 0.8], [0.85, 0.7], [0.95, 0.9]], dtype=torch.float32),
        "candidate_mask": torch.ones((3, 2), dtype=torch.bool),
        "reprojection_error": torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], dtype=torch.float32),
        "margin": torch.tensor([0.1, 0.2, 1.0], dtype=torch.float32),
        "metadata": {
            "reprojection_threshold_px": 8.0,
            "split_audit": {
                "audit_status": "passed",
            },
        },
    }

    selected, metadata = query_coverage_reservation_from_episode_cache(
        payload,
        num_gaussians=8,
        base_gaussian_id=torch.tensor([3, 1, 4, 6, 7], dtype=torch.long),
        score_threshold=0.5,
        margin_threshold=0.25,
        max_landmarks=2,
    )

    assert selected.tolist() == [1, 3]
    assert metadata["hard_query_count"] == 2
    assert metadata["covered_query_count"] == 2
    assert metadata["selected_count"] == 2
    assert metadata["split_audit"]["audit_status"] == "passed"


def test_coverage_constrained_topk_keeps_budget_and_spreads_cells():
    scores = torch.tensor([10.0, 9.0, 8.0, 1.0, 7.0, 6.0])
    xyz = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.1, 0.0],
            [0.2, 0.2, 0.0],
            [0.3, 0.3, 0.0],
            [5.0, 5.0, 0.0],
            [5.1, 5.1, 0.0],
        ],
        dtype=torch.float32,
    )

    selected = coverage_constrained_topk(scores=scores, xyz=xyz, budget=3, coverage_grid=2)

    assert selected.numel() == 3
    assert 0 in selected.tolist()
    assert 4 in selected.tolist()
    assert len(set(selected.tolist())) == 3


def test_resample_detector_landmarks_changes_indices_with_same_budget(tmp_path):
    detector = tmp_path / "detector"
    detector.mkdir()
    with (detector / "sampled_idx.pkl").open("wb") as handle:
        pickle.dump(torch.tensor([0, 1, 2], dtype=torch.long), handle)
    with (detector / "sampled_scores.pkl").open("wb") as handle:
        pickle.dump(
            {
                "sampled_scores": torch.tensor([0.2, 0.2, 0.2], dtype=torch.float32),
                "score_avg": torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2], dtype=torch.float32),
            },
            handle,
        )
    selector = torch.tensor([0.1, 0.2, 0.3, 0.9, 0.8], dtype=torch.float32)
    xyz = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    payload = resample_detector_landmarks(
        source_detector_dir=detector,
        selector=selector,
        xyz=xyz,
        budget="same_as_source",
        selector_weight=1.0,
        source_score_weight=0.0,
        coverage_grid=0,
    )

    assert payload["source_count"] == 3
    assert payload["output_count"] == 3
    assert payload["sampled_idx_changed"] is True
    assert payload["sampled_idx"].tolist() == [3, 4, 2]
    assert torch.allclose(payload["sampled_scores"], torch.tensor([0.9, 0.8, 0.3]))


def test_resample_detector_landmarks_can_use_hard_negative_risk(tmp_path):
    detector = tmp_path / "detector"
    detector.mkdir()
    with (detector / "sampled_idx.pkl").open("wb") as handle:
        pickle.dump(torch.tensor([0, 1], dtype=torch.long), handle)
    selector = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32)
    hard_negative_risk = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    xyz = torch.arange(9, dtype=torch.float32).reshape(3, 3)

    payload = resample_detector_landmarks(
        source_detector_dir=detector,
        selector=selector,
        xyz=xyz,
        budget=2,
        selector_weight=1.0,
        source_score_weight=0.0,
        hard_negative_risk=hard_negative_risk,
        hard_negative_weight=0.5,
    )

    assert payload["sampled_idx"].tolist() == [1, 2]
    assert payload["hard_negative_weight"] == 0.5
    assert torch.allclose(payload["hard_negative_risk"], hard_negative_risk)


def test_resample_detector_landmarks_can_use_positive_support(tmp_path):
    detector = tmp_path / "detector"
    detector.mkdir()
    with (detector / "sampled_idx.pkl").open("wb") as handle:
        pickle.dump(torch.tensor([0, 1], dtype=torch.long), handle)
    selector = torch.tensor([0.8, 0.7, 0.6], dtype=torch.float32)
    support = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    xyz = torch.arange(9, dtype=torch.float32).reshape(3, 3)

    payload = resample_detector_landmarks(
        source_detector_dir=detector,
        selector=selector,
        xyz=xyz,
        budget=2,
        selector_weight=1.0,
        source_score_weight=0.0,
        positive_support=support,
        positive_support_weight=0.35,
    )

    assert payload["sampled_idx"].tolist() == [1, 0]
    assert payload["positive_support_weight"] == 0.35
    assert torch.allclose(payload["positive_support"], support)


def test_resample_detector_landmarks_can_use_pose_information(tmp_path):
    detector = tmp_path / "detector"
    detector.mkdir()
    with (detector / "sampled_idx.pkl").open("wb") as handle:
        pickle.dump(torch.tensor([0, 1], dtype=torch.long), handle)
    selector = torch.tensor([0.8, 0.7, 0.6], dtype=torch.float32)
    pose_information = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    xyz = torch.arange(9, dtype=torch.float32).reshape(3, 3)

    payload = resample_detector_landmarks(
        source_detector_dir=detector,
        selector=selector,
        xyz=xyz,
        budget=2,
        selector_weight=1.0,
        source_score_weight=0.0,
        pose_information=pose_information,
        pose_information_weight=0.35,
    )

    assert payload["sampled_idx"].tolist() == [1, 0]
    assert payload["pose_information_weight"] == 0.35
    assert torch.allclose(payload["pose_information"], pose_information)


def test_resample_detector_landmarks_can_use_hard_query_support(tmp_path):
    detector = tmp_path / "detector"
    detector.mkdir()
    with (detector / "sampled_idx.pkl").open("wb") as handle:
        pickle.dump(torch.tensor([0, 1], dtype=torch.long), handle)
    selector = torch.tensor([0.8, 0.7, 0.6], dtype=torch.float32)
    hard_query_support = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    xyz = torch.arange(9, dtype=torch.float32).reshape(3, 3)

    payload = resample_detector_landmarks(
        source_detector_dir=detector,
        selector=selector,
        xyz=xyz,
        budget=2,
        selector_weight=1.0,
        source_score_weight=0.0,
        hard_query_support=hard_query_support,
        hard_query_support_weight=0.35,
    )

    assert payload["sampled_idx"].tolist() == [1, 0]
    assert payload["hard_query_support_weight"] == 0.35
    assert torch.allclose(payload["hard_query_support"], hard_query_support)


def test_resample_detector_landmarks_can_reserve_strict_support(tmp_path):
    detector = tmp_path / "detector"
    detector.mkdir()
    with (detector / "sampled_idx.pkl").open("wb") as handle:
        pickle.dump(torch.tensor([0, 1], dtype=torch.long), handle)
    selector = torch.tensor([0.9, 0.1, 0.8, 0.7], dtype=torch.float32)
    strict_support = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)
    xyz = torch.arange(12, dtype=torch.float32).reshape(4, 3)

    payload = resample_detector_landmarks(
        source_detector_dir=detector,
        selector=selector,
        xyz=xyz,
        budget=2,
        selector_weight=1.0,
        source_score_weight=0.0,
        candidate_pool="all_gaussians",
        strict_support=strict_support,
        strict_support_fraction=0.5,
        strict_support_metadata={"split_audit": {"audit_status": "passed"}},
    )

    assert payload["sampled_idx"].tolist() == [1, 0]
    assert payload["strict_support_fraction"] == 0.5
    assert payload["strict_support_reserved_count"] == 1
    assert torch.allclose(payload["strict_support"], strict_support)
    assert payload["strict_support_metadata"]["split_audit"]["audit_status"] == "passed"


def test_resample_detector_landmarks_can_retain_source_fraction_with_all_gaussians(tmp_path):
    detector = tmp_path / "detector"
    detector.mkdir()
    with (detector / "sampled_idx.pkl").open("wb") as handle:
        pickle.dump(torch.tensor([0, 1, 2, 3], dtype=torch.long), handle)
    selector = torch.tensor([0.1, 0.2, 0.3, 0.4, 1.0, 0.9], dtype=torch.float32)
    xyz = torch.arange(18, dtype=torch.float32).reshape(6, 3)

    payload = resample_detector_landmarks(
        source_detector_dir=detector,
        selector=selector,
        xyz=xyz,
        budget=4,
        selector_weight=1.0,
        source_score_weight=0.0,
        coverage_grid=0,
        candidate_pool="all_gaussians",
        preserve_source_order=False,
        source_retention_fraction=0.5,
    )

    assert payload["sampled_idx"].tolist() == [3, 2, 4, 5]
    assert payload["source_retention_fraction"] == 0.5
    assert payload["source_retained_count"] == 2


def test_resample_detector_landmarks_reserves_protected_source_indices_before_fill(tmp_path):
    detector = tmp_path / "detector"
    detector.mkdir()
    with (detector / "sampled_idx.pkl").open("wb") as handle:
        pickle.dump(torch.tensor([0, 1, 2, 3], dtype=torch.long), handle)
    selector = torch.tensor([0.1, 0.2, 0.3, 0.4, 1.0, 0.9], dtype=torch.float32)
    xyz = torch.arange(18, dtype=torch.float32).reshape(6, 3)

    payload = resample_detector_landmarks(
        source_detector_dir=detector,
        selector=selector,
        xyz=xyz,
        budget=4,
        selector_weight=1.0,
        source_score_weight=0.0,
        coverage_grid=0,
        candidate_pool="all_gaussians",
        preserve_source_order=False,
        protected_source_idx=torch.tensor([0, 2], dtype=torch.long),
        protected_source_fraction=1.0,
        protected_source_metadata={"source": "native8192"},
    )

    assert payload["sampled_idx"].tolist() == [0, 2, 4, 5]
    assert payload["protected_source_fraction"] == 1.0
    assert payload["protected_source_reserved_count"] == 2
    assert payload["protected_source_metadata"]["source"] == "native8192"


def test_resample_detector_landmarks_reserves_query_coverage_indices_before_fill(tmp_path):
    detector = tmp_path / "detector"
    detector.mkdir()
    with (detector / "sampled_idx.pkl").open("wb") as handle:
        pickle.dump(torch.tensor([0, 1, 2, 3], dtype=torch.long), handle)
    selector = torch.tensor([0.1, 0.2, 0.3, 0.4, 1.0, 0.9], dtype=torch.float32)
    xyz = torch.arange(18, dtype=torch.float32).reshape(6, 3)

    payload = resample_detector_landmarks(
        source_detector_dir=detector,
        selector=selector,
        xyz=xyz,
        budget=4,
        selector_weight=1.0,
        source_score_weight=0.0,
        coverage_grid=0,
        candidate_pool="all_gaussians",
        preserve_source_order=False,
        query_coverage_idx=torch.tensor([0, 2], dtype=torch.long),
        query_coverage_fraction=0.5,
        query_coverage_metadata={"source": "selfmap"},
    )

    assert payload["sampled_idx"].tolist() == [0, 2, 4, 5]
    assert payload["query_coverage_fraction"] == 0.5
    assert payload["query_coverage_reserved_count"] == 2
    assert payload["query_coverage_metadata"]["source"] == "selfmap"


def test_resample_detector_landmarks_deduplicates_and_clips_protected_source_indices(tmp_path):
    detector = tmp_path / "detector"
    detector.mkdir()
    with (detector / "sampled_idx.pkl").open("wb") as handle:
        pickle.dump(torch.tensor([0, 1, 2, 3], dtype=torch.long), handle)
    selector = torch.tensor([0.1, 0.2, 0.3, 0.4, 1.0], dtype=torch.float32)
    xyz = torch.arange(15, dtype=torch.float32).reshape(5, 3)

    payload = resample_detector_landmarks(
        source_detector_dir=detector,
        selector=selector,
        xyz=xyz,
        budget=3,
        selector_weight=1.0,
        source_score_weight=0.0,
        coverage_grid=0,
        candidate_pool="source_sampled",
        preserve_source_order=False,
        protected_source_idx=torch.tensor([2, 2, 99, 1, -1, 4], dtype=torch.long),
        protected_source_fraction=1.0,
    )

    assert payload["sampled_idx"].tolist() == [2, 1, 3]
    assert payload["protected_source_reserved_count"] == 2
    assert payload["protected_source_metadata"]["input_count"] == 6
    assert payload["protected_source_metadata"]["unique_count"] == 5
    assert payload["protected_source_metadata"]["candidate_count"] == 2


def test_resample_detector_landmarks_can_restrict_to_source_pool_and_preserve_order(tmp_path):
    detector = tmp_path / "detector"
    detector.mkdir()
    with (detector / "sampled_idx.pkl").open("wb") as handle:
        pickle.dump(torch.tensor([0, 1, 2, 3], dtype=torch.long), handle)
    selector = torch.tensor([0.1, 0.9, 0.8, 0.7, 1.0], dtype=torch.float32)
    xyz = torch.arange(15, dtype=torch.float32).reshape(5, 3)

    payload = resample_detector_landmarks(
        source_detector_dir=detector,
        selector=selector,
        xyz=xyz,
        budget=2,
        selector_weight=1.0,
        source_score_weight=0.0,
        coverage_grid=0,
        candidate_pool="source_sampled",
        preserve_source_order=True,
    )

    assert payload["sampled_idx"].tolist() == [1, 2]
    assert payload["output_count"] == 2
    assert payload["candidate_pool"] == "source_sampled"


def test_export_selector_resampled_map_writes_native_descriptor_map_and_manifest(tmp_path):
    source = tmp_path / "source"
    pc_dir = source / "point_cloud" / "iteration_30000"
    detector = source / "detector"
    pc_dir.mkdir(parents=True)
    detector.mkdir()
    _write_ply(pc_dir / "point_cloud.ply")
    with (detector / "sampled_idx.pkl").open("wb") as handle:
        pickle.dump(torch.tensor([0, 1, 2], dtype=torch.long), handle)
    checkpoint = tmp_path / "selector.pt"
    torch.save(
        {
            "export_descriptors": F.normalize(torch.ones((5, 2), dtype=torch.float32), p=2, dim=-1),
            "gate": torch.tensor([0.1, 0.2, 0.3, 0.9, 0.8], dtype=torch.float32),
        },
        checkpoint,
    )
    output = tmp_path / "resampled"
    args = build_argparser().parse_args(
        [
            "--source_map",
            str(source),
            "--checkpoint_path",
            str(checkpoint),
            "--output_map",
            str(output),
            "--descriptor_mode",
            "native",
            "--budget",
            "same_as_source",
            "--selector_weight",
            "1.0",
            "--source_score_weight",
            "0.0",
        ]
    )

    assert main(args) == 0

    with (output / "detector" / "sampled_idx.pkl").open("rb") as handle:
        sampled_idx = pickle.load(handle)
    manifest = __import__("json").loads((output / "selector_resampling_manifest.json").read_text(encoding="utf-8"))
    vertex = PlyData.read(str(output / "point_cloud" / "iteration_30000" / "point_cloud.ply"))["vertex"].data
    assert sampled_idx.tolist() == [0, 1, 2]
    assert manifest["sampled_idx_changed"] is False
    assert manifest["candidate_pool"] == "source_sampled"
    assert manifest["preserve_source_order"] is True
    assert manifest["descriptor_mode"] == "native"
    assert np.allclose([vertex["loc_0"][3], vertex["loc_1"][3]], [2.0, 0.0])


def test_export_selector_resampled_map_can_explicitly_use_all_gaussians(tmp_path):
    source = tmp_path / "source"
    pc_dir = source / "point_cloud" / "iteration_30000"
    detector = source / "detector"
    pc_dir.mkdir(parents=True)
    detector.mkdir()
    _write_ply(pc_dir / "point_cloud.ply")
    with (detector / "sampled_idx.pkl").open("wb") as handle:
        pickle.dump(torch.tensor([0, 1, 2], dtype=torch.long), handle)
    checkpoint = tmp_path / "selector.pt"
    torch.save(
        {
            "export_descriptors": F.normalize(torch.ones((5, 2), dtype=torch.float32), p=2, dim=-1),
            "gate": torch.tensor([0.1, 0.2, 0.3, 0.9, 0.8], dtype=torch.float32),
        },
        checkpoint,
    )
    output = tmp_path / "resampled"
    args = build_argparser().parse_args(
        [
            "--source_map",
            str(source),
            "--checkpoint_path",
            str(checkpoint),
            "--output_map",
            str(output),
            "--descriptor_mode",
            "native",
            "--budget",
            "same_as_source",
            "--selector_weight",
            "1.0",
            "--source_score_weight",
            "0.0",
            "--candidate_pool",
            "all_gaussians",
            "--no-preserve_source_order",
        ]
    )

    assert main(args) == 0

    with (output / "detector" / "sampled_idx.pkl").open("rb") as handle:
        sampled_idx = pickle.load(handle)
    manifest = __import__("json").loads((output / "selector_resampling_manifest.json").read_text(encoding="utf-8"))
    assert sampled_idx.tolist() == [3, 4, 2]
    assert manifest["sampled_idx_changed"] is True
    assert manifest["candidate_pool"] == "all_gaussians"
    assert manifest["preserve_source_order"] is False


def test_export_selector_resampled_map_records_hard_negative_cache(tmp_path):
    source = tmp_path / "source"
    pc_dir = source / "point_cloud" / "iteration_30000"
    detector = source / "detector"
    pc_dir.mkdir(parents=True)
    detector.mkdir()
    _write_ply(pc_dir / "point_cloud.ply")
    with (detector / "sampled_idx.pkl").open("wb") as handle:
        pickle.dump(torch.tensor([0, 1, 2], dtype=torch.long), handle)
    checkpoint = tmp_path / "selector.pt"
    torch.save(
        {
            "export_descriptors": F.normalize(torch.ones((5, 2), dtype=torch.float32), p=2, dim=-1),
            "gate": torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5], dtype=torch.float32),
        },
        checkpoint,
    )
    cache = tmp_path / "risk_cache.pt"
    torch.save(
        {
            "landmark_id": torch.tensor([[0, 1], [0, 2]], dtype=torch.long),
            "cosine": torch.tensor([[0.9, 0.2], [0.8, 0.2]], dtype=torch.float32),
            "candidate_mask": torch.ones((2, 2), dtype=torch.bool),
            "reprojection_error": torch.tensor([[20.0, 1.0], [18.0, 1.0]], dtype=torch.float32),
            "metadata": {
                "reprojection_threshold_px": 8.0,
                "split_audit": {
                    "audit_status": "passed",
                },
            },
        },
        cache,
    )
    output = tmp_path / "resampled"
    args = build_argparser().parse_args(
        [
            "--source_map",
            str(source),
            "--checkpoint_path",
            str(checkpoint),
            "--output_map",
            str(output),
            "--descriptor_mode",
            "native",
            "--budget",
            "same_as_source",
            "--selector_weight",
            "1.0",
            "--source_score_weight",
            "0.0",
            "--hard_negative_cache",
            str(cache),
            "--hard_negative_weight",
            "0.5",
            "--candidate_pool",
            "all_gaussians",
            "--no-preserve_source_order",
        ]
    )

    assert main(args) == 0

    with (output / "detector" / "sampled_idx.pkl").open("rb") as handle:
        sampled_idx = pickle.load(handle)
    manifest = __import__("json").loads((output / "selector_resampling_manifest.json").read_text(encoding="utf-8"))
    assert sampled_idx.tolist() == [1, 2, 3]
    assert manifest["hard_negative_cache"] == str(cache)
    assert manifest["hard_negative_weight"] == 0.5
    assert manifest["hard_negative_risk"]["split_audit"]["audit_status"] == "passed"


def test_export_selector_resampled_map_records_positive_support_cache(tmp_path):
    source = tmp_path / "source"
    pc_dir = source / "point_cloud" / "iteration_30000"
    detector = source / "detector"
    pc_dir.mkdir(parents=True)
    detector.mkdir()
    _write_ply(pc_dir / "point_cloud.ply")
    with (detector / "sampled_idx.pkl").open("wb") as handle:
        pickle.dump(torch.tensor([0, 1, 2], dtype=torch.long), handle)
    checkpoint = tmp_path / "selector.pt"
    torch.save(
        {
            "export_descriptors": F.normalize(torch.ones((5, 2), dtype=torch.float32), p=2, dim=-1),
            "gate": torch.tensor([0.8, 0.7, 0.6, 0.5, 0.4], dtype=torch.float32),
        },
        checkpoint,
    )
    cache = tmp_path / "support_cache.pt"
    torch.save(
        {
            "landmark_id": torch.tensor([[0, 1], [0, 2]], dtype=torch.long),
            "cosine": torch.tensor([[0.9, 0.2], [0.8, 0.7]], dtype=torch.float32),
            "candidate_mask": torch.ones((2, 2), dtype=torch.bool),
            "reprojection_error": torch.tensor([[1.0, 20.0], [2.0, 1.0]], dtype=torch.float32),
            "metadata": {
                "reprojection_threshold_px": 8.0,
                "split_audit": {
                    "audit_status": "passed",
                },
            },
        },
        cache,
    )
    output = tmp_path / "resampled"
    args = build_argparser().parse_args(
        [
            "--source_map",
            str(source),
            "--checkpoint_path",
            str(checkpoint),
            "--output_map",
            str(output),
            "--descriptor_mode",
            "native",
            "--budget",
            "same_as_source",
            "--selector_weight",
            "1.0",
            "--source_score_weight",
            "0.0",
            "--positive_support_cache",
            str(cache),
            "--positive_support_weight",
            "0.35",
            "--candidate_pool",
            "all_gaussians",
            "--no-preserve_source_order",
        ]
    )

    assert main(args) == 0

    with (output / "detector" / "sampled_idx.pkl").open("rb") as handle:
        sampled_idx = pickle.load(handle)
    manifest = __import__("json").loads((output / "selector_resampling_manifest.json").read_text(encoding="utf-8"))
    assert sampled_idx.tolist() == [0, 2, 1]
    assert manifest["positive_support_cache"] == str(cache)
    assert manifest["positive_support_weight"] == 0.35
    assert manifest["positive_support"]["split_audit"]["audit_status"] == "passed"


def test_export_selector_resampled_map_records_pose_information_cache(tmp_path):
    source = tmp_path / "source"
    pc_dir = source / "point_cloud" / "iteration_30000"
    detector = source / "detector"
    pc_dir.mkdir(parents=True)
    detector.mkdir()
    _write_ply(pc_dir / "point_cloud.ply")
    with (detector / "sampled_idx.pkl").open("wb") as handle:
        pickle.dump(torch.tensor([0, 1, 2], dtype=torch.long), handle)
    checkpoint = tmp_path / "selector.pt"
    torch.save(
        {
            "export_descriptors": F.normalize(torch.ones((5, 2), dtype=torch.float32), p=2, dim=-1),
            "gate": torch.tensor([0.8, 0.7, 0.6, 0.5, 0.4], dtype=torch.float32),
        },
        checkpoint,
    )
    cache = tmp_path / "pose_cache.pt"
    torch.save(
        {
            "landmark_id": torch.tensor([[0, 1], [0, 2]], dtype=torch.long),
            "cosine": torch.tensor([[0.9, 0.2], [0.8, 0.7]], dtype=torch.float32),
            "candidate_mask": torch.ones((2, 2), dtype=torch.bool),
            "reprojection_error": torch.tensor([[1.0, 20.0], [2.0, 1.0]], dtype=torch.float32),
            "query_score": torch.tensor([1.0, 1.0], dtype=torch.float32),
            "margin": torch.tensor([2.0, 1.0], dtype=torch.float32),
            "metadata": {
                "reprojection_threshold_px": 8.0,
                "split_audit": {
                    "audit_status": "passed",
                },
            },
        },
        cache,
    )
    output = tmp_path / "resampled"
    args = build_argparser().parse_args(
        [
            "--source_map",
            str(source),
            "--checkpoint_path",
            str(checkpoint),
            "--output_map",
            str(output),
            "--descriptor_mode",
            "native",
            "--budget",
            "same_as_source",
            "--selector_weight",
            "1.0",
            "--source_score_weight",
            "0.0",
            "--pose_information_cache",
            str(cache),
            "--pose_information_weight",
            "0.35",
            "--candidate_pool",
            "all_gaussians",
            "--no-preserve_source_order",
        ]
    )

    assert main(args) == 0

    with (output / "detector" / "sampled_idx.pkl").open("rb") as handle:
        sampled_idx = pickle.load(handle)
    manifest = __import__("json").loads((output / "selector_resampling_manifest.json").read_text(encoding="utf-8"))
    assert sampled_idx.tolist() == [0, 2, 1]
    assert manifest["pose_information_cache"] == str(cache)
    assert manifest["pose_information_weight"] == 0.35
    assert manifest["pose_information"]["split_audit"]["audit_status"] == "passed"


def test_export_selector_resampled_map_records_hard_query_support_cache(tmp_path):
    source = tmp_path / "source"
    pc_dir = source / "point_cloud" / "iteration_30000"
    detector = source / "detector"
    pc_dir.mkdir(parents=True)
    detector.mkdir()
    _write_ply(pc_dir / "point_cloud.ply")
    with (detector / "sampled_idx.pkl").open("wb") as handle:
        pickle.dump(torch.tensor([0, 1, 2], dtype=torch.long), handle)
    checkpoint = tmp_path / "selector.pt"
    torch.save(
        {
            "export_descriptors": F.normalize(torch.ones((5, 2), dtype=torch.float32), p=2, dim=-1),
            "gate": torch.tensor([0.8, 0.7, 0.6, 0.5, 0.4], dtype=torch.float32),
        },
        checkpoint,
    )
    cache = tmp_path / "hard_query_cache.pt"
    torch.save(
        {
            "landmark_id": torch.tensor([[0, 1], [0, 2]], dtype=torch.long),
            "cosine": torch.tensor([[0.9, 0.2], [0.8, 0.7]], dtype=torch.float32),
            "candidate_mask": torch.ones((2, 2), dtype=torch.bool),
            "reprojection_error": torch.tensor([[1.0, 20.0], [2.0, 1.0]], dtype=torch.float32),
            "margin": torch.tensor([1.0, 0.1], dtype=torch.float32),
            "metadata": {
                "reprojection_threshold_px": 8.0,
                "split_audit": {
                    "audit_status": "passed",
                },
            },
        },
        cache,
    )
    output = tmp_path / "resampled"
    args = build_argparser().parse_args(
        [
            "--source_map",
            str(source),
            "--checkpoint_path",
            str(checkpoint),
            "--output_map",
            str(output),
            "--descriptor_mode",
            "native",
            "--budget",
            "same_as_source",
            "--selector_weight",
            "1.0",
            "--source_score_weight",
            "0.0",
            "--hard_query_support_cache",
            str(cache),
            "--hard_query_support_weight",
            "0.35",
            "--candidate_pool",
            "all_gaussians",
            "--no-preserve_source_order",
        ]
    )

    assert main(args) == 0

    with (output / "detector" / "sampled_idx.pkl").open("rb") as handle:
        sampled_idx = pickle.load(handle)
    manifest = __import__("json").loads((output / "selector_resampling_manifest.json").read_text(encoding="utf-8"))
    assert sampled_idx.tolist() == [0, 2, 1]
    assert manifest["hard_query_support_cache"] == str(cache)
    assert manifest["hard_query_support_weight"] == 0.35
    assert manifest["hard_query_support"]["split_audit"]["audit_status"] == "passed"


def test_export_selector_resampled_map_records_query_coverage_cache(tmp_path):
    source = tmp_path / "source"
    pc_dir = source / "point_cloud" / "iteration_30000"
    detector = source / "detector"
    pc_dir.mkdir(parents=True)
    detector.mkdir()
    _write_ply(pc_dir / "point_cloud.ply")
    with (detector / "sampled_idx.pkl").open("wb") as handle:
        pickle.dump(torch.tensor([0, 1, 2], dtype=torch.long), handle)
    checkpoint = tmp_path / "selector.pt"
    torch.save(
        {
            "export_descriptors": F.normalize(torch.ones((5, 2), dtype=torch.float32), p=2, dim=-1),
            "gate": torch.tensor([0.1, 0.2, 0.3, 0.9, 0.8], dtype=torch.float32),
        },
        checkpoint,
    )
    cache = tmp_path / "query_coverage_cache.pt"
    torch.save(
        {
            "landmark_id": torch.tensor([[0, 1], [1, 2], [3, 4]], dtype=torch.long),
            "cosine": torch.tensor([[0.9, 0.8], [0.85, 0.7], [0.95, 0.9]], dtype=torch.float32),
            "candidate_mask": torch.ones((3, 2), dtype=torch.bool),
            "reprojection_error": torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], dtype=torch.float32),
            "margin": torch.tensor([0.1, 0.2, 1.0], dtype=torch.float32),
            "metadata": {
                "feedback_bank_split_name": "selfmap_train_rendered",
                "reprojection_threshold_px": 8.0,
                "split_audit": {
                    "audit_status": "passed",
                },
            },
        },
        cache,
    )
    output = tmp_path / "resampled"
    args = build_argparser().parse_args(
        [
            "--source_map",
            str(source),
            "--checkpoint_path",
            str(checkpoint),
            "--output_map",
            str(output),
            "--descriptor_mode",
            "native",
            "--budget",
            "same_as_source",
            "--selector_weight",
            "1.0",
            "--source_score_weight",
            "0.0",
            "--query_coverage_cache",
            str(cache),
            "--query_coverage_fraction",
            "0.67",
            "--query_coverage_margin_threshold",
            "0.25",
            "--candidate_pool",
            "all_gaussians",
            "--no-preserve_source_order",
        ]
    )

    assert main(args) == 0

    with (output / "detector" / "sampled_idx.pkl").open("rb") as handle:
        sampled_idx = pickle.load(handle)
    manifest = __import__("json").loads((output / "selector_resampling_manifest.json").read_text(encoding="utf-8"))
    audit_manifest = __import__("json").loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert sampled_idx.tolist() == [1, 0, 3]
    assert manifest["query_coverage_cache"] == str(cache)
    assert manifest["query_coverage_fraction"] == 0.67
    assert manifest["query_coverage_reserved_count"] == 2
    assert manifest["query_coverage"]["covered_query_count"] == 2
    assert manifest["query_coverage"]["split_audit"]["audit_status"] == "passed"
    assert audit_manifest["command"]
    assert audit_manifest["git_commit"]
    assert audit_manifest["scene"] == "source"
    assert audit_manifest["split"] == "selfmap_train_rendered"
    assert audit_manifest["map_path"] == str(output)
    assert audit_manifest["hyperparameters"]["query_coverage_fraction"] == 0.67
    assert audit_manifest["selector_feedback_enabled"] is True
    assert audit_manifest["rho_feedback_enabled"] is False


def test_export_selector_resampled_map_records_strict_support_cache(tmp_path):
    source = tmp_path / "source"
    pc_dir = source / "point_cloud" / "iteration_30000"
    detector = source / "detector"
    pc_dir.mkdir(parents=True)
    detector.mkdir()
    _write_ply(pc_dir / "point_cloud.ply")
    with (detector / "sampled_idx.pkl").open("wb") as handle:
        pickle.dump(torch.tensor([0, 1, 2], dtype=torch.long), handle)
    checkpoint = tmp_path / "selector.pt"
    torch.save(
        {
            "export_descriptors": F.normalize(torch.ones((5, 2), dtype=torch.float32), p=2, dim=-1),
            "gate": torch.tensor([0.8, 0.7, 0.1, 0.6, 0.5], dtype=torch.float32),
        },
        checkpoint,
    )
    cache = tmp_path / "strict_support_cache.pt"
    torch.save(
        {
            "landmark_id": torch.tensor([[2, 0], [2, 1]], dtype=torch.long),
            "cosine": torch.tensor([[0.9, 0.8], [0.95, 0.8]], dtype=torch.float32),
            "candidate_mask": torch.ones((2, 2), dtype=torch.bool),
            "reprojection_error": torch.tensor([[1.0, 20.0], [2.0, 20.0]], dtype=torch.float32),
            "metadata": {
                "reprojection_threshold_px": 8.0,
                "split_audit": {
                    "audit_status": "passed",
                },
            },
        },
        cache,
    )
    output = tmp_path / "resampled"
    args = build_argparser().parse_args(
        [
            "--source_map",
            str(source),
            "--checkpoint_path",
            str(checkpoint),
            "--output_map",
            str(output),
            "--descriptor_mode",
            "native",
            "--budget",
            "same_as_source",
            "--selector_weight",
            "1.0",
            "--source_score_weight",
            "0.0",
            "--strict_support_cache",
            str(cache),
            "--strict_support_fraction",
            "0.34",
            "--strict_support_score_threshold",
            "0.7",
            "--strict_support_reprojection_threshold_px",
            "3.0",
            "--candidate_pool",
            "all_gaussians",
            "--no-preserve_source_order",
        ]
    )

    assert main(args) == 0

    with (output / "detector" / "sampled_idx.pkl").open("rb") as handle:
        sampled_idx = pickle.load(handle)
    manifest = __import__("json").loads((output / "selector_resampling_manifest.json").read_text(encoding="utf-8"))
    assert sampled_idx.tolist() == [2, 0, 1]
    assert manifest["strict_support_cache"] == str(cache)
    assert manifest["strict_support_fraction"] == 0.34
    assert manifest["strict_support_reserved_count"] == 1
    assert manifest["strict_support_score_threshold"] == 0.7
    assert manifest["strict_support_reprojection_threshold_px"] == 3.0
    assert manifest["strict_support"]["split_audit"]["audit_status"] == "passed"


def test_export_selector_resampled_map_records_source_retention_fraction(tmp_path):
    source = tmp_path / "source"
    pc_dir = source / "point_cloud" / "iteration_30000"
    detector = source / "detector"
    pc_dir.mkdir(parents=True)
    detector.mkdir()
    _write_ply(pc_dir / "point_cloud.ply")
    with (detector / "sampled_idx.pkl").open("wb") as handle:
        pickle.dump(torch.tensor([0, 1, 2], dtype=torch.long), handle)
    checkpoint = tmp_path / "selector.pt"
    torch.save(
        {
            "export_descriptors": F.normalize(torch.ones((5, 2), dtype=torch.float32), p=2, dim=-1),
            "gate": torch.tensor([0.1, 0.2, 0.3, 0.9, 0.8], dtype=torch.float32),
        },
        checkpoint,
    )
    output = tmp_path / "resampled"
    args = build_argparser().parse_args(
        [
            "--source_map",
            str(source),
            "--checkpoint_path",
            str(checkpoint),
            "--output_map",
            str(output),
            "--descriptor_mode",
            "native",
            "--budget",
            "same_as_source",
            "--selector_weight",
            "1.0",
            "--source_score_weight",
            "0.0",
            "--candidate_pool",
            "all_gaussians",
            "--no-preserve_source_order",
            "--source_retention_fraction",
            "0.67",
        ]
    )

    assert main(args) == 0

    manifest = __import__("json").loads((output / "selector_resampling_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_retention_fraction"] == 0.67
    assert manifest["source_retained_count"] == 2


def test_export_selector_resampled_map_records_protected_source_map(tmp_path):
    source = tmp_path / "source"
    pc_dir = source / "point_cloud" / "iteration_30000"
    detector = source / "detector"
    pc_dir.mkdir(parents=True)
    detector.mkdir()
    _write_ply(pc_dir / "point_cloud.ply")
    with (detector / "sampled_idx.pkl").open("wb") as handle:
        pickle.dump(torch.tensor([0, 1, 2], dtype=torch.long), handle)
    protected = tmp_path / "protected"
    protected_detector = protected / "detector"
    protected_detector.mkdir(parents=True)
    with (protected_detector / "sampled_idx.pkl").open("wb") as handle:
        pickle.dump(torch.tensor([2, 0], dtype=torch.long), handle)
    checkpoint = tmp_path / "selector.pt"
    torch.save(
        {
            "export_descriptors": F.normalize(torch.ones((5, 2), dtype=torch.float32), p=2, dim=-1),
            "gate": torch.tensor([0.1, 0.2, 0.3, 0.9, 0.8], dtype=torch.float32),
        },
        checkpoint,
    )
    output = tmp_path / "resampled"
    args = build_argparser().parse_args(
        [
            "--source_map",
            str(source),
            "--checkpoint_path",
            str(checkpoint),
            "--output_map",
            str(output),
            "--descriptor_mode",
            "native",
            "--budget",
            "same_as_source",
            "--selector_weight",
            "1.0",
            "--source_score_weight",
            "0.0",
            "--candidate_pool",
            "all_gaussians",
            "--no-preserve_source_order",
            "--protected_source_map",
            str(protected),
            "--protected_source_fraction",
            "1.0",
        ]
    )

    assert main(args) == 0

    with (output / "detector" / "sampled_idx.pkl").open("rb") as handle:
        sampled_idx = pickle.load(handle)
    manifest = __import__("json").loads((output / "selector_resampling_manifest.json").read_text(encoding="utf-8"))
    assert sampled_idx.tolist() == [2, 0, 3]
    assert manifest["protected_source_map"] == str(protected)
    assert manifest["protected_source_fraction"] == 1.0
    assert manifest["protected_source_reserved_count"] == 2
    assert manifest["protected_source"]["candidate_count"] == 2


def test_export_selector_resampled_map_dry_run_does_not_write_map(tmp_path):
    source = tmp_path / "source"
    pc_dir = source / "point_cloud" / "iteration_30000"
    detector = source / "detector"
    pc_dir.mkdir(parents=True)
    detector.mkdir()
    _write_ply(pc_dir / "point_cloud.ply")
    with (detector / "sampled_idx.pkl").open("wb") as handle:
        pickle.dump(torch.tensor([0, 1, 2], dtype=torch.long), handle)
    checkpoint = tmp_path / "selector.pt"
    torch.save({"export_descriptors": torch.ones((5, 2)), "gate": torch.ones(5)}, checkpoint)
    output = tmp_path / "dry_run_map"
    args = build_argparser().parse_args(
        [
            "--source_map",
            str(source),
            "--checkpoint_path",
            str(checkpoint),
            "--output_map",
            str(output),
            "--dry_run",
        ]
    )

    assert main(args) == 0
    assert not output.exists()


def test_write_resampled_detector_payload_does_not_mutate_symlink_targets(tmp_path):
    from loc_gs.stdloc_native.selector_resampling import write_resampled_detector_payload

    source = tmp_path / "source_detector"
    output = tmp_path / "output_detector"
    source.mkdir()
    output.mkdir()
    with (source / "sampled_idx.pkl").open("wb") as handle:
        pickle.dump(torch.tensor([0, 1, 2], dtype=torch.long), handle)
    with (source / "sampled_scores.pkl").open("wb") as handle:
        pickle.dump({"sampled_scores": torch.tensor([0.1, 0.2, 0.3])}, handle)
    (output / "sampled_idx.pkl").symlink_to(source / "sampled_idx.pkl")
    (output / "sampled_scores.pkl").symlink_to(source / "sampled_scores.pkl")

    write_resampled_detector_payload(
        output,
        {
            "sampled_idx": torch.tensor([2, 1], dtype=torch.long),
            "sampled_scores": torch.tensor([0.9, 0.8], dtype=torch.float32),
            "score_avg": torch.tensor([0.1, 0.8, 0.9], dtype=torch.float32),
            "selector": torch.tensor([0.2, 0.3, 0.4], dtype=torch.float32),
        },
    )

    with (source / "sampled_idx.pkl").open("rb") as handle:
        source_idx = pickle.load(handle)
    with (output / "sampled_idx.pkl").open("rb") as handle:
        output_idx = pickle.load(handle)
    assert not (output / "sampled_idx.pkl").is_symlink()
    assert not (output / "sampled_scores.pkl").is_symlink()
    assert source_idx.tolist() == [0, 1, 2]
    assert output_idx.tolist() == [2, 1]
