import json
import pickle
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F


def _write_source_log(source, *, sampled_idx=None):
    source.mkdir()
    source_features = F.normalize(torch.eye(2, dtype=torch.float32), p=2, dim=-1)
    sampled = torch.tensor([5, 9], dtype=torch.long) if sampled_idx is None else sampled_idx
    with (source / "keypoints_features.pkl").open("wb") as handle:
        pickle.dump(source_features, handle)
    with (source / "keypoints_sampled_idx.pkl").open("wb") as handle:
        pickle.dump(sampled, handle)
    (source / "config.yaml").write_text("sample: {}\n", encoding="utf-8")
    return source_features, sampled


def test_build_ulfloc_solver_feedback_feature_log_copies_log_and_writes_audits(tmp_path):
    source = tmp_path / "source_log"
    source_features, sampled_idx = _write_source_log(source)
    observation_cache = tmp_path / "observations.pt"
    torch.save(
        {
            "schema_version": "synthetic_solver_feedback_feature_observation_cache_v1",
            "split_name": "selfmap_train",
            "sampled_idx": sampled_idx,
            "observation_descriptor_source": "ulf_original_multiview_feature_observation",
            "observations": {
                5: [
                    {
                        "descriptor": F.normalize(torch.tensor([0.8, 0.6], dtype=torch.float32), p=2, dim=0),
                        "positive_weight": 1.0,
                        "negative_weight": 0.0,
                    }
                ],
                9: [
                    {
                        "descriptor": torch.tensor([1.0, 0.0], dtype=torch.float32),
                        "positive_weight": 0.0,
                        "negative_weight": 2.0,
                    }
                ],
            },
        },
        observation_cache,
    )
    impact = tmp_path / "impact_attribution.pt"
    torch.save({"split_name": "selfmap_train", "metadata": {"record_count": 2}}, impact)
    output = tmp_path / "fused_log"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_ulfloc_solver_feedback_feature_log",
            "--input_log_dir",
            str(source),
            "--observation_cache",
            str(observation_cache),
            "--impact_attribution",
            str(impact),
            "--output_log_dir",
            str(output),
            "--scene",
            "ShopFacade",
            "--trust_alpha",
            "0.5",
            "--min_native_cosine",
            "0.0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    with (output / "keypoints_features.pkl").open("rb") as handle:
        saved_features = torch.as_tensor(pickle.load(handle)).float()
    with (output / "keypoints_sampled_idx.pkl").open("rb") as handle:
        saved_idx = torch.as_tensor(pickle.load(handle)).long()
    manifest = json.loads((output / "solver_feedback_feature_log_manifest.json").read_text())
    metrics = json.loads((output / "metrics_summary.json").read_text())
    split_audit = json.loads((output / "split_audit.json").read_text())

    assert torch.equal(saved_idx, sampled_idx)
    assert (output / "config.yaml").read_text(encoding="utf-8") == "sample: {}\n"
    assert not torch.allclose(saved_features[0], source_features[0])
    assert torch.allclose(saved_features[1], source_features[1])
    assert manifest["same_sampled_idx"] is True
    assert metrics["negative_view_excluded_count"] == 1
    assert split_audit["test_split_used"] is False
    assert split_audit["official_test_used"] is False
    assert (output / "manifest.json").exists()
    assert (output / "command.txt").exists()
    assert (output / "git_status.txt").exists()


def test_build_ulfloc_solver_feedback_feature_log_rejects_test_split_impact(tmp_path):
    source = tmp_path / "source_log"
    _source_features, sampled_idx = _write_source_log(source)
    observation_cache = tmp_path / "observations.pt"
    torch.save(
        {
            "schema_version": "synthetic_solver_feedback_feature_observation_cache_v1",
            "split_name": "selfmap_train",
            "sampled_idx": sampled_idx,
            "observations": {},
        },
        observation_cache,
    )
    impact = tmp_path / "impact_attribution.pt"
    torch.save({"split_name": "test"}, impact)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_ulfloc_solver_feedback_feature_log",
            "--input_log_dir",
            str(source),
            "--observation_cache",
            str(observation_cache),
            "--impact_attribution",
            str(impact),
            "--output_log_dir",
            str(tmp_path / "fused_log"),
            "--scene",
            "ShopFacade",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "test split impact" in result.stderr


def test_build_ulfloc_solver_feedback_feature_log_supports_contrastive_mode(tmp_path):
    source = tmp_path / "source_log"
    source_features, sampled_idx = _write_source_log(source)
    observation_cache = tmp_path / "observations.pt"
    torch.save(
        {
            "schema_version": "synthetic_solver_feedback_feature_observation_cache_v1",
            "split_name": "selfmap_train",
            "sampled_idx": sampled_idx,
            "observation_descriptor_source": "ulf_original_multiview_feature_observation",
            "observations": {
                9: [
                    {
                        "descriptor": torch.tensor([0.0, 1.0], dtype=torch.float32),
                        "positive_weight": 0.0,
                        "negative_weight": 2.0,
                    },
                    {
                        "descriptor": torch.tensor([1.0, 0.0], dtype=torch.float32),
                        "positive_weight": 2.0,
                        "negative_weight": 0.0,
                    },
                ],
            },
        },
        observation_cache,
    )
    impact = tmp_path / "impact_attribution.pt"
    torch.save({"split_name": "selfmap_train", "metadata": {"record_count": 2}}, impact)
    output = tmp_path / "contrastive_log"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_ulfloc_solver_feedback_feature_log",
            "--input_log_dir",
            str(source),
            "--observation_cache",
            str(observation_cache),
            "--impact_attribution",
            str(impact),
            "--output_log_dir",
            str(output),
            "--scene",
            "ShopFacade",
            "--fusion_mode",
            "contrastive",
            "--min_native_cosine",
            "0.0",
            "--contrastive_steps",
            "5",
            "--contrastive_lr",
            "0.2",
            "--negative_margin",
            "0.0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output / "metrics_summary.json").read_text())
    with (output / "keypoints_features.pkl").open("rb") as handle:
        saved_features = torch.as_tensor(pickle.load(handle)).float()

    assert metrics["descriptor_mode"] == "solver_feedback_contrastive_fusion_v1"
    assert metrics["fusion_mode"] == "contrastive"
    assert metrics["observation_descriptor_source"] == "ulf_original_multiview_feature_observation"
    assert metrics["negative_view_excluded_count"] == 1
    assert not torch.allclose(saved_features[1], source_features[1])


def test_contrastive_feature_log_prefers_solver_impact_over_visibility_weights(tmp_path):
    source = tmp_path / "source_log"
    source_features, sampled_idx = _write_source_log(source)
    observation_cache = tmp_path / "observations.pt"
    torch.save(
        {
            "schema_version": "synthetic_solver_feedback_feature_observation_cache_v1",
            "split_name": "selfmap_train",
            "sampled_idx": sampled_idx,
            "observation_id_space": "gaussian_id",
            "observation_descriptor_source": "ulf_original_multiview_feature_observation",
            "observations": {
                9: [
                    {
                        "source_view_id": "view_good",
                        "descriptor": torch.tensor([1.0, 0.0], dtype=torch.float32),
                        "positive_weight": 1.0,  # visibility, not solver feedback
                        "negative_weight": 0.0,
                    },
                    {
                        "source_view_id": "view_bad",
                        "descriptor": torch.tensor([0.0, 1.0], dtype=torch.float32),
                        "positive_weight": 1.0,  # would be a false positive if impact is ignored
                        "negative_weight": 0.0,
                    },
                ],
            },
        },
        observation_cache,
    )
    impact = tmp_path / "impact_attribution.pt"
    torch.save(
        {
            "split_name": "selfmap_train",
            "view_positive": {("9", "view_good"): {"weight": 2.0}},
            "view_negative": {("9", "view_bad"): {"weight": 3.0}},
            "metadata": {"record_count": 2},
        },
        impact,
    )
    output = tmp_path / "contrastive_log"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_ulfloc_solver_feedback_feature_log",
            "--input_log_dir",
            str(source),
            "--observation_cache",
            str(observation_cache),
            "--impact_attribution",
            str(impact),
            "--output_log_dir",
            str(output),
            "--scene",
            "ShopFacade",
            "--fusion_mode",
            "contrastive",
            "--min_native_cosine",
            "0.0",
            "--contrastive_steps",
            "5",
            "--contrastive_lr",
            "0.1",
            "--negative_margin",
            "0.0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output / "metrics_summary.json").read_text())
    with (output / "keypoints_features.pkl").open("rb") as handle:
        saved_features = torch.as_tensor(pickle.load(handle)).float()

    assert metrics["positive_view_candidate_count"] == 1
    assert metrics["negative_view_candidate_count"] == 1
    assert metrics["negative_view_selected_count"] == 1
    assert not torch.allclose(saved_features[1], source_features[1])


def test_contrastive_feature_log_can_use_weak_visibility_positive_scale(tmp_path):
    source = tmp_path / "source_log"
    source_features, sampled_idx = _write_source_log(source)
    observation_cache = tmp_path / "observations.pt"
    torch.save(
        {
            "schema_version": "synthetic_solver_feedback_feature_observation_cache_v1",
            "split_name": "selfmap_train",
            "sampled_idx": sampled_idx,
            "observation_id_space": "gaussian_id",
            "observation_descriptor_source": "ulf_original_multiview_feature_observation",
            "observations": {
                9: [
                    {
                        "source_view_id": "view_good",
                        "descriptor": torch.tensor([1.0, 0.0], dtype=torch.float32),
                        "positive_weight": 1.0,
                        "negative_weight": 0.0,
                    },
                    {
                        "source_view_id": "view_neutral_visible",
                        "descriptor": torch.tensor([0.7, 0.3], dtype=torch.float32),
                        "positive_weight": 1.0,
                        "negative_weight": 0.0,
                    },
                    {
                        "source_view_id": "view_bad",
                        "descriptor": torch.tensor([0.0, 1.0], dtype=torch.float32),
                        "positive_weight": 1.0,
                        "negative_weight": 0.0,
                    },
                ],
            },
        },
        observation_cache,
    )
    impact = tmp_path / "impact_attribution.pt"
    torch.save(
        {
            "split_name": "selfmap_train",
            "view_positive": {("9", "view_good"): {"weight": 2.0}},
            "view_negative": {("9", "view_bad"): {"weight": 3.0}},
            "metadata": {"record_count": 3},
        },
        impact,
    )
    output = tmp_path / "contrastive_log"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_ulfloc_solver_feedback_feature_log",
            "--input_log_dir",
            str(source),
            "--observation_cache",
            str(observation_cache),
            "--impact_attribution",
            str(impact),
            "--output_log_dir",
            str(output),
            "--scene",
            "ShopFacade",
            "--fusion_mode",
            "contrastive",
            "--min_native_cosine",
            "0.0",
            "--observation_positive_weight_scale",
            "0.25",
            "--contrastive_steps",
            "5",
            "--contrastive_lr",
            "0.1",
            "--negative_margin",
            "0.0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output / "metrics_summary.json").read_text())
    with (output / "keypoints_features.pkl").open("rb") as handle:
        saved_features = torch.as_tensor(pickle.load(handle)).float()

    assert metrics["observation_positive_weight_scale"] == pytest.approx(0.25)
    assert metrics["positive_view_candidate_count"] == 2
    assert metrics["negative_view_candidate_count"] == 1
    assert not torch.allclose(saved_features[1], source_features[1])


def test_build_ulfloc_solver_feedback_feature_log_filters_views_with_active_plan(tmp_path):
    source = tmp_path / "source_log"
    source_features, sampled_idx = _write_source_log(source)
    observation_cache = tmp_path / "observations.pt"
    torch.save(
        {
            "schema_version": "synthetic_solver_feedback_feature_observation_cache_v1",
            "split_name": "selfmap_train",
            "sampled_idx": sampled_idx,
            "observations": {
                5: [
                    {
                        "source_view_id": "view_keep",
                        "descriptor": torch.tensor([1.0, 0.0], dtype=torch.float32),
                        "positive_weight": 1.0,
                        "negative_weight": 0.0,
                    },
                    {
                        "source_view_id": "view_drop",
                        "descriptor": torch.tensor([0.0, 1.0], dtype=torch.float32),
                        "positive_weight": 1.0,
                        "negative_weight": 0.0,
                    },
                ],
            },
        },
        observation_cache,
    )
    active_plan = tmp_path / "landmark_fusion_plan.json"
    active_plan.write_text(
        json.dumps(
            {
                "schema_version": "active_fusion_plan_test_v1",
                "split_name": "selfmap_train",
                "landmark_fusion_plan": {
                    "5": {"selected_view_ids": ["view_keep"]},
                },
            }
        ),
        encoding="utf-8",
    )
    impact = tmp_path / "impact_attribution.pt"
    torch.save({"split_name": "selfmap_train", "metadata": {"record_count": 2}}, impact)
    output = tmp_path / "active_pruned_log"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_ulfloc_solver_feedback_feature_log",
            "--input_log_dir",
            str(source),
            "--observation_cache",
            str(observation_cache),
            "--impact_attribution",
            str(impact),
            "--active_fusion_plan",
            str(active_plan),
            "--output_log_dir",
            str(output),
            "--scene",
            "ShopFacade",
            "--trust_alpha",
            "1.0",
            "--min_native_cosine",
            "0.0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output / "metrics_summary.json").read_text())
    with (output / "keypoints_features.pkl").open("rb") as handle:
        saved_features = torch.as_tensor(pickle.load(handle)).float()

    assert torch.allclose(saved_features[0], source_features[0])
    assert metrics["active_fusion_plan_enabled"] is True
    assert metrics["active_fusion_plan_kept_observation_count"] == 1
    assert metrics["active_fusion_plan_filtered_observation_count"] == 1
