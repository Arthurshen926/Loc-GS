import json
from pathlib import Path

import pytest
import torch

from loc_gs.sparse.correspondences import SparseCandidateBatch
from loc_gs.students.descriptor_fusion import (
    DescriptorFusionConfig,
    DescriptorFusionModel,
    descriptor_fusion_score_rows,
    load_descriptor_fusion,
    train_descriptor_fusion_from_payload,
)


def _payload():
    return {
        "metadata": {"format": "listwise", "scene": "GreatCourt", "source_split_name": "train", "topk": 2},
        "query_desc": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        "landmark_desc": torch.tensor(
            [
                [[0.0, 1.0], [0.0, 1.0]],
                [[0.0, 1.0], [1.0, 0.0]],
            ],
            dtype=torch.float32,
        ),
        "landmark_id": torch.tensor([[10, 20], [10, 30]], dtype=torch.int64),
        "label": torch.tensor([0, 1], dtype=torch.int64),
        "candidate_mask": torch.ones((2, 2), dtype=torch.bool),
        "dense_consistent": torch.tensor([[True, False], [False, True]], dtype=torch.bool),
        "sparse_inlier": torch.tensor([[True, False], [False, True]], dtype=torch.bool),
        "solver_weight": torch.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=torch.float32),
        "label_roles": [["protected_support", "hard_negative"], ["hard_negative", "protected_support"]],
        "query_id": ["img.png::kp0", "img.png::kp1"],
        "image_id": ["img.png", "img.png"],
        "keypoint_id": ["kp0", "kp1"],
        "source_phase": ["train", "train"],
    }


def _batch() -> SparseCandidateBatch:
    return SparseCandidateBatch(
        scene="GreatCourt",
        split_name="train",
        query_id="img.png",
        keypoint_xy=[[1.0, 2.0], [3.0, 4.0]],
        candidate_landmark_ids=[[10, 20], [10, 30]],
        candidate_scores=[[0.1, 0.9], [0.8, 0.7]],
        candidate_valid_mask=[[True, True], [True, True]],
    )


def _descriptor_batch() -> SparseCandidateBatch:
    return SparseCandidateBatch(
        scene="GreatCourt",
        split_name="train",
        query_id="img.png",
        keypoint_xy=[[1.0, 2.0], [3.0, 4.0]],
        query_descriptors=[[1.0, 0.0], [0.0, 1.0]],
        candidate_landmark_ids=[[10, 20], [10, 30]],
        candidate_scores=[[0.1, 0.9], [0.8, 0.7]],
        candidate_valid_mask=[[True, True], [True, True]],
    )


def test_descriptor_fusion_moves_landmarks_toward_solver_useful_query_descriptors():
    model, summary = train_descriptor_fusion_from_payload(
        _payload(),
        DescriptorFusionConfig(trust_region=0.75, hard_negative_penalty=1.0),
    )

    assert summary["student_modules"] == ["descriptor_fusion"]
    assert summary["protected_support_count"] == 2
    assert summary["hard_negative_count"] == 2
    assert model.score_landmark(10) > model.score_landmark(20)
    assert model.score_landmark(30) > model.score_landmark(10)
    assert model.fused_descriptors["10"][0] > model.fused_descriptors["10"][1]
    rows = descriptor_fusion_score_rows(_batch(), model)
    assert rows[0][0] > rows[0][1]


def test_descriptor_fusion_scores_query_against_fused_descriptors():
    model = DescriptorFusionModel(
        fused_descriptors={"10": [1.0, 0.0], "20": [0.0, 1.0], "30": [0.0, 1.0]},
        landmark_scores={},
        negative_scores={},
    )

    rows = descriptor_fusion_score_rows(_descriptor_batch(), model)

    assert rows[0][0] > rows[0][1]
    assert rows[1][1] > rows[1][0]


def test_descriptor_fusion_json_roundtrip(tmp_path: Path):
    model, _summary = train_descriptor_fusion_from_payload(_payload())
    path = tmp_path / "descriptor_fusion.json"
    path.write_text(json.dumps(model.to_json_dict(), sort_keys=True), encoding="utf-8")

    loaded = load_descriptor_fusion(path)

    assert isinstance(loaded, DescriptorFusionModel)
    assert loaded.to_json_dict() == model.to_json_dict()


def test_descriptor_fusion_torch_roundtrip(tmp_path: Path):
    model, _summary = train_descriptor_fusion_from_payload(_payload())
    path = tmp_path / "descriptor_fusion.pt"
    torch.save(model.to_torch_dict(), path)

    loaded = load_descriptor_fusion(path)

    assert isinstance(loaded, DescriptorFusionModel)
    assert loaded.score_scale == model.score_scale
    assert loaded.landmark_scores == pytest.approx(model.landmark_scores)
    assert loaded.negative_scores == pytest.approx(model.negative_scores)
    assert loaded.fused_descriptors["10"] == pytest.approx(model.fused_descriptors["10"])
