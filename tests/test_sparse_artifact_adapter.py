from pathlib import Path

import pytest
import torch

from loc_gs.sparse.artifact_adapter import (
    load_listwise_candidate_artifact,
    write_candidate_batches_jsonl,
)


def _write_pair_cache(path: Path, *, split_name: str = "train") -> Path:
    payload = {
        "metadata": {
            "format": "listwise",
            "scene": "GreatCourt",
            "source_split_name": split_name,
            "feedback_bank_split_name": "selfmap_train_rendered",
            "topk": 3,
            "split_audit": {
                "audit_status": "passed",
                "checks": {
                    "image_id_disjointness": {"status": "passed", "overlap": []},
                    "feedback_bank_split": {"status": "passed", "split_name": "selfmap_train_rendered"},
                },
            },
        },
        "query_yx": torch.tensor([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=torch.float32),
        "query_desc": torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
            dtype=torch.float32,
        ),
        "landmark_desc": torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
                [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]],
                [[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]],
            ],
            dtype=torch.float32,
        ),
        "landmark_id": torch.tensor([[101, 102, 103], [201, 202, 203], [301, 302, 303]], dtype=torch.int64),
        "cosine": torch.tensor([[0.9, 0.8, 0.7], [0.6, 0.5, 0.4], [0.3, 0.2, 0.1]], dtype=torch.float32),
        "margin": torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
        "query_score": torch.tensor([0.95, 0.85, 0.75], dtype=torch.float32),
        "landmark_prior": torch.tensor(
            [[0.1, 0.9, 0.2], [0.3, 0.4, 0.5], [0.8, 0.2, 0.1]],
            dtype=torch.float32,
        ),
        "label": torch.tensor([1, 3, 0], dtype=torch.int64),
        "candidate_mask": torch.tensor(
            [[True, True, True], [True, False, True], [True, True, False]],
            dtype=torch.bool,
        ),
        "dense_consistent": torch.tensor(
            [[False, True, False], [False, False, False], [True, False, False]],
            dtype=torch.bool,
        ),
        "sparse_inlier": torch.tensor(
            [[False, True, False], [False, False, False], [True, False, False]],
            dtype=torch.bool,
        ),
        "reprojection_error": torch.tensor(
            [[15.0, 1.0, 20.0], [18.0, 17.0, 16.0], [0.5, 9.0, 12.0]],
            dtype=torch.float32,
        ),
        "solver_weight": torch.tensor(
            [[0.1, 2.0, 0.1], [0.1, 0.1, 0.1], [3.0, 0.1, 0.1]],
            dtype=torch.float32,
        ),
        "query_id": ["img_a.png::kp_0001", "img_a.png::kp_0002", "img_b.png::kp_0001"],
        "image_id": ["img_a.png", "img_a.png", "img_b.png"],
        "keypoint_id": ["kp_0001", "kp_0002", "kp_0001"],
        "source_phase": ["train", "train", "train"],
    }
    torch.save(payload, path)
    return path


def test_listwise_artifact_adapter_groups_rows_into_internal_candidate_batches(tmp_path: Path):
    path = _write_pair_cache(tmp_path / "pairs.pt")

    artifact = load_listwise_candidate_artifact(path)

    assert artifact.scene == "GreatCourt"
    assert artifact.split_name == "train"
    assert artifact.keypoint_count == 3
    assert artifact.topk == 3
    assert len(artifact.batches) == 2

    first = artifact.batches[0]
    assert first.query_id == "img_a.png"
    assert first.keypoint_xy == [[20.0, 10.0], [40.0, 30.0]]
    assert first.candidate_landmark_ids == [[101, 102, 103], [201, 202, 203]]
    assert first.teacher_labels == [1, None]
    assert first.candidate_geometric_correct == [[False, True, False], [False, False, False]]
    assert first.candidate_valid_mask == [[True, True, True], [True, False, True]]
    assert first.candidate_dense_consistent == [[False, True, False], [False, False, False]]
    assert first.candidate_sparse_inlier == [[False, True, False], [False, False, False]]
    assert first.candidate_reprojection_error_px == [[15.0, 1.0, 20.0], [18.0, 17.0, 16.0]]
    assert first.candidate_solver_weight is not None
    assert first.candidate_solver_weight[0] == pytest.approx([0.1, 2.0, 0.1])
    assert first.candidate_solver_weight[1] == pytest.approx([0.1, 0.1, 0.1])
    assert first.candidate_margin[0] == pytest.approx([0.1, 0.1, 0.1])
    assert first.candidate_margin[1] == pytest.approx([0.2, 0.2, 0.2])
    assert first.candidate_query_score[0] == pytest.approx([0.95, 0.95, 0.95])
    assert first.candidate_query_score[1] == pytest.approx([0.85, 0.85, 0.85])
    assert first.candidate_landmark_prior[0] == pytest.approx([0.1, 0.9, 0.2])
    assert first.candidate_landmark_prior[1] == pytest.approx([0.3, 0.4, 0.5])
    assert first.query_descriptors == [[1.0, 0.0], [0.0, 1.0]]
    assert first.candidate_landmark_descriptors is not None
    assert first.candidate_landmark_descriptors[0] == [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]

    summary = artifact.summarize_candidate_availability()
    assert summary["keypoint_count"] == 3
    assert summary["top1_correct"] == 1
    assert summary["topk_available"] == 2
    assert summary["oracle_gap"] == 1


def test_listwise_artifact_adapter_rejects_test_split(tmp_path: Path):
    path = _write_pair_cache(tmp_path / "pairs.pt", split_name="test")

    with pytest.raises(ValueError, match="test split"):
        load_listwise_candidate_artifact(path)


def test_candidate_batches_jsonl_exports_internal_input_preview(tmp_path: Path):
    path = _write_pair_cache(tmp_path / "pairs.pt")
    artifact = load_listwise_candidate_artifact(path)
    output = tmp_path / "candidate_batches.jsonl"

    count = write_candidate_batches_jsonl(artifact.batches, output, max_batches=1)

    assert count == 1
    text = output.read_text(encoding="utf-8")
    assert '"query_id": "img_a.png"' in text
    assert '"candidate_landmark_ids": [[101, 102, 103], [201, 202, 203]]' in text
    assert '"candidate_landmark_prior": [[0.10000000149011612, 0.8999999761581421, 0.20000000298023224]' in text
