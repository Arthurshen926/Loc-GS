import torch
import torch.nn.functional as F
import numpy as np
from plyfile import PlyData, PlyElement

from loc_gs.scripts.train_unified_lff import (
    build_argparser,
    load_unified_lff_training_tensors,
    main,
)


def _write_unified_cache(path, *, base: torch.Tensor) -> None:
    query_desc = F.normalize(
        torch.stack(
            [
                base[0] + 0.05 * torch.ones_like(base[0]),
                base[2] + 0.05 * torch.ones_like(base[2]),
                torch.randn_like(base[0]),
                base[1] + 0.05 * torch.ones_like(base[1]),
            ],
            dim=0,
        ),
        p=2,
        dim=-1,
    )
    candidate_landmark_ids = torch.tensor(
        [
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [1, 4, 0],
        ],
        dtype=torch.long,
    )
    payload = {
        "query_desc": query_desc,
        "candidate_landmark_ids": candidate_landmark_ids,
        "candidate_cosine": torch.rand(4, 3),
        "candidate_mask": torch.ones(4, 3, dtype=torch.bool),
        "listwise_label": torch.tensor([0, 1, 3, 0], dtype=torch.long),
        "pair_label": torch.tensor(
            [
                [True, False, False],
                [False, True, False],
                [False, False, False],
                [True, False, False],
            ]
        ),
        "gaussian_advantage_target": torch.tensor([0.9, 0.8, 0.2, 0.1, 0.5], dtype=torch.float32),
        "metadata": {"format": "selfmap_episode_v1"},
    }
    torch.save(payload, path)


def _write_descriptor_ply(path) -> None:
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("loc_0", "f4"), ("loc_1", "f4")]
    data = np.empty(2, dtype=dtype)
    data["x"] = [0.0, 1.0]
    data["y"] = [0.0, 1.0]
    data["z"] = [1.0, 1.0]
    data["loc_0"] = [3.0, 0.0]
    data["loc_1"] = [4.0, 2.0]
    PlyData([PlyElement.describe(data, "vertex")], text=True).write(path)


def test_train_unified_lff_parser_defaults_are_single_path_recipe():
    args = build_argparser().parse_args(
        [
            "--base_descriptor_path",
            "base.pt",
            "--episode_cache",
            "cache.pt",
            "--output_path",
            "unified.pt",
        ]
    )

    assert args.alpha_max == 0.05
    assert args.lambda_trust > 0.0
    assert args.lambda_gate > 0.0
    assert args.lambda_rank > 0.0
    assert args.batch_size >= 8192


def test_load_unified_lff_training_tensors_uses_landmark_ids_not_posthoc_descriptors(tmp_path):
    base = F.normalize(torch.randn(5, 4), p=2, dim=-1)
    base_path = tmp_path / "base.pt"
    cache_path = tmp_path / "cache.pt"
    torch.save({"descriptors": base}, base_path)
    _write_unified_cache(cache_path, base=base)

    tensors = load_unified_lff_training_tensors(base_path, [cache_path])

    assert tensors["base_descriptors"].shape == (5, 4)
    assert tensors["query_desc"].shape == (4, 4)
    assert tensors["candidate_landmark_ids"].shape == (4, 3)
    assert tensors["listwise_label"].tolist() == [0, 1, 3, 0]
    assert tensors["gaussian_advantage_target"].shape == (5,)


def test_load_unified_lff_training_tensors_accepts_existing_listwise_calibration_cache(tmp_path):
    base = F.normalize(torch.randn(5, 4), p=2, dim=-1)
    base_path = tmp_path / "base.pt"
    cache_path = tmp_path / "calibration_listwise.pt"
    torch.save({"descriptors": base}, base_path)
    torch.save(
        {
            "query_desc": F.normalize(torch.randn(3, 4), p=2, dim=-1),
            "landmark_id": torch.tensor([[0, 1, 2], [1, 2, 3], [4, 3, 2]], dtype=torch.long),
            "cosine": torch.tensor([[0.9, 0.7, 0.1], [0.8, 0.6, 0.4], [0.95, 0.3, 0.2]]),
            "candidate_mask": torch.ones(3, 3, dtype=torch.bool),
            "label": torch.tensor([0, 3, 0], dtype=torch.long),
            "metadata": {"format": "listwise"},
        },
        cache_path,
    )

    tensors = load_unified_lff_training_tensors(base_path, [cache_path])

    assert tensors["candidate_landmark_ids"].tolist() == [[0, 1, 2], [1, 2, 3], [4, 3, 2]]
    assert tensors["listwise_label"].tolist() == [0, 3, 0]
    assert tensors["pair_label"].tolist() == [[True, False, False], [False, False, False], [True, False, False]]
    assert tensors["gaussian_advantage_target"][1] < tensors["gaussian_advantage_target"][0]


def test_load_unified_lff_training_tensors_can_limit_gate_penalty_to_hard_false_positives(tmp_path):
    base = F.normalize(torch.randn(4, 4), p=2, dim=-1)
    base_path = tmp_path / "base.pt"
    cache_path = tmp_path / "calibration_listwise.pt"
    torch.save({"descriptors": base}, base_path)
    torch.save(
        {
            "query_desc": F.normalize(torch.randn(2, 4), p=2, dim=-1),
            "landmark_id": torch.tensor([[0, 1], [2, 1]], dtype=torch.long),
            "cosine": torch.tensor([[0.95, 0.20], [0.40, 0.30]], dtype=torch.float32),
            "candidate_mask": torch.ones(2, 2, dtype=torch.bool),
            "label": torch.tensor([0, 2], dtype=torch.long),
            "metadata": {"format": "listwise"},
        },
        cache_path,
    )

    all_negatives = load_unified_lff_training_tensors(base_path, [cache_path])
    hard_negatives = load_unified_lff_training_tensors(
        base_path,
        [cache_path],
        false_positive_score_threshold=0.5,
    )

    assert hard_negatives["gaussian_advantage_target"][1] > all_negatives["gaussian_advantage_target"][1]
    assert torch.allclose(
        hard_negatives["gaussian_advantage_target"][0],
        all_negatives["gaussian_advantage_target"][0],
    )


def test_load_unified_lff_training_tensors_can_use_native_ply_descriptor_bank(tmp_path):
    base_path = tmp_path / "native.ply"
    cache_path = tmp_path / "cache.pt"
    _write_descriptor_ply(base_path)
    base = F.normalize(torch.tensor([[3.0, 4.0], [0.0, 2.0]], dtype=torch.float32), p=2, dim=-1)
    torch.save(
        {
            "query_desc": base.clone(),
            "candidate_landmark_ids": torch.tensor([[0], [1]], dtype=torch.long),
            "listwise_label": torch.tensor([0, 0], dtype=torch.long),
        },
        cache_path,
    )

    tensors = load_unified_lff_training_tensors(base_path, [cache_path])

    assert torch.allclose(tensors["base_descriptors"], base)


def test_load_unified_lff_training_tensors_can_use_cache_embedded_landmark_bank(tmp_path):
    base = F.normalize(torch.randn(4, 3), p=2, dim=-1)
    cache_path = tmp_path / "cache_with_bank.pt"
    torch.save(
        {
            "base_landmark_desc": base,
            "base_gaussian_id": torch.tensor([10, 20, 30, 40], dtype=torch.long),
            "query_desc": base[:2].clone(),
            "landmark_id": torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
            "label": torch.tensor([0, 2], dtype=torch.long),
            "candidate_mask": torch.ones(2, 2, dtype=torch.bool),
        },
        cache_path,
    )

    tensors = load_unified_lff_training_tensors("", [cache_path])

    assert torch.allclose(tensors["base_descriptors"], base)
    assert tensors["base_gaussian_id"].tolist() == [10, 20, 30, 40]


def test_train_unified_lff_writes_export_aligned_descriptor_checkpoint(tmp_path):
    torch.manual_seed(7)
    base = F.normalize(torch.randn(5, 4), p=2, dim=-1)
    base_path = tmp_path / "base.pt"
    cache_path = tmp_path / "cache.pt"
    output_path = tmp_path / "unified_lff.pt"
    torch.save(base, base_path)
    _write_unified_cache(cache_path, base=base)
    args = build_argparser().parse_args(
        [
            "--base_descriptor_path",
            str(base_path),
            "--episode_cache",
            str(cache_path),
            "--output_path",
            str(output_path),
            "--epochs",
            "2",
            "--batch_size",
            "2",
            "--lr",
            "0.05",
            "--lambda_trust",
            "0.1",
            "--lambda_gate",
            "0.2",
            "--lambda_rank",
            "0.1",
            "--device",
            "cpu",
        ]
    )

    main(args)
    checkpoint = torch.load(output_path, map_location="cpu")

    assert checkpoint["config"]["model_type"] == "unified_lff_descriptor"
    assert checkpoint["config"]["alpha_max"] == 0.05
    assert "residual" in checkpoint["state_dict"]
    assert "gate_logit" in checkpoint["state_dict"]
    assert "selector_logit" in checkpoint["state_dict"]
    assert checkpoint["export_descriptors"].shape == base.shape
    assert checkpoint["gate"].shape == (base.shape[0],)
    assert checkpoint["residual_gate"].shape == (base.shape[0],)
    assert checkpoint["metadata"]["selector_gate_decoupled"] is True
    assert checkpoint["metadata"]["single_path_deployment"] is True
    assert len(checkpoint["metadata"]["history"]) == 2
