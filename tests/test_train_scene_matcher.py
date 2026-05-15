import torch

from loc_gs.localization.scene_matcher import load_scene_matcher
from loc_gs.scripts.train_scene_matcher import (
    build_argparser,
    listwise_reprojection_verifier_loss,
    listwise_soft_reprojection_loss,
    load_listwise_tensors,
    load_pair_tensors,
    main,
)


def _write_pairs(path):
    count = 16
    dim = 4
    payload = {
        "query_desc": torch.randn(count, dim),
        "landmark_desc": torch.randn(count, dim),
        "cosine": torch.randn(count),
        "margin": torch.rand(count),
        "landmark_prior": torch.rand(count),
        "label": torch.tensor([0, 1] * (count // 2), dtype=torch.float32),
    }
    torch.save(payload, path)


def _write_listwise_pairs(path, *, include_reprojection_error: bool = False):
    groups = 8
    topk = 3
    dim = 4
    payload = {
        "query_desc": torch.randn(groups, dim),
        "landmark_desc": torch.randn(groups, topk, dim),
        "cosine": torch.randn(groups, topk),
        "margin": torch.rand(groups),
        "query_score": torch.rand(groups),
        "landmark_prior": torch.rand(groups, topk),
        "candidate_mask": torch.ones(groups, topk, dtype=torch.bool),
        "label": torch.tensor([0, 1, 2, 3, 0, 1, 3, 2], dtype=torch.long),
    }
    if include_reprojection_error:
        payload["reprojection_error"] = torch.tensor(
            [
                [0.25, 4.0, 12.0],
                [6.0, 0.5, 8.0],
                [12.0, 6.0, 0.2],
                [14.0, 10.0, 8.0],
                [0.3, 5.0, 9.0],
                [8.0, 0.4, 7.0],
                [9.0, 11.0, 13.0],
                [7.0, 9.0, 0.6],
            ],
            dtype=torch.float32,
        )
    torch.save(payload, path)


def test_train_scene_matcher_loads_pair_tensors(tmp_path):
    path = tmp_path / "pairs.pt"
    _write_pairs(path)

    tensors = load_pair_tensors([path])

    assert tensors["query_desc"].shape == (16, 4)
    assert torch.allclose(tensors["query_score"], torch.ones(16))
    assert tensors["label"].sum().item() == 8


def test_train_scene_matcher_writes_reloadable_checkpoint(tmp_path):
    pair_path = tmp_path / "pairs.pt"
    output_path = tmp_path / "best.pt"
    _write_pairs(pair_path)
    args = build_argparser().parse_args(
        [
            "--pair_files",
            str(pair_path),
            "--output_path",
            str(output_path),
            "--epochs",
            "1",
            "--batch_size",
            "8",
            "--hidden_dim",
            "8",
            "--num_layers",
            "2",
            "--dropout",
            "0.1",
            "--include_raw_descriptors",
            "--balanced_batches",
            "--device",
            "cpu",
        ]
    )

    main(args)
    matcher = load_scene_matcher(output_path)

    assert output_path.exists()
    assert matcher.config["descriptor_dim"] == 4
    assert matcher.config["scalar_dim"] == 5
    assert matcher.config["include_raw_descriptors"] is True


def test_train_scene_matcher_writes_reloadable_listwise_checkpoint(tmp_path):
    pair_path = tmp_path / "listwise_pairs.pt"
    output_path = tmp_path / "best_listwise.pt"
    _write_listwise_pairs(pair_path)
    args = build_argparser().parse_args(
        [
            "--pair_files",
            str(pair_path),
            "--output_path",
            str(output_path),
            "--listwise",
            "--epochs",
            "1",
            "--batch_size",
            "4",
            "--hidden_dim",
            "8",
            "--num_layers",
            "2",
            "--device",
            "cpu",
        ]
    )

    main(args)
    matcher = load_scene_matcher(output_path)
    checkpoint = torch.load(output_path, map_location="cpu")

    assert output_path.exists()
    assert matcher.config["model_type"] == "listwise"
    assert matcher.config["descriptor_dim"] == 4
    assert matcher.config["scalar_dim"] == 5
    assert checkpoint["metadata"]["listwise_loss_balance"] == "binary"
    assert len(checkpoint["metadata"]["class_weight"]) == 4


def test_train_scene_matcher_loads_optional_listwise_reprojection_errors(tmp_path):
    missing_path = tmp_path / "listwise_missing_errors.pt"
    present_path = tmp_path / "listwise_with_errors.pt"
    _write_listwise_pairs(missing_path)
    _write_listwise_pairs(present_path, include_reprojection_error=True)

    missing = load_listwise_tensors([missing_path])
    present = load_listwise_tensors([present_path])

    assert torch.isinf(missing["reprojection_error"]).all()
    assert present["reprojection_error"].shape == (8, 3)
    assert torch.isfinite(present["reprojection_error"]).all()


def test_listwise_reprojection_verifier_loss_rewards_low_error_candidates():
    errors = torch.tensor([[0.1, 12.0]], dtype=torch.float32)
    mask = torch.ones_like(errors, dtype=torch.bool)
    good_logits = torch.tensor([[4.0, -4.0, 0.0]], dtype=torch.float32)
    bad_logits = torch.tensor([[-4.0, 4.0, 0.0]], dtype=torch.float32)

    good_loss = listwise_reprojection_verifier_loss(good_logits, errors, mask, sigma_px=4.0)
    bad_loss = listwise_reprojection_verifier_loss(bad_logits, errors, mask, sigma_px=4.0)

    assert good_loss < bad_loss


def test_listwise_reprojection_verifier_loss_ignores_masked_infinite_logits():
    errors = torch.tensor([[0.1, float("inf")]], dtype=torch.float32)
    mask = torch.tensor([[True, False]])
    logits = torch.tensor([[4.0, float("-inf"), 0.0]], dtype=torch.float32)

    loss = listwise_reprojection_verifier_loss(logits, errors, mask, sigma_px=4.0)

    assert torch.isfinite(loss)


def test_listwise_soft_reprojection_loss_prefers_low_error_candidates_and_dustbin():
    errors = torch.tensor(
        [
            [0.1, 12.0],
            [float("inf"), float("inf")],
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor([[True, True], [False, False]])
    good_logits = torch.tensor([[4.0, -4.0, 0.0], [-4.0, -4.0, 4.0]], dtype=torch.float32)
    bad_logits = torch.tensor([[-4.0, 4.0, 0.0], [4.0, 4.0, -4.0]], dtype=torch.float32)

    good_loss = listwise_soft_reprojection_loss(good_logits, errors, mask, sigma_px=4.0)
    bad_loss = listwise_soft_reprojection_loss(bad_logits, errors, mask, sigma_px=4.0)

    assert good_loss < bad_loss
    assert torch.isfinite(good_loss)


def test_listwise_soft_reprojection_loss_ignores_masked_infinite_logits():
    errors = torch.tensor([[0.1, float("inf")]], dtype=torch.float32)
    mask = torch.tensor([[True, False]])
    logits = torch.tensor([[4.0, float("-inf"), 0.0]], dtype=torch.float32)

    loss = listwise_soft_reprojection_loss(logits, errors, mask, sigma_px=4.0)

    assert torch.isfinite(loss)


def test_train_scene_matcher_listwise_can_use_reprojection_verifier_loss(tmp_path):
    pair_path = tmp_path / "listwise_pairs_with_errors.pt"
    output_path = tmp_path / "best_listwise_verifier.pt"
    _write_listwise_pairs(pair_path, include_reprojection_error=True)
    args = build_argparser().parse_args(
        [
            "--pair_files",
            str(pair_path),
            "--output_path",
            str(output_path),
            "--listwise",
            "--listwise_verifier_loss_weight",
            "0.25",
            "--listwise_verifier_sigma_px",
            "4.0",
            "--epochs",
            "1",
            "--batch_size",
            "4",
            "--hidden_dim",
            "8",
            "--num_layers",
            "2",
            "--device",
            "cpu",
        ]
    )

    main(args)
    checkpoint = torch.load(output_path, map_location="cpu")

    assert checkpoint["metadata"]["listwise_verifier_loss_weight"] == 0.25
    assert checkpoint["metadata"]["listwise_verifier_sigma_px"] == 4.0
    assert checkpoint["metadata"]["verifier_finite_ratio"] == 1.0


def test_train_scene_matcher_listwise_can_use_soft_reprojection_loss(tmp_path):
    pair_path = tmp_path / "listwise_pairs_with_soft_errors.pt"
    output_path = tmp_path / "best_listwise_soft.pt"
    _write_listwise_pairs(pair_path, include_reprojection_error=True)
    args = build_argparser().parse_args(
        [
            "--pair_files",
            str(pair_path),
            "--output_path",
            str(output_path),
            "--listwise",
            "--listwise_ce_loss_weight",
            "0.25",
            "--listwise_soft_reprojection_loss_weight",
            "1.5",
            "--listwise_soft_reprojection_sigma_px",
            "6.0",
            "--epochs",
            "1",
            "--batch_size",
            "4",
            "--hidden_dim",
            "8",
            "--num_layers",
            "2",
            "--device",
            "cpu",
        ]
    )

    main(args)
    checkpoint = torch.load(output_path, map_location="cpu")

    assert checkpoint["metadata"]["listwise_ce_loss_weight"] == 0.25
    assert checkpoint["metadata"]["listwise_soft_reprojection_loss_weight"] == 1.5
    assert checkpoint["metadata"]["listwise_soft_reprojection_sigma_px"] == 6.0


def test_train_scene_matcher_listwise_can_use_rank_gap_features(tmp_path):
    pair_path = tmp_path / "listwise_pairs.pt"
    output_path = tmp_path / "best_listwise_rank_gap.pt"
    _write_listwise_pairs(pair_path)
    args = build_argparser().parse_args(
        [
            "--pair_files",
            str(pair_path),
            "--output_path",
            str(output_path),
            "--listwise",
            "--listwise_extra_features",
            "query_score_rank_gap",
            "--epochs",
            "1",
            "--batch_size",
            "4",
            "--hidden_dim",
            "8",
            "--num_layers",
            "2",
            "--device",
            "cpu",
        ]
    )

    main(args)
    checkpoint = torch.load(output_path, map_location="cpu")
    matcher = load_scene_matcher(output_path)

    assert matcher.config["scalar_dim"] == 7
    assert matcher.config["listwise_extra_features"] == "query_score_rank_gap"
    assert checkpoint["metadata"]["listwise_extra_features"] == "query_score_rank_gap"
