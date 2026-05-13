import torch

from loc_gs.localization.scene_matcher import load_scene_matcher
from loc_gs.scripts.train_scene_matcher import build_argparser, load_pair_tensors, main


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
