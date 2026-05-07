import yaml
import numpy as np
import torch
from plyfile import PlyData, PlyElement

from loc_gs.config import load_config
from loc_gs.models.hybrid_gaussian import HybridFeatureGaussian


def test_config_loads_localization_training_fields(tmp_path):
    config_path = tmp_path / "loc.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "feature_type": "superpoint",
                "localization_loss_weight": 2.0,
                "sp_recon_aux_weight": 0.15,
                "geometry_unfreeze_epoch": 7,
                "train_geometry_xyz": True,
                "lr_geometry_xyz": 1.0e-5,
                "locability_prior_weight": 0.75,
                "sp_use_locability_input": True,
                "max_val_batches_per_epoch": 12,
            }
        ),
        encoding="utf-8",
    )

    cfg = load_config(str(config_path))

    assert cfg.localization_loss_weight == 2.0
    assert cfg.sp_recon_aux_weight == 0.15
    assert cfg.geometry_unfreeze_epoch == 7
    assert cfg.train_geometry_xyz is True
    assert cfg.lr_geometry_xyz == 1.0e-5
    assert cfg.locability_prior_weight == 0.75
    assert cfg.sp_use_locability_input is True
    assert cfg.max_val_batches_per_epoch == 12


def test_hybrid_gaussian_has_checkpointable_locability_parameter():
    model = HybridFeatureGaussian(latent_dim=4, output_dim=8)
    model.initialize_localization_attributes(3)

    with torch.no_grad():
        model._locability_logit.copy_(torch.tensor([[0.0], [1.0], [-1.0]]))

    locability = model.get_locability()
    assert locability.shape == (3, 1)
    assert torch.all((locability > 0.0) & (locability < 1.0))

    state = model.state_dict()
    restored = HybridFeatureGaussian(latent_dim=4, output_dim=8)
    restored.initialize_localization_attributes(3)
    restored.load_state_dict(state)

    assert torch.allclose(restored.get_locability_logits(), model.get_locability_logits())


def test_hybrid_gaussian_loads_stdloc_loc_features_from_ply(tmp_path):
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"),
        ("scale_1", "f4"),
        ("scale_2", "f4"),
        ("rot_0", "f4"),
        ("rot_1", "f4"),
        ("rot_2", "f4"),
        ("rot_3", "f4"),
        ("loc_0", "f4"),
        ("loc_1", "f4"),
    ]
    vertices = np.array(
        [
            (0, 0, 1, 0, 0, 0, 0.1, 0.2, 0.3, 0, -2, -2, -2, 1, 0, 0, 0, 3.0, 4.0),
            (1, 0, 1, 0, 0, 0, 0.1, 0.2, 0.3, 0, -2, -2, -2, 1, 0, 0, 0, 0.0, 5.0),
        ],
        dtype=dtype,
    )
    ply_path = tmp_path / "stdloc_loc_feature.ply"
    PlyData([PlyElement.describe(vertices, "vertex")]).write(ply_path)

    model = HybridFeatureGaussian(latent_dim=4, output_dim=8)
    model.load_from_ply(str(ply_path))

    loc_feature = model.get_ply_loc_feature()
    assert loc_feature.shape == (2, 2)
    assert torch.allclose(loc_feature.norm(dim=-1), torch.ones(2), atol=1e-6)
    assert torch.allclose(loc_feature[0], torch.tensor([0.6, 0.8]), atol=1e-6)
