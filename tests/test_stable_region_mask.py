import torch

from loc_gs.dense_support.stable_region_mask import build_stable_region_proxy_mask


def test_stable_region_proxy_mask_flags_glare_and_vegetation_without_external_model():
    image = torch.zeros(3, 4, 4, dtype=torch.float32)
    image[:, :2, :2] = 0.95
    image[1, 2:, :2] = 0.8
    image[0, 2:, :2] = 0.1
    image[2, 2:, :2] = 0.1
    image[:, :, 2:] = 0.4

    result = build_stable_region_proxy_mask(image)

    assert result["stable_mask"].shape == (4, 4)
    assert result["risk"].min() >= 0.0
    assert result["risk"].max() <= 1.0
    assert result["risk"][:2, :2].mean() > result["risk"][:, 2:].mean()
    assert result["risk"][2:, :2].mean() > result["risk"][:, 2:].mean()
    assert result["metadata"]["external_model"] == "none"
