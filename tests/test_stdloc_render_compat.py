import torch

from loc_gs.dense_support.stdloc_render_compat import visible_mask_from_gsplat_radii


def test_visible_mask_from_gsplat_radii_collapses_multichannel_radii():
    radii = torch.tensor([[[0.0, 0.0], [2.0, 0.0], [0.0, 3.0], [4.0, 5.0]]])

    mask = visible_mask_from_gsplat_radii(radii)

    assert mask.tolist() == [False, True, True, True]

