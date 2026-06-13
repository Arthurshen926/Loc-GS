import torch

from loc_gs.scripts.audit_ulfloc_scene_detector import _apply_audit_masks, _topk_xy_from_heatmap


def test_apply_audit_masks_matches_sparse_eval_mask_semantics():
    image = torch.ones(3, 4, 5, dtype=torch.float32)
    obj = torch.ones(4, 5, dtype=torch.bool)
    sky = torch.ones(4, 5, dtype=torch.bool)
    distort = torch.ones(4, 5, dtype=torch.bool)
    obj[1, 2] = False
    distort[2, 3] = False
    sky[0, 4] = False
    masks = {"q.png": (obj, sky, distort)}

    masked = _apply_audit_masks(image.clone(), masks, "q.png")

    assert masked[:, 1, 2].sum().item() == 0.0
    assert masked[:, 2, 3].sum().item() == 0.0
    assert masked[:, 0, 4].sum().item() == 0.0
    assert torch.allclose(masked[:, 3, 1], torch.ones(3))


def test_apply_audit_masks_is_noop_without_masks():
    image = torch.rand(3, 4, 5)

    masked = _apply_audit_masks(image, None, "q.png")

    assert masked is image


def test_topk_xy_from_heatmap_returns_detector_heatmap_peaks_in_xy_order():
    heatmap = torch.zeros(1, 1, 5, 6, dtype=torch.float32)
    heatmap[..., 1, 4] = 0.9
    heatmap[..., 3, 2] = 0.8

    xy, scores = _topk_xy_from_heatmap(heatmap, max_keypoints=2, nms_radius=0)

    assert xy.tolist() == [[4.0, 1.0], [2.0, 3.0]]
    assert torch.allclose(scores, torch.tensor([0.9, 0.8], dtype=torch.float32))
