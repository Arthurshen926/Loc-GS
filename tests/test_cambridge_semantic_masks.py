import pickle
from pathlib import Path

import torch

from loc_gs.data.cambridge_semantic_masks import (
    CambridgeSemanticMaskStore,
    compose_cambridge_semantic_valid_mask,
    resolve_cambridge_masks_path,
    resize_semantic_mask,
)


def _write_masks(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stuff_mask = torch.tensor([[True, True, False], [True, True, True]])
    sky_mask = torch.tensor([[True, False, True], [True, True, True]])
    distort_mask = torch.tensor([[True, True, True], [False, True, True]])
    with path.open("wb") as handle:
        pickle.dump({"seq1/frame00001.png": (stuff_mask, sky_mask, distort_mask)}, handle)


def test_compose_cambridge_semantic_valid_mask_removes_dynamic_sky_and_distort():
    stuff_mask = torch.tensor([[True, True, False], [True, True, True]])
    sky_mask = torch.tensor([[True, False, True], [True, True, True]])
    distort_mask = torch.tensor([[True, True, True], [False, True, True]])

    dynamic_only = compose_cambridge_semantic_valid_mask(
        (stuff_mask, sky_mask, distort_mask),
        remove_sky=False,
    )
    dynamic_and_sky = compose_cambridge_semantic_valid_mask(
        (stuff_mask, sky_mask, distort_mask),
        remove_sky=True,
    )

    assert dynamic_only.tolist() == [[True, True, False], [False, True, True]]
    assert dynamic_and_sky.tolist() == [[True, False, False], [False, True, True]]


def test_resolve_cambridge_masks_path_falls_back_to_processed_for_root_images(tmp_path):
    scene_root = tmp_path / "ShopFacade"
    _write_masks(scene_root / "processed" / "masks.pkl")

    assert resolve_cambridge_masks_path(scene_root, ".") == scene_root / "processed" / "masks.pkl"


def test_cambridge_semantic_mask_store_returns_audited_valid_mask(tmp_path):
    scene_root = tmp_path / "ShopFacade"
    mask_path = scene_root / "processed" / "masks.pkl"
    _write_masks(mask_path)

    store = CambridgeSemanticMaskStore.from_scene(scene_root, image_subdir="processed", remove_sky=True)
    mask = store.valid_mask("seq1/frame00001.png")

    assert store.loaded
    assert store.path == mask_path
    assert mask.shape == (2, 3)
    assert mask.tolist() == [[True, False, False], [False, True, True]]
    assert store.audit()["mask_path"] == str(mask_path)


def test_cambridge_semantic_mask_store_missing_image_returns_all_valid_and_tracks_miss(tmp_path):
    scene_root = tmp_path / "ShopFacade"
    _write_masks(scene_root / "processed" / "masks.pkl")
    store = CambridgeSemanticMaskStore.from_scene(scene_root, image_subdir="processed", remove_sky=True)

    mask = store.valid_mask("seq1/missing.png", fallback_hw=(4, 5))

    assert mask.shape == (4, 5)
    assert mask.all()
    assert store.audit()["missing_image_count"] == 1


def test_resize_semantic_mask_uses_nearest_neighbor():
    mask = torch.tensor([[True, False], [False, True]])

    resized = resize_semantic_mask(mask, height=4, width=4)

    assert resized.dtype == torch.bool
    assert resized.tolist() == [
        [True, True, False, False],
        [True, True, False, False],
        [False, False, True, True],
        [False, False, True, True],
    ]
