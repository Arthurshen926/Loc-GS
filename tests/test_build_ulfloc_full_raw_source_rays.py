from types import SimpleNamespace

import torch

from loc_gs.scripts.build_ulfloc_full_raw_source_rays import view_records_for_match_sources


def _camera(name: str) -> SimpleNamespace:
    return SimpleNamespace(
        image_name=name,
        image_width=100,
        image_height=80,
        FoVx=1.0,
        FoVy=0.8,
        world_view_transform=torch.eye(4),
    )


def test_view_records_for_match_sources_selects_only_observed_train_views():
    views = view_records_for_match_sources(
        [_camera("a.png"), _camera("b.png"), _camera("c.png")],
        [
            {"source_view_id": "b.png", "source_xy": [1.0, 2.0]},
            {"source_view_id": "c.png", "source_xy": [3.0, 4.0]},
        ],
        split_name="selfmap_train",
    )

    assert list(views) == ["b.png", "c.png"]
    assert views["b.png"]["source_view_id"] == "b.png"
    assert views["b.png"]["split_name"] == "selfmap_train"


def test_view_records_for_match_sources_max_views_limits_smoke_scope():
    views = view_records_for_match_sources(
        [_camera("a.png"), _camera("b.png"), _camera("c.png")],
        [
            {"source_view_id": "a.png", "source_xy": [1.0, 2.0]},
            {"source_view_id": "b.png", "source_xy": [3.0, 4.0]},
            {"source_view_id": "c.png", "source_xy": [5.0, 6.0]},
        ],
        split_name="selfmap_train",
        max_views=2,
    )

    assert list(views) == ["a.png", "b.png"]
