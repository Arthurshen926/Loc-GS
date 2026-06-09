import pytest

from loc_gs.feedback.projected_gaussian_rays import (
    projected_gaussians_to_sparse_source_rays,
    proxy_contribution,
)


def test_proxy_contribution_decays_with_distance_and_uses_opacity():
    near = proxy_contribution(distance_px=0.5, radius_px=4.0, opacity=0.8)
    far = proxy_contribution(distance_px=8.0, radius_px=4.0, opacity=0.8)

    assert near > far
    assert near <= 0.8
    assert far > 0.0


def test_projected_gaussians_to_sparse_source_rays_extracts_topk_near_sparse_pixel():
    matches = [
        {
            "query_id": "q1",
            "source_view_id": "s1",
            "source_xy": [10.0, 20.0],
            "query_xy": [10.0, 20.0],
        }
    ]
    projections = [
        {
            "projection_source": "full_raw_gaussians",
            "source_view_id": "s1",
            "gaussian_id": 0,
            "xy": [10.2, 20.1],
            "radius": 4.0,
            "opacity": 0.7,
            "depth": 5.0,
        },
        {
            "projection_source": "full_raw_gaussians",
            "source_view_id": "s1",
            "gaussian_id": 1,
            "xy": [12.0, 20.0],
            "radius": 6.0,
            "opacity": 0.9,
            "depth": 3.0,
        },
        {
            "projection_source": "full_raw_gaussians",
            "source_view_id": "s1",
            "gaussian_id": 2,
            "xy": [50.0, 20.0],
            "radius": 4.0,
            "opacity": 1.0,
            "depth": 2.0,
        },
    ]

    bundle = projected_gaussians_to_sparse_source_rays(
        matches,
        projections,
        split_name="selfmap_train",
        top_k=2,
        max_radius_px=8.0,
    )

    assert bundle["summary"]["match_count"] == 1
    assert bundle["summary"]["ray_count"] == 1
    ray = bundle["rays"][0]
    assert ray["pixel_xy"] == [10.0, 20.0]
    assert [item["gaussian_id"] for item in ray["contributors"]] == [1, 0]
    assert ray["rendered_depth"] == pytest.approx(3.0)
    assert ray["expected_depth"] > 3.0


def test_projected_gaussians_to_sparse_source_rays_drops_uncovered_sparse_pixel():
    bundle = projected_gaussians_to_sparse_source_rays(
        [{"source_view_id": "s1", "source_xy": [10.0, 20.0]}],
        [
            {
                "projection_source": "full_raw_gaussians",
                "source_view_id": "s1",
                "gaussian_id": 0,
                "xy": [100.0, 100.0],
                "radius": 2.0,
                "opacity": 1.0,
            }
        ],
        split_name="selfmap_train",
        max_radius_px=4.0,
    )

    assert bundle["rays"] == []
    assert bundle["summary"]["dropped_no_projection"] == 1


def test_projected_gaussians_to_sparse_source_rays_rejects_test_split():
    with pytest.raises(ValueError, match="test split"):
        projected_gaussians_to_sparse_source_rays([], [], split_name="test")


def test_projected_gaussians_to_sparse_source_rays_rejects_sampled_projection_source():
    with pytest.raises(ValueError, match="full_raw_gaussians"):
        projected_gaussians_to_sparse_source_rays(
            [{"source_view_id": "s1", "source_xy": [10.0, 20.0]}],
            [
                {
                    "projection_source": "sampled_idx_only",
                    "source_view_id": "s1",
                    "gaussian_id": 0,
                    "xy": [10.0, 20.0],
                    "radius": 2.0,
                }
            ],
            split_name="selfmap_train",
        )
