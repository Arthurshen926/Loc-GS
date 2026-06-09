import pytest

from loc_gs.feedback.cross_view_ray_observations import (
    build_cross_view_ray_observations,
    build_ray_contributor_index,
    find_ray_contributors,
    pixel_key,
)


def test_pixel_key_quantizes_image_coordinates():
    assert pixel_key([10.2, 20.7], quantization_px=1.0) == (10, 21)
    assert pixel_key([10.2, 20.7], quantization_px=0.5) == (20, 41)


def test_ray_index_supports_exact_and_radius_lookup():
    rays = [
        {
            "source_view_id": "s1",
            "pixel_xy": [10.0, 20.0],
            "contributors": [
                {"gaussian_id": 3, "contribution": 0.8, "xyz": [0.0, 0.0, 4.0]},
                {"gaussian_id": 4, "contribution": 0.2, "xyz": [1.0, 0.0, 4.0]},
            ],
        }
    ]
    index = build_ray_contributor_index(rays)

    exact = find_ray_contributors(index, "s1", [10.0, 20.0])
    nearby = find_ray_contributors(index, "s1", [10.4, 20.2], radius_px=1.0)

    assert exact is not None
    assert nearby is not None
    assert exact["contributors"][0]["gaussian_id"] == 3
    assert nearby["contributors"][0]["gaussian_id"] == 3
    assert find_ray_contributors(index, "s1", [12.0, 20.0], radius_px=0.5) is None


def test_build_cross_view_ray_observations_joins_matches_to_source_ray_contributors():
    matches = [
        {
            "query_id": "q1",
            "source_view_id": "s1",
            "split_name": "selfmap_train",
            "query_xy": [5.0, 6.0],
            "source_xy": [10.0, 20.0],
            "descriptor_score": 0.9,
            "local_geometry_score": 0.8,
            "pnp_inlier": True,
            "reprojection_error_px": 1.5,
            "expected_depth": 8.0,
        },
        {
            "query_id": "q1",
            "source_view_id": "s1",
            "split_name": "selfmap_train",
            "query_xy": [7.0, 8.0],
            "source_xy": [10.4, 20.2],
            "descriptor_score": 0.7,
            "pnp_inlier": False,
            "reprojection_error_px": 50.0,
        },
    ]
    rays = [
        {
            "source_view_id": "s1",
            "pixel_xy": [10.0, 20.0],
            "rendered_depth": 3.0,
            "contributors": [
                {"gaussian_id": 3, "contribution": 0.8, "depth": 3.0, "xyz": [0.0, 0.0, 3.0]},
                {"gaussian_id": 4, "contribution": 0.2, "depth": 8.0, "xyz": [1.0, 0.0, 8.0]},
            ],
        }
    ]

    bundle = build_cross_view_ray_observations(
        matches,
        rays,
        split_name="selfmap_train",
        radius_px=1.0,
    )

    assert bundle["summary"]["match_count"] == 2
    assert bundle["summary"]["observation_count"] == 2
    assert bundle["summary"]["dropped_no_contributors"] == 0
    first = bundle["observations"][0]
    assert first["query_id"] == "q1"
    assert first["source_view_id"] == "s1"
    assert first["query_xy"] == [5.0, 6.0]
    assert first["source_xy"] == [10.0, 20.0]
    assert first["ray_pixel_xy"] == [10.0, 20.0]
    assert first["contributors"][0]["gaussian_id"] == 3
    assert first["expected_depth"] == pytest.approx(8.0)
    assert first["rendered_depth"] == pytest.approx(3.0)


def test_build_cross_view_ray_observations_preserves_sparse_pnp_observation_fields():
    matches = [
        {
            "query_id": "q1",
            "source_view_id": "s1",
            "split_name": "selfmap_train",
            "query_xy": [5.0, 6.0],
            "query_xy_norm": [0.25, 0.75],
            "source_xy": [10.0, 20.0],
            "bearing": [0.1, -0.2, 1.0],
            "camera_xyz": [1.0, -0.5, 7.0],
            "descriptor_margin": 0.42,
            "local_geometry_score": 0.8,
            "ray_entropy": 0.31,
            "pnp_inlier": True,
            "reprojection_error_px": 1.5,
        }
    ]
    rays = [
        {
            "source_view_id": "s1",
            "pixel_xy": [10.0, 20.0],
            "contributors": [{"gaussian_id": 3, "contribution": 1.0, "depth": 7.0}],
            "expected_depth": 7.0,
            "rendered_depth": 7.0,
        }
    ]

    bundle = build_cross_view_ray_observations(matches, rays, split_name="selfmap_train")

    observation = bundle["observations"][0]
    assert observation["query_xy_norm"] == pytest.approx([0.25, 0.75])
    assert observation["bearing"] == pytest.approx([0.1, -0.2, 1.0])
    assert observation["camera_xyz"] == pytest.approx([1.0, -0.5, 7.0])
    assert observation["descriptor_margin"] == pytest.approx(0.42)
    assert observation["local_geometry_score"] == pytest.approx(0.8)
    assert observation["ray_entropy"] == pytest.approx(0.31)


def test_build_cross_view_ray_observations_reports_missing_rays():
    bundle = build_cross_view_ray_observations(
        [
            {
                "query_id": "q1",
                "source_view_id": "s1",
                "source_xy": [10.0, 20.0],
                "pnp_inlier": True,
            }
        ],
        [],
        split_name="selfmap_train",
    )

    assert bundle["observations"] == []
    assert bundle["summary"]["match_count"] == 1
    assert bundle["summary"]["dropped_no_contributors"] == 1


def test_build_cross_view_ray_observations_rejects_test_split():
    with pytest.raises(ValueError, match="test split"):
        build_cross_view_ray_observations([], [], split_name="test")
