import pytest

from loc_gs.feedback.raster_ray_contributors import (
    infer_pixel_xy,
    raster_intersections_to_ray_records,
)


def test_infer_pixel_xy_supports_pixel_xy_and_flat_pixel_id():
    assert infer_pixel_xy({"pixel_xy": [3.2, 4.7]}, width=10) == [3.2, 4.7]
    assert infer_pixel_xy({"pixel_id": 23}, width=10) == [3.0, 2.0]


def test_raster_intersections_group_topk_contributors_by_pixel():
    intersections = [
        {"pixel_xy": [10.0, 20.0], "gaussian_id": 0, "contribution": 0.2, "depth": 8.0, "xyz": [0, 0, 8]},
        {"pixel_xy": [10.0, 20.0], "gaussian_id": 1, "contribution": 0.8, "depth": 3.0, "xyz": [0, 0, 3]},
        {"pixel_xy": [10.0, 20.0], "gaussian_id": 2, "contribution": 0.1, "depth": 9.0, "xyz": [0, 1, 9]},
        {"pixel_xy": [11.0, 20.0], "gaussian_id": 3, "contribution": 0.7, "depth": 5.0, "xyz": [1, 0, 5]},
    ]

    bundle = raster_intersections_to_ray_records(
        intersections,
        source_view_id="s1",
        split_name="selfmap_train",
        width=100,
        top_k=2,
    )

    assert bundle["summary"]["intersection_count"] == 4
    assert bundle["summary"]["ray_count"] == 2
    ray = bundle["rays"][0]
    assert ray["source_view_id"] == "s1"
    assert ray["pixel_xy"] == [10.0, 20.0]
    assert [item["gaussian_id"] for item in ray["contributors"]] == [1, 0]
    assert ray["rendered_depth"] == pytest.approx(3.0)
    assert ray["expected_depth"] == pytest.approx((0.8 * 3.0 + 0.2 * 8.0) / 1.0)


def test_raster_intersections_use_opacity_proxy_when_contribution_missing():
    bundle = raster_intersections_to_ray_records(
        [
            {"pixel_id": 0, "gaussian_id": 0, "opacity": 0.1, "depth": 4.0},
            {"pixel_id": 0, "gaussian_id": 1, "opacity": 0.9, "depth": 5.0},
        ],
        source_view_id="s1",
        split_name="selfmap_train",
        width=10,
    )

    contributors = bundle["rays"][0]["contributors"]
    assert contributors[0]["gaussian_id"] == 1
    assert contributors[0]["contribution"] == pytest.approx(0.9)
    assert bundle["summary"]["weight_source"] == "contribution_or_opacity_proxy"


def test_raster_intersections_reject_test_split():
    with pytest.raises(ValueError, match="test split"):
        raster_intersections_to_ray_records([], source_view_id="s1", split_name="test", width=10)
