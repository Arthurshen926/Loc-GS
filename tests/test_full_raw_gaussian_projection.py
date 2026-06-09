import pytest

from loc_gs.feedback.full_raw_gaussian_projection import (
    build_sparse_source_rays_from_full_raw_gaussians,
    project_full_raw_gaussians_to_view,
    validate_full_raw_projection_bundle,
)


def test_project_full_raw_gaussians_to_view_audits_all_source_gaussians():
    bundle = project_full_raw_gaussians_to_view(
        gaussian_xyz=[
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
            [0.0, 0.0, -1.0],
            [20.0, 0.0, 2.0],
        ],
        world_to_camera=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        intrinsic=[
            [10.0, 0.0, 50.0],
            [0.0, 10.0, 50.0],
            [0.0, 0.0, 1.0],
        ],
        width=100,
        height=100,
        source_view_id="s1",
        split_name="selfmap_train",
        opacity=[0.2, 0.3, 0.4, 0.5],
        radius_px=[1.0, 2.0, 3.0, 4.0],
    )

    assert bundle["summary"]["projection_source"] == "full_raw_gaussians"
    assert bundle["summary"]["source_gaussian_count"] == 4
    assert bundle["summary"]["projected_gaussian_count"] == 2
    assert bundle["summary"]["dropped_behind_count"] == 1
    assert bundle["summary"]["dropped_out_of_frame_count"] == 1
    assert bundle["summary"]["sampled_idx_used"] is False
    assert [row["gaussian_id"] for row in bundle["projections"]] == [0, 1]
    assert bundle["projections"][0]["xy"] == pytest.approx([50.0, 50.0])
    assert bundle["projections"][1]["xy"] == pytest.approx([55.0, 50.0])
    assert bundle["projections"][1]["opacity"] == pytest.approx(0.3)
    assert bundle["projections"][1]["radius"] == pytest.approx(2.0)


def test_validate_full_raw_projection_bundle_rejects_sampled_subset_source():
    bundle = {
        "summary": {
            "projection_source": "sampled_idx_only",
            "source_gaussian_count": 2,
            "sampled_idx_used": True,
        },
        "projections": [],
    }

    with pytest.raises(ValueError, match="full_raw_gaussians"):
        validate_full_raw_projection_bundle(bundle)


def test_project_full_raw_gaussians_to_view_rejects_test_split():
    with pytest.raises(ValueError, match="test split"):
        project_full_raw_gaussians_to_view(
            gaussian_xyz=[[0.0, 0.0, 2.0]],
            world_to_camera=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            intrinsic=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            width=10,
            height=10,
            source_view_id="s1",
            split_name="test",
        )


def test_build_sparse_source_rays_from_full_raw_gaussians_streams_per_view_without_projection_dump():
    bundle = build_sparse_source_rays_from_full_raw_gaussians(
        matches=[
            {"source_view_id": "s1", "source_xy": [50.0, 50.0], "split_name": "selfmap_train"},
            {"source_view_id": "s1", "source_xy": [55.0, 50.0], "split_name": "selfmap_train"},
        ],
        gaussian_xyz=[
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
            [20.0, 0.0, 2.0],
        ],
        views_by_source={
            "s1": {
                "source_view_id": "s1",
                "width": 100,
                "height": 100,
                "world_to_camera": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "intrinsic": [
                    [10.0, 0.0, 50.0],
                    [0.0, 10.0, 50.0],
                    [0.0, 0.0, 1.0],
                ],
            }
        },
        split_name="selfmap_train",
        top_k=1,
        max_radius_px=4.0,
    )

    assert bundle["summary"]["projection_source"] == "full_raw_gaussians"
    assert bundle["summary"]["materialized_projection_jsonl"] is False
    assert bundle["summary"]["source_view_count_projected"] == 1
    assert bundle["summary"]["projected_gaussian_count"] == 2
    assert bundle["summary"]["ray_count"] == 2
    assert [ray["contributors"][0]["gaussian_id"] for ray in bundle["rays"]] == [0, 1]


def test_build_sparse_source_rays_from_full_raw_gaussians_selects_topk_by_proxy_score():
    bundle = build_sparse_source_rays_from_full_raw_gaussians(
        matches=[
            {"source_view_id": "s1", "source_xy": [50.0, 50.0], "split_name": "selfmap_train"},
        ],
        gaussian_xyz=[
            [0.0, 0.0, 2.0],
            [0.2, 0.0, 2.0],
            [4.0, 0.0, 2.0],
        ],
        views_by_source={
            "s1": {
                "source_view_id": "s1",
                "width": 100,
                "height": 100,
                "world_to_camera": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "intrinsic": [
                    [10.0, 0.0, 50.0],
                    [0.0, 10.0, 50.0],
                    [0.0, 0.0, 1.0],
                ],
            }
        },
        split_name="selfmap_train",
        opacity=[0.4, 1.0, 1.0],
        radius_px=[1.0, 1.0, 1.0],
        top_k=2,
        max_radius_px=8.0,
        radius_scale=8.0,
    )

    contributors = bundle["rays"][0]["contributors"]
    assert [item["gaussian_id"] for item in contributors] == [1, 0]
    assert contributors[0]["contribution"] > contributors[1]["contribution"]
