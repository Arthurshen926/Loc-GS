from types import SimpleNamespace

import pytest

from loc_gs.scripts.eval_sparse_conditioned_dense_control import (
    _resume_rows,
    _resume_partial_rows,
    _selected_camera_count,
    _preload_render_backend,
    summarize_comparison,
    summarize_pose_rows,
)


def test_summarize_pose_rows_reports_tail_metrics_and_recall():
    rows = [
        {"te": 1.0, "re": 0.1},
        {"te": 10.0, "re": 1.0},
        {"te": 200.0, "re": 2.0},
    ]

    summary = summarize_pose_rows(rows, te_key="te", re_key="re")

    assert summary["query_count"] == 3
    assert summary["median_te_cm"] == 10.0
    assert summary["R50cm5deg"] == 2 / 3
    assert summary["severe_rate_1m"] == 1 / 3
    assert summary["catastrophic_rate_5m"] == 0.0


def test_summarize_comparison_counts_regressions_and_render_labels():
    rows = [
        {
            "sparse_te_cm": 10.0,
            "base_dense_te_cm": 20.0,
            "sparse_conditioned_dense_te_cm": 14.0,
            "sparse_conditioned_label": "gated_base",
            "low_confidence_gated_base": True,
            "base_dense_reused": False,
            "base_reuse_reason": "base_requires_repair",
        },
        {
            "sparse_te_cm": 10.0,
            "base_dense_te_cm": 10.0,
            "sparse_conditioned_dense_te_cm": 65.0,
            "sparse_conditioned_label": "gated_ray_up-5.000",
            "low_confidence_gated_base": False,
            "base_dense_reused": True,
            "base_reuse_reason": "base_dense_high_confidence",
        },
    ]

    summary = summarize_comparison(rows)

    assert summary["regression_20cm_count"] == 1
    assert summary["regression_50cm_count"] == 1
    assert summary["improved_gt_0_5cm_count"] == 1
    assert summary["non_base_render_count"] == 2
    assert summary["low_confidence_gated_base_count"] == 1
    assert summary["guided_pose_count"] == 1
    assert summary["base_dense_worsened_count"] == 1
    assert summary["sparse_conditioned_dense_worsened_count"] == 1
    assert summary["base_dense_reused_count"] == 1
    assert summary["base_dense_high_confidence_reuse_count"] == 1
    assert summary["base_preflight_safe_reuse_count"] == 0


def test_preload_render_backend_reports_boolean_status():
    assert isinstance(_preload_render_backend(), bool)


def test_resume_rows_accepts_matching_partial_prefix(tmp_path):
    scene_dir = tmp_path / "Scene"
    scene_dir.mkdir()
    (scene_dir / "results.partial.json").write_text(
        '[{"query_index": 0, "image_name": "a.png"}, {"query_index": 1, "image_name": "b.png"}]',
        encoding="utf-8",
    )
    cameras = [SimpleNamespace(image_name="a.png"), SimpleNamespace(image_name="b.png"), SimpleNamespace(image_name="c.png")]

    rows = _resume_rows(scene_dir, cameras, resume_partial=True)

    assert [row["image_name"] for row in rows] == ["a.png", "b.png"]


def test_resume_partial_rows_does_not_need_to_materialize_cameras(tmp_path):
    scene_dir = tmp_path / "Scene"
    scene_dir.mkdir()
    (scene_dir / "results.partial.json").write_text(
        '[{"query_index": 0, "image_name": "a.png"}]',
        encoding="utf-8",
    )

    rows = _resume_partial_rows(scene_dir, resume_partial=True)

    assert rows == [{"query_index": 0, "image_name": "a.png"}]


def test_selected_camera_count_honors_stride_and_max_queries():
    assert _selected_camera_count([object()] * 10, query_stride=3, max_queries=0) == 4
    assert _selected_camera_count([object()] * 10, query_stride=3, max_queries=2) == 2


def test_resume_rows_rejects_mismatched_partial_prefix(tmp_path):
    scene_dir = tmp_path / "Scene"
    scene_dir.mkdir()
    (scene_dir / "results.partial.json").write_text(
        '[{"query_index": 0, "image_name": "wrong.png"}]',
        encoding="utf-8",
    )
    cameras = [SimpleNamespace(image_name="a.png")]

    with pytest.raises(ValueError, match="does not match camera prefix"):
        _resume_rows(scene_dir, cameras, resume_partial=True)
