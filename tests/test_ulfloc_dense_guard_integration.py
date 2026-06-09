from pathlib import Path


def test_ulfloc_has_optional_dense_transition_guard_hook():
    source = Path("/root/ULF-Loc/ulfloc.py").read_text(encoding="utf-8")

    assert "dense_transition_guard" in source
    assert "return_matches=self.dense_transition_guard_enabled" in source
    assert "should_accept_dense" in source
    assert "reject_dense_keep_sparse" in source


def test_ulfloc_sparse_matches_expose_guard_anchors():
    source = Path("/root/ULF-Loc/ulfloc.py").read_text(encoding="utf-8")

    assert '"p3d"' in source
    assert '"intrinsic"' in source
    assert '"image_size"' in source
    assert '"total_match_count"' in source


def test_ulfloc_results_use_jsonable_serializer_for_guard_payloads():
    source = Path("/root/ULF-Loc/ulfloc.py").read_text(encoding="utf-8")

    assert "def _jsonable" in source
    assert "json.dump(_jsonable(results)" in source


def test_ulfloc_dense_render_feedback_can_retry_only_after_guard_rejection():
    source = Path("/root/ULF-Loc/ulfloc.py").read_text(encoding="utf-8")

    assert "dense_render_feedback_activation" in source
    assert '"retry_on_reject"' in source
    assert '"retry_on_transition_risk"' in source
    assert "opacity_scale_override" in source
    assert "dense_render_feedback_retry" in source
    assert "native_dense_rejected_before_retry" in source
    assert "native_dense_risky_before_retry" in source
