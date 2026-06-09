import json
from pathlib import Path

import pytest

from loc_gs.scripts.prepare_ulfloc_log_alias import prepare_ulfloc_log_alias


def test_prepare_ulfloc_log_alias_links_required_files(tmp_path):
    source = tmp_path / "candidate_log"
    target = tmp_path / "dense_guard_log"
    source.mkdir()
    (source / "keypoints_sampled_idx.pkl").write_bytes(b"idx")
    (source / "keypoints_features.pkl").write_bytes(b"features")
    (source / "solver_feedback.pkl").write_bytes(b"feedback")

    manifest = prepare_ulfloc_log_alias(
        source_log_dir=source,
        target_log_dir=target,
        command="python -m alias",
    )

    assert (target / "keypoints_sampled_idx.pkl").is_symlink()
    assert (target / "keypoints_features.pkl").is_symlink()
    assert (target / "solver_feedback.pkl").is_symlink()
    assert (target / "keypoints_sampled_idx.pkl").read_bytes() == b"idx"
    assert manifest["schema"] == "loc_gs_ulfloc_log_alias_v1"
    assert manifest["missing_optional_files"] == []
    written = json.loads((target / "alias_manifest.json").read_text(encoding="utf-8"))
    assert written["source_log_dir"] == str(source.resolve())
    assert (target / "alias_command.txt").read_text(encoding="utf-8").strip() == "python -m alias"


def test_prepare_ulfloc_log_alias_allows_missing_optional_files(tmp_path):
    source = tmp_path / "candidate_log"
    target = tmp_path / "dense_guard_log"
    source.mkdir()
    (source / "keypoints_sampled_idx.pkl").write_bytes(b"idx")
    (source / "keypoints_features.pkl").write_bytes(b"features")

    manifest = prepare_ulfloc_log_alias(source_log_dir=source, target_log_dir=target)

    assert manifest["missing_optional_files"] == ["solver_feedback.pkl"]
    assert not (target / "solver_feedback.pkl").exists()


def test_prepare_ulfloc_log_alias_requires_required_files(tmp_path):
    source = tmp_path / "candidate_log"
    target = tmp_path / "dense_guard_log"
    source.mkdir()
    (source / "keypoints_sampled_idx.pkl").write_bytes(b"idx")

    with pytest.raises(FileNotFoundError):
        prepare_ulfloc_log_alias(source_log_dir=source, target_log_dir=target)


def test_prepare_ulfloc_log_alias_force_replaces_existing_symlink(tmp_path):
    source = tmp_path / "candidate_log"
    other = tmp_path / "other_log"
    target = tmp_path / "dense_guard_log"
    source.mkdir()
    other.mkdir()
    target.mkdir()
    for root, payload in ((source, b"idx"), (other, b"old")):
        (root / "keypoints_sampled_idx.pkl").write_bytes(payload)
        (root / "keypoints_features.pkl").write_bytes(payload)
    (target / "keypoints_sampled_idx.pkl").symlink_to((other / "keypoints_sampled_idx.pkl").resolve())

    prepare_ulfloc_log_alias(source_log_dir=source, target_log_dir=target, force=True)

    assert (target / "keypoints_sampled_idx.pkl").read_bytes() == b"idx"
