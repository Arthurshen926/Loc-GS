from pathlib import Path

import pytest

from loc_gs.sparse.audit import (
    ForbiddenRuntimeDependency,
    assert_internal_mainline_sources,
    reject_test_split,
)


def test_reject_test_split_blocks_training_labels():
    for split in ("test", "official_test", "cambridge_test"):
        with pytest.raises(ValueError, match="test split"):
            reject_test_split(split, purpose="teacher labels")


def test_reject_test_split_allows_non_test_and_unknown():
    assert reject_test_split("train_selfmap", purpose="teacher labels") == "train_selfmap"
    assert reject_test_split("", purpose="eval manifest") == "unknown"


def test_internal_mainline_audit_finds_external_ulfloc_import(tmp_path: Path):
    path = tmp_path / "bad.py"
    path.write_text(
        "import sys\n"
        "sys.path.insert(0, '/root/ULF-Loc')\n"
        "from ulfloc import ULFLoc\n",
        encoding="utf-8",
    )

    with pytest.raises(ForbiddenRuntimeDependency) as exc:
        assert_internal_mainline_sources([path])

    assert "/root/ULF-Loc" in str(exc.value)
    assert str(path) in str(exc.value)


def test_internal_mainline_audit_finds_vendored_stdloc_runtime(tmp_path: Path):
    path = tmp_path / "bad_stdloc.py"
    path.write_text(
        "STDLOC_ROOT = 'third_party/stdloc'\n"
        "subprocess.run(['python', 'third_party/stdloc/stdloc.py'])\n",
        encoding="utf-8",
    )

    with pytest.raises(ForbiddenRuntimeDependency, match="third_party/stdloc/stdloc.py"):
        assert_internal_mainline_sources([path])


def test_internal_mainline_audit_allows_reference_text_when_disabled(tmp_path: Path):
    path = tmp_path / "doc.py"
    path.write_text("REFERENCE = 'third_party/stdloc/configs/stdloc_cambridge.yaml'\n", encoding="utf-8")

    assert_internal_mainline_sources([path])


def test_internal_mainline_audit_cli_source_set_is_clean():
    from loc_gs.scripts.audit_internal_mainline import (
        internal_mainline_source_paths,
        run_internal_mainline_audit,
    )

    paths = internal_mainline_source_paths(Path("loc_gs"))
    assert any(str(path).endswith("eval_internal_sparse_cached.py") for path in paths)
    assert any(str(path).endswith("eval_sparse_distilled_cambridge.py") for path in paths)
    assert any(str(path).endswith("run_internal_sparse_gate.py") for path in paths)
    assert any(str(path).endswith("core/pnp.py") for path in paths)
    assert any(str(path).endswith("sparse/artifact_adapter.py") for path in paths)
    assert any(str(path).endswith("sparse/candidate_artifact_merge.py") for path in paths)
    assert any(str(path).endswith("sparse/cached_eval.py") for path in paths)
    assert any(str(path).endswith("sparse/candidate_completion_plan.py") for path in paths)
    assert any(str(path).endswith("sparse/candidate_coverage.py") for path in paths)
    assert any(str(path).endswith("sparse/landmarks.py") for path in paths)
    assert any(str(path).endswith("sparse/results_metrics.py") for path in paths)
    assert any(str(path).endswith("sparse/sparse_lgcv.py") for path in paths)
    assert any(str(path).endswith("sparse/pipeline.py") for path in paths)
    assert any(str(path).endswith("sparse/pose_map_frame_audit.py") for path in paths)
    assert any(str(path).endswith("sparse/real_inputs.py") for path in paths)
    assert any(str(path).endswith("students/descriptor_fusion.py") for path in paths)
    assert any(str(path).endswith("students/detector_student.py") for path in paths)
    assert any(str(path).endswith("students/landmark_selector.py") for path in paths)
    assert any(str(path).endswith("audit_internal_pose_map_frame.py") for path in paths)
    assert any(str(path).endswith("teacher/distillation_artifact.py") for path in paths)
    assert any(str(path).endswith("teacher/online_episode.py") for path in paths)
    assert any(str(path).endswith("build_internal_distillation_artifact.py") for path in paths)
    assert any(str(path).endswith("build_internal_candidate_completion_plan.py") for path in paths)
    assert any(str(path).endswith("build_internal_sparse_metrics_from_results.py") for path in paths)
    assert any(str(path).endswith("audit_internal_candidate_coverage.py") for path in paths)
    assert any(str(path).endswith("train_internal_sparse_students.py") for path in paths)
    assert any(str(path).endswith("merge_internal_candidate_artifacts.py") for path in paths)
    assert any(str(path).endswith("run_internal_sparse_smoke.py") for path in paths)
    assert any(str(path).endswith("teacher/solver_feedback.py") for path in paths)
    assert any(str(path).endswith("build_internal_solver_feedback_labels.py") for path in paths)
    assert any(str(path).endswith("simulation/query_sampler.py") for path in paths)
    assert any(str(path).endswith("build_internal_simulation_plan.py") for path in paths)
    assert any(str(path).endswith("training/sparse_candidate_scorer.py") for path in paths)
    assert any(str(path).endswith("train_internal_sparse_candidate_scorer.py") for path in paths)

    assert run_internal_mainline_audit(paths) == {
        "checked_file_count": len(paths),
        "forbidden_hit_count": 0,
    }
