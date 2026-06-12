import json
from pathlib import Path

from loc_gs.sparse.correspondences import SparseCandidateBatch
from loc_gs.students.detector_student import (
    DetectorStudentConfig,
    DetectorStudentModel,
    detector_score_rows,
    load_detector_student,
    train_detector_student,
)


def _batch() -> SparseCandidateBatch:
    return SparseCandidateBatch(
        scene="GreatCourt",
        split_name="train",
        query_id="img.png",
        keypoint_xy=[[10.0, 10.0], [90.0, 10.0], [10.0, 90.0]],
        candidate_landmark_ids=[[1, 2], [3, 4], [5, 6]],
        candidate_scores=[[0.8, 0.1], [0.7, 0.2], [0.6, 0.3]],
        candidate_valid_mask=[[True, True], [True, True], [True, True]],
        candidate_label_roles=[
            ["protected_support", "neutral"],
            ["hard_negative", "hard_negative"],
            ["positive_inlier", "neutral"],
        ],
        candidate_solver_weight=[[2.0, 1.0], [1.5, 1.0], [1.0, 1.0]],
    )


def test_detector_student_learns_cell_utility_from_solver_feedback():
    model, summary = train_detector_student([_batch()], DetectorStudentConfig(grid_size=2))

    assert summary["student_modules"] == ["detector_student"]
    assert summary["positive_keypoint_count"] == 2
    assert summary["hard_negative_keypoint_count"] == 1
    assert model.score_keypoint([10.0, 10.0]) > model.score_keypoint([90.0, 10.0])
    assert model.score_keypoint([10.0, 90.0]) > model.score_keypoint([90.0, 10.0])
    rows = detector_score_rows(_batch(), model)
    assert rows[0][0] == rows[0][1]
    assert rows[0][0] > rows[1][0]


def test_detector_student_json_roundtrip(tmp_path: Path):
    model, _summary = train_detector_student([_batch()], DetectorStudentConfig(grid_size=2))
    path = tmp_path / "detector_student.json"
    path.write_text(json.dumps(model.to_json_dict(), sort_keys=True), encoding="utf-8")

    loaded = load_detector_student(path)

    assert isinstance(loaded, DetectorStudentModel)
    assert loaded.to_json_dict() == model.to_json_dict()
