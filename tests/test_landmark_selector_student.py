import json
from pathlib import Path

from loc_gs.sparse.correspondences import SparseCandidateBatch
from loc_gs.students.landmark_selector import (
    LandmarkSelectorConfig,
    LandmarkSelectorModel,
    landmark_selector_score_rows,
    load_landmark_selector,
    train_landmark_selector,
)


def _batch() -> SparseCandidateBatch:
    return SparseCandidateBatch(
        scene="GreatCourt",
        split_name="train",
        query_id="img.png",
        keypoint_xy=[[1.0, 2.0], [3.0, 4.0]],
        candidate_landmark_ids=[[10, 20], [10, 30]],
        candidate_scores=[[0.1, 0.9], [0.8, 0.7]],
        candidate_valid_mask=[[True, True], [True, True]],
        candidate_geometric_correct=[[True, False], [True, False]],
        candidate_dense_consistent=[[True, False], [True, False]],
        candidate_sparse_inlier=[[True, False], [True, False]],
        candidate_solver_weight=[[2.0, 1.0], [2.0, 1.0]],
        candidate_label_roles=[["protected_support", "hard_negative"], ["protected_support", "hard_negative"]],
    )


def test_landmark_selector_learns_solver_useful_priors_and_conflicts():
    model, summary = train_landmark_selector([_batch()], LandmarkSelectorConfig(conflict_penalty=0.25))

    assert summary["student_modules"] == ["landmark_selector", "conflict_graph"]
    assert summary["protected_support_count"] == 2
    assert summary["hard_negative_count"] == 2
    assert model.score_landmark(10) > model.score_landmark(20)
    assert model.score_landmark(10) > model.score_landmark(30)
    assert model.conflict_weight(10, 20) > 0.0
    rows = landmark_selector_score_rows(_batch(), model)
    assert rows[0][0] > rows[0][1]


def test_landmark_selector_json_roundtrip(tmp_path: Path):
    model, _summary = train_landmark_selector([_batch()])
    path = tmp_path / "landmark_selector.json"
    path.write_text(json.dumps(model.to_json_dict(), sort_keys=True), encoding="utf-8")

    loaded = load_landmark_selector(path)

    assert isinstance(loaded, LandmarkSelectorModel)
    assert loaded.to_json_dict() == model.to_json_dict()
