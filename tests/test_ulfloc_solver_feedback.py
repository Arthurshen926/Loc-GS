import importlib.util
import pickle
from pathlib import Path

import pytest
import torch


def _load_ulfloc_solver_feedback():
    module_path = Path("/root/ULF-Loc/utils/solver_feedback.py")
    spec = importlib.util.spec_from_file_location("ulfloc_solver_feedback", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_ulfloc_solver_feedback_alpha_zero_preserves_scores(tmp_path):
    module = _load_ulfloc_solver_feedback()
    artifact = tmp_path / "feedback.pkl"
    with artifact.open("wb") as f:
        pickle.dump(
            {
                "schema_version": "ulfloc_solver_feedback_v1",
                "split_name": "selfmap_train",
                "landmark_weights": torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32),
            },
            f,
        )

    feedback = module.load_solver_feedback(artifact, num_landmarks=3)
    score = torch.tensor([2.0, 4.0, 8.0], dtype=torch.float32)

    assert torch.allclose(module.apply_landmark_feedback(score, feedback.landmark_weights, alpha=0.0), score)
    assert torch.allclose(
        module.apply_landmark_feedback(score, feedback.landmark_weights, alpha=0.5),
        torch.tensor([1.5, 4.0, 12.0], dtype=torch.float32),
    )


def test_ulfloc_solver_feedback_rejects_test_split(tmp_path):
    module = _load_ulfloc_solver_feedback()
    artifact = tmp_path / "feedback.pkl"
    with artifact.open("wb") as f:
        pickle.dump(
            {
                "schema_version": "ulfloc_solver_feedback_v1",
                "split_name": "test",
                "landmark_weights": torch.ones(3, dtype=torch.float32),
            },
            f,
        )

    with pytest.raises(ValueError, match="test split"):
        module.load_solver_feedback(artifact, num_landmarks=3)


def test_ulfloc_solver_feedback_expands_sparse_landmark_ids(tmp_path):
    module = _load_ulfloc_solver_feedback()
    artifact = tmp_path / "feedback.pkl"
    with artifact.open("wb") as f:
        pickle.dump(
            {
                "schema_version": "ulfloc_solver_feedback_v1",
                "split_name": "selfmap_train",
                "landmark_ids": torch.tensor([1, 3], dtype=torch.long),
                "landmark_weights": torch.tensor([1.5, 0.25], dtype=torch.float32),
            },
            f,
        )

    feedback = module.load_solver_feedback(artifact, num_landmarks=5)

    assert torch.allclose(feedback.landmark_weights, torch.tensor([1.0, 1.5, 1.0, 0.25, 1.0]))
    assert torch.allclose(
        module.gather_sampled_weights(feedback.landmark_weights, torch.tensor([3, 0, 1])),
        torch.tensor([0.25, 1.0, 1.5]),
    )


def test_ulfloc_dense_render_opacity_scale_defaults_to_disabled():
    module = _load_ulfloc_solver_feedback()

    assert module.build_dense_render_opacity_scale(
        num_landmarks=5,
        sampled_idx=torch.tensor([1, 3]),
        mode="none",
        non_selected_opacity=0.1,
    ) is None


def test_ulfloc_dense_render_opacity_scale_keeps_sampled_core():
    module = _load_ulfloc_solver_feedback()

    scale = module.build_dense_render_opacity_scale(
        num_landmarks=5,
        sampled_idx=torch.tensor([1, 3]),
        mode="sampled",
        non_selected_opacity=0.25,
    )

    assert torch.allclose(scale, torch.tensor([0.25, 1.0, 0.25, 1.0, 0.25]))


def test_ulfloc_dense_render_opacity_scale_can_use_solver_weights():
    module = _load_ulfloc_solver_feedback()

    scale = module.build_dense_render_opacity_scale(
        num_landmarks=5,
        sampled_idx=torch.tensor([1, 3]),
        mode="sampled_solver_feedback",
        non_selected_opacity=0.25,
        solver_weights=torch.tensor([1.0, 2.0, 1.0, 0.5, 1.0]),
        alpha=0.5,
    )

    assert torch.allclose(scale, torch.tensor([0.25, 1.5, 0.25, 0.75, 0.25]))


def test_ulfloc_renderer_accepts_dense_opacity_scale_hook():
    renderer_source = Path("/root/ULF-Loc/gaussian_renderer/__init__.py").read_text(encoding="utf-8")

    assert "opacity_scale=None" in renderer_source
    assert "opacities = opacity.squeeze(-1)" in renderer_source
    assert "opacities = opacities * opacity_scale" in renderer_source


def test_ulfloc_dense_path_passes_solver_feedback_opacity_scale_to_render():
    ulfloc_source = Path("/root/ULF-Loc/ulfloc.py").read_text(encoding="utf-8")

    assert "build_dense_render_opacity_scale" in ulfloc_source
    assert "self.dense_render_opacity_scale" in ulfloc_source
    assert "opacity_scale_override" in ulfloc_source
    assert "self.dense_render_feedback_activation == \"always\"" in ulfloc_source


def test_ulfloc_solver_feedback_loads_view_landmark_weights(tmp_path):
    module = _load_ulfloc_solver_feedback()
    artifact = tmp_path / "feedback.pkl"
    with artifact.open("wb") as f:
        pickle.dump(
            {
                "schema_version": "ulfloc_solver_feedback_v1",
                "split_name": "selfmap_train",
                "landmark_weights": torch.ones(5, dtype=torch.float32),
                "view_landmark_weights": {
                    "img_0001": {
                        "landmark_ids": torch.tensor([1, 3], dtype=torch.long),
                        "weights": torch.tensor([1.5, 0.5], dtype=torch.float32),
                    },
                    "img_0002": {"2": 1.75},
                },
            },
            f,
        )

    feedback = module.load_solver_feedback(artifact, num_landmarks=5)

    assert feedback.has_view_landmark_weights() is True
    gathered = module.gather_sampled_view_weights(
        feedback,
        torch.tensor([0, 1, 2, 3], dtype=torch.long),
        ["img_0001", "img_0002", "missing"],
    )
    assert torch.allclose(
        gathered,
        torch.tensor(
            [
                [1.0, 1.5, 1.0, 0.5],
                [1.0, 1.0, 1.75, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        ),
    )


def test_view_dependent_weights_can_change_normalized_feature_fusion(tmp_path):
    module = _load_ulfloc_solver_feedback()
    artifact = tmp_path / "feedback.pkl"
    with artifact.open("wb") as f:
        pickle.dump(
            {
                "schema_version": "ulfloc_solver_feedback_v1",
                "split_name": "selfmap_train",
                "landmark_weights": torch.ones(1, dtype=torch.float32),
                "view_landmark_weights": {
                    "good_view": {"landmark_ids": [0], "weights": [2.0]},
                    "bad_view": {"landmark_ids": [0], "weights": [0.25]},
                },
            },
            f,
        )
    feedback = module.load_solver_feedback(artifact, num_landmarks=1)
    view_weights = module.gather_sampled_view_weights(feedback, torch.tensor([0]), ["good_view", "bad_view"])
    features = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=torch.float32)
    base_weights = torch.ones((2, 1, 1), dtype=torch.float32)

    native = (features * base_weights).sum(dim=0) / base_weights.sum(dim=0)
    solver = (features * view_weights[:, :, None]).sum(dim=0) / view_weights[:, :, None].sum(dim=0)

    assert torch.allclose(native, torch.tensor([[0.5, 0.5]]))
    assert solver[0, 0] > 0.85
    assert solver[0, 1] < 0.15
