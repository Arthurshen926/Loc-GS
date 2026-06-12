"""Training-time teacher schemas for sparse-dense distilled localization."""

from .inlier_precision_feedback import (
    InlierPrecisionFeedbackConfig,
    InlierPrecisionFeedbackRow,
    apply_inlier_precision_feedback_to_solver_rows,
    build_inlier_precision_feedback,
    load_inlier_precision_feedback_rows,
)
from .labels import TeacherLabelBatch, build_teacher_labels

__all__ = [
    "InlierPrecisionFeedbackConfig",
    "InlierPrecisionFeedbackRow",
    "TeacherLabelBatch",
    "apply_inlier_precision_feedback_to_solver_rows",
    "build_inlier_precision_feedback",
    "build_teacher_labels",
    "load_inlier_precision_feedback_rows",
]
