"""Training-time teacher schemas for sparse-dense distilled localization."""

from .labels import TeacherLabelBatch, build_teacher_labels

__all__ = ["TeacherLabelBatch", "build_teacher_labels"]
