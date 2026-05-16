from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool_value(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _xy_value(value: Any) -> tuple[float | None, float | None]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return _float_or_none(value[0]), _float_or_none(value[1])
    return None, None


@dataclass
class FeedbackMatchRecord:
    scene: str = ""
    query_id: str = ""
    source_view_id: str = ""
    pose_source: str = ""
    keypoint_xy: tuple[float | None, float | None] = (None, None)
    matched_landmark_id: str = ""
    matched_gaussian_id: str = ""
    descriptor_score: float | None = None
    detector_score: float | None = None
    match_rank: int = 0
    pnp_inlier: bool = False
    reprojection_error_px: float | None = None
    depth_consistency: float | None = None
    visibility_score: float | None = None
    pose_error_t_cm: float | None = None
    pose_error_r_deg: float | None = None
    pnp_success: bool = False
    dense_refine_success: bool = False
    jacobian_info_trace: float | None = None
    jacobian_info_logdet_proxy: float | None = None

    @classmethod
    def from_mapping(cls, item: dict[str, Any] | "FeedbackMatchRecord") -> "FeedbackMatchRecord":
        if isinstance(item, FeedbackMatchRecord):
            return item
        keypoint_xy = _xy_value(item.get("keypoint_xy", item.get("keypoint")))
        return cls(
            scene=str(item.get("scene", "")),
            query_id=str(item.get("query_id", "")),
            source_view_id=str(item.get("source_view_id", "")),
            pose_source=str(item.get("pose_source", "")),
            keypoint_xy=keypoint_xy,
            matched_landmark_id=str(item.get("matched_landmark_id", item.get("landmark_id", ""))),
            matched_gaussian_id=str(item.get("matched_gaussian_id", item.get("gaussian_id", ""))),
            descriptor_score=_float_or_none(item.get("descriptor_score")),
            detector_score=_float_or_none(item.get("detector_score")),
            match_rank=_int_value(item.get("match_rank"), default=0),
            pnp_inlier=_bool_value(item.get("pnp_inlier"), default=False),
            reprojection_error_px=_float_or_none(item.get("reprojection_error_px")),
            depth_consistency=_float_or_none(item.get("depth_consistency")),
            visibility_score=_float_or_none(item.get("visibility_score")),
            pose_error_t_cm=_float_or_none(item.get("pose_error_t_cm")),
            pose_error_r_deg=_float_or_none(item.get("pose_error_r_deg")),
            pnp_success=_bool_value(item.get("pnp_success"), default=False),
            dense_refine_success=_bool_value(item.get("dense_refine_success"), default=False),
            jacobian_info_trace=_float_or_none(item.get("jacobian_info_trace")),
            jacobian_info_logdet_proxy=_float_or_none(item.get("jacobian_info_logdet_proxy")),
        )

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["keypoint_xy"] = list(self.keypoint_xy)
        return out


@dataclass
class FeedbackPoseRecord:
    scene: str = ""
    query_id: str = ""
    source_view_id: str = ""
    pose_source: str = ""
    pose_error_t_cm: float | None = None
    pose_error_r_deg: float | None = None
    pnp_success: bool = False
    dense_refine_success: bool = False
    num_matches: int = 0
    num_inliers: int = 0

    @classmethod
    def from_mapping(cls, item: dict[str, Any] | "FeedbackPoseRecord") -> "FeedbackPoseRecord":
        if isinstance(item, FeedbackPoseRecord):
            return item
        return cls(
            scene=str(item.get("scene", "")),
            query_id=str(item.get("query_id", "")),
            source_view_id=str(item.get("source_view_id", "")),
            pose_source=str(item.get("pose_source", "")),
            pose_error_t_cm=_float_or_none(item.get("pose_error_t_cm")),
            pose_error_r_deg=_float_or_none(item.get("pose_error_r_deg")),
            pnp_success=_bool_value(item.get("pnp_success"), default=False),
            dense_refine_success=_bool_value(item.get("dense_refine_success"), default=False),
            num_matches=_int_value(item.get("num_matches"), default=0),
            num_inliers=_int_value(item.get("num_inliers"), default=0),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FeedbackEpisode:
    scene: str = ""
    query_id: str = ""
    source_view_id: str = ""
    pose_source: str = ""
    split_name: str = ""
    matches: list[FeedbackMatchRecord] = field(default_factory=list)
    pose: FeedbackPoseRecord | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, item: dict[str, Any] | "FeedbackEpisode") -> "FeedbackEpisode":
        if isinstance(item, FeedbackEpisode):
            return item
        return cls(
            scene=str(item.get("scene", "")),
            query_id=str(item.get("query_id", "")),
            source_view_id=str(item.get("source_view_id", "")),
            pose_source=str(item.get("pose_source", "")),
            split_name=str(item.get("split_name", "")),
            matches=[FeedbackMatchRecord.from_mapping(record) for record in item.get("matches", [])],
            pose=FeedbackPoseRecord.from_mapping(item["pose"]) if isinstance(item.get("pose"), dict) else None,
            metadata=dict(item.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene": self.scene,
            "query_id": self.query_id,
            "source_view_id": self.source_view_id,
            "pose_source": self.pose_source,
            "split_name": self.split_name,
            "matches": [record.to_dict() for record in self.matches],
            "pose": None if self.pose is None else self.pose.to_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass
class FeedbackBankSummary:
    record_count: int = 0
    scene_count: int = 0
    scenes: list[str] = field(default_factory=list)
    pnp_inlier_rate: float = 0.0
    hard_negative_rate: float = 0.0
    pnp_success_rate: float = 0.0
    dense_refine_success_rate: float = 0.0
    mean_pose_error_t_cm: float | None = None
    mean_pose_error_r_deg: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
