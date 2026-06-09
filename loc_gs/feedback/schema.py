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


def _int_pair_value(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return None


def _vector_value(value: Any, *, length: int) -> tuple[float, ...] | None:
    if not isinstance(value, (list, tuple)) or len(value) < int(length):
        return None
    out: list[float] = []
    for idx in range(int(length)):
        item = _float_or_none(value[idx])
        if item is None:
            return None
        out.append(float(item))
    return tuple(out)


@dataclass
class FeedbackMatchRecord:
    scene: str = ""
    split_name: str = ""
    query_id: str = ""
    image_id: str = ""
    source_view_id: str = ""
    pose_source: str = ""
    source_role: str = ""
    keypoint_id: str = ""
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
    query_sparse_te_cm: float | None = None
    pose_error_r_deg: float | None = None
    pnp_success: bool = False
    pose_success: bool = False
    dense_refine_success: bool = False
    dense_transition: str = ""
    dense_delta_te_cm: float | None = None
    jacobian_info_trace: float | None = None
    jacobian_info_logdet_proxy: float | None = None
    query_xy_norm: tuple[float, float] | None = None
    bearing: tuple[float, float, float] | None = None
    camera_xyz: tuple[float, float, float] | None = None
    depth_m: float | None = None
    descriptor_margin: float | None = None
    local_geometry_score: float | None = None
    ray_artifact_score: float | None = None
    ray_entropy: float | None = None
    image_cell: tuple[int, int] | None = None
    depth_bin: int | None = None

    @classmethod
    def from_mapping(cls, item: dict[str, Any] | "FeedbackMatchRecord") -> "FeedbackMatchRecord":
        if isinstance(item, FeedbackMatchRecord):
            return item
        keypoint_xy = _xy_value(item.get("keypoint_xy", item.get("keypoint")))
        pnp_success = _bool_value(item.get("pnp_success"), default=False)
        pose_success = _bool_value(item.get("pose_success"), default=pnp_success)
        pose_error_t_cm = _float_or_none(item.get("pose_error_t_cm"))
        query_sparse_te_cm = _float_or_none(item.get("query_sparse_te_cm", pose_error_t_cm))
        return cls(
            scene=str(item.get("scene", "")),
            split_name=str(item.get("split_name", "")),
            query_id=str(item.get("query_id", "")),
            image_id=str(item.get("image_id", "")),
            source_view_id=str(item.get("source_view_id", "")),
            pose_source=str(item.get("pose_source", "")),
            source_role=str(item.get("source_role", item.get("pose_source", ""))),
            keypoint_id=str(item.get("keypoint_id", "")),
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
            pose_error_t_cm=pose_error_t_cm,
            query_sparse_te_cm=query_sparse_te_cm,
            pose_error_r_deg=_float_or_none(item.get("pose_error_r_deg")),
            pnp_success=pnp_success,
            pose_success=pose_success,
            dense_refine_success=_bool_value(item.get("dense_refine_success"), default=False),
            dense_transition=str(item.get("dense_transition", "")),
            dense_delta_te_cm=_float_or_none(item.get("dense_delta_te_cm")),
            jacobian_info_trace=_float_or_none(item.get("jacobian_info_trace")),
            jacobian_info_logdet_proxy=_float_or_none(item.get("jacobian_info_logdet_proxy")),
            query_xy_norm=_vector_value(item.get("query_xy_norm"), length=2),  # type: ignore[arg-type]
            bearing=_vector_value(item.get("bearing"), length=3),  # type: ignore[arg-type]
            camera_xyz=_vector_value(item.get("camera_xyz"), length=3),  # type: ignore[arg-type]
            depth_m=_float_or_none(item.get("depth_m")),
            descriptor_margin=_float_or_none(item.get("descriptor_margin")),
            local_geometry_score=_float_or_none(item.get("local_geometry_score")),
            ray_artifact_score=_float_or_none(item.get("ray_artifact_score")),
            ray_entropy=_float_or_none(item.get("ray_entropy")),
            image_cell=_int_pair_value(item.get("image_cell")),
            depth_bin=None if item.get("depth_bin") is None else _int_value(item.get("depth_bin"), default=-1),
        )

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["keypoint_xy"] = list(self.keypoint_xy)
        out["landmark_id"] = self.matched_landmark_id
        out["gaussian_id"] = self.matched_gaussian_id
        if self.query_xy_norm is not None:
            out["query_xy_norm"] = list(self.query_xy_norm)
        if self.bearing is not None:
            out["bearing"] = list(self.bearing)
        if self.camera_xyz is not None:
            out["camera_xyz"] = list(self.camera_xyz)
        if self.image_cell is not None:
            out["image_cell"] = list(self.image_cell)
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
