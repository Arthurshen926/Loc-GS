from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from loc_gs.simulation.query_sampler import SimulatedQuerySpec
from loc_gs.sparse.artifact_adapter import CachedCandidateArtifact
from loc_gs.sparse.audit import reject_test_split
from loc_gs.sparse.rerank import summarize_candidate_availability
from loc_gs.teacher.distillation_artifact import SolverFeedbackRow


@dataclass(frozen=True)
class OnlineEpisodeConfig:
    max_keypoints_per_episode: int | None = None
    require_rendered_rgb: bool = False


@dataclass(frozen=True)
class OnlineSparseDenseEpisode:
    scene: str
    split_name: str
    episode_id: str
    synthetic_query_id: str
    source_image_id: str
    render_engine: str
    candidate_keypoint_count: int
    source_candidate_keypoint_count: int
    top1_correct: int
    topk_available: int
    oracle_gap: int
    feedback_matched: bool
    dense_helped: bool
    distill_weight: float
    render_rgb_path: str | None = None
    render_ready: bool = False

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": "internal_online_sparse_dense_episode_v1",
            **asdict(self),
        }


class OnlineEpisodeList(list[OnlineSparseDenseEpisode]):
    def __init__(self, values: Sequence[OnlineSparseDenseEpisode], *, missing_candidate_count: int) -> None:
        super().__init__(values)
        self.missing_candidate_count = int(missing_candidate_count)


@dataclass(frozen=True)
class OnlineTeacherObservationRow:
    scene: str
    split_name: str
    synthetic_query_id: str
    source_image_id: str
    keypoint_id: str
    query_yx: tuple[float, float]
    landmark_ids: tuple[int, ...]
    candidate_scores: tuple[float, ...]
    candidate_mask: tuple[bool, ...]
    geometric_correct: tuple[bool, ...]
    dense_consistent: tuple[bool, ...]
    sparse_inlier: tuple[bool, ...]
    reprojection_error_px: tuple[float, ...]
    solver_weight: tuple[float, ...]
    label_roles: tuple[str, ...]
    query_desc: tuple[float, ...] | None = None
    landmark_desc: tuple[tuple[float, ...], ...] | None = None
    margin: tuple[float, ...] | None = None
    query_score: tuple[float, ...] | None = None
    landmark_prior: tuple[float, ...] | None = None
    dense_helped: bool = False
    distill_weight: float = 0.0
    render_rgb_path: str | None = None
    render_ready: bool = False
    render_engine: str = "3dgs"

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any], *, row_index: int) -> "OnlineTeacherObservationRow":
        split = reject_test_split(str(row.get("split_name") or "unknown"), purpose="online teacher observation")
        scene = str(row.get("scene") or "unknown")
        synthetic_query_id = str(row.get("synthetic_query_id") or "")
        if not synthetic_query_id:
            raise ValueError(f"online teacher observation row {row_index} is missing synthetic_query_id")
        keypoint_id = str(row.get("keypoint_id") or f"kp{row_index:06d}")
        query_yx = _float_tuple(row.get("query_yx"), name="query_yx", row_index=row_index)
        if len(query_yx) != 2:
            raise ValueError(f"online teacher observation row {row_index} query_yx must have length 2")
        landmark_ids = tuple(int(value) for value in _required_sequence(row, "landmark_ids", row_index=row_index))
        topk = len(landmark_ids)
        if topk <= 0:
            raise ValueError(f"online teacher observation row {row_index} must contain at least one landmark")
        scores = _float_tuple(row.get("candidate_scores"), name="candidate_scores", row_index=row_index, expected=topk)
        mask = _bool_tuple(row.get("candidate_mask", [True] * topk), name="candidate_mask", row_index=row_index, expected=topk)
        geometric = _bool_tuple(
            row.get("geometric_correct", row.get("candidate_geometric_correct", [False] * topk)),
            name="geometric_correct",
            row_index=row_index,
            expected=topk,
        )
        dense = _bool_tuple(
            row.get("dense_consistent", row.get("candidate_dense_consistent", geometric)),
            name="dense_consistent",
            row_index=row_index,
            expected=topk,
        )
        sparse = _bool_tuple(
            row.get("sparse_inlier", row.get("candidate_sparse_inlier", geometric)),
            name="sparse_inlier",
            row_index=row_index,
            expected=topk,
        )
        reprojection = _float_tuple(
            row.get("reprojection_error_px", row.get("reprojection_error", [0.0] * topk)),
            name="reprojection_error_px",
            row_index=row_index,
            expected=topk,
        )
        solver_weight = _float_tuple(
            row.get("solver_weight", row.get("candidate_solver_weight", [1.0] * topk)),
            name="solver_weight",
            row_index=row_index,
            expected=topk,
        )
        roles = tuple(
            str(value)
            for value in _sequence_with_default(
                row.get("label_roles", row.get("candidate_label_roles")),
                ["positive_inlier" if value else "neutral" for value in geometric],
                row_index=row_index,
                expected=topk,
                name="label_roles",
            )
        )
        query_desc = None
        if row.get("query_desc") is not None:
            query_desc = _float_tuple(row.get("query_desc"), name="query_desc", row_index=row_index)
        landmark_desc = None
        if row.get("landmark_desc") is not None:
            raw_desc = _required_sequence(row, "landmark_desc", row_index=row_index)
            if len(raw_desc) != topk:
                raise ValueError(f"online teacher observation row {row_index} landmark_desc length must match topk")
            landmark_desc = tuple(
                _float_tuple(desc, name="landmark_desc", row_index=row_index)
                for desc in raw_desc
            )
        return cls(
            scene=scene,
            split_name=split,
            synthetic_query_id=synthetic_query_id,
            source_image_id=str(row.get("source_image_id") or ""),
            keypoint_id=keypoint_id,
            query_yx=(float(query_yx[0]), float(query_yx[1])),
            landmark_ids=landmark_ids,
            candidate_scores=scores,
            candidate_mask=mask,
            geometric_correct=geometric,
            dense_consistent=dense,
            sparse_inlier=sparse,
            reprojection_error_px=reprojection,
            solver_weight=solver_weight,
            label_roles=roles,
            query_desc=query_desc,
            landmark_desc=landmark_desc,
            margin=_optional_float_tuple(row.get("margin"), row_index=row_index, expected=topk, name="margin"),
            query_score=_optional_float_tuple(row.get("query_score"), row_index=row_index, expected=topk, name="query_score"),
            landmark_prior=_optional_float_tuple(
                row.get("landmark_prior"),
                row_index=row_index,
                expected=topk,
                name="landmark_prior",
            ),
            dense_helped=bool(row.get("dense_helped", False)),
            distill_weight=float(row.get("distill_weight", 0.0)),
            render_rgb_path=None if not row.get("render_rgb_path") else str(row.get("render_rgb_path")),
            render_ready=bool(row.get("render_ready", False)),
            render_engine=str(row.get("render_engine") or "3dgs"),
        )

    @property
    def label_index(self) -> int:
        for idx, value in enumerate(self.geometric_correct):
            if bool(value):
                return int(idx)
        return -1


def build_online_sparse_dense_episodes(
    specs: Sequence[SimulatedQuerySpec],
    artifact: CachedCandidateArtifact,
    feedback_rows: Sequence[SolverFeedbackRow],
    *,
    cfg: OnlineEpisodeConfig | None = None,
    render_records: Sequence[Mapping[str, Any]] | None = None,
) -> OnlineEpisodeList:
    if cfg is None:
        cfg = OnlineEpisodeConfig()
    artifact_split = reject_test_split(artifact.split_name, purpose="online sparse-dense episode generation")
    batches_by_query = {batch.query_id: batch for batch in artifact.batches}
    feedback_by_query = {row.query_id: row for row in feedback_rows}
    render_by_query = {
        str(record.get("synthetic_query_id")): record
        for record in (render_records or [])
        if record.get("synthetic_query_id")
    }
    for feedback in feedback_rows:
        reject_test_split(feedback.split_name, purpose="online sparse-dense episode generation")
        if feedback.scene not in {"unknown", artifact.scene}:
            raise ValueError(f"feedback scene mismatch: expected {artifact.scene}, got {feedback.scene}")

    episodes: list[OnlineSparseDenseEpisode] = []
    missing_candidate_count = 0
    for idx, spec in enumerate(specs):
        split = reject_test_split(spec.split_name, purpose="online sparse-dense episode generation")
        if spec.scene != artifact.scene:
            raise ValueError(f"simulation scene mismatch: expected {artifact.scene}, got {spec.scene}")
        batch = batches_by_query.get(spec.source_image_id)
        if batch is None:
            missing_candidate_count += 1
            continue
        limit = batch.keypoint_count
        if cfg.max_keypoints_per_episode is not None:
            limit = min(limit, max(0, int(cfg.max_keypoints_per_episode)))
        correct_rows = (batch.candidate_geometric_correct or [])[:limit]
        availability = summarize_candidate_availability(
            [[{"geometric_correct": bool(value)} for value in row] for row in correct_rows]
        )
        feedback = feedback_by_query.get(spec.source_image_id)
        render_record = render_by_query.get(spec.synthetic_query_id)
        render_rgb_path = None if render_record is None else str(render_record.get("rgb_path") or "")
        render_ready = bool(render_record is not None and render_record.get("rgb_exists"))
        if bool(cfg.require_rendered_rgb) and not render_ready:
            raise ValueError(f"missing rendered RGB for online episode: {spec.synthetic_query_id}")
        episodes.append(
            OnlineSparseDenseEpisode(
                scene=artifact.scene,
                split_name=artifact_split if artifact_split != "unknown" else split,
                episode_id=f"episode/{idx:06d}",
                synthetic_query_id=spec.synthetic_query_id,
                source_image_id=spec.source_image_id,
                render_engine=spec.render_engine,
                candidate_keypoint_count=int(limit),
                source_candidate_keypoint_count=int(batch.keypoint_count),
                top1_correct=int(availability["top1_correct"]),
                topk_available=int(availability["topk_available"]),
                oracle_gap=int(availability["oracle_gap"]),
                feedback_matched=bool(feedback is not None),
                dense_helped=bool(feedback.dense_helped) if feedback is not None else False,
                distill_weight=float(feedback.distill_weight) if feedback is not None else 0.0,
                render_rgb_path=render_rgb_path or None,
                render_ready=render_ready,
            )
        )
    return OnlineEpisodeList(episodes, missing_candidate_count=missing_candidate_count)


def summarize_online_sparse_dense_episodes(episodes: Sequence[OnlineSparseDenseEpisode]) -> dict[str, object]:
    missing = int(getattr(episodes, "missing_candidate_count", 0))
    return {
        "schema_version": "internal_online_sparse_dense_episode_summary_v1",
        "online_episode_count": int(len(episodes)),
        "missing_candidate_count": missing,
        "candidate_keypoint_count": int(sum(episode.candidate_keypoint_count for episode in episodes)),
        "source_image_count": int(len({episode.source_image_id for episode in episodes})),
        "dense_helped_episode_count": int(sum(1 for episode in episodes if episode.dense_helped)),
        "feedback_matched_episode_count": int(sum(1 for episode in episodes if episode.feedback_matched)),
        "render_ready_episode_count": int(sum(1 for episode in episodes if episode.render_ready)),
        "missing_render_episode_count": int(sum(1 for episode in episodes if not episode.render_ready)),
        "top1_correct": int(sum(episode.top1_correct for episode in episodes)),
        "topk_available": int(sum(episode.topk_available for episode in episodes)),
        "oracle_gap": int(sum(episode.oracle_gap for episode in episodes)),
        "render_engine": "3dgs",
    }


def build_online_candidate_payload(
    payload: Mapping[str, Any],
    episodes: Sequence[OnlineSparseDenseEpisode],
) -> dict[str, Any]:
    image_ids = _as_list(_required(payload, "image_id"))
    row_count = len(image_ids)
    rows_by_image: dict[str, list[int]] = {}
    for row_idx, image_id in enumerate(image_ids):
        rows_by_image.setdefault(str(image_id), []).append(row_idx)

    selected_indices: list[int] = []
    episode_ids: list[str] = []
    synthetic_query_ids: list[str] = []
    for episode in episodes:
        source_rows = rows_by_image.get(episode.source_image_id, [])
        for row_idx in source_rows[: int(episode.candidate_keypoint_count)]:
            selected_indices.append(row_idx)
            episode_ids.append(episode.episode_id)
            synthetic_query_ids.append(episode.synthetic_query_id)
    if not selected_indices:
        raise ValueError("online episodes did not select any candidate rows")

    selected = torch.as_tensor(selected_indices, dtype=torch.long)
    out: dict[str, Any] = {}
    for key, value in payload.items():
        out[key] = _select_rows(value, selected, selected_indices, row_count)
    out["online_episode_id"] = episode_ids
    out["synthetic_query_id"] = synthetic_query_ids
    metadata = dict(out.get("metadata", {}) if isinstance(out.get("metadata"), Mapping) else {})
    metadata.update(
        {
            "online_episode_count": int(len(episodes)),
            "online_candidate_row_count": int(len(selected_indices)),
            "training_source": "online_3dgs_sparse_dense_episode",
        }
    )
    out["metadata"] = metadata
    return out


def load_online_teacher_observation_rows(path: str | Path) -> list[OnlineTeacherObservationRow]:
    rows: list[OnlineTeacherObservationRow] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, Mapping):
            raise ValueError(f"online teacher observation row {line_number} must be a JSON object")
        rows.append(OnlineTeacherObservationRow.from_mapping(row, row_index=line_number))
    return rows


def build_online_candidate_payload_from_observations(
    observations: Sequence[OnlineTeacherObservationRow],
    *,
    scene: str,
    split_name: str,
) -> tuple[dict[str, Any], dict[str, object]]:
    split = reject_test_split(split_name, purpose="direct online teacher observation training")
    if not observations:
        raise ValueError("at least one online teacher observation is required")
    topk_values = {len(row.landmark_ids) for row in observations}
    if len(topk_values) != 1:
        raise ValueError(f"online teacher observations must have a fixed topk, got {sorted(topk_values)}")
    topk = int(next(iter(topk_values)))
    for row_idx, row in enumerate(observations):
        row_split = reject_test_split(row.split_name, purpose="direct online teacher observation row")
        if row_split != split:
            raise ValueError(f"online teacher observation split mismatch at row {row_idx}: expected {split}, got {row_split}")
        if row.scene != str(scene):
            raise ValueError(f"online teacher observation scene mismatch at row {row_idx}: expected {scene}, got {row.scene}")
    all_query_desc = all(row.query_desc is not None for row in observations)
    all_landmark_desc = all(row.landmark_desc is not None for row in observations)
    payload: dict[str, Any] = {
        "metadata": {
            "format": "listwise",
            "scene": str(scene),
            "split_name": split,
            "source_split_name": split,
            "topk": int(topk),
            "training_source": "online_3dgs_student_teacher_observation_stream",
            "candidate_binding_mode": "direct_online_teacher_observations",
            "source_candidate_reuse_enabled": False,
            "dense_teacher_enabled": True,
            "dense_inference_enabled": False,
            "external_runtime_dependency": "forbidden",
            "split_audit": {
                "schema_version": "internal_split_audit_v1",
                "audit_status": "passed",
                "split_name": split,
                "official_test_used": False,
                "test_split_used": False,
            },
        },
        "query_yx": torch.tensor([list(row.query_yx) for row in observations], dtype=torch.float32),
        "landmark_id": torch.tensor([list(row.landmark_ids) for row in observations], dtype=torch.int64),
        "cosine": torch.tensor([list(row.candidate_scores) for row in observations], dtype=torch.float32),
        "label": torch.tensor([row.label_index for row in observations], dtype=torch.int64),
        "candidate_mask": torch.tensor([list(row.candidate_mask) for row in observations], dtype=torch.bool),
        "dense_consistent": torch.tensor([list(row.dense_consistent) for row in observations], dtype=torch.bool),
        "sparse_inlier": torch.tensor([list(row.sparse_inlier) for row in observations], dtype=torch.bool),
        "reprojection_error": torch.tensor([list(row.reprojection_error_px) for row in observations], dtype=torch.float32),
        "solver_weight": torch.tensor([list(row.solver_weight) for row in observations], dtype=torch.float32),
        "label_roles": [list(row.label_roles) for row in observations],
        "query_id": [f"{row.synthetic_query_id}::{row.keypoint_id}" for row in observations],
        "image_id": [row.synthetic_query_id for row in observations],
        "keypoint_id": [row.keypoint_id for row in observations],
        "source_phase": [split] * len(observations),
        "source_image_id": [row.source_image_id for row in observations],
        "online_episode_id": [f"episode/{idx:06d}" for idx, _row in enumerate(observations)],
        "synthetic_query_id": [row.synthetic_query_id for row in observations],
    }
    if all_query_desc:
        payload["query_desc"] = torch.tensor([list(row.query_desc or ()) for row in observations], dtype=torch.float32)
    if all_landmark_desc:
        payload["landmark_desc"] = torch.tensor(
            [[list(desc) for desc in (row.landmark_desc or ())] for row in observations],
            dtype=torch.float32,
        )
    _add_optional_tensor(payload, observations, "margin", "margin")
    _add_optional_tensor(payload, observations, "query_score", "query_score")
    _add_optional_tensor(payload, observations, "landmark_prior", "landmark_prior")
    summary = summarize_online_teacher_observations(observations, scene=scene, split_name=split)
    return payload, summary


def summarize_online_teacher_observations(
    observations: Sequence[OnlineTeacherObservationRow],
    *,
    scene: str,
    split_name: str,
) -> dict[str, object]:
    split = reject_test_split(split_name, purpose="direct online teacher observation summary")
    topk_values = [len(row.landmark_ids) for row in observations]
    return {
        "schema_version": "internal_online_teacher_observation_summary_v1",
        "scene": str(scene),
        "split_name": split,
        "online_observation_count": int(len(observations)),
        "online_episode_count": int(len({row.synthetic_query_id for row in observations})),
        "synthetic_query_count": int(len({row.synthetic_query_id for row in observations})),
        "source_image_count": int(len({row.source_image_id for row in observations if row.source_image_id})),
        "candidate_keypoint_count": int(len(observations)),
        "topk": 0 if not topk_values else int(max(topk_values)),
        "dense_helped_episode_count": int(
            len({row.synthetic_query_id for row in observations if row.dense_helped})
        ),
        "render_ready_episode_count": int(
            len({row.synthetic_query_id for row in observations if row.render_ready})
        ),
        "missing_render_episode_count": int(
            len({row.synthetic_query_id for row in observations if not row.render_ready})
        ),
        "candidate_binding_mode": "direct_online_teacher_observations",
        "source_candidate_reuse_enabled": False,
        "render_engine": "3dgs",
    }


def build_online_episodes_from_observations(
    observations: Sequence[OnlineTeacherObservationRow],
    *,
    scene: str,
    split_name: str,
) -> OnlineEpisodeList:
    split = reject_test_split(split_name, purpose="direct online teacher observation episode export")
    by_query: dict[str, list[OnlineTeacherObservationRow]] = {}
    for row in observations:
        if row.scene != str(scene):
            raise ValueError(f"online teacher observation scene mismatch: expected {scene}, got {row.scene}")
        row_split = reject_test_split(row.split_name, purpose="direct online teacher observation episode export")
        if row_split != split:
            raise ValueError(f"online teacher observation split mismatch: expected {split}, got {row_split}")
        by_query.setdefault(row.synthetic_query_id, []).append(row)
    episodes: list[OnlineSparseDenseEpisode] = []
    for idx, (synthetic_query_id, rows) in enumerate(by_query.items()):
        top1 = sum(1 for row in rows if row.geometric_correct and bool(row.geometric_correct[0]))
        topk = sum(1 for row in rows if any(bool(value) for value in row.geometric_correct))
        episodes.append(
            OnlineSparseDenseEpisode(
                scene=str(scene),
                split_name=split,
                episode_id=f"episode/{idx:06d}",
                synthetic_query_id=synthetic_query_id,
                source_image_id=str(rows[0].source_image_id),
                render_engine=str(rows[0].render_engine),
                candidate_keypoint_count=int(len(rows)),
                source_candidate_keypoint_count=0,
                top1_correct=int(top1),
                topk_available=int(topk),
                oracle_gap=int(topk - top1),
                feedback_matched=True,
                dense_helped=any(row.dense_helped for row in rows),
                distill_weight=max(float(row.distill_weight) for row in rows),
                render_rgb_path=rows[0].render_rgb_path,
                render_ready=all(row.render_ready for row in rows),
            )
        )
    return OnlineEpisodeList(episodes, missing_candidate_count=0)


def _required(payload: Mapping[str, Any], key: str) -> Any:
    if key not in payload:
        raise ValueError(f"candidate artifact is missing required field: {key}")
    return payload[key]


def _as_list(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _select_rows(value: Any, selected: torch.Tensor, selected_indices: Sequence[int], row_count: int) -> Any:
    if hasattr(value, "detach") and getattr(value, "shape", (None,))[0] == row_count:
        return value.detach().cpu().index_select(0, selected)
    if isinstance(value, list) and len(value) == row_count:
        return [value[idx] for idx in selected_indices]
    return value


def _required_sequence(row: Mapping[str, Any], key: str, *, row_index: int) -> Sequence[Any]:
    value = row.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"online teacher observation row {row_index} {key} must be a sequence")
    return value


def _sequence_with_default(
    value: Any,
    default: Sequence[Any],
    *,
    row_index: int,
    expected: int,
    name: str,
) -> Sequence[Any]:
    if value is None:
        value = default
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"online teacher observation row {row_index} {name} must be a sequence")
    if len(value) != int(expected):
        raise ValueError(f"online teacher observation row {row_index} {name} must have length {expected}")
    return value


def _float_tuple(
    value: Any,
    *,
    name: str,
    row_index: int,
    expected: int | None = None,
) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"online teacher observation row {row_index} {name} must be a sequence")
    if expected is not None and len(value) != int(expected):
        raise ValueError(f"online teacher observation row {row_index} {name} must have length {expected}")
    return tuple(float(item) for item in value)


def _optional_float_tuple(
    value: Any,
    *,
    row_index: int,
    expected: int,
    name: str,
) -> tuple[float, ...] | None:
    if value is None:
        return None
    return _float_tuple(value, name=name, row_index=row_index, expected=expected)


def _bool_tuple(
    value: Any,
    *,
    name: str,
    row_index: int,
    expected: int,
) -> tuple[bool, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"online teacher observation row {row_index} {name} must be a sequence")
    if len(value) != int(expected):
        raise ValueError(f"online teacher observation row {row_index} {name} must have length {expected}")
    return tuple(bool(item) for item in value)


def _add_optional_tensor(
    payload: dict[str, Any],
    observations: Sequence[OnlineTeacherObservationRow],
    attr: str,
    key: str,
) -> None:
    values = [getattr(row, attr) for row in observations]
    if not all(value is not None for value in values):
        return
    payload[key] = torch.tensor([list(value or ()) for value in values], dtype=torch.float32)
