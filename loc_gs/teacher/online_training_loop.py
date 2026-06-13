from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from loc_gs.simulation.query_sampler import SimulatedQuerySpec
from loc_gs.sparse.audit import reject_test_split
from loc_gs.teacher.online_episode import (
    OnlineTeacherObservationRow,
    summarize_online_teacher_observations,
)


@dataclass(frozen=True)
class Online3DGSTrainingLoop:
    render_query: Callable[[SimulatedQuerySpec], Mapping[str, Any]]
    run_sparse_student: Callable[[SimulatedQuerySpec, Mapping[str, Any]], Sequence[Mapping[str, Any]]]
    run_sparse_dense_teacher: Callable[
        [SimulatedQuerySpec, Mapping[str, Any], Sequence[Mapping[str, Any]]],
        Sequence[Mapping[str, Any]],
    ]


def run_online_3dgs_training_loop(
    specs: Sequence[SimulatedQuerySpec],
    loop: Online3DGSTrainingLoop,
    *,
    scene: str,
    split_name: str,
) -> tuple[list[OnlineTeacherObservationRow], dict[str, object]]:
    split = reject_test_split(split_name, purpose="online 3DGS sparse-dense training loop")
    observations: list[OnlineTeacherObservationRow] = []
    for spec_idx, spec in enumerate(specs):
        spec_split = reject_test_split(spec.split_name, purpose="online 3DGS sparse-dense training loop")
        if spec_split != split:
            raise ValueError(f"online 3DGS spec split mismatch at row {spec_idx}: expected {split}, got {spec_split}")
        if spec.scene != str(scene):
            raise ValueError(f"online 3DGS spec scene mismatch at row {spec_idx}: expected {scene}, got {spec.scene}")
        rendered = dict(loop.render_query(spec))
        sparse_candidates = list(loop.run_sparse_student(spec, rendered))
        teacher_rows = list(loop.run_sparse_dense_teacher(spec, rendered, sparse_candidates))
        for row_idx, row in enumerate(teacher_rows):
            merged = {
                **dict(row),
                "scene": str(scene),
                "split_name": split,
                "synthetic_query_id": spec.synthetic_query_id,
                "source_image_id": spec.source_image_id,
                "render_rgb_path": rendered.get("render_rgb_path", row.get("render_rgb_path")),
                "render_ready": bool(rendered.get("render_ready", row.get("render_ready", False))),
                "render_engine": str(rendered.get("render_engine", row.get("render_engine", "3dgs"))),
            }
            observations.append(
                OnlineTeacherObservationRow.from_mapping(
                    merged,
                    row_index=spec_idx * max(1, len(teacher_rows)) + row_idx + 1,
                )
            )
    summary = summarize_online_teacher_observations(observations, scene=scene, split_name=split)
    summary.update(
        {
            "schema_version": "internal_online_3dgs_training_loop_summary_v1",
            "online_loop_stage_order": ["render_query", "run_sparse_student", "run_sparse_dense_teacher"],
            "source_candidate_reuse_enabled": False,
        }
    )
    return observations, summary
