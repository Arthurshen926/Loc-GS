from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from loc_gs.sparse.audit import reject_test_split


def build_render_teacher_jobs(
    episodes: Sequence[Mapping[str, Any]],
    *,
    scene: str,
    split_name: str,
    require_render_ready: bool = False,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    split = reject_test_split(split_name, purpose="internal render teacher jobs")
    jobs: list[dict[str, object]] = []
    missing_render = 0
    for idx, episode in enumerate(episodes):
        row_split = reject_test_split(str(episode.get("split_name") or split), purpose="internal render teacher job row")
        if row_split != split:
            raise ValueError(f"render teacher job split mismatch at row {idx}: expected {split}, got {row_split}")
        row_scene = str(episode.get("scene") or scene)
        if row_scene != str(scene):
            raise ValueError(f"render teacher job scene mismatch at row {idx}: expected {scene}, got {row_scene}")
        render_ready = bool(episode.get("render_ready"))
        if not render_ready:
            missing_render += 1
            if require_render_ready:
                raise ValueError(f"missing rendered RGB for teacher job: {episode.get('synthetic_query_id')}")
            continue
        synthetic_query_id = str(episode.get("synthetic_query_id") or "")
        if not synthetic_query_id:
            raise ValueError(f"online episode row {idx} is missing synthetic_query_id")
        jobs.append(
            {
                "schema_version": "internal_render_teacher_job_v1",
                "scene": str(scene),
                "split_name": split,
                "job_id": f"render_teacher_job_{len(jobs):06d}",
                "episode_id": str(episode.get("episode_id") or ""),
                "synthetic_query_id": synthetic_query_id,
                "source_image_id": str(episode.get("source_image_id") or ""),
                "render_rgb_path": str(episode.get("render_rgb_path") or ""),
                "candidate_keypoint_count": int(episode.get("candidate_keypoint_count") or 0),
                "sparse_teacher_stage": "internal_sparse_pnp_trace",
                "dense_teacher_stage": "internal_stdloc_style_dense_refinement_trace",
                "teacher_signal": "solver_feedback_dense_delta",
                "inference_usage": "training_teacher_only",
                "dense_teacher_enabled": True,
                "dense_inference_enabled": False,
                "external_runtime_dependency": "forbidden",
            }
        )
    summary = {
        "schema_version": "internal_render_teacher_job_summary_v1",
        "scene": str(scene),
        "split_name": split,
        "episode_count": int(len(episodes)),
        "teacher_job_count": int(len(jobs)),
        "missing_render_episode_count": int(missing_render),
        "require_render_ready": bool(require_render_ready),
        "dense_teacher_enabled": True,
        "dense_inference_enabled": False,
        "external_runtime_dependency": "forbidden",
    }
    return summary, jobs


def load_online_episode_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"online episode row {line_number} must be a JSON object")
        rows.append(row)
    return rows
