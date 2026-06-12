import json
from pathlib import Path

import pytest

from loc_gs.teacher.render_teacher_jobs import build_render_teacher_jobs
from loc_gs.scripts.build_internal_render_teacher_jobs import main


def _episode(idx: int, *, render_ready: bool) -> dict[str, object]:
    return {
        "schema_version": "internal_online_sparse_dense_episode_v1",
        "scene": "GreatCourt",
        "split_name": "train_selfmap",
        "episode_id": f"episode/{idx:06d}",
        "synthetic_query_id": f"sim/GreatCourt/train_selfmap/{idx:06d}",
        "source_image_id": f"seq1/frame{idx:05d}.png",
        "render_engine": "3dgs",
        "render_rgb_path": f"/renders/{idx:06d}.png",
        "render_ready": render_ready,
        "candidate_keypoint_count": 128,
    }


def test_render_teacher_jobs_emit_training_only_sparse_dense_work_items():
    summary, jobs = build_render_teacher_jobs(
        [_episode(0, render_ready=True), _episode(1, render_ready=False)],
        scene="GreatCourt",
        split_name="train_selfmap",
        require_render_ready=False,
    )

    assert summary["schema_version"] == "internal_render_teacher_job_summary_v1"
    assert summary["episode_count"] == 2
    assert summary["teacher_job_count"] == 1
    assert summary["missing_render_episode_count"] == 1
    assert jobs[0]["schema_version"] == "internal_render_teacher_job_v1"
    assert jobs[0]["synthetic_query_id"] == "sim/GreatCourt/train_selfmap/000000"
    assert jobs[0]["inference_usage"] == "training_teacher_only"
    assert jobs[0]["dense_teacher_enabled"] is True
    assert jobs[0]["dense_inference_enabled"] is False
    assert jobs[0]["external_runtime_dependency"] == "forbidden"


def test_render_teacher_jobs_can_require_render_ready():
    with pytest.raises(ValueError, match="missing rendered RGB"):
        build_render_teacher_jobs(
            [_episode(0, render_ready=False)],
            scene="GreatCourt",
            split_name="train_selfmap",
            require_render_ready=True,
        )


def test_render_teacher_jobs_reject_test_split():
    with pytest.raises(ValueError, match="test split"):
        build_render_teacher_jobs(
            [_episode(0, render_ready=True)],
            scene="GreatCourt",
            split_name="test",
        )


def test_render_teacher_jobs_cli_writes_manifest_and_jsonl(tmp_path: Path):
    episodes = tmp_path / "online_episodes.jsonl"
    episodes.write_text(
        "\n".join(json.dumps(row) for row in [_episode(0, render_ready=True), _episode(1, render_ready=False)])
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "teacher_jobs"

    rc = main(
        [
            "--scene",
            "GreatCourt",
            "--split_name",
            "train_selfmap",
            "--online_episodes",
            str(episodes),
            "--output_dir",
            str(out),
        ]
    )

    assert rc == 0
    summary = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    jobs = [json.loads(line) for line in (out / "teacher_jobs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert summary["teacher_job_count"] == 1
    assert summary["missing_render_episode_count"] == 1
    assert manifest["schema_version"] == "internal_render_teacher_job_manifest_v1"
    assert manifest["dense_teacher_enabled"] is True
    assert manifest["dense_inference_enabled"] is False
    assert jobs[0]["episode_id"] == "episode/000000"
