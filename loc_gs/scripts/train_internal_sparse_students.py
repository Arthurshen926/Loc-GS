#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import torch

from loc_gs.core.camera import load_camera_records
from loc_gs.simulation.query_sampler import SimulationSamplerConfig, sample_simulated_queries
from loc_gs.sparse.artifact_adapter import load_listwise_candidate_artifact
from loc_gs.sparse.audit import reject_test_split
from loc_gs.teacher.distillation_artifact import (
    DistillationArtifactConfig,
    build_distillation_payload,
    load_solver_feedback_label_rows,
)
from loc_gs.teacher.online_episode import (
    OnlineEpisodeConfig,
    build_online_candidate_payload,
    build_online_sparse_dense_episodes,
    summarize_online_sparse_dense_episodes,
)
from loc_gs.training.sparse_candidate_scorer import CandidateScorerConfig, train_candidate_scorer


def _git_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), text=True).strip()
    except Exception as exc:
        return f"unknown: {exc}"


def _write_jsonl(path: str | Path, rows: Iterable[Mapping[str, object]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train internal sparse student modules from online sparse-dense episodes.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--cameras_json", type=Path, required=True)
    parser.add_argument("--candidate_artifact", type=Path, required=True)
    parser.add_argument("--solver_feedback_labels", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sample_count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--translation_std_m", type=float, default=0.25)
    parser.add_argument("--yaw_std_deg", type=float, default=5.0)
    parser.add_argument("--pitch_std_deg", type=float, default=2.0)
    parser.add_argument("--roll_std_deg", type=float, default=2.0)
    parser.add_argument("--max_keypoints_per_episode", type=int, default=None)
    parser.add_argument("--dense_consistency_reprojection_px", type=float, default=4.0)
    parser.add_argument("--sparse_inlier_reprojection_px", type=float, default=8.0)
    parser.add_argument("--hard_negative_reprojection_px", type=float, default=8.0)
    parser.add_argument("--max_solver_weight", type=float, default=4.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--rank_feature_scale", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    split = reject_test_split(str(args.split_name), purpose="online sparse student training")
    sim_cfg = SimulationSamplerConfig(
        sample_count=int(args.sample_count),
        seed=int(args.seed),
        translation_std_m=float(args.translation_std_m),
        yaw_std_deg=float(args.yaw_std_deg),
        pitch_std_deg=float(args.pitch_std_deg),
        roll_std_deg=float(args.roll_std_deg),
    )
    feedback_rows = load_solver_feedback_label_rows(args.solver_feedback_labels)
    artifact = load_listwise_candidate_artifact(args.candidate_artifact)
    camera_records = load_camera_records(args.cameras_json)
    candidate_source_ids = {batch.query_id for batch in artifact.batches}
    camera_records = {image_id: record for image_id, record in camera_records.items() if image_id in candidate_source_ids}
    if not camera_records:
        raise ValueError("no cameras overlap candidate artifact source images")
    specs = sample_simulated_queries(camera_records, scene=str(args.scene), split_name=split, cfg=sim_cfg)
    episodes = build_online_sparse_dense_episodes(
        specs,
        artifact,
        feedback_rows,
        cfg=OnlineEpisodeConfig(
            max_keypoints_per_episode=None
            if args.max_keypoints_per_episode is None
            else int(args.max_keypoints_per_episode)
        ),
    )

    payload = torch.load(args.candidate_artifact, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"candidate artifact must contain a dict payload: {args.candidate_artifact}")
    distilled_payload, distillation_summary = build_distillation_payload(
        payload,
        feedback_rows,
        scene=str(args.scene),
        split_name=artifact.split_name,
        cfg=DistillationArtifactConfig(
            dense_consistency_reprojection_px=float(args.dense_consistency_reprojection_px),
            sparse_inlier_reprojection_px=float(args.sparse_inlier_reprojection_px),
            hard_negative_reprojection_px=float(args.hard_negative_reprojection_px),
            max_solver_weight=float(args.max_solver_weight),
        ),
    )
    online_payload = build_online_candidate_payload(distilled_payload, episodes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    online_artifact_path = args.output_dir / "online_distilled_candidates.pt"
    torch.save(online_payload, online_artifact_path)
    online_artifact = load_listwise_candidate_artifact(online_artifact_path)
    scorer_cfg = CandidateScorerConfig(
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        rank_feature_scale=float(args.rank_feature_scale),
    )
    model, scorer_summary = train_candidate_scorer(online_artifact, scorer_cfg)

    episode_summary = summarize_online_sparse_dense_episodes(episodes)
    summary = {
        "schema_version": "internal_online_sparse_student_training_summary_v1",
        "scene": str(args.scene),
        "split_name": split,
        "student_modules": ["correspondence_scorer"],
        **episode_summary,
        "distillation": distillation_summary,
        "candidate_scorer": scorer_summary,
    }
    manifest = {
        "schema_version": "internal_online_sparse_student_training_manifest_v1",
        "method": "sparse_dense_distilled_internal",
        "scene": str(args.scene),
        "split_name": split,
        "git_commit": _git_commit(),
        "command": [sys.executable, "-m", "loc_gs.scripts.train_internal_sparse_students", *(argv or sys.argv[1:])],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inference_stage": "online_sparse_student_training",
        "dense_teacher_enabled": True,
        "dense_inference_enabled": False,
        "external_runtime_dependency": "forbidden",
        "cameras_json": str(args.cameras_json),
        "candidate_artifact": str(args.candidate_artifact),
        "solver_feedback_labels": str(args.solver_feedback_labels),
        "online_distilled_candidate_artifact": str(online_artifact_path),
        "hyperparameters": {
            "sample_count": int(args.sample_count),
            "seed": int(args.seed),
            "translation_std_m": float(args.translation_std_m),
            "yaw_std_deg": float(args.yaw_std_deg),
            "pitch_std_deg": float(args.pitch_std_deg),
            "roll_std_deg": float(args.roll_std_deg),
            "max_keypoints_per_episode": args.max_keypoints_per_episode,
            "dense_consistency_reprojection_px": float(args.dense_consistency_reprojection_px),
            "sparse_inlier_reprojection_px": float(args.sparse_inlier_reprojection_px),
            "hard_negative_reprojection_px": float(args.hard_negative_reprojection_px),
            "max_solver_weight": float(args.max_solver_weight),
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "rank_feature_scale": float(args.rank_feature_scale),
            "feature_names": list(scorer_cfg.feature_names),
            "camera_sampling_source": "candidate_artifact_sources",
            "candidate_source_image_count": int(len(candidate_source_ids)),
            "sampled_camera_count": int(len(camera_records)),
        },
    }
    split_audit = artifact.metadata.get("split_audit")
    if not isinstance(split_audit, dict):
        split_audit = {"audit_status": "unknown", "reason": "candidate artifact did not include split_audit"}

    _write_jsonl(args.output_dir / "online_episodes.jsonl", [episode.to_json_dict() for episode in episodes])
    (args.output_dir / "model.json").write_text(
        json.dumps(model.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "split_audit.json").write_text(
        json.dumps(split_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
