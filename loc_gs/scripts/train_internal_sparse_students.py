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
from loc_gs.simulation.render_manifest import load_simulation_plan_rows
from loc_gs.simulation.query_sampler import SimulationSamplerConfig, sample_simulated_queries
from loc_gs.sparse.artifact_adapter import load_listwise_candidate_artifact
from loc_gs.sparse.audit import reject_test_split
from loc_gs.students.candidate_mlp_scorer import CandidateMLPScorerConfig, train_candidate_mlp_scorer
from loc_gs.students.descriptor_fusion import DescriptorFusionConfig, train_descriptor_fusion_from_payload
from loc_gs.students.detector_student import DetectorStudentConfig, train_detector_student
from loc_gs.students.landmark_selector import LandmarkSelectorConfig, train_landmark_selector
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
    parser.add_argument("--render_manifest", type=Path, default=None)
    parser.add_argument("--require_rendered_rgb", action="store_true")
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
    parser.add_argument("--candidate_mlp_hidden_dim", type=int, default=32)
    parser.add_argument("--candidate_mlp_learning_rate", type=float, default=0.03)
    parser.add_argument("--candidate_mlp_listwise_loss_weight", type=float, default=1.0)
    parser.add_argument("--landmark_conflict_penalty", type=float, default=0.1)
    parser.add_argument("--descriptor_trust_region", type=float, default=0.25)
    parser.add_argument("--detector_grid_size", type=int, default=8)
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
    render_records = load_simulation_plan_rows(args.render_manifest) if args.render_manifest is not None else None
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
            else int(args.max_keypoints_per_episode),
            require_rendered_rgb=bool(args.require_rendered_rgb),
        ),
        render_records=render_records,
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
    candidate_mlp_model, candidate_mlp_summary = train_candidate_mlp_scorer(
        online_artifact,
        CandidateMLPScorerConfig(
            epochs=int(args.epochs),
            learning_rate=float(args.candidate_mlp_learning_rate),
            hidden_dim=int(args.candidate_mlp_hidden_dim),
            seed=int(args.seed),
            listwise_loss_weight=float(args.candidate_mlp_listwise_loss_weight),
            rank_feature_scale=float(args.rank_feature_scale),
        ),
    )
    selector_model, selector_summary = train_landmark_selector(
        online_artifact.batches,
        LandmarkSelectorConfig(conflict_penalty=float(args.landmark_conflict_penalty)),
    )
    descriptor_model, descriptor_summary = train_descriptor_fusion_from_payload(
        online_payload,
        DescriptorFusionConfig(trust_region=float(args.descriptor_trust_region)),
    )
    detector_model, detector_summary = train_detector_student(
        online_artifact.batches,
        DetectorStudentConfig(grid_size=int(args.detector_grid_size)),
    )

    episode_summary = summarize_online_sparse_dense_episodes(episodes)
    student_modules = [
        "correspondence_scorer",
        "candidate_mlp_scorer",
        "landmark_selector",
        "conflict_graph",
        "descriptor_fusion",
        "detector_student",
    ]
    summary = {
        "schema_version": "internal_online_sparse_student_training_summary_v1",
        "scene": str(args.scene),
        "split_name": split,
        "student_modules": student_modules,
        **episode_summary,
        "distillation": distillation_summary,
        "candidate_scorer": scorer_summary,
        "candidate_mlp_scorer": candidate_mlp_summary,
        "landmark_selector": selector_summary,
        "descriptor_fusion": descriptor_summary,
        "detector_student": detector_summary,
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
        "render_manifest": None if args.render_manifest is None else str(args.render_manifest),
        "online_distilled_candidate_artifact": str(online_artifact_path),
        "candidate_mlp_scorer": str(args.output_dir / "candidate_mlp_scorer.pt"),
        "hyperparameters": {
            "sample_count": int(args.sample_count),
            "seed": int(args.seed),
            "translation_std_m": float(args.translation_std_m),
            "yaw_std_deg": float(args.yaw_std_deg),
            "pitch_std_deg": float(args.pitch_std_deg),
            "roll_std_deg": float(args.roll_std_deg),
            "max_keypoints_per_episode": args.max_keypoints_per_episode,
            "require_rendered_rgb": bool(args.require_rendered_rgb),
            "dense_consistency_reprojection_px": float(args.dense_consistency_reprojection_px),
            "sparse_inlier_reprojection_px": float(args.sparse_inlier_reprojection_px),
            "hard_negative_reprojection_px": float(args.hard_negative_reprojection_px),
            "max_solver_weight": float(args.max_solver_weight),
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "rank_feature_scale": float(args.rank_feature_scale),
            "candidate_mlp_hidden_dim": int(args.candidate_mlp_hidden_dim),
            "candidate_mlp_learning_rate": float(args.candidate_mlp_learning_rate),
            "candidate_mlp_listwise_loss_weight": float(args.candidate_mlp_listwise_loss_weight),
            "landmark_conflict_penalty": float(args.landmark_conflict_penalty),
            "descriptor_trust_region": float(args.descriptor_trust_region),
            "detector_grid_size": int(args.detector_grid_size),
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
    torch.save(candidate_mlp_model.to_torch_dict(), args.output_dir / "candidate_mlp_scorer.pt")
    (args.output_dir / "landmark_selector.json").write_text(
        json.dumps(selector_model.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "conflict_graph.json").write_text(
        json.dumps(selector_model.to_conflict_graph_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "descriptor_fusion.json").write_text(
        json.dumps(descriptor_model.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "detector_student.json").write_text(
        json.dumps(detector_model.to_json_dict(), indent=2, sort_keys=True) + "\n",
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
