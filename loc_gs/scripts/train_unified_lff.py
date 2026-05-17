#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from loc_gs.localization.selfmap_episode_cache import build_gaussian_advantage_labels
from loc_gs.models.unified_lff import UnifiedLFFDescriptor


REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse_paths(items: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        paths.extend(Path(part) for part in str(item).split(",") if part)
    if not paths:
        raise ValueError("at least one episode cache is required")
    return paths


def _load_descriptor_bank(path: str | Path) -> torch.Tensor:
    descriptor_path = Path(path)
    if descriptor_path.suffix.lower() == ".ply":
        from loc_gs.stdloc_native.lff_export import _load_ply_loc_features

        return F.normalize(_load_ply_loc_features(descriptor_path).float(), p=2, dim=-1)
    payload: Any = torch.load(descriptor_path, map_location="cpu")
    if isinstance(payload, torch.Tensor):
        desc = payload
    elif isinstance(payload, dict):
        desc = None
        for key in ("descriptors", "base_descriptors", "landmark_desc", "features", "export_descriptors"):
            if key in payload:
                desc = torch.as_tensor(payload[key])
                break
        if desc is None:
            raise KeyError(f"{path} does not contain a descriptor bank")
    else:
        raise ValueError("descriptor bank must be a tensor or a dict containing descriptors")
    desc = torch.as_tensor(desc).float()
    if desc.dim() != 2:
        raise ValueError("descriptor bank must have shape [num_landmarks, descriptor_dim]")
    return F.normalize(desc, p=2, dim=-1)


def _first_positive_or_dustbin(pair_label: torch.Tensor) -> torch.Tensor:
    if pair_label.dim() != 2:
        raise ValueError("pair_label must have shape [N,K]")
    n, topk = int(pair_label.shape[0]), int(pair_label.shape[1])
    out = torch.full((n,), topk, dtype=torch.long)
    any_pos = pair_label.bool().any(dim=1)
    first = pair_label.float().argmax(dim=1).long()
    out[any_pos] = first[any_pos]
    return out


def _pair_label_from_listwise(labels: torch.Tensor, topk: int) -> torch.Tensor:
    label = labels.long().reshape(-1)
    out = torch.zeros(label.shape[0], int(topk), dtype=torch.bool)
    rows = torch.where(label < int(topk))[0]
    if rows.numel() > 0:
        out[rows, label[rows]] = True
    return out


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _git_status_text() -> str:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"git status unavailable: {exc}\n"
    return result.stdout


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _scene_from_cache_metadata(cache_metadata: list[dict[str, Any]]) -> str:
    scenes = sorted({str(item.get("scene", "")).strip() for item in cache_metadata if str(item.get("scene", "")).strip()})
    return scenes[0] if len(scenes) == 1 else "unknown"


def _split_from_cache_metadata(cache_metadata: list[dict[str, Any]]) -> str:
    audited = {
        str(item.get("feedback_bank_split_name", "")).strip()
        for item in cache_metadata
        if str(item.get("feedback_bank_split_name", "")).strip()
    }
    if len(audited) == 1:
        return next(iter(audited))
    phase_names: set[str] = set()
    for item in cache_metadata:
        phase_counts = item.get("phase_counts", {})
        if isinstance(phase_counts, dict):
            phase_names.update(str(key) for key, value in phase_counts.items() if int(value) > 0)
    if phase_names and phase_names <= {"train", "rendered"}:
        return "selfmap_train_rendered_unknown_ids"
    return "unknown"


def _training_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "loc_gs.scripts.train_unified_lff",
        "--base_descriptor_path",
        str(args.base_descriptor_path),
        "--episode_cache",
        *[str(path) for path in args.episode_cache],
        "--output_path",
        str(args.output_path),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--alpha_max",
        str(args.alpha_max),
        "--init_gate",
        str(args.init_gate),
        "--temperature",
        str(args.temperature),
        "--lambda_trust",
        str(args.lambda_trust),
        "--lambda_gate",
        str(args.lambda_gate),
        "--lambda_pair",
        str(args.lambda_pair),
        "--lambda_rank",
        str(args.lambda_rank),
        "--rank_margin",
        str(args.rank_margin),
        "--trust_l1_weight",
        str(args.trust_l1_weight),
        "--selector_bias_weight",
        str(args.selector_bias_weight),
        "--lambda_selector_listwise",
        str(args.lambda_selector_listwise),
        "--lambda_selector_gate",
        str(args.lambda_selector_gate),
        "--lambda_selector_hard_negative",
        str(args.lambda_selector_hard_negative),
        "--lambda_selector_budget",
        str(args.lambda_selector_budget),
        "--lambda_selector_coverage",
        str(args.lambda_selector_coverage),
        "--false_positive_score_threshold",
        str(args.false_positive_score_threshold),
        "--pose_target_weight",
        str(args.pose_target_weight),
        "--pose_target_reprojection_threshold_px",
        str(args.pose_target_reprojection_threshold_px),
        "--pose_target_score_threshold",
        str(args.pose_target_score_threshold),
        "--lambda_selector_pose_pair",
        str(args.lambda_selector_pose_pair),
        "--selector_pose_pair_margin",
        str(args.selector_pose_pair_margin),
        "--pose_pair_reprojection_threshold_px",
        str(args.pose_pair_reprojection_threshold_px),
        "--pose_pair_score_threshold",
        str(args.pose_pair_score_threshold),
        "--num_workers",
        str(args.num_workers),
        "--device",
        str(args.device),
    ]
    if bool(getattr(args, "selector_only", False)):
        command.append("--selector_only")
    if bool(getattr(args, "freeze_descriptor_residual", False)):
        command.append("--freeze_descriptor_residual")
    return command


def _command_text(command: list[str]) -> str:
    return " ".join(str(part) for part in command).rstrip() + "\n"


def _split_audit_from_training_cache(cache_metadata: list[dict[str, Any]]) -> dict[str, Any]:
    audits = [dict(item.get("split_audit", {})) for item in cache_metadata if isinstance(item.get("split_audit"), dict)]
    if audits and len(audits) == len(cache_metadata):
        statuses = {str(audit.get("audit_status", "unknown")) for audit in audits}
        if len(audits) == 1:
            return audits[0]
        if "failed" in statuses:
            status = "failed"
        elif "unknown" in statuses:
            status = "unknown"
        else:
            status = "passed"
        return {
            "audit_status": status,
            "checks": {
                "cache_audits": {
                    "status": status,
                    "per_cache": audits,
                }
            },
        }
    split = _split_from_cache_metadata(cache_metadata)
    return {
        "audit_status": "unknown",
        "checks": {
            "image_id_disjointness": {
                "status": "unknown",
                "reason": "episode cache metadata does not include complete source/test image id lists",
                "overlap": [],
            },
            "feedback_bank_split": {
                "status": "passed" if split != "unknown" and "test" not in split.lower() else "unknown",
                "split_name": split,
            },
            "quality_gate": {
                "status": "passed",
                "mode": "disabled",
                "per_query_branch_selection": False,
            },
        },
    }


def _write_training_audit_bundle(
    *,
    output_path: Path,
    args: argparse.Namespace,
    cache_paths: list[Path],
    cache_metadata: list[dict[str, Any]],
    history: list[dict[str, float]],
    selector_only: bool,
    residual_frozen: bool,
) -> None:
    root = output_path.parent
    scene = _scene_from_cache_metadata(cache_metadata)
    split = _split_from_cache_metadata(cache_metadata)
    command = _training_command(args)
    manifest = {
        "git_commit": _git_commit(),
        "timestamp_utc": _utc_timestamp(),
        "command": command,
        "scene": scene,
        "split": split,
        "checkpoint_path": str(output_path),
        "map_path": "",
        "data_roots": [],
        "episode_cache": [str(path) for path in cache_paths],
        "hyperparameters": {
            "selector_only": selector_only,
            "residual_frozen": residual_frozen,
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "alpha_max": float(args.alpha_max),
            "init_gate": float(args.init_gate),
            "temperature": float(args.temperature),
            "lambda_trust": float(args.lambda_trust),
            "lambda_gate": float(args.lambda_gate),
            "lambda_pair": float(args.lambda_pair),
            "lambda_rank": float(args.lambda_rank),
            "rank_margin": float(args.rank_margin),
            "trust_l1_weight": float(args.trust_l1_weight),
            "selector_bias_weight": float(args.selector_bias_weight),
            "lambda_selector_listwise": float(args.lambda_selector_listwise),
            "lambda_selector_gate": float(args.lambda_selector_gate),
            "lambda_selector_hard_negative": float(args.lambda_selector_hard_negative),
            "lambda_selector_budget": float(args.lambda_selector_budget),
            "lambda_selector_coverage": float(args.lambda_selector_coverage),
            "false_positive_score_threshold": float(args.false_positive_score_threshold),
            "pose_target_weight": float(args.pose_target_weight),
            "pose_target_reprojection_threshold_px": float(args.pose_target_reprojection_threshold_px),
            "pose_target_score_threshold": float(args.pose_target_score_threshold),
            "lambda_selector_pose_pair": float(args.lambda_selector_pose_pair),
            "selector_pose_pair_margin": float(args.selector_pose_pair_margin),
            "pose_pair_reprojection_threshold_px": float(args.pose_pair_reprojection_threshold_px),
            "pose_pair_score_threshold": float(args.pose_pair_score_threshold),
            "num_workers": int(args.num_workers),
        },
        "feedback": {
            "residual": not selector_only,
            "selector": True,
            "rho": False,
        },
        "single_path_deployment": True,
        "branch_selection": False,
        "diagnostic": split == "unknown" or split.endswith("unknown_ids"),
    }
    split_audit = _split_audit_from_training_cache(cache_metadata)
    metrics_summary = {
        "scene": scene,
        "split": split,
        "checkpoint_path": str(output_path),
        "history": history,
        "final": history[-1] if history else {},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (root / "command.txt").write_text(_command_text(command), encoding="utf-8")
    (root / "metrics_summary.json").write_text(
        json.dumps(metrics_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (root / "split_audit.json").write_text(json.dumps(split_audit, indent=2, sort_keys=True), encoding="utf-8")
    (root / "git_status.txt").write_text(_git_status_text(), encoding="utf-8")


def _candidate_field(
    payload: dict[str, Any],
    key: str,
    *,
    reference: torch.Tensor,
    default: float | None = None,
) -> torch.Tensor:
    if key in payload:
        tensor = torch.as_tensor(payload[key]).float()
    elif default is not None:
        tensor = torch.full(reference.shape, float(default), dtype=torch.float32)
    else:
        raise KeyError(key)
    if tensor.shape == reference.shape:
        return tensor
    if tensor.dim() == 1 and tensor.shape[0] == reference.shape[0]:
        return tensor[:, None].expand(reference.shape).clone()
    if tensor.numel() == 1:
        return tensor.reshape(1, 1).expand(reference.shape).clone()
    raise ValueError(f"{key} must have shape {tuple(reference.shape)} or [{reference.shape[0]}]")


def _pose_reliability_target_from_payload(
    payload: dict[str, Any],
    *,
    path: Path,
    landmark_ids: torch.Tensor,
    candidate_mask: torch.Tensor,
    candidate_cosine: torch.Tensor,
    num_landmarks: int,
    reprojection_threshold_px: float,
    score_threshold: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    error_key = "candidate_reprojection_error" if "candidate_reprojection_error" in payload else "reprojection_error"
    if error_key not in payload:
        raise KeyError(f"{path} is missing reprojection_error for pose-target selector training")
    threshold = float(reprojection_threshold_px)
    if threshold <= 0.0:
        metadata = payload.get("metadata", {})
        if isinstance(metadata, dict) and float(metadata.get("reprojection_threshold_px", 0.0)) > 0.0:
            threshold = float(metadata["reprojection_threshold_px"])
        else:
            raise ValueError("pose_target_reprojection_threshold_px must be positive")
    errors = _candidate_field(payload, error_key, reference=landmark_ids).float()
    query_score = _candidate_field(payload, "query_score", reference=landmark_ids, default=1.0).clamp_min(0.0)
    margin = _candidate_field(payload, "margin", reference=landmark_ids, default=1.0).clamp_min(0.0)
    finite = torch.isfinite(errors)
    valid = (
        candidate_mask
        & finite
        & (errors >= 0.0)
        & (errors <= threshold)
        & (candidate_cosine >= float(score_threshold))
    )
    closeness = (1.0 - (errors.clamp_min(0.0) / max(threshold, 1e-8))).clamp(0.0, 1.0)
    confidence = (closeness * query_score * margin).masked_fill(~valid, 0.0)
    signal = torch.zeros(int(num_landmarks), dtype=torch.float32)
    flat_valid = valid.reshape(-1)
    if bool(flat_valid.any()):
        signal.scatter_add_(0, landmark_ids.reshape(-1)[flat_valid], confidence.reshape(-1)[flat_valid])
    signal_max = float(signal.max().item()) if signal.numel() else 0.0
    normalized = signal / signal_max if signal_max > 0.0 else signal
    target = (0.5 + 0.5 * normalized).clamp(0.0, 1.0)
    summary = {
        "path": str(path),
        "reprojection_threshold_px": threshold,
        "score_threshold": float(score_threshold),
        "positive_pairs": int(flat_valid.sum().item()),
        "positive_landmarks": int((signal > 0.0).sum().item()),
        "signal_max": signal_max,
    }
    return target, summary


def _pose_pair_utility_from_payload(
    payload: dict[str, Any],
    *,
    path: Path,
    landmark_ids: torch.Tensor,
    candidate_mask: torch.Tensor,
    candidate_cosine: torch.Tensor,
    reprojection_threshold_px: float,
    score_threshold: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    error_key = "candidate_reprojection_error" if "candidate_reprojection_error" in payload else "reprojection_error"
    if error_key not in payload:
        raise KeyError(f"{path} is missing reprojection_error for selector pose-pair supervision")
    threshold = float(reprojection_threshold_px)
    if threshold <= 0.0:
        metadata = payload.get("metadata", {})
        if isinstance(metadata, dict) and float(metadata.get("reprojection_threshold_px", 0.0)) > 0.0:
            threshold = float(metadata["reprojection_threshold_px"])
        else:
            raise ValueError("pose_pair_reprojection_threshold_px must be positive or present in cache metadata")
    errors = _candidate_field(payload, error_key, reference=landmark_ids).float()
    query_score = _candidate_field(payload, "query_score", reference=landmark_ids, default=1.0).clamp_min(0.0)
    margin = _candidate_field(payload, "margin", reference=landmark_ids, default=1.0).clamp_min(0.0)
    finite = torch.isfinite(errors)
    valid = (
        candidate_mask
        & finite
        & (errors >= 0.0)
        & (errors <= threshold)
        & (candidate_cosine >= float(score_threshold))
    )
    closeness = (1.0 - (errors.clamp_min(0.0) / max(threshold, 1e-8))).clamp(0.0, 1.0)
    utility = (closeness * query_score * margin).masked_fill(~valid, 0.0).clamp(0.0, 1.0)
    positive_ids = landmark_ids[utility > 0.0]
    summary = {
        "path": str(path),
        "reprojection_threshold_px": threshold,
        "score_threshold": float(score_threshold),
        "positive_pairs": int((utility > 0.0).sum().item()),
        "positive_landmarks": int(torch.unique(positive_ids).numel()) if positive_ids.numel() else 0,
        "utility_max": float(utility.max().item()) if utility.numel() else 0.0,
    }
    return utility, summary


def load_unified_lff_training_tensors(
    base_descriptor_path: str | Path,
    episode_cache_paths: list[str | Path],
    *,
    false_positive_score_threshold: float | None = 0.0,
    pose_target_weight: float = 0.0,
    pose_target_reprojection_threshold_px: float = 8.0,
    pose_target_score_threshold: float = 0.0,
    pose_pair_reprojection_threshold_px: float = 0.0,
    pose_pair_score_threshold: float = 0.0,
) -> dict[str, Any]:
    loaded_payloads = [(Path(path), torch.load(Path(path), map_location="cpu")) for path in episode_cache_paths]
    cache_metadata = [dict(payload.get("metadata", {})) for _, payload in loaded_payloads]
    pose_weight = min(max(float(pose_target_weight), 0.0), 1.0)
    pose_pair_enabled = float(pose_pair_reprojection_threshold_px) > 0.0
    pose_target_summaries: list[dict[str, Any]] = []
    pose_pair_summaries: list[dict[str, Any]] = []
    base_gaussian_id = torch.empty(0, dtype=torch.long)
    if str(base_descriptor_path):
        base = _load_descriptor_bank(base_descriptor_path)
    else:
        base = None
        for path, payload in loaded_payloads:
            if "base_landmark_desc" not in payload:
                continue
            candidate_base = F.normalize(torch.as_tensor(payload["base_landmark_desc"]).float(), p=2, dim=-1)
            if candidate_base.dim() != 2:
                raise ValueError(f"{path} base_landmark_desc must have shape [N,D]")
            if base is None:
                base = candidate_base
                if "base_gaussian_id" in payload:
                    base_gaussian_id = torch.as_tensor(payload["base_gaussian_id"]).long().reshape(-1)
            elif candidate_base.shape != base.shape:
                raise ValueError("all embedded base_landmark_desc tensors must have the same shape")
        if base is None:
            raise ValueError("base_descriptor_path is empty and no episode cache contains base_landmark_desc")
    if base_gaussian_id.numel() and base_gaussian_id.shape[0] != int(base.shape[0]):
        raise ValueError("base_gaussian_id must have one value per base descriptor")
    query_chunks: list[torch.Tensor] = []
    id_chunks: list[torch.Tensor] = []
    mask_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    pair_label_chunks: list[torch.Tensor] = []
    cosine_chunks: list[torch.Tensor] = []
    pose_utility_chunks: list[torch.Tensor] = []
    gate_targets: list[torch.Tensor] = []
    for path, payload in loaded_payloads:
        if "query_desc" not in payload:
            raise KeyError(f"{path} is missing query_desc")
        if "candidate_landmark_ids" in payload:
            ids = torch.as_tensor(payload["candidate_landmark_ids"]).long()
        elif "landmark_id" in payload:
            ids = torch.as_tensor(payload["landmark_id"]).long()
        else:
            raise KeyError(f"{path} is missing candidate_landmark_ids or landmark_id")
        if ids.dim() != 2:
            raise ValueError("candidate_landmark_ids must have shape [N,K]")
        n, topk = int(ids.shape[0]), int(ids.shape[1])
        query = F.normalize(torch.as_tensor(payload["query_desc"]).float(), p=2, dim=-1)
        if query.shape[0] != n:
            raise ValueError("query_desc and candidate_landmark_ids disagree on N")
        if ids.numel() and (int(ids.min().item()) < 0 or int(ids.max().item()) >= int(base.shape[0])):
            raise IndexError("candidate_landmark_ids reference descriptors outside the base bank")
        mask = torch.as_tensor(payload.get("candidate_mask", torch.ones(n, topk, dtype=torch.bool))).bool()
        if mask.shape != ids.shape:
            raise ValueError("candidate_mask must match candidate_landmark_ids")
        if "listwise_label" in payload:
            label = torch.as_tensor(payload["listwise_label"]).long().reshape(-1)
        elif "label" in payload and torch.as_tensor(payload["label"]).dim() == 1:
            label = torch.as_tensor(payload["label"]).long().reshape(-1)
        elif "pair_label" in payload:
            label = _first_positive_or_dustbin(torch.as_tensor(payload["pair_label"]).bool())
        else:
            raise KeyError(f"{path} is missing listwise_label, label, or pair_label")
        if label.shape[0] != n:
            raise ValueError("listwise_label must have one label per query")
        label = label.clamp(0, topk)
        if "pair_label" in payload:
            pair_label = torch.as_tensor(payload["pair_label"]).bool()
        else:
            pair_label = _pair_label_from_listwise(label, topk)
        if pair_label.shape != ids.shape:
            raise ValueError("pair_label must match candidate_landmark_ids")
        if "candidate_cosine" in payload:
            cosine = torch.as_tensor(payload["candidate_cosine"]).float()
        elif "cosine" in payload:
            cosine = torch.as_tensor(payload["cosine"]).float()
        else:
            cosine = torch.zeros(n, topk, dtype=torch.float32)
        if cosine.shape != ids.shape:
            raise ValueError("candidate_cosine must match candidate_landmark_ids")
        if pose_pair_enabled:
            pose_utility, pose_pair_summary = _pose_pair_utility_from_payload(
                payload,
                path=path,
                landmark_ids=ids,
                candidate_mask=mask,
                candidate_cosine=cosine,
                reprojection_threshold_px=float(pose_pair_reprojection_threshold_px),
                score_threshold=float(pose_pair_score_threshold),
            )
            pose_pair_summaries.append(pose_pair_summary)
        else:
            pose_utility = torch.zeros_like(cosine)
        if "gaussian_advantage_target" in payload:
            target = torch.as_tensor(payload["gaussian_advantage_target"]).float().reshape(-1)
            if target.shape[0] != int(base.shape[0]):
                raise ValueError("gaussian_advantage_target must have one value per base descriptor")
        else:
            valid = mask.reshape(-1)
            stats = build_gaussian_advantage_labels(
                ids.reshape(-1)[valid],
                pair_label.reshape(-1)[valid],
                num_landmarks=int(base.shape[0]),
                pair_scores=cosine.reshape(-1)[valid],
                false_positive_score_threshold=false_positive_score_threshold,
                false_positive_weight=1.0,
            )
            target = stats["target"]
        if pose_weight > 0.0:
            pose_target, pose_summary = _pose_reliability_target_from_payload(
                payload,
                path=path,
                landmark_ids=ids,
                candidate_mask=mask,
                candidate_cosine=cosine,
                num_landmarks=int(base.shape[0]),
                reprojection_threshold_px=float(pose_target_reprojection_threshold_px),
                score_threshold=float(pose_target_score_threshold),
            )
            target = ((1.0 - pose_weight) * target + pose_weight * pose_target).clamp(0.0, 1.0)
            pose_target_summaries.append(pose_summary)
        gate_targets.append(target)
        query_chunks.append(query)
        id_chunks.append(ids)
        mask_chunks.append(mask)
        label_chunks.append(label)
        pair_label_chunks.append(pair_label)
        cosine_chunks.append(cosine)
        pose_utility_chunks.append(pose_utility)
    if not query_chunks:
        raise ValueError("episode caches are empty")
    if gate_targets:
        gate_target = torch.stack(gate_targets, dim=0).mean(dim=0).clamp(0.0, 1.0)
    else:
        gate_target = torch.full((int(base.shape[0]),), 0.5, dtype=torch.float32)
    pose_target_summary: dict[str, Any] = {
        "enabled": pose_weight > 0.0,
        "weight": pose_weight,
        "reprojection_threshold_px": float(pose_target_reprojection_threshold_px),
        "score_threshold": float(pose_target_score_threshold),
        "positive_pairs": int(sum(item.get("positive_pairs", 0) for item in pose_target_summaries)),
        "positive_landmarks": int(sum(item.get("positive_landmarks", 0) for item in pose_target_summaries)),
        "per_cache": pose_target_summaries,
    }
    pose_pair_summary: dict[str, Any] = {
        "enabled": pose_pair_enabled,
        "reprojection_threshold_px": float(pose_pair_reprojection_threshold_px),
        "score_threshold": float(pose_pair_score_threshold),
        "positive_pairs": int(sum(item.get("positive_pairs", 0) for item in pose_pair_summaries)),
        "positive_landmarks": int(sum(item.get("positive_landmarks", 0) for item in pose_pair_summaries)),
        "per_cache": pose_pair_summaries,
    }
    return {
        "base_descriptors": base,
        "base_gaussian_id": base_gaussian_id,
        "query_desc": torch.cat(query_chunks, dim=0),
        "candidate_landmark_ids": torch.cat(id_chunks, dim=0),
        "candidate_mask": torch.cat(mask_chunks, dim=0),
        "listwise_label": torch.cat(label_chunks, dim=0),
        "pair_label": torch.cat(pair_label_chunks, dim=0),
        "candidate_cosine": torch.cat(cosine_chunks, dim=0),
        "candidate_pose_utility": torch.cat(pose_utility_chunks, dim=0),
        "gaussian_advantage_target": gate_target,
        "pose_target_summary": pose_target_summary,
        "pose_pair_summary": pose_pair_summary,
        "cache_metadata": cache_metadata,
    }


def _ranking_loss(candidate_logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, margin: float) -> torch.Tensor:
    topk = int(candidate_logits.shape[1])
    positive = labels < topk
    if not bool(positive.any()):
        return candidate_logits.sum() * 0.0
    rows = torch.where(positive)[0]
    cols = labels[rows]
    pos_score = candidate_logits[rows, cols]
    neg_mask = mask[rows].clone()
    neg_mask[torch.arange(rows.numel(), device=rows.device), cols] = False
    hard_neg = candidate_logits[rows].masked_fill(~neg_mask, -1e4).amax(dim=1)
    valid = hard_neg > -9999.0
    if not bool(valid.any()):
        return candidate_logits.sum() * 0.0
    return F.relu(float(margin) - pos_score[valid] + hard_neg[valid]).mean()


def _pose_pair_ranking_loss(
    candidate_selector_logits: torch.Tensor,
    pose_utility: torch.Tensor,
    mask: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    if candidate_selector_logits.shape != pose_utility.shape or candidate_selector_logits.shape != mask.shape:
        raise ValueError("selector logits, pose utility, and mask must have the same shape")
    utility = pose_utility.to(device=candidate_selector_logits.device, dtype=torch.float32)
    valid = mask.to(device=candidate_selector_logits.device, dtype=torch.bool) & torch.isfinite(utility)
    score_delta = candidate_selector_logits[:, :, None] - candidate_selector_logits[:, None, :]
    utility_delta = utility[:, :, None] - utility[:, None, :]
    pair_mask = valid[:, :, None] & valid[:, None, :] & (utility_delta > 1e-6)
    if not bool(pair_mask.any()):
        return candidate_selector_logits.sum() * 0.0
    weights = utility_delta[pair_mask].detach()
    losses = F.softplus(float(margin) - score_delta[pair_mask]) * weights
    return losses.sum() / weights.sum().clamp_min(1e-8)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train export-aligned Unified LFF-v2 descriptor gates.")
    parser.add_argument("--base_descriptor_path", required=True)
    parser.add_argument("--episode_cache", nargs="+", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--alpha_max", type=float, default=0.05)
    parser.add_argument("--init_gate", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lambda_trust", type=float, default=1.0)
    parser.add_argument("--lambda_gate", type=float, default=1.0)
    parser.add_argument("--lambda_pair", type=float, default=0.25)
    parser.add_argument("--lambda_rank", type=float, default=0.25)
    parser.add_argument("--rank_margin", type=float, default=0.1)
    parser.add_argument("--trust_l1_weight", type=float, default=1.0)
    parser.add_argument(
        "--selector_only",
        action="store_true",
        help="Train only the localization selector; keep native descriptors as the export payload.",
    )
    parser.add_argument(
        "--freeze_descriptor_residual",
        action="store_true",
        help="Freeze descriptor residual and residual gate parameters.",
    )
    parser.add_argument(
        "--selector_bias_weight",
        type=float,
        default=1.0,
        help="Weight for selector logits added to native cosine logits in selector-only mode.",
    )
    parser.add_argument("--lambda_selector_listwise", type=float, default=1.0)
    parser.add_argument("--lambda_selector_gate", type=float, default=1.0)
    parser.add_argument("--lambda_selector_hard_negative", type=float, default=0.25)
    parser.add_argument("--lambda_selector_budget", type=float, default=0.0)
    parser.add_argument("--lambda_selector_coverage", type=float, default=0.0)
    parser.add_argument("--lambda_selector_pose_pair", type=float, default=0.0)
    parser.add_argument("--selector_pose_pair_margin", type=float, default=0.2)
    parser.add_argument(
        "--false_positive_score_threshold",
        type=float,
        default=0.0,
        help="Only negative candidates at or above this cosine contribute to Gaussian gate suppression.",
    )
    parser.add_argument(
        "--pose_target_weight",
        type=float,
        default=0.0,
        help="Blend weight for pose-reliability selector targets from audited self-map reprojection errors.",
    )
    parser.add_argument(
        "--pose_target_reprojection_threshold_px",
        type=float,
        default=8.0,
        help="Reprojection threshold used to reward pose-reliable selector landmarks when pose_target_weight > 0.",
    )
    parser.add_argument(
        "--pose_target_score_threshold",
        type=float,
        default=0.0,
        help="Minimum candidate cosine for pose-target rewards when pose_target_weight > 0.",
    )
    parser.add_argument(
        "--pose_pair_reprojection_threshold_px",
        type=float,
        default=0.0,
        help="Reprojection threshold for pair-level selector pose supervision; 0 disables it.",
    )
    parser.add_argument(
        "--pose_pair_score_threshold",
        type=float,
        default=0.0,
        help="Minimum candidate cosine for pair-level selector pose supervision.",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(args: argparse.Namespace | None = None) -> None:
    args = build_argparser().parse_args() if args is None else args
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cache_paths = _parse_paths(list(args.episode_cache))
    tensors = load_unified_lff_training_tensors(
        args.base_descriptor_path,
        cache_paths,
        false_positive_score_threshold=float(args.false_positive_score_threshold),
        pose_target_weight=float(args.pose_target_weight),
        pose_target_reprojection_threshold_px=float(args.pose_target_reprojection_threshold_px),
        pose_target_score_threshold=float(args.pose_target_score_threshold),
        pose_pair_reprojection_threshold_px=(
            float(args.pose_pair_reprojection_threshold_px)
            if float(args.lambda_selector_pose_pair) > 0.0
            else 0.0
        ),
        pose_pair_score_threshold=float(args.pose_pair_score_threshold),
    )
    dataset = TensorDataset(
        tensors["query_desc"].float(),
        tensors["candidate_landmark_ids"].long(),
        tensors["candidate_mask"].bool(),
        tensors["listwise_label"].long(),
        tensors["pair_label"].bool(),
        tensors["candidate_pose_utility"].float(),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
    )
    model = UnifiedLFFDescriptor(
        tensors["base_descriptors"],
        alpha_max=float(args.alpha_max),
        init_gate=float(args.init_gate),
        init_selector=0.5,
    ).to(device)
    selector_only = bool(getattr(args, "selector_only", False))
    residual_frozen = selector_only or bool(getattr(args, "freeze_descriptor_residual", False))
    if residual_frozen:
        model.residual.requires_grad_(False)
        model.gate_logit.requires_grad_(False)
    frozen_residual_gate_logit = float(model.gate_logit.detach().flatten()[0].cpu())
    dustbin_logit = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=device))
    trainable_model_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_model_params + [dustbin_logit],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    gate_target = tensors["gaussian_advantage_target"].to(device)
    topk = int(tensors["candidate_landmark_ids"].shape[1])
    temp = max(float(args.temperature), 1e-6)
    history: list[dict[str, float]] = []
    for epoch in range(int(args.epochs)):
        total = 0
        sums = {
            "loss": 0.0,
            "ce_loss": 0.0,
            "pair_loss": 0.0,
            "rank_loss": 0.0,
            "trust_loss": 0.0,
            "gate_loss": 0.0,
            "selector_listwise_loss": 0.0,
            "selector_gate_loss": 0.0,
            "selector_hard_negative_loss": 0.0,
            "selector_budget_loss": 0.0,
            "selector_coverage_loss": 0.0,
            "selector_pose_pair_loss": 0.0,
        }
        correct = 0
        for query_desc, candidate_ids, candidate_mask, listwise_label, pair_label, pose_utility in tqdm(
            loader,
            desc=f"UnifiedLFF epoch {epoch + 1}",
            dynamic_ncols=True,
        ):
            query_desc = query_desc.to(device)
            candidate_ids = candidate_ids.to(device)
            candidate_mask = candidate_mask.to(device)
            listwise_label = listwise_label.to(device).clamp(0, topk)
            pair_label = pair_label.to(device)
            pose_utility = pose_utility.to(device)
            batch = int(query_desc.shape[0])
            q = F.normalize(query_desc.float(), p=2, dim=-1)
            if selector_only:
                flat_ids = candidate_ids.reshape(-1)
                candidate_desc = model.base_descriptors[flat_ids].view(batch, topk, -1)
                native_logits = (candidate_desc * q[:, None, :]).sum(dim=-1) / temp
                selector_bias = model.selector_logit[flat_ids].view(batch, topk)
                candidate_logits_raw = native_logits + float(args.selector_bias_weight) * selector_bias
            else:
                candidate_desc = model(candidate_ids.reshape(-1)).view(batch, topk, -1)
                candidate_logits_raw = (candidate_desc * q[:, None, :]).sum(dim=-1) / temp
            candidate_logits = candidate_logits_raw.masked_fill(~candidate_mask, -1e4)
            dustbin = dustbin_logit.expand(batch, 1)
            logits = torch.cat([candidate_logits, dustbin], dim=1)
            ce_loss = F.cross_entropy(logits, listwise_label)
            if selector_only:
                pair_loss = logits.sum() * 0.0
                rank_loss = _ranking_loss(candidate_logits_raw, listwise_label, candidate_mask, float(args.rank_margin))
                trust = logits.sum() * 0.0
                gate_loss = F.binary_cross_entropy_with_logits(model.selector_logit, gate_target)
                selector_budget_loss = model.selector().mean()
                selector_coverage_loss = logits.sum() * 0.0
                selector_pose_pair_loss = _pose_pair_ranking_loss(
                    selector_bias,
                    pose_utility,
                    candidate_mask,
                    margin=float(args.selector_pose_pair_margin),
                )
                loss = (
                    float(args.lambda_selector_listwise) * ce_loss
                    + float(args.lambda_selector_gate) * gate_loss
                    + float(args.lambda_selector_hard_negative) * rank_loss
                    + float(args.lambda_selector_budget) * selector_budget_loss
                    + float(args.lambda_selector_coverage) * selector_coverage_loss
                    + float(args.lambda_selector_pose_pair) * selector_pose_pair_loss
                )
            else:
                pair_valid = candidate_mask
                if bool(pair_valid.any()):
                    pair_logits = candidate_logits_raw[pair_valid] - dustbin_logit
                    pair_loss = F.binary_cross_entropy_with_logits(pair_logits, pair_label[pair_valid].float())
                else:
                    pair_loss = logits.sum() * 0.0
                rank_loss = _ranking_loss(candidate_logits_raw, listwise_label, candidate_mask, float(args.rank_margin))
                trust = model.trust_region_loss(l1_weight=float(args.trust_l1_weight))["loss"]
                gate_loss = F.binary_cross_entropy_with_logits(model.selector_logit, gate_target)
                selector_budget_loss = logits.sum() * 0.0
                selector_coverage_loss = logits.sum() * 0.0
                selector_pose_pair_loss = logits.sum() * 0.0
                loss = (
                    ce_loss
                    + float(args.lambda_pair) * pair_loss
                    + float(args.lambda_rank) * rank_loss
                    + float(args.lambda_trust) * trust
                    + float(args.lambda_gate) * gate_loss
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += batch
            correct += int((logits.detach().argmax(dim=-1) == listwise_label).sum().item())
            for key, value in (
                ("loss", loss),
                ("ce_loss", ce_loss),
                ("pair_loss", pair_loss),
                ("rank_loss", rank_loss),
                ("trust_loss", trust),
                ("gate_loss", gate_loss),
                ("selector_listwise_loss", ce_loss if selector_only else logits.sum() * 0.0),
                ("selector_gate_loss", gate_loss if selector_only else logits.sum() * 0.0),
                ("selector_hard_negative_loss", rank_loss if selector_only else logits.sum() * 0.0),
                ("selector_budget_loss", selector_budget_loss),
                ("selector_coverage_loss", selector_coverage_loss),
                ("selector_pose_pair_loss", selector_pose_pair_loss),
            ):
                sums[key] += float(value.detach().cpu()) * batch
        row = {key: value / max(1, total) for key, value in sums.items()}
        row["epoch"] = float(epoch + 1)
        row["accuracy"] = correct / max(1, total)
        history.append(row)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_descriptors = model.base_descriptors.detach().cpu() if selector_only else model().detach().cpu()
    torch.save(
        {
            "config": {
                "model_type": "unified_lff_descriptor",
                "num_landmarks": model.num_landmarks,
                "descriptor_dim": model.descriptor_dim,
                "alpha_max": float(args.alpha_max),
                "temperature": float(args.temperature),
                "selector_only": selector_only,
            },
            "state_dict": model.state_dict(),
            "readout_state_dict": {"dustbin_logit": dustbin_logit.detach().cpu()},
            "export_descriptors": export_descriptors,
            "gate": model.selector().detach().cpu(),
            "residual_gate": model.gate().detach().cpu(),
            "base_gaussian_id": tensors["base_gaussian_id"].detach().cpu(),
            "metadata": {
                "episode_cache": [str(path) for path in cache_paths],
                "base_descriptor_path": str(args.base_descriptor_path),
                "samples": int(tensors["query_desc"].shape[0]),
                "topk": topk,
                "single_path_deployment": True,
                "branch_selection": False,
                "selector_gate_decoupled": True,
                "selector_only": selector_only,
                "residual_frozen": residual_frozen,
                "descriptor_mode": "native_required" if selector_only else "bounded_residual",
                "selector_bias_weight": float(args.selector_bias_weight),
                "frozen_residual_gate_logit": frozen_residual_gate_logit,
                "loss_weights": {
                    "trust": float(args.lambda_trust),
                    "gate": float(args.lambda_gate),
                    "pair": float(args.lambda_pair),
                    "rank": float(args.lambda_rank),
                    "selector_listwise": float(args.lambda_selector_listwise),
                    "selector_gate": float(args.lambda_selector_gate),
                    "selector_hard_negative": float(args.lambda_selector_hard_negative),
                    "selector_budget": float(args.lambda_selector_budget),
                    "selector_coverage": float(args.lambda_selector_coverage),
                    "selector_pose_pair": float(args.lambda_selector_pose_pair),
                },
                "false_positive_score_threshold": float(args.false_positive_score_threshold),
                "pose_target": tensors["pose_target_summary"],
                "pose_pair": tensors["pose_pair_summary"],
                "selector_pose_pair_margin": float(args.selector_pose_pair_margin),
                "history": history,
            },
        },
        output_path,
    )
    _write_training_audit_bundle(
        output_path=output_path,
        args=args,
        cache_paths=cache_paths,
        cache_metadata=list(tensors.get("cache_metadata", [])),
        history=history,
        selector_only=selector_only,
        residual_frozen=residual_frozen,
    )
    print(f"[unified_lff] wrote {output_path}")


if __name__ == "__main__":
    main()
