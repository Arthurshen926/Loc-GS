#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from loc_gs.feedback.io import save_feedback_bank, summarize_feedback_bank
from loc_gs.feedback.labels import derive_scene_reliability_baseline_relative
from loc_gs.feedback.schema import FeedbackMatchRecord
from loc_gs.scripts.locgsctl import summarize_path


def _json_dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_repo_root()),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _output_paths(output_path: str | Path) -> dict[str, Path]:
    output = Path(output_path)
    if output.suffix.lower() in {".jsonl", ".json"}:
        output_dir = output.parent
        bank = output
    else:
        output_dir = output
        bank = output_dir / "feedback_bank.jsonl"
    return {
        "output_dir": output_dir,
        "feedback_bank": bank,
        "feedback_summary": output_dir / "feedback_summary.json",
        "manifest": output_dir / "manifest.json",
    }


def _to_cpu(payload: Any) -> Any:
    if torch.is_tensor(payload):
        return payload.detach().cpu()
    return payload


def _tensor(payload: dict[str, Any], key: str, default: Any = None) -> torch.Tensor | None:
    value = payload.get(key, default)
    if value is None:
        return None
    return torch.as_tensor(value).detach().cpu()


def _scalar(value: Any) -> float | None:
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        value = value.reshape(-1)[0].item()
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _dense_pose_metrics(summary_path: str | Path) -> dict[str, float | None]:
    summary = summarize_path(summary_path)
    dense = summary.get("dense", {})
    if not isinstance(dense, dict):
        return {"pose_error_t_cm": None, "pose_error_r_deg": None}
    return {
        "pose_error_t_cm": _scalar(dense.get("median_te_cm")),
        "pose_error_r_deg": _scalar(dense.get("median_re_deg")),
    }


def _gaussian_id(
    payload: dict[str, Any],
    *,
    landmark_id: int,
    flat_index: int,
    row: int | None = None,
    col: int | None = None,
) -> str:
    for key in ("gaussian_id", "matched_gaussian_id", "base_gaussian_id"):
        tensor = _tensor(payload, key)
        if tensor is None or tensor.numel() == 0:
            continue
        if row is not None and col is not None and tensor.dim() >= 2 and row < tensor.shape[0] and col < tensor.shape[1]:
            return str(int(tensor[row, col].item()))
        flat = tensor.reshape(-1)
        if key == "base_gaussian_id" and 0 <= landmark_id < flat.numel():
            return str(int(flat[landmark_id].item()))
        if flat.numel() > flat_index:
            return str(int(flat[flat_index].item()))
        if 0 <= landmark_id < flat.numel():
            return str(int(flat[landmark_id].item()))
    return str(landmark_id)


def _query_xy(query_yx: torch.Tensor | None, row: int) -> list[float | None]:
    if query_yx is None or query_yx.numel() == 0 or row >= query_yx.shape[0]:
        return [None, None]
    y = _scalar(query_yx[row, 0])
    x = _scalar(query_yx[row, 1])
    return [x, y]


def _listwise_records(
    payload: dict[str, Any],
    *,
    scene: str,
    source_view_id: str,
    pose_source: str,
    pose_metrics: dict[str, float | None],
) -> list[FeedbackMatchRecord]:
    cosine = _tensor(payload, "cosine")
    label = _tensor(payload, "label")
    if cosine is None or label is None:
        raise KeyError("listwise pair cache requires cosine and label tensors")
    if cosine.dim() != 2:
        raise ValueError("listwise cosine must have shape [num_queries, topk]")
    num_queries, topk = int(cosine.shape[0]), int(cosine.shape[1])
    labels = label.long().reshape(-1)
    if labels.numel() != num_queries:
        raise ValueError("listwise label must have one value per query")
    landmark_ids = _tensor(payload, "landmark_id")
    if landmark_ids is None:
        landmark_ids = torch.arange(num_queries * topk, dtype=torch.long).reshape(num_queries, topk)
    candidate_mask = _tensor(payload, "candidate_mask", torch.ones_like(cosine, dtype=torch.bool)).bool()
    reprojection_error = _tensor(payload, "reprojection_error", torch.full_like(cosine, float("nan"))).float()
    query_score = _tensor(payload, "query_score", torch.ones(num_queries)).float().reshape(-1)
    landmark_prior = _tensor(payload, "landmark_prior", torch.zeros_like(cosine)).float()
    query_yx = _tensor(payload, "query_yx")
    records: list[FeedbackMatchRecord] = []
    for row in range(num_queries):
        query_id = f"query_{row:06d}"
        pnp_success = int(labels[row].item()) < topk
        for col in range(topk):
            if not bool(candidate_mask[row, col].item()):
                continue
            landmark_id = int(landmark_ids[row, col].item())
            flat_index = row * topk + col
            records.append(
                FeedbackMatchRecord.from_mapping(
                    {
                        "scene": scene,
                        "query_id": query_id,
                        "source_view_id": source_view_id,
                        "pose_source": pose_source,
                        "keypoint_xy": _query_xy(query_yx, row),
                        "matched_landmark_id": str(landmark_id),
                        "matched_gaussian_id": _gaussian_id(
                            payload,
                            landmark_id=landmark_id,
                            flat_index=flat_index,
                            row=row,
                            col=col,
                        ),
                        "descriptor_score": _scalar(cosine[row, col]),
                        "detector_score": _scalar(query_score[row]) if query_score.numel() > row else None,
                        "match_rank": col + 1,
                        "pnp_inlier": int(labels[row].item()) == col,
                        "reprojection_error_px": _scalar(reprojection_error[row, col]),
                        "depth_consistency": None,
                        "visibility_score": _scalar(landmark_prior[row, col]),
                        "pose_error_t_cm": pose_metrics["pose_error_t_cm"],
                        "pose_error_r_deg": pose_metrics["pose_error_r_deg"],
                        "pnp_success": pnp_success,
                        "dense_refine_success": pose_metrics["pose_error_t_cm"] is not None,
                    }
                )
            )
    return records


def _pairwise_records(
    payload: dict[str, Any],
    *,
    scene: str,
    source_view_id: str,
    pose_source: str,
    pose_metrics: dict[str, float | None],
) -> list[FeedbackMatchRecord]:
    label = _tensor(payload, "label")
    if label is None:
        raise KeyError("pair cache requires label tensor")
    labels = label.float().reshape(-1)
    count = int(labels.numel())
    cosine = _tensor(payload, "cosine", torch.full((count,), float("nan"))).float().reshape(-1)
    query_score = _tensor(payload, "query_score", torch.ones(count)).float().reshape(-1)
    landmark_prior = _tensor(payload, "landmark_prior", torch.zeros(count)).float().reshape(-1)
    landmark_ids = _tensor(payload, "landmark_id", torch.arange(count, dtype=torch.long)).long().reshape(-1)
    records: list[FeedbackMatchRecord] = []
    for idx in range(count):
        landmark_id = int(landmark_ids[idx].item()) if landmark_ids.numel() > idx else idx
        inlier = bool(labels[idx].item() >= 0.5)
        records.append(
            FeedbackMatchRecord.from_mapping(
                {
                    "scene": scene,
                    "query_id": f"pair_{idx:06d}",
                    "source_view_id": source_view_id,
                    "pose_source": pose_source,
                    "keypoint_xy": [None, None],
                    "matched_landmark_id": str(landmark_id),
                    "matched_gaussian_id": _gaussian_id(payload, landmark_id=landmark_id, flat_index=idx),
                    "descriptor_score": _scalar(cosine[idx]) if cosine.numel() > idx else None,
                    "detector_score": _scalar(query_score[idx]) if query_score.numel() > idx else None,
                    "match_rank": 1,
                    "pnp_inlier": inlier,
                    "reprojection_error_px": 0.0 if inlier else None,
                    "depth_consistency": None,
                    "visibility_score": _scalar(landmark_prior[idx]) if landmark_prior.numel() > idx else None,
                    "pose_error_t_cm": pose_metrics["pose_error_t_cm"],
                    "pose_error_r_deg": pose_metrics["pose_error_r_deg"],
                    "pnp_success": inlier,
                    "dense_refine_success": pose_metrics["pose_error_t_cm"] is not None,
                }
            )
        )
    return records


def load_pair_cache_records(
    pair_cache: str | Path,
    *,
    scene: str,
    selfmap_summary: str | Path,
) -> tuple[list[FeedbackMatchRecord], dict[str, Any]]:
    path = Path(pair_cache)
    if not path.exists():
        raise FileNotFoundError(f"pair cache not found: {path}")
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"pair cache must be a dict: {path}")
    payload = {key: _to_cpu(value) for key, value in payload.items()}
    metadata = dict(payload.get("metadata", {}))
    pair_format = str(metadata.get("format", "listwise" if _tensor(payload, "cosine") is not None and _tensor(payload, "cosine").dim() == 2 else "pair"))
    pose_metrics = _dense_pose_metrics(selfmap_summary)
    source_view_id = path.name
    pose_source = "selfmap_pair_cache"
    if pair_format == "listwise":
        records = _listwise_records(
            payload,
            scene=scene,
            source_view_id=source_view_id,
            pose_source=pose_source,
            pose_metrics=pose_metrics,
        )
    else:
        records = _pairwise_records(
            payload,
            scene=scene,
            source_view_id=source_view_id,
            pose_source=pose_source,
            pose_metrics=pose_metrics,
        )
    return records, metadata


def _dry_run_payload(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    fields = sorted(FeedbackMatchRecord.__dataclass_fields__.keys())
    return {
        "dry_run": True,
        "scene": args.scene,
        "split_name": args.split_name,
        "inputs": {
            "pair_cache": args.pair_cache,
            "selfmap_summary": args.selfmap_summary,
            "baseline_summary": args.baseline_summary,
        },
        "would_write": {key: str(value) for key, value in paths.items() if key != "output_dir"},
        "schema": {"record_fields": fields},
    }


def export_feedback_bank(args: argparse.Namespace) -> dict[str, Any]:
    if str(args.split_name).strip().lower() == "test":
        raise ValueError("feedback bank export cannot use Cambridge test split")
    paths = _output_paths(args.output_path)
    if args.dry_run:
        return _dry_run_payload(args, paths)

    records, pair_metadata = load_pair_cache_records(
        args.pair_cache,
        scene=args.scene,
        selfmap_summary=args.selfmap_summary,
    )
    manifest: dict[str, Any] = {
        "scene": args.scene,
        "split_name": args.split_name,
        "pair_cache": str(args.pair_cache),
        "selfmap_summary": str(args.selfmap_summary),
        "baseline_summary": str(args.baseline_summary),
        "pair_cache_metadata": pair_metadata,
        "source": "export_feedback_bank_from_cambridge",
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv) if sys.argv else "",
    }
    if args.baseline_summary:
        manifest["baseline_relative"] = derive_scene_reliability_baseline_relative(
            {"scene": args.scene, **summarize_path(args.selfmap_summary)},
            summarize_path(args.baseline_summary),
        )

    save_feedback_bank(paths["feedback_bank"], records, manifest)
    summary = summarize_feedback_bank(paths["feedback_bank"])
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    paths["feedback_summary"].write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    paths["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "dry_run": False,
        "feedback_bank": str(paths["feedback_bank"]),
        "feedback_summary": str(paths["feedback_summary"]),
        "manifest": str(paths["manifest"]),
        "record_count": len(records),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export Cambridge self-localization and pair-cache signals to a Loc-GS feedback bank."
    )
    parser.add_argument("--scene", required=True)
    parser.add_argument("--pair_cache", required=True)
    parser.add_argument("--selfmap_summary", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--baseline_summary", default="")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main(args: argparse.Namespace | None = None) -> int:
    args = build_argparser().parse_args() if args is None else args
    result = export_feedback_bank(args)
    print(_json_dump(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
