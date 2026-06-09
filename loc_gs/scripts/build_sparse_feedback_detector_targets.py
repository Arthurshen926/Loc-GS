#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

import torch

from loc_gs.reporting.artifact_audit import write_artifact_audit_bundle
from loc_gs.stdloc_native.detector_target_refinement import build_lsf_detector_target


def _command() -> str:
    return " ".join(shlex.quote(part) for part in sys.argv)


def _load_pair_cache(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("pair_cache must contain a dict payload")
    return payload


def _split_audit(metadata: dict[str, Any]) -> dict[str, Any]:
    audit = metadata.get("split_audit")
    if isinstance(audit, dict) and "audit_status" in audit:
        return dict(audit)
    split = str(metadata.get("source_split_name", metadata.get("feedback_bank_split_name", ""))).strip()
    if split.lower() == "test":
        return {
            "audit_status": "failed",
            "split_name": split,
            "reason": "detector refinement targets must not be built from test split pair caches",
        }
    return {
        "audit_status": "unknown" if not split else "passed",
        "split_name": split or "unknown",
        "reason": "non-test split metadata found" if split else "split metadata missing",
    }


def _reject_test_cache(metadata: dict[str, Any]) -> None:
    for key in ("source_split_name", "feedback_bank_split_name", "split_name"):
        if str(metadata.get(key, "")).strip().lower() == "test":
            raise ValueError("cannot build sparse feedback detector targets from a test split pair cache")
    audit = _split_audit(metadata)
    if str(audit.get("audit_status", "")).lower() == "failed":
        raise ValueError("cannot build sparse feedback detector targets from a cache with failed split audit")


def _positive_rows(
    payload: dict[str, Any],
    *,
    sigma_px: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if "query_yx" not in payload:
        raise KeyError("listwise pair cache is missing query_yx")
    if "label" not in payload:
        raise KeyError("listwise pair cache is missing label")
    labels = torch.as_tensor(payload["label"], dtype=torch.long).reshape(-1)
    yx = torch.as_tensor(payload["query_yx"], dtype=torch.float32).reshape(-1, 2)
    if labels.shape[0] != yx.shape[0]:
        raise ValueError("label and query_yx row counts must match")
    rows = torch.arange(labels.shape[0], dtype=torch.long)
    topk = 0
    if "cosine" in payload:
        cosine = torch.as_tensor(payload["cosine"], dtype=torch.float32)
        if cosine.dim() != 2 or cosine.shape[0] != labels.shape[0]:
            raise ValueError("cosine must have shape [N,K]")
        topk = int(cosine.shape[1])
    else:
        topk = int(labels.max().item()) + 1 if labels.numel() else 0
        cosine = torch.ones(labels.shape[0], max(topk, 1), dtype=torch.float32)
    if topk <= 0:
        return rows.new_empty((0,)), yx.new_empty((0, 2)), yx.new_empty((0,)), yx.new_empty((0,))

    mask = torch.ones(labels.shape[0], topk, dtype=torch.bool)
    if "candidate_mask" in payload:
        mask = torch.as_tensor(payload["candidate_mask"], dtype=torch.bool)
        if mask.shape != (labels.shape[0], topk):
            raise ValueError("candidate_mask must match cosine shape")
    reproj = torch.zeros(labels.shape[0], topk, dtype=torch.float32)
    if "reprojection_error" in payload:
        reproj = torch.as_tensor(payload["reprojection_error"], dtype=torch.float32)
        if reproj.shape != (labels.shape[0], topk):
            raise ValueError("reprojection_error must match cosine shape")
    solver_validity = torch.ones(labels.shape[0], topk, dtype=torch.float32)
    if "solver_validity" in payload:
        solver_validity = torch.as_tensor(payload["solver_validity"], dtype=torch.float32)
        if solver_validity.shape != (labels.shape[0], topk):
            raise ValueError("solver_validity must match cosine shape")
    positive = (labels >= 0) & (labels < topk)
    positive = positive & mask[rows, labels.clamp(0, max(topk - 1, 0))]
    if "reprojection_error" in payload:
        positive = positive & torch.isfinite(reproj[rows, labels.clamp(0, max(topk - 1, 0))])
    if not bool(positive.any()):
        return rows.new_empty((0,)), yx.new_empty((0, 2)), yx.new_empty((0,)), yx.new_empty((0,))
    pos_rows = rows[positive]
    pos_cols = labels[positive]
    score = cosine[pos_rows, pos_cols].float().clamp_min(0.0)
    validity = solver_validity[pos_rows, pos_cols].float().clamp(0.0, 1.0)
    if "reprojection_error" in payload:
        sigma = max(float(sigma_px), 1e-6)
        score = score * torch.exp(-reproj[pos_rows, pos_cols].clamp_min(0.0) / sigma)
    return pos_rows, yx[pos_rows], score.clamp_min(0.0), validity


def build_sparse_feedback_detector_targets(
    pair_cache: dict[str, Any],
    *,
    height: int,
    width: int,
    sigma_px: float = 1.0,
    score_sigma_px: float = 4.0,
    solver_validity_power: float = 0.0,
) -> dict[str, Any]:
    metadata = dict(pair_cache.get("metadata", {}))
    _reject_test_cache(metadata)
    image_ids = [str(item) for item in pair_cache.get("image_id", [])]
    positive_indices_tensor, yx_all, weights_all, solver_validity_all = _positive_rows(
        pair_cache,
        sigma_px=float(score_sigma_px),
    )
    if not image_ids:
        image_ids = [f"row_{idx:06d}" for idx in range(int(torch.as_tensor(pair_cache["label"]).numel()))]
    if len(image_ids) != int(torch.as_tensor(pair_cache["label"]).numel()):
        raise ValueError("image_id must have one entry per listwise row")

    targets: dict[str, dict[str, Any]] = {}
    for out_row, source_row in enumerate(positive_indices_tensor.tolist()):
        image_id = image_ids[int(source_row)]
        entry = targets.setdefault(image_id, {"yx": [], "weights": [], "solver_validity": []})
        entry["yx"].append(yx_all[out_row])
        entry["weights"].append(weights_all[out_row])
        entry["solver_validity"].append(solver_validity_all[out_row])

    materialized: dict[str, dict[str, Any]] = {}
    for image_id, entry in targets.items():
        points = torch.stack(entry["yx"], dim=0)
        weights = torch.stack(entry["weights"], dim=0)
        solver_validity = torch.stack(entry["solver_validity"], dim=0)
        heatmap, weight_map, target_meta = build_lsf_detector_target(
            projected_yx=points,
            support_weights=weights,
            solver_validity_weights=solver_validity,
            solver_validity_power=float(solver_validity_power),
            height=int(height),
            width=int(width),
            sigma_px=float(sigma_px),
        )
        materialized[image_id] = {
            "heatmap": heatmap[0, 0].contiguous(),
            "weight": weight_map[0, 0].contiguous(),
            "positive_count": int(points.shape[0]),
            "target_metadata": target_meta,
        }

    return {
        "targets": materialized,
        "metadata": {
            "format": "sparse_feedback_detector_targets_v1",
            "source_pair_cache_format": str(metadata.get("format", "")),
            "landmark_candidate_source": str(metadata.get("landmark_candidate_source", "")),
            "height": int(height),
            "width": int(width),
            "sigma_px": float(sigma_px),
            "score_sigma_px": float(score_sigma_px),
            "solver_validity_power": float(solver_validity_power),
            "positive_rows": int(sum(entry["positive_count"] for entry in materialized.values())),
            "image_count": int(len(materialized)),
            "source_metadata": metadata,
        },
        "split_audit": _split_audit(metadata),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build LSF-aware detector target heatmaps from sparse all-Gaussian feedback pairs.")
    parser.add_argument("--pair_cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--sigma_px", type=float, default=1.0)
    parser.add_argument("--score_sigma_px", type=float, default=4.0)
    parser.add_argument("--solver_validity_power", type=float, default=0.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    pair_cache_path = Path(args.pair_cache)
    output = Path(args.output)
    payload = _load_pair_cache(pair_cache_path)
    artifact = build_sparse_feedback_detector_targets(
        payload,
        height=int(args.height),
        width=int(args.width),
        sigma_px=float(args.sigma_px),
        score_sigma_px=float(args.score_sigma_px),
        solver_validity_power=float(args.solver_validity_power),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    manifest = {
        "artifact": str(output),
        "pair_cache": str(pair_cache_path),
        "height": int(args.height),
        "width": int(args.width),
        "sigma_px": float(args.sigma_px),
        "score_sigma_px": float(args.score_sigma_px),
        "solver_validity_power": float(args.solver_validity_power),
        "metadata": artifact["metadata"],
        "split_audit": artifact["split_audit"],
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_artifact_audit_bundle(
        output.parent,
        manifest=manifest,
        command=_command(),
        metrics_summary=artifact["metadata"],
        split_audit=artifact["split_audit"],
    )
    print(json.dumps({"output": str(output), "manifest": str(manifest_path), **artifact["metadata"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
