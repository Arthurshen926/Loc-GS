from __future__ import annotations

import json
import math
import pickle
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from loc_gs.reporting.artifact_audit import artifact_split_audit, write_artifact_audit_bundle


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_repo_root(),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _load_source_idx(source_map: Path) -> torch.Tensor:
    path = source_map / "detector" / "sampled_idx.pkl"
    if not path.exists():
        raise FileNotFoundError(f"missing source sampled_idx.pkl: {path}")
    return torch.as_tensor(_load_pickle(path), dtype=torch.long).reshape(-1).cpu()


def _rank_normalize(values: torch.Tensor) -> torch.Tensor:
    tensor = values.float().reshape(-1).cpu()
    finite = torch.isfinite(tensor)
    out = torch.zeros_like(tensor)
    if not bool(finite.any().item()):
        return out
    valid_indices = torch.where(finite)[0]
    valid_values = tensor[finite]
    if valid_values.numel() == 1:
        out[valid_indices] = 1.0
        return out
    order = torch.argsort(valid_values, descending=False, stable=True)
    ranks = torch.empty_like(valid_values, dtype=torch.float32)
    ranks[order] = torch.arange(valid_values.numel(), dtype=torch.float32)
    out[valid_indices] = ranks / float(valid_values.numel() - 1)
    return out.clamp(0.0, 1.0)


def _rank_normalize_observed(values: torch.Tensor, observed_count: torch.Tensor | None) -> torch.Tensor:
    tensor = values.float().reshape(-1).cpu()
    if observed_count is None:
        observed = tensor > 0
    else:
        observed_tensor = observed_count.reshape(-1).cpu()
        if observed_tensor.numel() != tensor.numel():
            raise ValueError(
                f"observed_count length {observed_tensor.numel()} does not match support_score length {tensor.numel()}"
            )
        observed = observed_tensor > 0
    out = torch.zeros_like(tensor)
    if bool(observed.any().item()):
        ranked = _rank_normalize(tensor[observed])
        out[torch.where(observed)[0]] = ranked
    return out.clamp(0.0, 1.0)


def _load_detector_score_avg(source_map: Path, source_idx: torch.Tensor, size: int) -> tuple[torch.Tensor, dict[str, Any]] | None:
    path = source_map / "detector" / "sampled_scores.pkl"
    if not path.exists():
        return None
    payload = _load_pickle(path)
    if isinstance(payload, dict) and "score_avg" in payload:
        score_avg = torch.as_tensor(payload["score_avg"], dtype=torch.float32).reshape(-1).cpu()
        if score_avg.numel() == int(size):
            return score_avg, {"kind": "detector_score_avg", "path": str(path)}
        raise ValueError(f"score_avg length {score_avg.numel()} does not match support size {size}")
    sampled_scores = (
        torch.as_tensor(payload.get("sampled_scores"), dtype=torch.float32).reshape(-1).cpu()
        if isinstance(payload, dict) and "sampled_scores" in payload
        else torch.as_tensor(payload, dtype=torch.float32).reshape(-1).cpu()
    )
    if sampled_scores.numel() != source_idx.numel():
        return None
    scores = torch.zeros((int(size),), dtype=torch.float32)
    valid = (source_idx >= 0) & (source_idx < int(size))
    scores[source_idx[valid]] = sampled_scores[valid]
    return scores, {"kind": "detector_sampled_scores", "path": str(path)}


def _load_point_cloud_locability(source_map: Path, size: int) -> tuple[torch.Tensor, dict[str, Any]] | None:
    candidates = [
        source_map / "point_cloud" / "iteration_30000" / "point_cloud.ply",
        source_map / "input.ply",
    ]
    for path in candidates:
        if not path.exists():
            continue
        from plyfile import PlyData

        data = PlyData.read(str(path))["vertex"].data
        names = data.dtype.names or ()
        if "locability_logit" not in names:
            continue
        logits = torch.as_tensor(data["locability_logit"], dtype=torch.float32).reshape(-1).cpu()
        if logits.numel() != int(size):
            raise ValueError(f"locability_logit length {logits.numel()} does not match support size {size}")
        return torch.sigmoid(logits), {"kind": "point_cloud_locability_logit", "path": str(path)}
    return None


def _load_native_score(
    source_map: Path,
    source_idx: torch.Tensor,
    size: int,
    *,
    score_source: str = "auto",
) -> tuple[torch.Tensor, dict[str, Any]]:
    score_source = str(score_source or "auto")
    if score_source not in {"auto", "detector", "point_cloud"}:
        raise ValueError(f"unsupported score_source: {score_source}")
    if score_source in {"auto", "detector"}:
        detector = _load_detector_score_avg(source_map, source_idx, size)
        if detector is not None:
            return detector
        if score_source == "detector":
            raise FileNotFoundError(f"{source_map} has no detector score_avg compatible with support size {size}")
    if score_source in {"auto", "point_cloud"}:
        point_cloud = _load_point_cloud_locability(source_map, size)
        if point_cloud is not None:
            return point_cloud
        if score_source == "point_cloud":
            raise FileNotFoundError(f"{source_map} has no point_cloud locability_logit compatible with support size {size}")
    raise FileNotFoundError(
        f"{source_map} has neither detector/sampled_scores.pkl score_avg nor point_cloud locability_logit"
    )


def _load_support(path: Path | None) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if path is None:
        return {}, {}
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("solver consensus support artifact must be a dict")
    if "support_score" not in payload:
        raise KeyError("solver consensus support artifact must contain support_score")
    tensors: dict[str, torch.Tensor] = {
        "support_score": torch.as_tensor(payload["support_score"], dtype=torch.float32).reshape(-1).cpu()
    }
    size = int(tensors["support_score"].numel())
    for key in ("hard_negative_risk", "dense_worsen_risk"):
        if key in payload:
            tensor = torch.as_tensor(payload[key], dtype=torch.float32).reshape(-1).cpu()
            if tensor.numel() != size:
                raise ValueError(f"{key} length {tensor.numel()} does not match support_score length {size}")
            tensors[key] = tensor
        else:
            tensors[key] = torch.zeros((size,), dtype=torch.float32)
    if "observed_count" in payload:
        observed = torch.as_tensor(payload["observed_count"], dtype=torch.long).reshape(-1).cpu()
        if observed.numel() != size:
            raise ValueError(f"observed_count length {observed.numel()} does not match support_score length {size}")
        tensors["observed_count"] = observed
    metadata = dict(payload.get("metadata", {})) if isinstance(payload.get("metadata", {}), dict) else {}
    split = str(metadata.get("split", metadata.get("split_name", ""))).strip().lower()
    if split == "test":
        raise ValueError("test split support artifacts are not allowed for LSF v2 native proposals")
    return tensors, metadata


def _load_localization_support_field(path: Path | None) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if path is None:
        return {}, {}
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("localization support field artifact must be a dict")
    metadata = dict(payload.get("metadata", {})) if isinstance(payload.get("metadata", {}), dict) else {}
    split = str(metadata.get("split_name", metadata.get("split", ""))).strip().lower()
    if split == "test":
        raise ValueError("test split localization support fields are not allowed")
    tensors: dict[str, torch.Tensor] = {}
    for key in (
        "support_for_selection",
        "support_score",
        "hard_negative_risk",
        "dense_worsen_risk",
        "solver_set_utility",
        "pose_information",
        "pose_information_gain",
        "hard_query_support",
        "visibility_stability",
        "ambiguity_risk",
    ):
        if key in payload:
            tensors[key] = torch.as_tensor(payload[key], dtype=torch.float32).reshape(-1).cpu()
    if "support_for_selection" not in tensors and "support_score" not in tensors:
        raise KeyError("localization support field must contain support_for_selection or support_score")
    base = tensors["support_for_selection"] if "support_for_selection" in tensors else tensors["support_score"]
    size = int(base.numel())
    for key, tensor in tensors.items():
        if tensor.numel() != size:
            raise ValueError(f"{key} length {tensor.numel()} does not match localization support field length {size}")
    return tensors, metadata


def _command_from_argv() -> str:
    return " ".join(shlex.quote(part) for part in sys.argv) if sys.argv else ""


def build_lsf_v2_native_proposal(
    *,
    source_map: str | Path,
    output_dir: str | Path,
    solver_consensus_support_path: str | Path | None = None,
    localization_support_field_path: str | Path | None = None,
    candidate_pool_size: int = 4096,
    safe_support_threshold: float = 0.5,
    max_hard_negative_risk: float = 0.5,
    max_dense_worsen_risk: float = 0.5,
    min_native_rank: float = 0.75,
    score_source: str = "auto",
    native_weight: float = 0.75,
    support_weight: float = 0.20,
    hard_negative_weight: float = 0.35,
    dense_worsen_weight: float = 0.25,
    support_score_mode: str = "raw",
    solver_utility_weight: float = 0.0,
    pose_information_weight: float = 0.0,
    hard_query_support_weight: float = 0.0,
    visibility_weight: float = 0.0,
    ambiguity_weight: float = 0.0,
) -> dict[str, Any]:
    """Build paper-safe proposal tensors from native scores and audited v2 LSF support."""

    source = Path(source_map)
    support_path = Path(solver_consensus_support_path) if solver_consensus_support_path else None
    lsf_path = Path(localization_support_field_path) if localization_support_field_path else None
    output = Path(output_dir)
    tensors, support_metadata = _load_support(support_path)
    lsf_tensors, lsf_metadata = _load_localization_support_field(lsf_path)
    if not tensors and not lsf_tensors:
        raise ValueError("solver_consensus_support_path or localization_support_field_path is required")
    support_score_mode = str(support_score_mode or "raw")
    if support_score_mode not in {"raw", "rank_observed"}:
        raise ValueError(f"unsupported support_score_mode: {support_score_mode}")
    raw_support = (
        lsf_tensors.get("support_for_selection", lsf_tensors.get("support_score"))
        if lsf_tensors
        else tensors["support_score"]
    ).clamp(0.0, 1.0)
    support = (
        raw_support
        if support_score_mode == "raw" or lsf_tensors
        else _rank_normalize_observed(raw_support, tensors.get("observed_count")).clamp(0.0, 1.0)
    )
    size = int(support.numel())
    hard_negative = (
        lsf_tensors.get("hard_negative_risk", tensors.get("hard_negative_risk", torch.zeros((size,), dtype=torch.float32)))
    ).clamp(0.0, 1.0)
    dense_worsen = (
        lsf_tensors.get("dense_worsen_risk", tensors.get("dense_worsen_risk", torch.zeros((size,), dtype=torch.float32)))
    ).clamp(0.0, 1.0)
    solver_utility = lsf_tensors.get("solver_set_utility", torch.zeros((size,), dtype=torch.float32)).clamp(0.0, 1.0)
    pose_information = lsf_tensors.get(
        "pose_information", lsf_tensors.get("pose_information_gain", torch.zeros((size,), dtype=torch.float32))
    ).clamp(0.0, 1.0)
    hard_query_support = lsf_tensors.get("hard_query_support", torch.zeros((size,), dtype=torch.float32)).clamp(0.0, 1.0)
    visibility = lsf_tensors.get("visibility_stability", torch.zeros((size,), dtype=torch.float32)).clamp(0.0, 1.0)
    ambiguity = lsf_tensors.get("ambiguity_risk", torch.zeros((size,), dtype=torch.float32)).clamp(0.0, 1.0)

    source_idx = _load_source_idx(source)
    if source_idx.numel() and (int(source_idx.min().item()) < 0 or int(source_idx.max().item()) >= size):
        raise ValueError("source sampled_idx contains ids outside support_score length")
    source_mask = torch.zeros((size,), dtype=torch.bool)
    if source_idx.numel():
        source_mask[source_idx] = True

    native_score, score_source_metadata = _load_native_score(source, source_idx, size, score_source=score_source)
    native_rank = _rank_normalize(native_score)
    if not math.isfinite(float(min_native_rank)):
        raise ValueError("min_native_rank must be finite")

    selector = (
        float(native_weight) * native_rank
        + float(support_weight) * support
        + float(solver_utility_weight) * solver_utility
        + float(pose_information_weight) * pose_information
        + float(hard_query_support_weight) * hard_query_support
        + float(visibility_weight) * visibility
        - float(hard_negative_weight) * hard_negative
        - float(dense_worsen_weight) * dense_worsen
        - float(ambiguity_weight) * ambiguity
    ).clamp(0.0, 1.0)
    candidate_mask = (
        (~source_mask)
        & (native_rank >= float(min_native_rank))
        & (hard_negative <= float(max_hard_negative_risk))
        & (dense_worsen <= float(max_dense_worsen_risk))
    )
    candidate_ids = torch.where(candidate_mask)[0]
    if candidate_ids.numel():
        order = sorted(
            (int(idx) for idx in candidate_ids.tolist()),
            key=lambda idx: (-float(selector[idx].item()), -float(native_rank[idx].item()), idx),
        )
        order = order[: max(0, int(candidate_pool_size))]
        candidate_pool = torch.tensor(order, dtype=torch.long)
    else:
        candidate_pool = torch.empty((0,), dtype=torch.long)

    safe_mask = (
        source_mask
        & (support >= float(safe_support_threshold))
        & (hard_negative <= float(max_hard_negative_risk))
        & (dense_worsen <= float(max_dense_worsen_risk))
    )
    safe_core = torch.where(safe_mask)[0].long()
    merged_risk = torch.maximum(hard_negative, dense_worsen).float()

    output.mkdir(parents=True, exist_ok=True)
    selector_path = output / "selector_lsf_v2_native_proposal.pt"
    candidate_path = output / f"candidate_pool_lsf_v2_native_top{int(candidate_pool_size)}.pt"
    safe_core_path = output / "safe_core_lsf_v2_source.pt"
    hard_negative_path = output / "hard_negative_risk_lsf_v2.pt"
    dense_worsen_path = output / "dense_worsen_risk_lsf_v2.pt"
    merged_risk_path = output / "merged_risk_lsf_v2.pt"
    native_rank_path = output / "native_rank_score.pt"
    torch.save(selector.float(), selector_path)
    torch.save(candidate_pool.cpu(), candidate_path)
    torch.save(safe_core.cpu(), safe_core_path)
    torch.save(hard_negative.float(), hard_negative_path)
    torch.save(dense_worsen.float(), dense_worsen_path)
    torch.save(merged_risk.float(), merged_risk_path)
    torch.save(native_rank.float(), native_rank_path)

    split_name = str(support_metadata.get("split", support_metadata.get("split_name", "unknown"))) or "unknown"
    if split_name == "unknown" and lsf_metadata:
        split_name = str(lsf_metadata.get("split_name", lsf_metadata.get("split", "unknown"))) or "unknown"
    scene_name = str(support_metadata.get("scene", lsf_metadata.get("scene", "")))
    summary = {
        "scene": scene_name,
        "split_name": split_name,
        "test_split_used": split_name.strip().lower() == "test",
        "num_gaussians": size,
        "source_count": int(source_idx.numel()),
        "candidate_pool_count": int(candidate_pool.numel()),
        "safe_core_count": int(safe_core.numel()),
        "candidate_pool": str(candidate_path),
        "selector": str(selector_path),
        "safe_core": str(safe_core_path),
        "merged_risk": str(merged_risk_path),
        "score_source": score_source_metadata,
        "support_score_mode": support_score_mode,
        "localization_support_field_path": "" if lsf_path is None else str(lsf_path),
    }
    manifest = {
        "method": "lsf_v2_native_proposal",
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": _command_from_argv(),
        "source_map": str(source),
        "solver_consensus_support_path": "" if support_path is None else str(support_path),
        "localization_support_field_path": "" if lsf_path is None else str(lsf_path),
        "output_dir": str(output),
        "single_path_deployment": True,
        "branch_selection": False,
        "split_name": split_name,
        "test_split_used": summary["test_split_used"],
        "score_source": score_source_metadata,
        "support_metadata": support_metadata,
        "localization_support_field": lsf_metadata,
        "hyperparameters": {
            "candidate_pool_size": int(candidate_pool_size),
            "safe_support_threshold": float(safe_support_threshold),
            "max_hard_negative_risk": float(max_hard_negative_risk),
            "max_dense_worsen_risk": float(max_dense_worsen_risk),
            "min_native_rank": float(min_native_rank),
            "score_source": str(score_source),
            "native_weight": float(native_weight),
            "support_weight": float(support_weight),
            "hard_negative_weight": float(hard_negative_weight),
            "dense_worsen_weight": float(dense_worsen_weight),
            "support_score_mode": support_score_mode,
            "solver_utility_weight": float(solver_utility_weight),
            "pose_information_weight": float(pose_information_weight),
            "hard_query_support_weight": float(hard_query_support_weight),
            "visibility_weight": float(visibility_weight),
            "ambiguity_weight": float(ambiguity_weight),
        },
        "artifacts": {
            "selector": str(selector_path),
            "candidate_pool": str(candidate_path),
            "safe_core": str(safe_core_path),
            "hard_negative_risk": str(hard_negative_path),
            "dense_worsen_risk": str(dense_worsen_path),
            "merged_risk": str(merged_risk_path),
            "native_rank": str(native_rank_path),
        },
        "summary": summary,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    split_audit = artifact_split_audit(
        support_metadata,
        lsf_metadata,
        branch_selection=False,
    )
    metrics_summary = {
        "num_gaussians": size,
        "source_count": int(source_idx.numel()),
        "candidate_pool_count": int(candidate_pool.numel()),
        "safe_core_count": int(safe_core.numel()),
        "support_nonzero": int((support > 0).sum().item()),
        "hard_query_support_nonzero": int((hard_query_support > 0).sum().item()),
    }
    write_artifact_audit_bundle(
        output,
        manifest=manifest,
        command=manifest["command"],
        metrics_summary=metrics_summary,
        split_audit=split_audit,
    )
    return summary
