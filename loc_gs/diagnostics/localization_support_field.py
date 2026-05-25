from __future__ import annotations

import json
import math
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from loc_gs.reporting.artifact_audit import artifact_split_audit, write_artifact_audit_bundle


SCHEMA = "localization_support_field_v1"
_UTILITY_FIELDS = (
    "support",
    "viable_tuple_mass",
    "logdet_H",
    "min_eigenvalue",
    "dense_worsen_risk",
    "ambiguity",
)
_DEFAULT_WEIGHTS = {
    "support": 1.0,
    "viable_tuple_mass": 1.0,
    "logdet_H": 1.0,
    "min_eigenvalue": 1.0,
    "dense_worsen_risk": -1.0,
    "ambiguity": -1.0,
}
_ALIASES = {
    "min_eigen": "min_eigenvalue",
    "min_eigen_H": "min_eigenvalue",
    "lambda_min": "min_eigenvalue",
    "dense_worsen": "dense_worsen_risk",
    "dense_risk": "dense_worsen_risk",
}


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


def _command_from_argv() -> str:
    return " ".join(shlex.quote(part) for part in sys.argv) if sys.argv else ""


def _as_tensor(payload: Mapping[str, Any], key: str, *, size: int, dtype: torch.dtype, default: float = 0.0) -> torch.Tensor:
    if key not in payload:
        return torch.full((int(size),), default, dtype=dtype)
    tensor = torch.as_tensor(payload[key], dtype=dtype).reshape(-1).cpu()
    if tensor.numel() != int(size):
        raise ValueError(f"{key} length {tensor.numel()} does not match support length {size}")
    return tensor


def _rank_normalize(values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    tensor = values.float().reshape(-1).cpu()
    valid = torch.isfinite(tensor)
    if mask is not None:
        valid = valid & mask.bool().reshape(-1).cpu()
    out = torch.zeros_like(tensor)
    if not bool(valid.any().item()):
        return out
    indices = torch.where(valid)[0]
    selected = tensor[indices]
    if selected.numel() == 1:
        out[indices] = 1.0
        return out
    order = torch.argsort(selected, descending=False, stable=True)
    ranks = torch.empty_like(selected)
    ranks[order] = torch.arange(selected.numel(), dtype=torch.float32)
    out[indices] = ranks / float(selected.numel() - 1)
    return out.clamp(0.0, 1.0)


def _positive_max_normalize(values: torch.Tensor) -> torch.Tensor:
    tensor = values.float().reshape(-1).cpu()
    out = torch.zeros_like(tensor)
    positive = torch.isfinite(tensor) & (tensor > 0)
    if not bool(positive.any().item()):
        return out
    max_value = float(tensor[positive].max().item())
    if max_value <= 0.0:
        return out
    out[positive] = tensor[positive] / max_value
    return out.clamp(0.0, 1.0)


def _load_support(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("solver consensus support artifact must be a dict")
    if "support_score" not in payload:
        raise KeyError("solver consensus support artifact must contain support_score")
    support = torch.as_tensor(payload["support_score"], dtype=torch.float32).reshape(-1).cpu()
    size = int(support.numel())
    tensors = {
        "support_score": support.clamp(0.0, 1.0),
        "observed_count": _as_tensor(payload, "observed_count", size=size, dtype=torch.long, default=0),
        "inlier_consensus": _as_tensor(payload, "inlier_consensus", size=size, dtype=torch.float32, default=0.0),
        "hard_negative_risk": _as_tensor(payload, "hard_negative_risk", size=size, dtype=torch.float32, default=0.0).clamp(0.0, 1.0),
        "dense_worsen_risk": _as_tensor(payload, "dense_worsen_risk", size=size, dtype=torch.float32, default=0.0).clamp(0.0, 1.0),
    }
    metadata = dict(payload.get("metadata", {})) if isinstance(payload.get("metadata", {}), dict) else {}
    split = str(metadata.get("split", metadata.get("split_name", ""))).strip().lower()
    if split == "test":
        raise ValueError("test split support artifacts are not allowed")
    return tensors, metadata


def _metric(metrics: Mapping[str, Any], name: str) -> float:
    keys = [name] + [alias for alias, canonical in _ALIASES.items() if canonical == name]
    for key in keys:
        if key not in metrics:
            continue
        try:
            value = float(metrics[key])
        except (TypeError, ValueError):
            return 0.0
        return value if math.isfinite(value) else 0.0
    return 0.0


def _weights(payload: Mapping[str, Any] | None) -> dict[str, float]:
    out = dict(_DEFAULT_WEIGHTS)
    if not payload:
        return out
    thresholds = payload.get("thresholds", {}) if isinstance(payload, Mapping) else {}
    source = thresholds.get("cvar_weights", {}) if isinstance(thresholds, Mapping) else {}
    if not isinstance(source, Mapping):
        return out
    for key, value in source.items():
        canonical = _ALIASES.get(str(key), str(key))
        if canonical not in out:
            continue
        try:
            out[canonical] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _load_solver_constraints(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    split = str(payload.get("metadata", {}).get("split_name", payload.get("split_name", ""))).strip().lower()
    if split == "test":
        raise ValueError("test split solver constraints are not allowed")
    return payload


def _accumulate_query_metrics(
    payload: Mapping[str, Any] | None,
    *,
    size: int,
) -> dict[str, torch.Tensor]:
    utility = torch.zeros((int(size),), dtype=torch.float32)
    candidate_utility = torch.zeros_like(utility)
    source_utility = torch.zeros_like(utility)
    pose_information = torch.zeros_like(utility)
    ambiguity = torch.zeros_like(utility)
    dense_query_risk = torch.zeros_like(utility)
    hard_query_support = torch.zeros_like(utility)
    if not payload:
        return {
            "solver_set_utility": utility,
            "candidate_gain_utility": candidate_utility,
            "source_loss_utility": source_utility,
            "pose_information": pose_information,
            "hard_query_support": hard_query_support,
            "ambiguity_risk": ambiguity,
            "query_dense_worsen_risk": dense_query_risk,
        }
    weights = _weights(payload)

    def add_table(table_name: str, target: torch.Tensor) -> None:
        table = payload.get(table_name, {})
        if not isinstance(table, Mapping):
            return
        for gid_raw, query_map in table.items():
            gid = int(gid_raw)
            if gid < 0 or gid >= int(size) or not isinstance(query_map, Mapping):
                continue
            for metrics in query_map.values():
                if not isinstance(metrics, Mapping):
                    continue
                score = 0.0
                for field in _UTILITY_FIELDS:
                    score += float(weights[field]) * _metric(metrics, field)
                target[gid] += float(score)
                pose_information[gid] += max(0.0, _metric(metrics, "logdet_H")) + max(
                    0.0, _metric(metrics, "min_eigenvalue")
                )
                hard_query_support[gid] += max(0.0, _metric(metrics, "support")) + max(
                    0.0, _metric(metrics, "viable_tuple_mass")
                )
                ambiguity[gid] += max(0.0, _metric(metrics, "ambiguity"))
                dense_query_risk[gid] += max(0.0, _metric(metrics, "dense_worsen_risk"))

    add_table("candidate_gain", candidate_utility)
    add_table("source_loss", source_utility)
    utility = candidate_utility + source_utility
    return {
        "solver_set_utility": _positive_max_normalize(utility),
        "candidate_gain_utility": _positive_max_normalize(candidate_utility),
        "source_loss_utility": _positive_max_normalize(source_utility),
        "pose_information": _positive_max_normalize(pose_information),
        "hard_query_support": _positive_max_normalize(hard_query_support),
        "ambiguity_risk": _positive_max_normalize(ambiguity),
        "query_dense_worsen_risk": _positive_max_normalize(dense_query_risk),
    }


def build_localization_support_field(
    *,
    solver_consensus_support_path: str | Path,
    solver_admissibility_path: str | Path | None = None,
    support_score_mode: str = "raw",
) -> dict[str, Any]:
    """Build a canonical LSF artifact from solver feedback support and optional set-level constraints."""

    support_tensors, support_metadata = _load_support(solver_consensus_support_path)
    support_score_mode = str(support_score_mode or "raw")
    if support_score_mode not in {"raw", "rank_observed"}:
        raise ValueError(f"unsupported support_score_mode: {support_score_mode}")
    observed = support_tensors["observed_count"] > 0
    support_for_selection = (
        support_tensors["support_score"]
        if support_score_mode == "raw"
        else _rank_normalize(support_tensors["support_score"], observed)
    )
    constraints = _load_solver_constraints(solver_admissibility_path)
    set_level = _accumulate_query_metrics(constraints, size=int(support_for_selection.numel()))
    split_name = str(
        support_metadata.get("split_name", support_metadata.get("split", "unknown"))
    )
    metadata = {
        "schema": SCHEMA,
        "scene": str(support_metadata.get("scene", "")),
        "split_name": split_name,
        "support_score_mode": support_score_mode,
        "source_solver_consensus_support": str(solver_consensus_support_path),
        "source_solver_admissibility": "" if solver_admissibility_path is None else str(solver_admissibility_path),
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": _command_from_argv(),
        "support_metadata": support_metadata,
        "solver_admissibility_metadata": {} if constraints is None else dict(constraints.get("metadata", {})),
        "support_vector_fields": [
            "inlier_consensus",
            "hard_query_support",
            "pose_information_gain",
            "ambiguity_risk",
            "hard_negative_risk",
            "dense_worsen_risk",
            "visibility_stability",
        ],
        "single_path_deployment": True,
        "branch_selection": False,
    }
    visibility_stability = _positive_max_normalize(support_tensors["observed_count"].float())
    pose_information_gain = set_level["pose_information"].float().clamp(0.0, 1.0)
    return {
        "support_score": support_tensors["support_score"].float(),
        "support_for_selection": support_for_selection.float().clamp(0.0, 1.0),
        "observed_count": support_tensors["observed_count"].long(),
        "inlier_consensus": support_tensors["inlier_consensus"].float().clamp(0.0, 1.0),
        "hard_query_support": set_level["hard_query_support"].float().clamp(0.0, 1.0),
        "pose_information_gain": pose_information_gain,
        "visibility_stability": visibility_stability,
        "hard_negative_risk": support_tensors["hard_negative_risk"].float().clamp(0.0, 1.0),
        "dense_worsen_risk": torch.maximum(
            support_tensors["dense_worsen_risk"].float().clamp(0.0, 1.0),
            set_level["query_dense_worsen_risk"].float().clamp(0.0, 1.0),
        ),
        "solver_set_utility": set_level["solver_set_utility"].float().clamp(0.0, 1.0),
        "candidate_gain_utility": set_level["candidate_gain_utility"].float().clamp(0.0, 1.0),
        "source_loss_utility": set_level["source_loss_utility"].float().clamp(0.0, 1.0),
        "pose_information": set_level["pose_information"].float().clamp(0.0, 1.0),
        "ambiguity_risk": set_level["ambiguity_risk"].float().clamp(0.0, 1.0),
        "metadata": metadata,
    }


def write_localization_support_field(
    *,
    solver_consensus_support_path: str | Path,
    output_dir: str | Path,
    solver_admissibility_path: str | Path | None = None,
    support_score_mode: str = "raw",
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = build_localization_support_field(
        solver_consensus_support_path=solver_consensus_support_path,
        solver_admissibility_path=solver_admissibility_path,
        support_score_mode=support_score_mode,
    )
    artifact_path = output / "localization_support_field.pt"
    torch.save(payload, artifact_path)
    manifest = {
        "artifact": "localization_support_field.pt",
        "artifact_path": str(artifact_path),
        "schema": SCHEMA,
        "scene": payload["metadata"].get("scene", ""),
        "split_name": payload["metadata"].get("split_name", ""),
        "support_score_mode": support_score_mode,
        "solver_consensus_support_path": str(solver_consensus_support_path),
        "solver_admissibility_path": "" if solver_admissibility_path is None else str(solver_admissibility_path),
        "git_commit": payload["metadata"].get("git_commit", "unknown"),
        "timestamp_utc": payload["metadata"].get("timestamp_utc", ""),
        "command": payload["metadata"].get("command", ""),
        "data_root": "unknown",
        "checkpoint_path": "unknown",
        "map_path": "unknown",
        "hyperparameters": {"support_score_mode": support_score_mode},
        "split_audit": artifact_split_audit(
            payload["metadata"],
            payload["metadata"].get("support_metadata", {}),
            payload["metadata"].get("solver_admissibility_metadata", {}),
            branch_selection=False,
        ),
        "single_path_deployment": True,
        "branch_selection": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    metrics_summary = {
        "num_gaussians": int(payload["support_score"].numel()),
        "observed_count_nonzero": int((payload["observed_count"] > 0).sum().item()),
        "support_nonzero": int((payload["support_score"] > 0).sum().item()),
        "hard_query_support_nonzero": int((payload["hard_query_support"] > 0).sum().item()),
        "pose_information_nonzero": int((payload["pose_information_gain"] > 0).sum().item()),
    }
    write_artifact_audit_bundle(
        output,
        manifest=manifest,
        command=str(manifest.get("command", "")),
        metrics_summary=metrics_summary,
        split_audit=manifest["split_audit"],
    )
    return manifest
