from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


SCHEMA_VERSION = "ulfloc_solver_feedback_v1"


def _load_support_artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"solver support artifact not found: {path}")
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("solver support artifact must contain a dictionary payload")
    if "support_score" not in payload:
        raise KeyError("solver support artifact must contain support_score")
    return payload


def _metadata(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata", {})
    return dict(metadata) if isinstance(metadata, dict) else {}


def _split_name(payload: dict[str, Any]) -> str:
    metadata = _metadata(payload)
    split = str(
        metadata.get(
            "split_name",
            metadata.get("split", payload.get("split_name", payload.get("split", "unknown"))),
        )
    ).strip()
    return split or "unknown"


def _tensor(payload: dict[str, Any], key: str, length: int, default: float) -> torch.Tensor:
    if key not in payload:
        return torch.full((length,), float(default), dtype=torch.float32)
    value = torch.as_tensor(payload[key], dtype=torch.float32).reshape(-1).cpu()
    if value.numel() != length:
        raise ValueError(f"{key} length {value.numel()} does not match support_score length {length}")
    return value


def _long_tensor(payload: dict[str, Any], key: str, length: int, default: int) -> torch.Tensor:
    if key not in payload:
        return torch.full((length,), int(default), dtype=torch.long)
    value = torch.as_tensor(payload[key], dtype=torch.long).reshape(-1).cpu()
    if value.numel() != length:
        raise ValueError(f"{key} length {value.numel()} does not match support_score length {length}")
    return value


def _build_view_landmark_weights(
    payload: dict[str, Any],
    *,
    length: int,
    min_weight: float,
    max_weight: float,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    raw_positive = payload.get("per_query_support", payload.get("view_landmark_support", {}))
    raw_negative = payload.get(
        "per_query_negative_support",
        payload.get("view_landmark_negative_support", payload.get("per_query_risk", {})),
    )
    if not isinstance(raw_positive, dict):
        raw_positive = {}
    if not isinstance(raw_negative, dict):
        raw_negative = {}

    view_weights: dict[str, dict[str, torch.Tensor]] = {}
    entry_count = 0
    negative_entry_count = 0
    weight_lo = max(0.0, min(1.0, float(min_weight)))
    weight_hi = max(1.0, float(max_weight))

    views = sorted({str(view) for view in raw_positive} | {str(view) for view in raw_negative})
    for raw_view in views:
        raw_support = raw_positive.get(raw_view, {})
        raw_risk = raw_negative.get(raw_view, {})
        if not isinstance(raw_support, dict):
            raw_support = {}
        if not isinstance(raw_risk, dict):
            raw_risk = {}
        values_by_gid: dict[int, tuple[float, float]] = {}
        for raw_gid, raw_value in raw_support.items():
            try:
                gid = int(raw_gid)
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if gid < 0 or gid >= int(length) or value <= 0.0:
                continue
            positive, negative = values_by_gid.get(gid, (0.0, 0.0))
            values_by_gid[gid] = (positive + float(value), negative)
        for raw_gid, raw_value in raw_risk.items():
            try:
                gid = int(raw_gid)
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if gid < 0 or gid >= int(length) or value <= 0.0:
                continue
            positive, negative = values_by_gid.get(gid, (0.0, 0.0))
            values_by_gid[gid] = (positive, negative + float(value))
            negative_entry_count += 1

        if not values_by_gid:
            continue
        pos_max = max((positive for positive, _negative in values_by_gid.values()), default=0.0)
        neg_max = max((negative for _positive, negative in values_by_gid.values()), default=0.0)
        rows: list[tuple[int, float]] = []
        for gid, (positive, negative) in values_by_gid.items():
            pos_norm = float(positive) / max(float(pos_max), 1.0e-6) if positive > 0.0 else 0.0
            neg_norm = float(negative) / max(float(neg_max), 1.0e-6) if negative > 0.0 else 0.0
            net = max(-1.0, min(1.0, pos_norm - neg_norm))
            if net >= 0.0:
                weight = 1.0 + (weight_hi - 1.0) * net
            else:
                weight = 1.0 - (1.0 - weight_lo) * (-net)
            rows.append((gid, max(weight_lo, min(weight_hi, float(weight)))))
        if not rows:
            continue
        rows.sort(key=lambda item: item[0])
        ids = torch.as_tensor([gid for gid, _value in rows], dtype=torch.long)
        weights = torch.as_tensor([value for _gid, value in rows], dtype=torch.float32)
        view_weights[str(raw_view)] = {
            "landmark_ids": ids,
            "weights": weights,
        }
        entry_count += int(ids.numel())

    return view_weights, {
        "view_count": int(len(view_weights)),
        "entry_count": int(entry_count),
        "negative_entry_count": int(negative_entry_count),
        "min_weight": float(weight_lo),
        "max_weight": float(weight_hi),
        "source_key": (
            "per_query_support+per_query_negative_support"
            if "per_query_negative_support" in payload
            else ("per_query_support" if "per_query_support" in payload else "view_landmark_support")
        ),
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _git_status() -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], text=True)
    except Exception as exc:
        return f"git status unavailable: {exc}\n"


def build_ulfloc_solver_feedback(
    support_payload: dict[str, Any],
    *,
    scene: str,
    alpha: float,
    risk_penalty: float,
    min_weight: float,
    max_weight: float,
    guarded: bool = False,
    min_boost_support: float = 0.0,
    min_boost_positive_count: int = 0,
    max_boost_risk: float = 1.0,
    min_deboost_risk: float = 1.0,
    max_deboost_positive_count: int = 0,
) -> dict[str, Any]:
    support = torch.as_tensor(support_payload["support_score"], dtype=torch.float32).reshape(-1).cpu().clamp(0.0, 1.0)
    count = int(support.numel())
    hard_negative = _tensor(support_payload, "hard_negative_risk", count, 0.0).clamp(0.0, 1.0)
    dense_worsen = _tensor(support_payload, "dense_worsen_risk", count, 0.0).clamp(0.0, 1.0)
    artifact = _tensor(support_payload, "artifact_risk", count, 0.0).clamp(0.0, 1.0)
    observed_count = torch.as_tensor(
        support_payload.get("observed_count", torch.ones((count,), dtype=torch.long)),
        dtype=torch.long,
    ).reshape(-1).cpu()
    if observed_count.numel() != count:
        raise ValueError(
            f"observed_count length {observed_count.numel()} does not match support_score length {count}"
        )

    positive_observed_count = _long_tensor(support_payload, "positive_observed_count", count, 0)
    risk = torch.maximum(torch.maximum(hard_negative, dense_worsen), artifact)
    signal = support - (float(risk_penalty) * risk)
    weights = 1.0 + (float(alpha) * signal)
    weights = weights.clamp(float(min_weight), float(max_weight)).to(torch.float32)
    weights[observed_count <= 0] = 1.0
    guarded_summary = None
    if bool(guarded):
        observed = observed_count > 0
        boost_mask = (
            observed
            & (weights > 1.0)
            & (support >= float(min_boost_support))
            & (positive_observed_count >= int(min_boost_positive_count))
            & (risk <= float(max_boost_risk))
        )
        deboost_mask = (
            observed
            & (weights < 1.0)
            & (risk >= float(min_deboost_risk))
            & (positive_observed_count <= int(max_deboost_positive_count))
        )
        guarded_weights = torch.ones_like(weights)
        guarded_weights[boost_mask | deboost_mask] = weights[boost_mask | deboost_mask]
        weights = guarded_weights
        guarded_summary = {
            "boosted_count": int(boost_mask.sum().item()),
            "deboosted_count": int(deboost_mask.sum().item()),
            "neutral_count": int((~(boost_mask | deboost_mask)).sum().item()),
        }

    metadata = _metadata(support_payload)
    split_name = _split_name(support_payload)
    if split_name.lower() == "test":
        raise ValueError("refusing to export ULF-Loc solver feedback from test split")
    view_landmark_weights, view_weight_summary = _build_view_landmark_weights(
        support_payload,
        length=count,
        min_weight=float(min_weight),
        max_weight=float(max_weight),
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "scene": scene,
        "split_name": split_name,
        "num_landmarks": count,
        "landmark_weights": weights,
        "view_landmark_weights": view_landmark_weights,
        "landmark_support": support,
        "landmark_risk": risk,
        "observed_count": observed_count,
        "positive_observed_count": positive_observed_count,
        "metadata": {
            "source_solver_support_metadata": metadata,
            "hyperparameters": {
                "alpha": float(alpha),
                "risk_penalty": float(risk_penalty),
                "min_weight": float(min_weight),
                "max_weight": float(max_weight),
                "guarded": bool(guarded),
                "min_boost_support": float(min_boost_support),
                "min_boost_positive_count": int(min_boost_positive_count),
                "max_boost_risk": float(max_boost_risk),
                "min_deboost_risk": float(min_deboost_risk),
                "max_deboost_positive_count": int(max_deboost_positive_count),
            },
            "guarded_summary": guarded_summary
            if guarded_summary is not None
            else {"boosted_count": 0, "deboosted_count": 0, "neutral_count": int(count)},
            "view_landmark_weight_summary": view_weight_summary,
        },
    }


def _summary(artifact: dict[str, Any]) -> dict[str, Any]:
    weights = torch.as_tensor(artifact["landmark_weights"], dtype=torch.float32)
    risk = torch.as_tensor(artifact["landmark_risk"], dtype=torch.float32)
    support = torch.as_tensor(artifact["landmark_support"], dtype=torch.float32)
    observed = torch.as_tensor(artifact["observed_count"], dtype=torch.long) > 0
    summary = {
        "schema_version": artifact["schema_version"],
        "scene": artifact["scene"],
        "split_name": artifact["split_name"],
        "num_landmarks": int(weights.numel()),
        "observed_landmarks": int(observed.sum().item()),
        "weight_min": float(weights.min().item()) if weights.numel() else 0.0,
        "weight_max": float(weights.max().item()) if weights.numel() else 0.0,
        "weight_mean": float(weights.mean().item()) if weights.numel() else 0.0,
        "support_mean_observed": float(support[observed].mean().item()) if observed.any() else 0.0,
        "risk_mean_observed": float(risk[observed].mean().item()) if observed.any() else 0.0,
    }
    view_summary = artifact.get("metadata", {}).get("view_landmark_weight_summary")
    if isinstance(view_summary, dict):
        summary.update({f"view_landmark_weights_{key}": value for key, value in view_summary.items()})
    guarded_summary = artifact.get("metadata", {}).get("guarded_summary")
    if isinstance(guarded_summary, dict):
        summary.update({f"guarded_{key}": value for key, value in guarded_summary.items()})
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export ULF-Loc compatible solver-feedback weights.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--solver_support", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--risk_penalty", type=float, default=1.0)
    parser.add_argument("--min_weight", type=float, default=0.25)
    parser.add_argument("--max_weight", type=float, default=1.75)
    parser.add_argument("--guarded", action="store_true")
    parser.add_argument("--min_boost_support", type=float, default=0.0)
    parser.add_argument("--min_boost_positive_count", type=int, default=0)
    parser.add_argument("--max_boost_risk", type=float, default=1.0)
    parser.add_argument("--min_deboost_risk", type=float, default=1.0)
    parser.add_argument("--max_deboost_positive_count", type=int, default=0)
    return parser


def main(args: argparse.Namespace | None = None) -> int:
    if args is None:
        args = build_argparser().parse_args()

    support_payload = _load_support_artifact(Path(args.solver_support))
    artifact = build_ulfloc_solver_feedback(
        support_payload,
        scene=str(args.scene),
        alpha=float(args.alpha),
        risk_penalty=float(args.risk_penalty),
        min_weight=float(args.min_weight),
        max_weight=float(args.max_weight),
        guarded=bool(args.guarded),
        min_boost_support=float(args.min_boost_support),
        min_boost_positive_count=int(args.min_boost_positive_count),
        max_boost_risk=float(args.max_boost_risk),
        min_deboost_risk=float(args.min_deboost_risk),
        max_deboost_positive_count=int(args.max_deboost_positive_count),
    )
    summary = _summary(artifact)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = output_dir / "solver_feedback.pkl"
    with artifact_path.open("wb") as f:
        pickle.dump(artifact, f)

    split_audit = {
        "status": "passed" if artifact["split_name"].lower() != "test" else "failed",
        "split_name": artifact["split_name"],
        "test_split_used": artifact["split_name"].lower() == "test",
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "command": " ".join(sys.argv),
        "scene": str(args.scene),
        "split_name": artifact["split_name"],
        "test_split_used": bool(split_audit["test_split_used"]),
        "solver_support": str(Path(args.solver_support)),
        "artifact_path": str(artifact_path),
        "hyperparameters": artifact["metadata"]["hyperparameters"],
        "summary": summary,
    }

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "split_audit.json").write_text(json.dumps(split_audit, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "command.txt").write_text(manifest["command"] + "\n", encoding="utf-8")
    (output_dir / "git_status.txt").write_text(_git_status(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
