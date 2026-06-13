#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import numpy as np

from loc_gs.feedback.sparse_feedback_v4 import build_sparse_feedback_v4


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _git_status(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=str(root), text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"git status unavailable: {exc}\n"


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_payload(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".pt":
        payload = _torch_load(path)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"sparse trace payload must be a dict: {path}")
    return payload


def _json_key(key: Any) -> str:
    if isinstance(key, tuple):
        return "::".join(str(part) for part in key)
    return str(key)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {_json_key(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _compact_feedback_for_json(feedback: Mapping[str, Any], *, max_correspondences: int = 5000) -> dict[str, Any]:
    out = {str(key): value for key, value in feedback.items()}
    compact_rows: list[dict[str, Any]] = []
    query_descriptor_dim = 0
    landmark_descriptor_dim = 0
    descriptor_count = 0
    rows = list(feedback.get("correspondences", []))
    max_rows = max(0, int(max_correspondences))
    for row in rows[:max_rows]:
        if not isinstance(row, Mapping):
            compact_rows.append(row)
            continue
        compact_row = {
            str(key): value
            for key, value in row.items()
            if str(key) not in {"query_descriptor", "landmark_descriptor"}
        }
        if "query_descriptor" in row:
            descriptor_count += 1
            if query_descriptor_dim <= 0:
                query_descriptor_dim = int(torch.as_tensor(row.get("query_descriptor", [])).reshape(-1).numel())
        if "landmark_descriptor" in row and landmark_descriptor_dim <= 0:
            landmark_descriptor_dim = int(torch.as_tensor(row.get("landmark_descriptor", [])).reshape(-1).numel())
        compact_rows.append(compact_row)
    out["correspondences"] = compact_rows
    out["json_correspondence_count"] = int(len(compact_rows))
    out["json_correspondence_limit"] = int(max_rows)
    out["json_correspondences_truncated"] = bool(len(rows) > len(compact_rows))
    out["correspondence_descriptor_count"] = int(descriptor_count)
    out["query_descriptor_dim"] = int(query_descriptor_dim)
    out["landmark_descriptor_dim"] = int(landmark_descriptor_dim)
    out["correspondence_descriptors_omitted_from_json"] = True
    return out


def _strip_descriptor_fields(row: Any) -> Any:
    if not isinstance(row, Mapping):
        return row
    return {
        str(key): value
        for key, value in row.items()
        if str(key) not in {"query_descriptor", "landmark_descriptor"}
    }


def _label_only_feedback(feedback: Mapping[str, Any]) -> dict[str, Any]:
    out = {str(key): value for key, value in feedback.items()}
    out["correspondences"] = [_strip_descriptor_fields(row) for row in feedback.get("correspondences", [])]
    out["descriptor_fields_stripped"] = True
    return out


def _query_tokens_from_feedback(feedback: Mapping[str, Any]) -> dict[str, Any]:
    grouped: dict[str, dict[int, torch.Tensor]] = {}
    descriptor_dim = 0
    for row in feedback.get("correspondences", []):
        if not isinstance(row, Mapping) or "query_descriptor" not in row:
            continue
        query_id = str(row.get("query_id", "")).strip()
        if not query_id:
            continue
        try:
            keypoint_index = int(row.get("query_keypoint_index", row.get("match_row_index", len(grouped.get(query_id, {})))))
        except (TypeError, ValueError):
            keypoint_index = len(grouped.get(query_id, {}))
        descriptor = torch.as_tensor(row.get("query_descriptor", []), dtype=torch.float32).reshape(-1)
        if descriptor.numel() == 0:
            continue
        descriptor = torch.nn.functional.normalize(descriptor, p=2, dim=0)
        descriptor_dim = max(descriptor_dim, int(descriptor.numel()))
        xy = torch.as_tensor(row.get("keypoint_xy", row.get("query_xy", [0.0, 0.0])), dtype=torch.float32).reshape(-1)
        if xy.numel() < 2:
            xy = torch.zeros((2,), dtype=torch.float32)
        score = float(row.get("detector_score", 0.0) or 0.0)
        token = torch.cat([descriptor, xy[:2] / 640.0, torch.tensor([score], dtype=torch.float32)], dim=0)
        grouped.setdefault(query_id, {}).setdefault(keypoint_index, token)

    query_tokens = {
        query_id: torch.stack([token for _idx, token in sorted(rows.items())], dim=0)
        for query_id, rows in grouped.items()
        if rows
    }
    split_audit = feedback.get("split_audit", {})
    return {
        "schema_version": "sparse_solver_feedback_query_tokens_v1",
        "split_name": str(feedback.get("split_name", feedback.get("split", ""))),
        "query_tokens": query_tokens,
        "query_token_dim": int(descriptor_dim + 3) if descriptor_dim else 0,
        "query_count": int(len(query_tokens)),
        "query_token_source_role": "per_keypoint_runtime_schema",
        "split_audit": split_audit if isinstance(split_audit, Mapping) else {},
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build split-safe sparse solver Feedback v4 artifacts.")
    parser.add_argument("--trace", required=True, type=Path, help="Sparse trace payload (.pt or .json).")
    parser.add_argument("--output_dir", required=True, type=Path, help="Directory for Feedback v4 artifacts.")
    parser.add_argument("--protected_te_cm", default=10.0, type=float)
    parser.add_argument("--hard_te_cm", default=20.0, type=float)
    parser.add_argument("--low_reprojection_px", default=4.0, type=float)
    parser.add_argument("--high_score_threshold", default=0.8, type=float)
    parser.add_argument("--harmful_regression_cm", default=20.0, type=float)
    parser.add_argument("--min_descriptor_margin", default=0.05, type=float)
    parser.add_argument("--risky_competitor_max_margin", default=0.05, type=float)
    parser.add_argument("--max_json_correspondences", default=5000, type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)
    command = list(sys.argv if argv is None else [sys.argv[0], *argv])
    repo = Path(__file__).resolve().parents[2]
    trace_path = Path(args.trace)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = _load_payload(trace_path)
    feedback = build_sparse_feedback_v4(
        payload,
        protected_te_cm=float(args.protected_te_cm),
        hard_te_cm=float(args.hard_te_cm),
        low_reprojection_px=float(args.low_reprojection_px),
        high_score_threshold=float(args.high_score_threshold),
        harmful_regression_cm=float(args.harmful_regression_cm),
        min_descriptor_margin=float(args.min_descriptor_margin),
        risky_competitor_max_margin=float(args.risky_competitor_max_margin),
    )

    torch.save(feedback, output_dir / "feedback_v4.pt")
    torch.save(_label_only_feedback(feedback), output_dir / "feedback_v4_label_only.pt")
    torch.save(_query_tokens_from_feedback(feedback), output_dir / "query_tokens.pt")
    (output_dir / "feedback_v4.json").write_text(
        json.dumps(
            _json_safe(_compact_feedback_for_json(feedback, max_correspondences=int(args.max_json_correspondences))),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(_json_safe(feedback["metrics"]), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "split_audit.json").write_text(
        json.dumps(_json_safe(feedback["split_audit"]), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema_version": "sparse_feedback_v4_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(repo),
        "command": command,
        "trace": str(trace_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "split_name": feedback["split_name"],
        "test_split_used": False,
        "hyperparameters": {
            "protected_te_cm": float(args.protected_te_cm),
            "hard_te_cm": float(args.hard_te_cm),
            "low_reprojection_px": float(args.low_reprojection_px),
            "high_score_threshold": float(args.high_score_threshold),
            "harmful_regression_cm": float(args.harmful_regression_cm),
            "min_descriptor_margin": float(args.min_descriptor_margin),
            "risky_competitor_max_margin": float(args.risky_competitor_max_margin),
            "max_json_correspondences": int(args.max_json_correspondences),
        },
        "outputs": {
            "feedback_v4_pt": str((output_dir / "feedback_v4.pt").resolve()),
            "feedback_v4_label_only_pt": str((output_dir / "feedback_v4_label_only.pt").resolve()),
            "query_tokens_pt": str((output_dir / "query_tokens.pt").resolve()),
            "feedback_v4_json": str((output_dir / "feedback_v4.json").resolve()),
            "metrics_summary": str((output_dir / "metrics_summary.json").resolve()),
            "split_audit": str((output_dir / "split_audit.json").resolve()),
        },
        "metrics": feedback["metrics"],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "command.txt").write_text(
        " ".join(shlex.quote(part) for part in command) + "\n",
        encoding="utf-8",
    )
    (output_dir / "git_status.txt").write_text(_git_status(repo), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
