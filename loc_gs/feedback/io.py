from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from loc_gs.feedback.labels import derive_hard_negative_labels
from loc_gs.feedback.schema import FeedbackBankSummary, FeedbackMatchRecord


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


def _manifest(metadata: dict[str, Any]) -> dict[str, Any]:
    manifest = dict(metadata)
    manifest.setdefault("git_commit", _git_commit())
    manifest.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat())
    manifest.setdefault("scene", str(metadata.get("scene", "")))
    manifest.setdefault("split_name", str(metadata.get("split_name", metadata.get("split", ""))))
    manifest.setdefault("command", metadata.get("command", ""))
    return manifest


def _record_dict(record: FeedbackMatchRecord | dict[str, Any]) -> dict[str, Any]:
    return FeedbackMatchRecord.from_mapping(record).to_dict()


def save_feedback_bank(
    path: str | Path,
    records: Iterable[FeedbackMatchRecord | dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Save a feedback bank and include manifest metadata in the file."""
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(dict(metadata or {}))
    record_dicts = [_record_dict(record) for record in records]
    if dst.suffix.lower() == ".jsonl":
        with dst.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"type": "manifest", "manifest": manifest}, sort_keys=True) + "\n")
            for record in record_dicts:
                handle.write(json.dumps({"type": "record", "record": record}, sort_keys=True) + "\n")
    else:
        payload = {"manifest": manifest, "records": record_dicts}
        dst.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {"manifest": manifest, "record_count": len(record_dicts), "path": str(dst)}


def _load_jsonl(path: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        item = json.loads(raw)
        if item.get("type") == "manifest":
            manifest = dict(item.get("manifest", {}))
        elif item.get("type") == "record":
            records.append(FeedbackMatchRecord.from_mapping(item.get("record", {})).to_dict())
        else:
            records.append(FeedbackMatchRecord.from_mapping(item).to_dict())
    return {"manifest": manifest, "records": records}


def load_feedback_bank(path: str | Path) -> dict[str, Any]:
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"feedback bank not found: {src}")
    if src.suffix.lower() == ".jsonl":
        return _load_jsonl(src)
    data = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"feedback bank must be a JSON object: {src}")
    records = [FeedbackMatchRecord.from_mapping(record).to_dict() for record in data.get("records", [])]
    return {"manifest": dict(data.get("manifest", {})), "records": records}


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def summarize_feedback_bank(path: str | Path) -> dict[str, Any]:
    bank = load_feedback_bank(path)
    records = [FeedbackMatchRecord.from_mapping(record) for record in bank.get("records", [])]
    count = len(records)
    scenes = sorted({record.scene for record in records if record.scene})
    hard_negative_labels = derive_hard_negative_labels(records)
    pose_t = [record.pose_error_t_cm for record in records if record.pose_error_t_cm is not None]
    pose_r = [record.pose_error_r_deg for record in records if record.pose_error_r_deg is not None]
    summary = FeedbackBankSummary(
        record_count=count,
        scene_count=len(scenes),
        scenes=scenes,
        pnp_inlier_rate=(sum(1 for record in records if record.pnp_inlier) / count) if count else 0.0,
        hard_negative_rate=(sum(hard_negative_labels) / count) if count else 0.0,
        pnp_success_rate=(sum(1 for record in records if record.pnp_success) / count) if count else 0.0,
        dense_refine_success_rate=(sum(1 for record in records if record.dense_refine_success) / count) if count else 0.0,
        mean_pose_error_t_cm=_mean([float(v) for v in pose_t]),
        mean_pose_error_r_deg=_mean([float(v) for v in pose_r]),
    ).to_dict()
    summary["manifest"] = bank.get("manifest", {})
    return summary
