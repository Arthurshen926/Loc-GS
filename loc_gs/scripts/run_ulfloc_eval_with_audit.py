#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from loc_gs.export.manifest import write_export_eval_audit_bundle
from loc_gs.scripts.export_ulfloc_sparse_feedback import (
    load_dataset_image_list,
    resolve_ulfloc_source_path_for_loader,
)


RECALL_THRESHOLDS = {
    "R50cm5deg": (50.0, 5.0),
    "R15cm5deg": (15.0, 5.0),
    "R10cm5deg": (10.0, 5.0),
    "R5cm5deg": (5.0, 5.0),
    "R2cm2deg": (2.0, 2.0),
}


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _stage_pairs(results: Sequence[Any], stage_name: str) -> tuple[list[float], list[float], list[float]]:
    te_key = f"{stage_name}_TE"
    re_key = f"{stage_name}_AE"
    inliers: list[float] = []
    tes: list[float] = []
    res: list[float] = []
    for row in results:
        if not isinstance(row, Mapping):
            continue
        te = _safe_float(row.get(te_key))
        re = _safe_float(row.get(re_key))
        if te is None or re is None:
            continue
        tes.append(te)
        res.append(re)
        stage_payload = row.get(stage_name)
        if stage_name == "dense" and isinstance(stage_payload, Sequence) and stage_payload:
            stage_payload = stage_payload[-1]
        if isinstance(stage_payload, Mapping):
            inlier = _safe_float(stage_payload.get("inliers"))
            if inlier is not None:
                inliers.append(inlier)
    return tes, res, inliers


def _image_names_from_ulfloc_output_log(output_dir: Path) -> list[str]:
    log_path = output_dir / "output.log"
    if not log_path.exists():
        return []
    names: list[str] = []
    prefix = "Localize image:"
    for raw in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line.startswith(prefix):
            continue
        name = line[len(prefix) :].strip()
        if name:
            names.append(name)
    return names


def _attach_result_image_names_from_log(output_dir: Path, results: list[Any]) -> str:
    if not results or all(isinstance(row, Mapping) and row.get("image_name") for row in results):
        return "results.json:image_name"

    log_names = _image_names_from_ulfloc_output_log(output_dir)
    if len(log_names) != len(results):
        return "unknown"

    repaired: list[Any] = []
    for row, image_name in zip(results, log_names):
        if isinstance(row, Mapping):
            updated = dict(row)
            updated.setdefault("image_name", image_name)
            repaired.append(updated)
        else:
            repaired.append(row)
    results[:] = repaired
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return "output.log:Localize image"


def _cvar_upper_tail(values: Sequence[float], alpha: float = 0.1) -> float | None:
    if not values:
        return None
    count = max(1, int(math.ceil(len(values) * float(alpha))))
    return float(np.mean(sorted(values)[-count:]))


def _stage_metrics_from_results(results: Sequence[Any], stage_name: str) -> dict[str, float]:
    tes, res, inliers = _stage_pairs(results, stage_name)
    if not tes:
        return {}
    te_array = np.asarray(tes, dtype=np.float64)
    re_array = np.asarray(res, dtype=np.float64)
    metrics: dict[str, float] = {
        "median_te_cm": float(np.median(te_array)),
        "median_re_deg": float(np.median(re_array)),
        "p90_te_cm": float(np.percentile(te_array, 90)),
        "p95_te_cm": float(np.percentile(te_array, 95)),
        "cvar10_te_cm": float(_cvar_upper_tail(tes, 0.1) or 0.0),
        "severe_rate_1m": float(np.mean(te_array > 100.0)),
        "catastrophic_rate_5m": float(np.mean(te_array > 500.0)),
    }
    for name, (max_te, max_re) in RECALL_THRESHOLDS.items():
        metrics[name] = float(np.mean((te_array <= max_te) & (re_array <= max_re)))
    if inliers:
        metrics["avg_inliers"] = float(np.mean(np.asarray(inliers, dtype=np.float64)))
    return metrics


def _paired_sparse_dense_metrics(results: Sequence[Any]) -> dict[str, float]:
    sparse_tes: list[float] = []
    sparse_res: list[float] = []
    dense_tes: list[float] = []
    dense_res: list[float] = []
    for row in results:
        if not isinstance(row, Mapping):
            continue
        sparse_te = _safe_float(row.get("sparse_TE"))
        sparse_re = _safe_float(row.get("sparse_AE"))
        dense_te = _safe_float(row.get("dense_TE"))
        dense_re = _safe_float(row.get("dense_AE"))
        if sparse_te is None or sparse_re is None or dense_te is None or dense_re is None:
            continue
        sparse_tes.append(sparse_te)
        sparse_res.append(sparse_re)
        dense_tes.append(dense_te)
        dense_res.append(dense_re)

    if not sparse_tes:
        return {}

    sparse_te_array = np.asarray(sparse_tes, dtype=np.float64)
    sparse_re_array = np.asarray(sparse_res, dtype=np.float64)
    dense_te_array = np.asarray(dense_tes, dtype=np.float64)
    dense_re_array = np.asarray(dense_res, dtype=np.float64)
    delta_te = dense_te_array - sparse_te_array
    sparse_correct = (sparse_te_array <= 10.0) & (sparse_re_array <= 5.0)
    dense_correct = (dense_te_array <= 10.0) & (dense_re_array <= 5.0)
    return {
        "pair_count": int(delta_te.size),
        "dense_worsened_count": int(np.sum(delta_te > 0.0)),
        "dense_worsened_rate": float(np.mean(delta_te > 0.0)),
        "dense_worsened_5cm_count": int(np.sum(delta_te > 5.0)),
        "dense_regression_20cm_count": int(np.sum(delta_te > 20.0)),
        "dense_regression_50cm_count": int(np.sum(delta_te > 50.0)),
        "dense_improved_20cm_count": int(np.sum(delta_te < -20.0)),
        "dense_improved_50cm_count": int(np.sum(delta_te < -50.0)),
        "median_dense_minus_sparse_te_cm": float(np.median(delta_te)),
        "mean_dense_minus_sparse_te_cm": float(np.mean(delta_te)),
        "sparse_correct_dense_wrong_count": int(np.sum(sparse_correct & ~dense_correct)),
        "sparse_wrong_dense_correct_count": int(np.sum(~sparse_correct & dense_correct)),
    }


def _dense_guard_metrics(results: Sequence[Any]) -> dict[str, float]:
    guard_entry_count = 0
    accept_dense_count = 0
    reject_dense_keep_sparse_count = 0
    retry_entry_count = 0
    native_dense_rejected_before_retry_count = 0
    native_dense_risky_before_retry_count = 0
    selected_sparse_count = 0
    selected_dense_count = 0
    selected_dense_feedback_retry_count = 0

    for row in results:
        if not isinstance(row, Mapping):
            continue
        dense_payload = row.get("dense")
        if isinstance(dense_payload, Mapping):
            dense_entries: Sequence[Any] = [dense_payload]
        elif isinstance(dense_payload, Sequence) and not isinstance(dense_payload, (str, bytes)):
            dense_entries = dense_payload
        else:
            dense_entries = []

        for entry in dense_entries:
            if not isinstance(entry, Mapping):
                continue
            guard = entry.get("dense_transition_guard")
            if isinstance(guard, Mapping):
                guard_entry_count += 1
                decision = str(guard.get("decision", "")).strip()
                if decision == "accept_dense":
                    accept_dense_count += 1
                elif decision == "reject_dense_keep_sparse":
                    reject_dense_keep_sparse_count += 1
            if bool(entry.get("dense_render_feedback_retry", False)):
                retry_entry_count += 1
            selected_source = str(entry.get("selected_pose_source", "")).strip()
            if selected_source == "sparse":
                selected_sparse_count += 1
            elif selected_source in {"dense", "dense_feedback_retry"}:
                selected_dense_count += 1
            if selected_source == "dense_feedback_retry":
                selected_dense_feedback_retry_count += 1
            if selected_source == "native_dense_rejected_before_retry" or bool(
                entry.get("native_dense_rejected_before_retry", False)
            ):
                native_dense_rejected_before_retry_count += 1
            if selected_source == "native_dense_risky_before_retry" or bool(
                entry.get("native_dense_risky_before_retry", False)
            ):
                native_dense_risky_before_retry_count += 1

    if guard_entry_count == 0:
        return {}
    return {
        "guard_entry_count": int(guard_entry_count),
        "accept_dense_count": int(accept_dense_count),
        "reject_dense_keep_sparse_count": int(reject_dense_keep_sparse_count),
        "retry_entry_count": int(retry_entry_count),
        "native_dense_rejected_before_retry_count": int(native_dense_rejected_before_retry_count),
        "native_dense_risky_before_retry_count": int(native_dense_risky_before_retry_count),
        "selected_sparse_count": int(selected_sparse_count),
        "selected_dense_count": int(selected_dense_count),
        "selected_dense_feedback_retry_count": int(selected_dense_feedback_retry_count),
    }


def metrics_from_ulfloc_results(results: Sequence[Any]) -> dict[str, dict[str, float]]:
    metrics = {
        "sparse": _stage_metrics_from_results(results, "sparse"),
        "dense": _stage_metrics_from_results(results, "dense"),
        "paired_sparse_dense": _paired_sparse_dense_metrics(results),
    }
    guard_metrics = _dense_guard_metrics(results)
    if guard_metrics:
        metrics["dense_guard"] = guard_metrics
    return metrics


def _stage_metrics_from_summary(summary_stage: Mapping[str, Any]) -> dict[str, float]:
    aliases = {
        "median_te_cm": ("median_te_cm", "median_te"),
        "median_re_deg": ("median_re_deg", "median_ae", "median_re"),
        "R50cm5deg": ("R50cm5deg", "recall_50cm_5d"),
        "R10cm5deg": ("R10cm5deg", "recall_10cm_5d"),
        "R5cm5deg": ("R5cm5deg", "recall_5cm_5d"),
        "R2cm2deg": ("R2cm2deg", "recall_2cm_2d"),
    }
    out: dict[str, float] = {}
    for target, names in aliases.items():
        for name in names:
            value = _safe_float(summary_stage.get(name))
            if value is not None:
                out[target] = value
                break
    return out


def _merge_summary_and_results(summary: Mapping[str, Any], result_metrics: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for stage_name in ("sparse", "dense"):
        stage_summary = summary.get(stage_name, {})
        stage_metrics = (
            _stage_metrics_from_summary(stage_summary)
            if isinstance(stage_summary, Mapping)
            else {}
        )
        stage_metrics.update(dict(result_metrics.get(stage_name, {})))
        merged[stage_name] = stage_metrics
    for key, value in result_metrics.items():
        if key not in merged:
            merged[key] = dict(value)
    return merged


def _load_feedback_manifest(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    manifest_path = path / "manifest.json" if path.is_dir() else path
    if not manifest_path.exists():
        raise FileNotFoundError(f"feedback manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"feedback manifest must be a JSON object: {manifest_path}")
    return payload


def _image_id_set(path: Path | None) -> set[str] | None:
    if path is None or not path.exists():
        return None
    return set(load_dataset_image_list(path))


def _manifest_train_lists(payload: Mapping[str, Any] | None) -> list[Path]:
    if not isinstance(payload, Mapping):
        return []
    out: list[Path] = []
    raw = payload.get("train_list")
    if raw:
        out.append(Path(str(raw)))
    for key in ("feedback_manifest", "source_manifest"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            out.extend(_manifest_train_lists(value))
    source_manifests = payload.get("source_manifests")
    if isinstance(source_manifests, Mapping):
        for value in source_manifests.values():
            if isinstance(value, Mapping):
                out.extend(_manifest_train_lists(value))
    return out


def _feedback_image_ids_from_manifest(feedback_manifest: Mapping[str, Any] | None) -> set[str] | None:
    ids: set[str] = set()
    for train_list in _manifest_train_lists(feedback_manifest):
        loaded = _image_id_set(train_list)
        if loaded:
            ids.update(loaded)
    return ids or None


def split_audit_for_ulfloc_eval(
    *,
    split_name: str,
    feedback_manifest: Mapping[str, Any] | None,
    query_image_ids: set[str] | None = None,
    official_test_image_ids: set[str] | None = None,
    feedback_image_ids: set[str] | None = None,
) -> dict[str, Any]:
    normalized_split = str(split_name).strip() or "unknown"
    official_test = normalized_split.lower() == "test"
    checks: dict[str, dict[str, Any]] = {
        "query_split": {
            "status": "passed",
            "split_name": normalized_split,
            "official_test_used": official_test,
            "paper_safe_for_tuning": not official_test,
        },
        "real_query_path": {
            "status": "passed",
            "path": "ULF-Loc native sparse PnP followed by LGCV dense refinement",
            "branch_selection": False,
        },
    }
    feedback = dict(feedback_manifest or {})
    feedback_split = str(feedback.get("split_name", feedback.get("split", ""))).strip()
    if feedback_split.lower() == "test":
        checks["feedback_bank_split"] = {
            "status": "failed",
            "split_name": feedback_split,
            "reason": "feedback/solver-feedback source must not be test",
        }
    elif feedback_split:
        checks["feedback_bank_split"] = {"status": "passed", "split_name": feedback_split}
    else:
        checks["feedback_bank_split"] = {
            "status": "passed",
            "split_name": "none",
            "reason": "native ULF-Loc eval has no feedback bank",
        }

    if not official_test and query_image_ids is not None and official_test_image_ids is not None:
        overlap = sorted(query_image_ids & official_test_image_ids)
        checks["query_official_test_disjointness"] = {
            "status": "failed" if overlap else "passed",
            "overlap": overlap,
        }
    elif not official_test:
        checks["query_official_test_disjointness"] = {
            "status": "unknown",
            "reason": "query image ids or official test image ids are unavailable",
        }

    if not official_test and feedback_split and feedback_split.lower() != "test":
        if query_image_ids is not None and feedback_image_ids is not None:
            overlap = sorted(query_image_ids & feedback_image_ids)
            checks["feedback_eval_disjointness"] = {
                "status": "failed" if overlap else "passed",
                "overlap": overlap,
                "feedback_image_count": int(len(feedback_image_ids)),
                "query_image_count": int(len(query_image_ids)),
            }
        else:
            checks["feedback_eval_disjointness"] = {
                "status": "unknown",
                "reason": "feedback image ids or query image ids are unavailable",
            }

    statuses = {check["status"] for check in checks.values()}
    audit_status = "failed" if "failed" in statuses else "unknown" if "unknown" in statuses else "passed"
    return {
        "audit_status": audit_status,
        "split_name": normalized_split,
        "official_test_used": official_test,
        "paper_safe_for_tuning": (not official_test and audit_status == "passed"),
        "checks": checks,
    }


def _git_status(cwd: Path) -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=str(cwd), text=True)
    except Exception as exc:
        return f"git status unavailable: {exc}\n"


def write_ulfloc_eval_audit(
    *,
    output_dir: Path,
    command: Sequence[str],
    scene: str,
    split_name: str,
    source_path: Path,
    model_path: Path,
    cfg: Path,
    feedback_manifest: Mapping[str, Any] | None,
    ulf_root: Path,
    hyperparameters: Mapping[str, Any],
    query_list: Path | None = None,
    query_image_ids: set[str] | None = None,
    official_test_image_ids: set[str] | None = None,
    feedback_image_ids: set[str] | None = None,
    eval_source_path: Path | None = None,
) -> dict[str, str]:
    summary_path = output_dir / "summary.json"
    results_path = output_dir / "results.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"ULF-Loc summary.json not found: {summary_path}")
    if not results_path.exists():
        raise FileNotFoundError(f"ULF-Loc results.json not found: {results_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if not isinstance(summary, Mapping):
        raise ValueError(f"ULF-Loc summary.json must contain an object: {summary_path}")
    if not isinstance(results, list):
        raise ValueError(f"ULF-Loc results.json must contain a list: {results_path}")
    query_result_order_source = _attach_result_image_names_from_log(output_dir, results)

    split_audit = split_audit_for_ulfloc_eval(
        split_name=split_name,
        feedback_manifest=feedback_manifest,
        query_image_ids=query_image_ids,
        official_test_image_ids=official_test_image_ids,
        feedback_image_ids=feedback_image_ids,
    )
    metrics = _merge_summary_and_results(summary, metrics_from_ulfloc_results(results))
    metrics["query_count"] = len(results)
    manifest = {
        "method": "ulfloc_native_eval_with_locgs_audit",
        "scene": str(scene),
        "split": str(split_name),
        "checkpoint_path": str(model_path),
        "map_path": str(model_path),
        "data_roots": [str(source_path)],
        "eval_source_path": str(eval_source_path) if eval_source_path is not None else str(source_path),
        "query_list": str(query_list) if query_list is not None else None,
        "query_image_count": len(query_image_ids) if query_image_ids is not None else None,
        "query_result_order_source": query_result_order_source,
        "feedback_image_count": len(feedback_image_ids) if feedback_image_ids is not None else None,
        "cfg": str(cfg),
        "hyperparameters": dict(hyperparameters),
        "residual_feedback_enabled": False,
        "selector_feedback_enabled": bool(feedback_manifest),
        "rho_feedback_enabled": False,
        "branch_selection": False,
        "single_path_deployment": True,
        "real_query_path": "ULF-Loc native: sparse PnP -> LGCV dense refinement",
        "official_test_used": split_audit["official_test_used"],
        "paper_safe_for_tuning": split_audit["paper_safe_for_tuning"],
        "feedback_manifest": dict(feedback_manifest or {}),
    }
    paths = write_export_eval_audit_bundle(
        output_dir,
        manifest=manifest,
        command=list(command),
        metrics_summary=metrics,
        split_audit=split_audit,
    )
    ulf_status_path = output_dir / "ulf_git_status.txt"
    ulf_status_path.write_text(_git_status(ulf_root), encoding="utf-8")
    paths["ulf_git_status"] = str(ulf_status_path)
    return paths


def create_ulfloc_eval_source_overlay(
    *,
    source_path: Path,
    query_list: Path,
    overlay_root: Path,
    split_name: str,
) -> Path:
    query_images = load_dataset_image_list(query_list)
    if not query_images:
        raise ValueError(f"query_list contains no image rows: {query_list}")

    source = Path(source_path)
    overlay_name = f"cambridge_{source.name}_{str(split_name).replace('/', '_')}"
    overlay = Path(overlay_root) / overlay_name
    if overlay.exists() or overlay.is_symlink():
        if overlay.is_symlink() or overlay.is_file():
            overlay.unlink()
        else:
            shutil.rmtree(overlay)
    overlay.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        if child.name == "dataset_test.txt":
            continue
        target = overlay / child.name
        target.symlink_to(child.resolve(), target_is_directory=child.is_dir())
    query_lines = [
        raw.strip()
        for raw in Path(query_list).read_text(encoding="utf-8").splitlines()
        if raw.strip() and not raw.strip().startswith("#")
    ]
    (overlay / "dataset_test.txt").write_text("\n".join(query_lines) + "\n", encoding="utf-8")
    return overlay


def resolve_eval_source_path(args: argparse.Namespace) -> Path:
    query_list = getattr(args, "query_list", None)
    if query_list is not None:
        return create_ulfloc_eval_source_overlay(
            source_path=Path(args.source_path),
            query_list=Path(query_list),
            overlay_root=Path(args.model_path) / "_locgs_eval_source_overlays",
            split_name=str(args.split_name),
        )
    return resolve_ulfloc_source_path_for_loader(
        Path(args.source_path),
        Path(args.model_path) / "_ulf_loader_links",
    )


def build_ulfloc_eval_command(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    source_path = resolve_eval_source_path(args)
    command = [
        str(args.python),
        str(Path(args.ulf_root) / "ulfloc.py"),
        "-s",
        str(source_path),
        "-m",
        str(args.model_path),
        "--images",
        str(args.images),
        "-f",
        str(args.feature_type),
        "--data_device",
        str(args.data_device),
        "--longest_edge",
        str(args.longest_edge),
        "--iteration",
        str(args.iteration),
        "--cfg",
        str(Path(args.cfg).resolve()),
    ]
    if args.prefix:
        command.extend(["--prefix", str(args.prefix)])
    env = os.environ.copy()
    if str(args.cuda_visible_devices).strip():
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices).strip()
    return command, env


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"ULF config must contain a YAML mapping: {path}")
    return payload


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run native ULF-Loc eval and write Loc-GS audit files.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--source_path", required=True, type=Path)
    parser.add_argument("--model_path", required=True, type=Path)
    parser.add_argument("--cfg", required=True, type=Path)
    parser.add_argument("--ulf_root", default="/root/ULF-Loc", type=Path)
    parser.add_argument("--python", default="/root/miniconda3/envs/cybersim_agent/bin/python")
    parser.add_argument("--split_name", default="test")
    parser.add_argument("--images", default="processed")
    parser.add_argument("--feature_type", default="sp")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--data_device", default="cpu")
    parser.add_argument("--longest_edge", default=640, type=int)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--cuda_visible_devices", default="")
    parser.add_argument("--feedback_manifest", default=None, type=Path)
    parser.add_argument("--audit_output_dir", default=None, type=Path)
    parser.add_argument("--query_list", default=None, type=Path)
    parser.add_argument("--skip_run", action="store_true")
    return parser


def main(args: argparse.Namespace | None = None) -> int:
    if args is None:
        args = build_argparser().parse_args()

    config = _load_config(Path(args.cfg))
    command, env = build_ulfloc_eval_command(args)
    if not args.skip_run:
        subprocess.run(command, cwd=str(args.ulf_root), env=env, check=True)

    eval_dir = Path(args.audit_output_dir) if args.audit_output_dir else Path(args.model_path) / str(config["log_name"])
    feedback_manifest = _load_feedback_manifest(args.feedback_manifest)
    query_list = Path(args.query_list) if args.query_list is not None else None
    query_image_ids = _image_id_set(query_list) if query_list is not None else None
    official_test_image_ids = _image_id_set(Path(args.source_path) / "dataset_test.txt")
    feedback_image_ids = _feedback_image_ids_from_manifest(feedback_manifest)
    eval_source_path = resolve_eval_source_path(args)
    write_ulfloc_eval_audit(
        output_dir=eval_dir,
        command=command,
        scene=str(args.scene),
        split_name=str(args.split_name),
        source_path=Path(args.source_path),
        model_path=Path(args.model_path),
        cfg=Path(args.cfg),
        feedback_manifest=feedback_manifest,
        ulf_root=Path(args.ulf_root),
        hyperparameters={
            "images": str(args.images),
            "feature_type": str(args.feature_type),
            "iteration": int(args.iteration),
            "data_device": str(args.data_device),
            "longest_edge": int(args.longest_edge),
            "log_name": str(config.get("log_name", "")),
        },
        query_list=query_list,
        query_image_ids=query_image_ids,
        official_test_image_ids=official_test_image_ids,
        feedback_image_ids=feedback_image_ids,
        eval_source_path=eval_source_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
