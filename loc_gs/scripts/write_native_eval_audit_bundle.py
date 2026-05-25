#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from loc_gs.export.manifest import audit_split_usage, write_export_eval_audit_bundle


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _third_party_stdloc_evaluator_modified() -> bool:
    evaluator_paths = [
        "third_party/stdloc/stdloc.py",
        "third_party/stdloc/utils",
    ]
    try:
        result = subprocess.run(
            ["git", "status", "--short", "--", *evaluator_paths],
            cwd=str(_repo_root()),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return True
    return bool(result.stdout.strip())


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _stage_metrics(stage: Mapping[str, Any]) -> dict[str, Any]:
    aliases = {
        "median_re_deg": ("median_re_deg", "median_ae", "median_re", "median_rotation_deg"),
        "median_te_cm": ("median_te_cm", "median_te", "median_translation_cm"),
        "recall_5m_10deg": ("recall_5m_10deg", "recall_5m_10d"),
        "recall_2m_5deg": ("recall_2m_5deg", "recall_2m_5d"),
        "recall_10cm_5deg": ("recall_10cm_5deg", "recall_10cm_5d", "r10"),
        "recall_5cm_5deg": ("recall_5cm_5deg", "recall_5cm_5d", "r5"),
        "recall_2cm_2deg": ("recall_2cm_2deg", "recall_2cm_2d", "r2"),
        "avg_inliers": ("avg_inliers", "mean_inliers"),
    }
    out: dict[str, Any] = {}
    for target, names in aliases.items():
        for name in names:
            if name in stage:
                out[target] = stage[name]
                break
    return out


def _metrics_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for stage_name in ("sparse", "dense"):
        stage = summary.get(stage_name, {})
        if isinstance(stage, Mapping):
            metrics[stage_name] = _stage_metrics(stage)
    return metrics


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _result_recall_metrics(results: list[Any], stage_name: str) -> dict[str, float]:
    pairs: list[tuple[float, float]] = []
    ae_key = f"{stage_name}_AE"
    te_key = f"{stage_name}_TE"
    for item in results:
        if not isinstance(item, Mapping):
            continue
        ae = _float_or_none(item.get(ae_key))
        te = _float_or_none(item.get(te_key))
        if ae is None or te is None:
            continue
        pairs.append((float(te), float(ae)))
    if not pairs:
        return {}

    def _recall(max_te_cm: float, max_re_deg: float) -> float:
        return float(
            sum(1 for te, re in pairs if te <= float(max_te_cm) and re <= float(max_re_deg))
            / len(pairs)
        )

    return {
        "recall_10cm_5deg": _recall(10.0, 5.0),
        "recall_5cm_5deg": _recall(5.0, 5.0),
        "recall_2cm_2deg": _recall(2.0, 2.0),
    }


def _result_metrics(root: Path) -> dict[str, dict[str, float]]:
    results_path = root / "results.json"
    if not results_path.exists():
        return {}
    try:
        results = json.loads(results_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(results, list):
        return {}
    return {
        stage_name: metrics
        for stage_name in ("sparse", "dense")
        if (metrics := _result_recall_metrics(results, stage_name))
    }


def _merge_missing_metrics(
    metrics: dict[str, Any],
    supplemental: Mapping[str, Mapping[str, Any]],
) -> None:
    for stage_name, stage_metrics in supplemental.items():
        current = metrics.setdefault(stage_name, {})
        if not isinstance(current, dict):
            continue
        for key, value in stage_metrics.items():
            current.setdefault(key, value)


def _directory_size_bytes(path: Path) -> int | None:
    if not path.exists():
        return None
    if path.is_file():
        return int(path.stat().st_size)
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += int(item.stat().st_size)
        except OSError:
            continue
    return total


def _sampled_count(map_path: Path) -> int | None:
    sampled_idx = map_path / "detector" / "sampled_idx.pkl"
    if not sampled_idx.exists():
        return None
    try:
        with sampled_idx.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return None
    try:
        return int(len(payload))
    except TypeError:
        try:
            return int(payload.numel())
        except AttributeError:
            return None


def _map_manifest(map_path: Path) -> dict[str, Any]:
    for name in (
        "manifest.json",
        "lsf_solver_aware_manifest.json",
        "selector_resampling_manifest.json",
        "clean_source_manifest.json",
        "soft_prior_manifest.json",
        "lff_feature_manifest.json",
        "unified_lff_manifest.json",
    ):
        path = map_path / name
        if path.exists():
            try:
                payload = _load_json(path)
            except Exception:
                continue
            payload.setdefault("__manifest_name", name)
            return payload
    return {}


def _copied_cfg_path(root: Path) -> Path | None:
    for name in (
        "stdloc_cambridge_opencv.yaml",
        "stdloc_cambridge.yaml",
        "stdloc_soft_prior.yaml",
    ):
        path = root / name
        if path.exists():
            return path
    candidates = sorted(root.glob("*.yaml"))
    return candidates[0] if candidates else None


def _cfg_hyperparameters(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"cfg": ""}
    params: dict[str, Any] = {"cfg": str(path)}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return params
    if not isinstance(payload, Mapping):
        return params
    sparse = payload.get("sparse", {})
    dense = payload.get("dense", {})
    if isinstance(sparse, Mapping):
        if "solver" in sparse:
            params["sparse_solver"] = sparse["solver"]
        if "landmark_path" in sparse:
            params["sparse_landmark_path"] = sparse["landmark_path"]
        if "detector_path" in sparse:
            params["sparse_detector_path"] = sparse["detector_path"]
    if isinstance(dense, Mapping):
        if "solver" in dense:
            params["dense_solver"] = dense["solver"]
        if "iters" in dense:
            params["dense_iters"] = dense["iters"]
    return params


def _evaluator_safety(cfg_hparams: Mapping[str, Any]) -> dict[str, Any]:
    cfg_path = str(cfg_hparams.get("cfg", ""))
    cfg_name = Path(cfg_path).name if cfg_path else ""
    sparse_solver = str(cfg_hparams.get("sparse_solver", "")).strip()
    dense_solver = str(cfg_hparams.get("dense_solver", "")).strip()
    if sparse_solver and dense_solver and sparse_solver == dense_solver:
        evaluator_variant = sparse_solver
    elif sparse_solver or dense_solver:
        evaluator_variant = f"sparse={sparse_solver or 'unknown'},dense={dense_solver or 'unknown'}"
    else:
        evaluator_variant = "unknown"
    modified = _third_party_stdloc_evaluator_modified()
    cfg_is_vendored_stdloc_default = (
        cfg_name == "stdloc_cambridge.yaml"
        and sparse_solver == "poselib"
        and dense_solver == "poselib"
    )
    uses_poselib_single_path = bool(
        sparse_solver == "poselib"
        and dense_solver == "poselib"
    )
    uses_opencv_variant = bool(
        sparse_solver in {"opencv_prosac", "opencv_prosac_magsac"}
        or dense_solver in {"opencv_prosac", "opencv_prosac_magsac"}
    )
    return {
        "cfg_path": cfg_path,
        "cfg_is_vendored_stdloc_default": cfg_is_vendored_stdloc_default,
        "evaluator_variant": evaluator_variant,
        "third_party_stdloc_evaluator_modified": modified,
        "uses_poselib_single_path": uses_poselib_single_path,
        "uses_opencv_variant": uses_opencv_variant,
        "paper_safe_fixed_poselib_candidate": bool(
            uses_poselib_single_path and not modified
        ),
        "official_stdloc_parity_candidate": bool(
            cfg_is_vendored_stdloc_default and not modified
        ),
    }


def _manifest_from_artifact_path(path_text: Any) -> dict[str, Any]:
    if not path_text:
        return {}
    path = Path(str(path_text))
    root = path.parent if path.suffix else path
    if path.name == "feedback_bank.jsonl":
        root = path.parent
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        return _load_json(manifest_path)
    except Exception:
        return {}


def _split_audit_from_manifest_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    split_audit = payload.get("split_audit")
    if isinstance(split_audit, Mapping):
        return dict(split_audit)
    pair_cache = payload.get("pair_cache_metadata")
    if isinstance(pair_cache, Mapping):
        split_audit = pair_cache.get("split_audit")
        if isinstance(split_audit, Mapping):
            return dict(split_audit)
    support = payload.get("support_metadata")
    if isinstance(support, Mapping):
        split_audit = support.get("split_audit")
        if isinstance(split_audit, Mapping):
            return dict(split_audit)
        feedback_bank = support.get("source_feedback_bank")
        if feedback_bank:
            upstream = _manifest_from_artifact_path(feedback_bank)
            split_audit = _split_audit_from_manifest_payload(upstream)
            if split_audit:
                return split_audit
        audit = support.get("audit")
        if isinstance(audit, Mapping) and audit.get("path"):
            upstream = _manifest_from_artifact_path(audit.get("path"))
            split_audit = _split_audit_from_manifest_payload(upstream)
            if split_audit:
                return split_audit
    return {}


def _upstream_split_audit(map_manifest: Mapping[str, Any]) -> dict[str, Any]:
    split_audit = _split_audit_from_manifest_payload(map_manifest)
    if split_audit:
        return split_audit
    selector_manifest = _manifest_from_artifact_path(map_manifest.get("selector_path"))
    split_audit = _split_audit_from_manifest_payload(selector_manifest)
    if split_audit:
        return split_audit
    support_manifest = _manifest_from_artifact_path(map_manifest.get("solver_consensus_support_path"))
    return _split_audit_from_manifest_payload(support_manifest)


def _audit_status_from_checks(checks: Mapping[str, Mapping[str, Any]]) -> str:
    statuses = {str(check.get("status", "unknown")) for check in checks.values()}
    if "failed" in statuses:
        return "failed"
    if "unknown" in statuses:
        return "unknown"
    return "passed"


def _read_cambridge_split_image_ids(scene_root: Path, split: str) -> list[str] | None:
    path = scene_root / ("dataset_test.txt" if str(split) == "test" else "dataset_train.txt")
    if not path.exists():
        return None
    ids: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("Visual") or line.startswith("ImageFile"):
            continue
        parts = line.split()
        if parts:
            ids.append(str(parts[0]))
    return sorted(set(ids))


def _read_map_camera_image_ids(map_path: Path) -> list[str] | None:
    cameras_json = map_path / "cameras.json"
    if not cameras_json.exists():
        return None
    try:
        payload = json.loads(cameras_json.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, Mapping):
        entries = list(payload.values())
    else:
        return None
    ids: list[str] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        name = entry.get("img_name") or entry.get("image_name") or entry.get("name")
        if name:
            ids.append(str(name))
    return sorted(set(ids))


def _resolve_scene_root(data_root: str, scene: str) -> Path | None:
    if not data_root:
        return None
    root = Path(data_root)
    if (root / "dataset_train.txt").exists() or (root / "dataset_test.txt").exists():
        return root
    scene_root = root / scene
    if (scene_root / "dataset_train.txt").exists() or (scene_root / "dataset_test.txt").exists():
        return scene_root
    return root


def _native_map_camera_split_audit(
    *,
    scene: str,
    data_root: str,
    map_path: str,
    feedback_split: str,
    quality_gate_mode: str,
) -> dict[str, Any]:
    if not map_path:
        return {}
    scene_root = _resolve_scene_root(data_root, scene)
    if scene_root is None:
        return {}
    train_ids = _read_cambridge_split_image_ids(scene_root, "train")
    test_ids = _read_cambridge_split_image_ids(scene_root, "test")
    map_ids = _read_map_camera_image_ids(Path(map_path))
    if train_ids is None or test_ids is None or map_ids is None:
        return {}
    train_set = set(train_ids)
    map_set = set(map_ids)
    outside_train = sorted(map_set - train_set)
    audit = audit_split_usage(
        selfmap_image_ids=map_ids,
        calibration_image_ids=[],
        test_image_ids=test_ids,
        feedback_bank_manifest={"split_name": feedback_split},
        quality_gate={"mode": quality_gate_mode, "per_query_branch_selection": False},
    )
    audit["source"] = "map_cameras_vs_cambridge_test_split"
    audit["scene_root"] = str(scene_root)
    audit["map_path"] = str(map_path)
    audit["map_camera_count"] = int(len(map_ids))
    audit["train_image_count"] = int(len(train_ids))
    audit["test_image_count"] = int(len(test_ids))
    audit["checks"]["map_camera_train_membership"] = {
        "status": "failed" if outside_train else "passed",
        "outside_train_count": int(len(outside_train)),
        "outside_train_sample": outside_train[:20],
    }
    audit["audit_status"] = _audit_status_from_checks(audit["checks"])
    return audit


def _combine_upstream_and_map_camera_audits(
    upstream: Mapping[str, Any],
    map_camera: Mapping[str, Any],
) -> dict[str, Any]:
    if upstream and not map_camera:
        return dict(upstream)
    if map_camera and not upstream:
        return dict(map_camera)
    if not upstream and not map_camera:
        return {}
    checks: dict[str, Any] = {}
    upstream_checks = upstream.get("checks")
    if isinstance(upstream_checks, Mapping):
        checks.update({str(key): dict(value) for key, value in upstream_checks.items() if isinstance(value, Mapping)})
    map_checks = map_camera.get("checks")
    if isinstance(map_checks, Mapping):
        checks.update({str(key): dict(value) for key, value in map_checks.items() if isinstance(value, Mapping)})
    return {
        "audit_status": _audit_status_from_checks(checks),
        "source": "upstream_plus_map_cameras_vs_cambridge_test_split",
        "checks": checks,
        "upstream_audit": dict(upstream),
        "map_camera_audit": dict(map_camera),
    }


def _native_train_eval_split_audit(
    *,
    scene: str,
    data_root: str,
    eval_split: str,
    quality_gate_mode: str,
) -> dict[str, Any]:
    if str(eval_split) != "train":
        return {}
    scene_root = _resolve_scene_root(data_root, scene)
    if scene_root is None:
        return {}
    train_ids = _read_cambridge_split_image_ids(scene_root, "train")
    test_ids = _read_cambridge_split_image_ids(scene_root, "test")
    if train_ids is None or test_ids is None:
        return {}
    audit = audit_split_usage(
        selfmap_image_ids=train_ids,
        calibration_image_ids=[],
        test_image_ids=test_ids,
        feedback_bank_manifest={"split_name": "native_eval_train"},
        quality_gate={"mode": quality_gate_mode, "per_query_branch_selection": False},
    )
    audit["source"] = "dataset_train_test_disjoint_native_eval"
    audit["scene_root"] = str(scene_root)
    audit["train_image_count"] = int(len(train_ids))
    audit["test_image_count"] = int(len(test_ids))
    return audit


def _map_reporting_fields(map_path_text: str) -> dict[str, Any]:
    map_path = Path(map_path_text)
    fields: dict[str, Any] = {}
    size_bytes = _directory_size_bytes(map_path)
    if size_bytes is not None:
        fields["map_size_bytes"] = int(size_bytes)
        fields["map_size_mb"] = float(size_bytes / (1024.0 * 1024.0))
    count = _sampled_count(map_path)
    if count is not None:
        fields["sampled_count"] = int(count)
        fields["landmark_count"] = int(count)
    return fields


def write_native_eval_audit_bundle(
    run_dir: str | Path,
    *,
    scene: str,
    split: str,
    data_root: str,
    checkpoint_path: str,
    map_path: str,
    command: Sequence[str] | str,
    feedback_split: str = "selfmap_train",
    quality_gate_mode: str = "disabled",
    run_role: str = "",
    notes: str = "",
) -> dict[str, str]:
    root = Path(run_dir)
    summary_path = root / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found: {summary_path}")
    summary = _load_json(summary_path)
    metrics = _metrics_summary(summary)
    _merge_missing_metrics(metrics, _result_metrics(root))
    eval_split = str(summary.get("eval_split", split))
    max_test_cameras = summary.get("max_test_cameras")
    test_stride = summary.get("test_stride")
    resolved_map = str(map_path or summary.get("model_path", ""))
    map_reporting = _map_reporting_fields(resolved_map) if resolved_map else {}
    map_manifest = _map_manifest(Path(resolved_map)) if resolved_map else {}
    if map_reporting:
        metrics.update(map_reporting)

    split_audit = _combine_upstream_and_map_camera_audits(
        _upstream_split_audit(map_manifest),
        _native_map_camera_split_audit(
            scene=scene,
            data_root=data_root,
            map_path=resolved_map,
            feedback_split=feedback_split,
            quality_gate_mode=quality_gate_mode,
        ),
    )
    if not split_audit:
        split_audit = _native_train_eval_split_audit(
            scene=scene,
            data_root=data_root,
            eval_split=eval_split,
            quality_gate_mode=quality_gate_mode,
        )
    if not split_audit:
        split_audit = audit_split_usage(
            selfmap_image_ids=None,
            calibration_image_ids=None,
            test_image_ids=None,
            feedback_bank_manifest={"split_name": feedback_split},
            quality_gate={"mode": quality_gate_mode, "per_query_branch_selection": False},
        )
    cfg_hparams = _cfg_hyperparameters(_copied_cfg_path(root))
    evaluator_safety = _evaluator_safety(cfg_hparams)
    hyperparameters = {
        **cfg_hparams,
        "eval_split": eval_split,
        "max_test_cameras": max_test_cameras,
        "test_stride": test_stride,
    }
    manifest = {
        "scene": scene,
        "split": eval_split,
        "checkpoint_path": str(checkpoint_path),
        "map_path": resolved_map,
        "data_roots": [str(data_root)] if data_root else [],
        "hyperparameters": hyperparameters,
        "feedback": {
            "residual": False,
            "selector": False,
            "rho": False,
        },
        "feedback_enabled": False,
        "residual_enabled": False,
        "selector_enabled": False,
        "rho": False,
        "rho_feedback_enabled": False,
        "single_path_evaluator": True,
        "evaluator_safety": evaluator_safety,
        "quality_gate": {
            "mode": quality_gate_mode,
            "per_query_branch_selection": False,
        },
        "split_audit": split_audit,
        "notes": notes,
    }
    if str(run_role).strip():
        manifest["run_role"] = str(run_role).strip()
    manifest.update(map_reporting)
    if map_manifest:
        if map_manifest.get("__manifest_name"):
            manifest["map_manifest_type"] = map_manifest["__manifest_name"]
        for source_key, target_key in (
            ("method", "map_method"),
            ("rho", "map_rho"),
            ("source_map", "map_source_map"),
            ("output_cfg_path", "map_output_cfg_path"),
            ("checkpoint_path", "map_checkpoint_path"),
        ):
            if source_key in map_manifest:
                manifest[target_key] = map_manifest[source_key]
    for key in ("solver_aware", "selected_edits"):
        if isinstance(map_manifest.get(key), Mapping):
            manifest[key] = dict(map_manifest[key])
    for key in ("same_budget", "branch_selection", "single_path_deployment"):
        if key in map_manifest:
            manifest[key] = map_manifest[key]
            manifest[f"map_{key}"] = map_manifest[key]
    return write_export_eval_audit_bundle(
        root,
        manifest=manifest,
        command=command,
        metrics_summary=metrics,
        split_audit=split_audit,
        git_diff_text=None,
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write manifest/audit files for a native STDLoc eval directory containing summary.json."
    )
    parser.add_argument("run_dir")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--map-path", default="")
    parser.add_argument("--feedback-split", default="selfmap_train")
    parser.add_argument("--quality-gate-mode", default="disabled")
    parser.add_argument("--run-role", choices=["main_candidate", "ablation", "diagnostic", "rejected"], default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--command", nargs=argparse.REMAINDER, default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--command" in argv:
        command_idx = argv.index("--command")
        if command_idx + 1 < len(argv) and argv[command_idx + 1] == "--":
            del argv[command_idx + 1]
    parser = build_argparser()
    args = parser.parse_args(argv)
    command = list(args.command) if args.command else sys.argv
    if command[:1] == ["--"]:
        command = command[1:]
    paths = write_native_eval_audit_bundle(
        args.run_dir,
        scene=args.scene,
        split=args.split,
        data_root=args.data_root,
        checkpoint_path=args.checkpoint_path,
        map_path=args.map_path,
        command=command,
        feedback_split=args.feedback_split,
        quality_gate_mode=args.quality_gate_mode,
        run_role=args.run_role,
        notes=args.notes,
    )
    print(json.dumps(paths, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
