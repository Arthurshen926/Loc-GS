from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any


EXPECTED_ULF_FILES = (
    "README.md",
    "requirements.txt",
    "train.py",
    "ulfloc.py",
    "configs/ulfloc_cambridge.yaml",
    "scripts/train_cambridge.sh",
    "scripts/evaluate_cambridge.sh",
)

CAMBRIDGE_SCENES = (
    "GreatCourt",
    "KingsCollege",
    "OldHospital",
    "ShopFacade",
    "StMarysChurch",
)

PAPER_ULF_CAMBRIDGE = {
    "avg_dense_median_te_cm": 8.3,
    "avg_dense_median_re_deg": 0.13,
    "avg_r15cm5deg": 0.720,
    "avg_r10cm5deg": 0.622,
}


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _first_float(payload: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = _as_float(payload.get(key))
        if value is not None:
            return value
    return None


def extract_pose_metrics(payload: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Normalize ULF/Loc-GS metric summary variants into one schema."""

    out: dict[str, dict[str, float]] = {}
    for stage in ("sparse", "dense"):
        raw = payload.get(stage, {})
        if not isinstance(raw, dict):
            continue
        stage_metrics: dict[str, float] = {}
        aliases = {
            "median_te_cm": ("median_te_cm", "median_te", "median_translation_cm"),
            "median_re_deg": ("median_re_deg", "median_ae", "median_rotation_deg"),
            "R50cm5deg": ("R50cm5deg", "recall_50cm_5d", "recall_50cm_5deg"),
            "R15cm5deg": ("R15cm5deg", "recall_15cm_5d", "recall_15cm_5deg"),
            "R10cm5deg": ("R10cm5deg", "recall_10cm_5d", "recall_10cm_5deg"),
            "R5cm5deg": ("R5cm5deg", "recall_5cm_5d", "recall_5cm_5deg"),
            "R2cm2deg": ("R2cm2deg", "recall_2cm_2d", "recall_2cm_2deg"),
            "avg_inliers": ("avg_inliers", "mean_inliers"),
            "p90_te_cm": ("p90_te_cm", "p90_te"),
            "p95_te_cm": ("p95_te_cm", "p95_te"),
            "cvar10_te_cm": ("cvar10_te_cm", "tail_cvar10_te_cm"),
        }
        for target_key, source_keys in aliases.items():
            value = _first_float(raw, source_keys)
            if value is not None:
                stage_metrics[target_key] = value
        out[stage] = stage_metrics
    return out


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _git_text(cwd: Path, args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=str(cwd), text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def _infer_scene(path: Path, payloads: list[dict[str, Any]]) -> str:
    for payload in payloads:
        for key in ("scene", "scene_name"):
            value = payload.get(key)
            if value:
                return str(value)
    parts = set(path.parts)
    for scene in CAMBRIDGE_SCENES:
        if scene in parts:
            return scene
    return "unknown"


def _infer_split(path: Path, payloads: list[dict[str, Any]]) -> str:
    for payload in payloads:
        for key in ("split", "split_name", "eval_split"):
            value = payload.get(key)
            if value:
                return str(value)
        split_audit = payload.get("split_audit")
        if isinstance(split_audit, dict):
            value = split_audit.get("split_name") or split_audit.get("split")
            if value:
                return str(value)
    lowered_parts = [part.lower() for part in path.parts]
    if any("train_dev" in part or "traindev" in part for part in lowered_parts):
        return "train_dev"
    if "test" in lowered_parts:
        return "test"
    if any("train" in part for part in lowered_parts):
        return "train"
    return "unknown"


def _scan_result_dir(result_root: Path) -> list[dict[str, Any]]:
    if not result_root.exists():
        return []
    dirs: set[Path] = set()
    for name in ("manifest.json", "summary.json", "metrics_summary.json"):
        for path in result_root.rglob(name):
            dirs.add(path.parent)

    runs: list[dict[str, Any]] = []
    for run_dir in sorted(dirs):
        manifest = _read_json(run_dir / "manifest.json")
        summary = _read_json(run_dir / "summary.json")
        metrics_summary = _read_json(run_dir / "metrics_summary.json")
        payloads = [manifest, metrics_summary, summary]
        metrics = extract_pose_metrics(summary)
        metrics_from_summary = extract_pose_metrics(metrics_summary)
        for stage, stage_metrics in metrics_from_summary.items():
            metrics.setdefault(stage, {}).update(stage_metrics)
        scene = _infer_scene(run_dir, payloads)
        split = _infer_split(run_dir, payloads)
        group = _infer_recipe_group(run_dir, scene)
        runs.append(
            {
                "path": str(run_dir),
                "recipe_group": group,
                "scene": scene,
                "split": split,
                "has_manifest": bool(manifest),
                "has_summary": bool(summary),
                "has_metrics_summary": bool(metrics_summary),
                "metrics": metrics,
            }
        )
    return runs


def _infer_recipe_group(path: Path, scene: str) -> str:
    if scene and scene != "unknown":
        parts = list(path.parts)
        if scene in parts:
            idx = len(parts) - 1 - list(reversed(parts)).index(scene)
            return str(Path(*parts[:idx]))
    if path.name.lower() in {"test", "train", "train_dev", "validation", "val"}:
        return str(path.parent)
    return str(path)


def _summarize_results(runs: list[dict[str, Any]]) -> dict[str, Any]:
    scenes = sorted({run["scene"] for run in runs if run.get("scene") and run["scene"] != "unknown"})
    test_scenes = sorted({run["scene"] for run in runs if str(run.get("split")).lower() == "test"})
    train_dev_scenes = sorted(
        {
            run["scene"]
            for run in runs
            if "train" in str(run.get("split")).lower() or "dev" in str(run.get("split")).lower()
        }
    )
    dense_values = [
        _as_float(run.get("metrics", {}).get("dense", {}).get("median_te_cm"))
        for run in runs
        if _as_float(run.get("metrics", {}).get("dense", {}).get("median_te_cm")) is not None
    ]
    sparse_values = [
        _as_float(run.get("metrics", {}).get("sparse", {}).get("median_te_cm"))
        for run in runs
        if _as_float(run.get("metrics", {}).get("sparse", {}).get("median_te_cm")) is not None
    ]
    complete_test_groups: list[dict[str, Any]] = []
    by_group: dict[str, dict[str, float]] = {}
    for run in runs:
        if str(run.get("split")).lower() != "test":
            continue
        scene = str(run.get("scene", "unknown"))
        if scene not in CAMBRIDGE_SCENES:
            continue
        value = _as_float(run.get("metrics", {}).get("dense", {}).get("median_te_cm"))
        if value is None:
            continue
        group = str(run.get("recipe_group", "unknown"))
        scene_values = by_group.setdefault(group, {})
        current = scene_values.get(scene)
        if current is None or value < current:
            scene_values[scene] = value
    for group, scene_values in sorted(by_group.items()):
        if all(scene in scene_values for scene in CAMBRIDGE_SCENES):
            avg = sum(scene_values[scene] for scene in CAMBRIDGE_SCENES) / float(len(CAMBRIDGE_SCENES))
            complete_test_groups.append(
                {
                    "recipe_group": group,
                    "avg_dense_median_te_cm": avg,
                    "scene_dense_median_te_cm": dict(scene_values),
                }
            )
    complete_test_groups.sort(key=lambda item: float(item["avg_dense_median_te_cm"]))

    return {
        "run_count": int(len(runs)),
        "scene_count": int(len(scenes)),
        "scenes": scenes,
        "official_test_scene_count": int(len(test_scenes)),
        "test_scenes": test_scenes,
        "complete_test_recipe_group_count": int(len(complete_test_groups)),
        "best_five_scene_test_avg_te_cm": complete_test_groups[0]["avg_dense_median_te_cm"]
        if complete_test_groups
        else None,
        "complete_test_recipe_groups": complete_test_groups,
        "train_dev_scene_count": int(len(train_dev_scenes)),
        "train_dev_scenes": train_dev_scenes,
        "best_dense_single_scene_te_cm": min(dense_values) if dense_values else None,
        "best_sparse_single_scene_te_cm": min(sparse_values) if sparse_values else None,
        "best_dense_avg_te_cm": complete_test_groups[0]["avg_dense_median_te_cm"]
        if complete_test_groups
        else (min(dense_values) if dense_values else None),
        "best_sparse_avg_te_cm": min(sparse_values) if sparse_values else None,
        "runs": runs,
    }


def _config_status(ulf_root: Path, runs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    config_paths = sorted((ulf_root / "configs").glob("ulfloc_cambridge*.yaml"))
    text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in config_paths if path.exists())
    lowered = text.lower()
    command_text = ""
    mask_file_hits = 0
    mask_file_checks = 0
    for run in runs or []:
        manifest = _read_json(Path(str(run["path"])) / "manifest.json")
        command = manifest.get("command", "")
        if isinstance(command, list):
            command_text += " " + " ".join(str(item) for item in command)
        elif command:
            command_text += " " + str(command)
        images = str(manifest.get("hyperparameters", {}).get("images", "processed"))
        for root in manifest.get("data_roots", []) if isinstance(manifest.get("data_roots"), list) else []:
            mask_file_checks += 1
            if (Path(str(root)) / images / "masks.pkl").exists():
                mask_file_hits += 1
    lowered_command = command_text.lower()
    masks_available_for_runs = bool(mask_file_checks > 0 and mask_file_hits == mask_file_checks)
    return {
        "config_paths": [str(path) for path in config_paths],
        "uses_processed_images": "processed" in lowered or "--images processed" in lowered_command,
        "mask_config_present": "mask" in lowered or "mask" in lowered_command or masks_available_for_runs,
        "mentions_cambridge": "cambridge" in lowered,
        "command_uses_processed_images": "--images processed" in lowered_command,
        "command_has_no_masks_flag": "--no_masks" in lowered_command or "--no-masks" in lowered_command,
        "mask_file_checks": int(mask_file_checks),
        "mask_file_hits": int(mask_file_hits),
        "masks_available_for_runs": masks_available_for_runs,
    }


def build_ulfloc_reproduction_audit(
    *,
    ulf_root: str | Path = "/root/ULF-Loc",
    result_root: str | Path | None = None,
) -> dict[str, Any]:
    ulf_root = Path(ulf_root)
    result_root = Path(result_root) if result_root is not None else ulf_root / "outputs"
    expected = {rel: (ulf_root / rel).exists() for rel in EXPECTED_ULF_FILES}
    git_status = _git_text(ulf_root, ["status", "--short"]) if ulf_root.exists() else "unknown"
    runs = _scan_result_dir(result_root)
    report = {
        "schema_version": "ulfloc_reproduction_audit_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "paper_reference": dict(PAPER_ULF_CAMBRIDGE),
        "checkout": {
            "path": str(ulf_root),
            "checkout_exists": ulf_root.exists(),
            "git_head": _git_text(ulf_root, ["rev-parse", "HEAD"]) if ulf_root.exists() else "unknown",
            "git_status_short": git_status,
            "has_local_modifications": bool(git_status and git_status != "unknown"),
            "expected_files": expected,
            "missing_files": [rel for rel, exists in expected.items() if not exists],
        },
        "config": _config_status(ulf_root, runs) if ulf_root.exists() else {},
        "environment": {
            "python": sys.executable,
            "package_versions": {
                "torch": _package_version("torch"),
                "gsplat": _package_version("gsplat"),
                "opencv-python": _package_version("opencv-python"),
                "numpy": _package_version("numpy"),
            },
        },
        "results": _summarize_results(runs),
    }
    report["gaps"] = classify_ulfloc_reproduction_gaps(report)
    return report


def classify_ulfloc_reproduction_gaps(report: dict[str, Any]) -> list[dict[str, str]]:
    gaps: list[dict[str, str]] = []
    checkout = report.get("checkout", {})
    results = report.get("results", {})
    config = report.get("config", {})
    package_versions = report.get("environment", {}).get("package_versions", {})

    if checkout.get("missing_files"):
        gaps.append(
            {
                "id": "incomplete_ulfloc_checkout",
                "severity": "critical",
                "evidence": ", ".join(checkout.get("missing_files", [])),
                "recommendation": "restore the official ULF-Loc checkout before claiming reproduction.",
            }
        )
    if checkout.get("has_local_modifications"):
        gaps.append(
            {
                "id": "ulf_checkout_has_local_modifications",
                "severity": "high",
                "evidence": "git status is not clean",
                "recommendation": "run native ULF from a clean worktree or report it as a modified baseline.",
            }
        )
    if int(results.get("complete_test_recipe_group_count", 0)) <= 0:
        gaps.append(
            {
                "id": "missing_five_scene_official_test_reproduction",
                "severity": "critical",
                "evidence": (
                    f"complete test recipe groups found: {results.get('complete_test_recipe_group_count', 0)}; "
                    f"union test scenes found: {results.get('official_test_scene_count', 0)}/5"
                ),
                "recommendation": "produce frozen five-scene Cambridge test outputs with audit files.",
            }
        )
    if int(results.get("train_dev_scene_count", 0)) > 0 and int(results.get("official_test_scene_count", 0)) == 0:
        gaps.append(
            {
                "id": "only_train_dev_or_diagnostic_outputs_found",
                "severity": "medium",
                "evidence": "train/dev outputs exist but no official-test reproduction was found",
                "recommendation": "use train/dev only for debugging; keep paper comparison to paper-reported numbers until frozen test is run.",
            }
        )
    if package_versions.get("gsplat") in {None, "", "unknown"}:
        gaps.append(
            {
                "id": "unknown_gsplat_version",
                "severity": "high",
                "evidence": "gsplat is not visible to the active Python environment",
                "recommendation": "match ULF-Loc's required gsplat stack before attributing accuracy gaps to the method.",
            }
        )
    if not bool(config.get("uses_processed_images")) or not bool(config.get("mask_config_present")):
        gaps.append(
            {
                "id": "processed_images_or_masks_not_confirmed",
                "severity": "high",
                "evidence": "config/manifest scan did not confirm both processed images and mask usage",
                "recommendation": "verify Cambridge processed images and Mask2Former masks are used for native ULF runs.",
            }
        )
    dense = _as_float(results.get("best_five_scene_test_avg_te_cm"))
    if dense is not None and dense > PAPER_ULF_CAMBRIDGE["avg_dense_median_te_cm"] * 1.25:
        gaps.append(
            {
                "id": "local_dense_metric_far_from_paper",
                "severity": "high",
                "evidence": f"best local dense median TE {dense:.3f}cm vs paper avg {PAPER_ULF_CAMBRIDGE['avg_dense_median_te_cm']:.3f}cm",
                "recommendation": "audit config, checkpoint, feature fusion, landmark count, and evaluator path before adding Loc-GS changes.",
            }
        )
    return gaps


def audit_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ULF-Loc Reproduction Audit",
        "",
        f"- checkout: `{report.get('checkout', {}).get('path', '')}`",
        f"- git head: `{report.get('checkout', {}).get('git_head', 'unknown')}`",
        f"- runs scanned: `{report.get('results', {}).get('run_count', 0)}`",
        f"- scenes scanned: `{report.get('results', {}).get('scene_count', 0)}`",
        f"- official-test scenes: `{report.get('results', {}).get('official_test_scene_count', 0)}/5`",
        f"- complete test recipe groups: `{report.get('results', {}).get('complete_test_recipe_group_count', 0)}`",
        f"- best five-scene test avg dense TE: `{report.get('results', {}).get('best_five_scene_test_avg_te_cm')}`",
        f"- best single-scene dense TE: `{report.get('results', {}).get('best_dense_single_scene_te_cm')}`",
        "",
        "## Gaps",
        "",
    ]
    gaps = report.get("gaps", [])
    if not gaps:
        lines.append("No reproduction blockers were detected by this audit.")
    for gap in gaps:
        lines.append(f"- `{gap['severity']}` `{gap['id']}`: {gap['evidence']}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This audit does not prove ULF-Loc paper parity. It identifies whether local outputs are suitable for diagnosing the local reproduction gap before Loc-GS modifications are compared.",
        ]
    )
    return "\n".join(lines) + "\n"
