#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from loc_gs.data.colmap_pose_repair import cambridge_pose_qnorm_issue_map


CAMBRIDGE_SCENES = (
    "GreatCourt",
    "KingsCollege",
    "OldHospital",
    "ShopFacade",
    "StMarysChurch",
)

STAGE_NAMES = ("sparse", "dense")

RECALL_THRESHOLDS = {
    "R50cm5deg": (50.0, 5.0),
    "R15cm5deg": (15.0, 5.0),
    "R10cm5deg": (10.0, 5.0),
    "R5cm5deg": (5.0, 5.0),
    "R2cm2deg": (2.0, 2.0),
}

PAPER_REPORTED_MEDIANS = {
    "AS/SIFT": {
        "GreatCourt": (24.0, 0.13),
        "KingsCollege": (13.0, 0.22),
        "OldHospital": (20.0, 0.36),
        "ShopFacade": (4.0, 0.21),
        "StMarysChurch": (8.0, 0.25),
        "Avg": (14.0, 0.23),
    },
    "HLoc SP+SG": {
        "GreatCourt": (17.7, 0.11),
        "KingsCollege": (11.0, 0.20),
        "OldHospital": (15.1, 0.31),
        "ShopFacade": (4.2, 0.20),
        "StMarysChurch": (7.0, 0.22),
        "Avg": (11.0, 0.21),
    },
    "DSAC*": {
        "GreatCourt": (33.0, 0.21),
        "KingsCollege": (17.9, 0.31),
        "OldHospital": (21.1, 0.40),
        "ShopFacade": (5.2, 0.24),
        "StMarysChurch": (15.4, 0.51),
        "Avg": (18.5, 0.33),
    },
    "ACE": {
        "GreatCourt": (28.0, 0.10),
        "KingsCollege": (18.0, 0.30),
        "OldHospital": (25.0, 0.50),
        "ShopFacade": (5.0, 0.30),
        "StMarysChurch": (9.0, 0.30),
        "Avg": (17.0, 0.30),
    },
    "NeuMap": {
        "GreatCourt": (6.0, 0.10),
        "KingsCollege": (14.0, 0.19),
        "OldHospital": (19.0, 0.36),
        "ShopFacade": (6.0, 0.25),
        "StMarysChurch": (17.0, 0.53),
        "Avg": (12.0, 0.29),
    },
    "GLACE": {
        "GreatCourt": (19.0, 0.10),
        "KingsCollege": (19.0, 0.30),
        "OldHospital": (17.0, 0.40),
        "ShopFacade": (4.0, 0.20),
        "StMarysChurch": (9.0, 0.30),
        "Avg": (14.0, 0.30),
    },
    "NeRFMatch": {
        "GreatCourt": (19.6, 0.09),
        "KingsCollege": (12.5, 0.23),
        "OldHospital": (20.9, 0.38),
        "ShopFacade": (8.4, 0.40),
        "StMarysChurch": (10.9, 0.35),
        "Avg": (14.5, 0.29),
    },
    "GSplatLoc": {
        "KingsCollege": (31.0, 0.49),
        "OldHospital": (16.0, 0.68),
        "ShopFacade": (4.0, 0.34),
        "StMarysChurch": (14.0, 0.42),
        "Avg": (16.0, 0.48),
    },
    "GSFFs-PR Feature": {
        "KingsCollege": (17.0, 0.26),
        "OldHospital": (18.0, 0.36),
        "ShopFacade": (4.0, 0.25),
        "StMarysChurch": (8.0, 0.26),
        "Avg": (12.0, 0.30),
    },
    "STDLoc paper": {
        "GreatCourt": (15.7, 0.06),
        "KingsCollege": (15.0, 0.17),
        "OldHospital": (11.9, 0.21),
        "ShopFacade": (3.0, 0.13),
        "StMarysChurch": (4.7, 0.14),
        "Avg": (10.1, 0.14),
    },
    "ACE+GS-CPR": {
        "KingsCollege": (20.0, 0.29),
        "OldHospital": (21.0, 0.40),
        "ShopFacade": (5.0, 0.24),
        "StMarysChurch": (13.0, 0.40),
        "Avg": (15.0, 0.33),
    },
    "ULF-Loc": {
        "GreatCourt": (7.49, 0.04),
        "KingsCollege": (17.03, 0.19),
        "OldHospital": (10.26, 0.19),
        "ShopFacade": (2.83, 0.14),
        "StMarysChurch": (3.65, 0.11),
        "Avg": (8.3, 0.13),
    },
}

PAPER_REPORTED_RECALLS = {
    "HLoc SP+SG": (91.4, 64.8, 52.0),
    "ACE": (78.7, 43.1, 31.5),
    "GLACE": (91.0, 62.8, 47.6),
    "STDLoc paper": (95.4, 70.8, 59.9),
    "GLACE+GS-CPR": (92.5, 65.5, 50.7),
    "ACE+GS-CPR": (84.6, 56.8, 42.6),
    "ULF-Loc": (93.7, 72.0, 62.2),
}


@dataclass(frozen=True)
class RunRoots:
    baseline_root: Path
    candidate_root: Path
    data_root: Path
    output_dir: Path
    baseline_label: str
    candidate_label: str


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    position = min(1.0, max(0.0, float(q))) * float(len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - float(low)
    return round(float((1.0 - weight) * ordered[low] + weight * ordered[high]), 12)


def _tail_cvar(values: Sequence[float], fraction: float = 0.10) -> float | None:
    if not values:
        return None
    ordered = sorted((float(v) for v in values), reverse=True)
    count = max(1, int(math.ceil(len(ordered) * max(0.0, float(fraction)))))
    return float(sum(ordered[:count]) / count)


def _mean(values: Iterable[float | None]) -> float | None:
    filtered = [float(v) for v in values if v is not None]
    if not filtered:
        return None
    return float(sum(filtered) / len(filtered))


def _result_stage_row(row: Mapping[str, Any], stage: str) -> dict[str, float | None]:
    te = _safe_float(row.get(f"{stage}_TE"))
    re = _safe_float(row.get(f"{stage}_AE"))
    nested = row.get(stage)
    inliers = None
    if stage == "dense" and isinstance(nested, list) and nested:
        inliers = _safe_float(nested[-1].get("inliers") if isinstance(nested[-1], Mapping) else None)
    elif isinstance(nested, Mapping):
        inliers = _safe_float(nested.get("inliers"))
    return {"te_cm": te, "re_deg": re, "inliers": inliers}


def load_query_results(run_dir: Path) -> list[dict[str, Any]]:
    raw = _load_json(run_dir / "results.json")
    if isinstance(raw, Mapping) and isinstance(raw.get("results"), list):
        raw = raw["results"]
    if not isinstance(raw, list):
        raise ValueError(f"results.json must contain a list: {run_dir / 'results.json'}")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            continue
        row: dict[str, Any] = {"query_index": index}
        for stage in STAGE_NAMES:
            stage_row = _result_stage_row(item, stage)
            for key, value in stage_row.items():
                row[f"{stage}_{key}"] = value
        rows.append(row)
    return rows


def _recall(rows: Sequence[Mapping[str, Any]], stage: str, max_te_cm: float, max_re_deg: float) -> float | None:
    pairs = [
        (row.get(f"{stage}_te_cm"), row.get(f"{stage}_re_deg"))
        for row in rows
        if row.get(f"{stage}_te_cm") is not None and row.get(f"{stage}_re_deg") is not None
    ]
    if not pairs:
        return None
    hit = sum(1 for te, re in pairs if float(te) <= max_te_cm and float(re) <= max_re_deg)
    return float(hit / len(pairs))


def summarize_run(run_dir: Path) -> dict[str, Any]:
    rows = load_query_results(run_dir)
    summary_path = run_dir / "metrics_summary.json"
    file_summary = _load_json(summary_path) if summary_path.exists() else {}
    split_audit_path = run_dir / "split_audit.json"
    split_audit = _load_json(split_audit_path) if split_audit_path.exists() else {}
    out: dict[str, Any] = {
        "run_dir": str(run_dir),
        "query_count": len(rows),
        "sampled_count": file_summary.get("sampled_count"),
        "landmark_count": file_summary.get("landmark_count"),
        "map_size_mb": file_summary.get("map_size_mb"),
        "audit_status": split_audit.get("audit_status", "missing"),
    }
    for stage in STAGE_NAMES:
        tes = [float(row[f"{stage}_te_cm"]) for row in rows if row.get(f"{stage}_te_cm") is not None]
        res = [float(row[f"{stage}_re_deg"]) for row in rows if row.get(f"{stage}_re_deg") is not None]
        inliers = [
            float(row[f"{stage}_inliers"])
            for row in rows
            if row.get(f"{stage}_inliers") is not None
        ]
        stage_out: dict[str, Any] = {
            "median_te_cm": _median(tes),
            "median_re_deg": _median(res),
            "p90_te_cm": _percentile(tes, 0.90),
            "p95_te_cm": _percentile(tes, 0.95),
            "cvar10_te_cm": _tail_cvar(tes, 0.10),
            "severe_rate_1m": (
                float(sum(1 for value in tes if value > 100.0) / len(tes)) if tes else None
            ),
            "catastrophic_rate_5m": (
                float(sum(1 for value in tes if value > 500.0) / len(tes)) if tes else None
            ),
            "avg_inliers": _mean(inliers),
        }
        for name, (te_thr, re_thr) in RECALL_THRESHOLDS.items():
            stage_out[name] = _recall(rows, stage, te_thr, re_thr)
        out[stage] = stage_out
    return out


def _read_test_image_names(data_root: Path, scene: str) -> list[str]:
    split_path = data_root / scene / "dataset_test.txt"
    if not split_path.exists():
        return []
    names: list[str] = []
    for line in split_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        first = stripped.split()[0].strip(",")
        if Path(first).suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            continue
        names.append(first)
    return sorted(names)


def _resolve_image_path(data_root: Path, scene: str, image_name: str) -> str:
    scene_root = data_root / scene
    for candidate in (scene_root / "processed" / image_name, scene_root / image_name):
        if candidate.exists():
            return str(candidate)
    return str(scene_root / "processed" / image_name)


def _threshold_hit(te: float | None, re: float | None, te_thr: float, re_thr: float) -> bool | None:
    if te is None or re is None:
        return None
    return float(te) <= te_thr and float(re) <= re_thr


def build_qualitative_cases(
    *,
    scene: str,
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    data_root: Path,
    worse_cm: float = 5.0,
    better_cm: float = 5.0,
) -> list[dict[str, Any]]:
    image_names = _read_test_image_names(data_root, scene)
    qnorm_issues = cambridge_pose_qnorm_issue_map(data_root / scene, split="test")
    count = min(len(baseline_rows), len(candidate_rows))
    cases: list[dict[str, Any]] = []
    for index in range(count):
        baseline = baseline_rows[index]
        candidate = candidate_rows[index]
        image_name = image_names[index] if index < len(image_names) else f"query_{index:06d}"
        row: dict[str, Any] = {
            "scene": scene,
            "query_index": index,
            "image_name": image_name,
            "image_path": _resolve_image_path(data_root, scene, image_name),
        }
        categories: list[str] = []
        qnorm_issue = qnorm_issues.get(image_name)
        if qnorm_issue is not None:
            row["pose_qnorm"] = round(float(qnorm_issue.qnorm), 6)
            row["pose_qnorm_delta"] = round(float(qnorm_issue.delta), 6)
            categories.append("pose_qnorm_outlier")
        for stage in STAGE_NAMES:
            for metric in ("te_cm", "re_deg", "inliers"):
                b_value = baseline.get(f"{stage}_{metric}")
                c_value = candidate.get(f"{stage}_{metric}")
                row[f"baseline_{stage}_{metric}"] = b_value
                row[f"candidate_{stage}_{metric}"] = c_value
                if b_value is not None and c_value is not None:
                    row[f"delta_{stage}_{metric}"] = float(c_value) - float(b_value)

        delta_dense = row.get("delta_dense_te_cm")
        delta_sparse = row.get("delta_sparse_te_cm")
        if delta_dense is not None and float(delta_dense) >= worse_cm:
            categories.append("candidate_worse_dense")
        if delta_dense is not None and float(delta_dense) <= -better_cm:
            categories.append("candidate_better_dense")
        if (
            delta_sparse is not None
            and delta_dense is not None
            and float(delta_sparse) <= -1.0
            and float(delta_dense) >= 1.0
        ):
            categories.append("sparse_help_dense_hurt")

        cand_sparse_te = row.get("candidate_sparse_te_cm")
        cand_dense_te = row.get("candidate_dense_te_cm")
        base_sparse_te = row.get("baseline_sparse_te_cm")
        base_dense_te = row.get("baseline_dense_te_cm")
        if cand_sparse_te is not None and cand_dense_te is not None and float(cand_dense_te) - float(cand_sparse_te) >= worse_cm:
            categories.append("candidate_dense_refinement_worsens")
        if base_sparse_te is not None and base_dense_te is not None and float(base_dense_te) - float(base_sparse_te) >= worse_cm:
            categories.append("baseline_dense_refinement_worsens")
        cand_dense_re = row.get("candidate_dense_re_deg")
        if cand_dense_te is not None and (float(cand_dense_te) >= 50.0 or (cand_dense_re is not None and float(cand_dense_re) >= 2.0)):
            categories.append("hard_failure")

        for name, te_thr, re_thr in (("R10", 10.0, 5.0), ("R5", 5.0, 5.0), ("R2", 2.0, 2.0)):
            base_hit = _threshold_hit(row.get("baseline_dense_te_cm"), row.get("baseline_dense_re_deg"), te_thr, re_thr)
            cand_hit = _threshold_hit(row.get("candidate_dense_te_cm"), row.get("candidate_dense_re_deg"), te_thr, re_thr)
            if base_hit is True and cand_hit is False:
                categories.append(f"{name}_loss")
            elif base_hit is False and cand_hit is True:
                categories.append(f"{name}_gain")

        row["categories"] = categories
        if categories:
            cases.append(row)
    return cases


def _paired_hard_metrics(
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    stage: str = "dense",
    regression_20cm: float = 20.0,
    regression_50cm: float = 50.0,
    sparse_correct_te_cm: float = 5.0,
    sparse_correct_re_deg: float = 5.0,
    dense_wrong_te_cm: float = 10.0,
    dense_wrong_re_deg: float = 5.0,
    dense_worsen_margin_cm: float = 5.0,
) -> dict[str, Any]:
    count = min(len(baseline_rows), len(candidate_rows))
    regression_20 = 0
    regression_50 = 0
    improvement_20 = 0
    sparse_correct_dense_wrong = 0
    candidate_dense_worsened = 0
    baseline_dense_worsened = 0
    for index in range(count):
        baseline = baseline_rows[index]
        candidate = candidate_rows[index]
        b_te = _safe_float(baseline.get(f"{stage}_te_cm"))
        c_te = _safe_float(candidate.get(f"{stage}_te_cm"))
        if b_te is not None and c_te is not None:
            delta = c_te - b_te
            if delta >= regression_20cm:
                regression_20 += 1
            if delta >= regression_50cm:
                regression_50 += 1
            if delta <= -regression_20cm:
                improvement_20 += 1

        c_sparse_te = _safe_float(candidate.get("sparse_te_cm"))
        c_sparse_re = _safe_float(candidate.get("sparse_re_deg"))
        c_dense_te = _safe_float(candidate.get("dense_te_cm"))
        c_dense_re = _safe_float(candidate.get("dense_re_deg"))
        if (
            c_sparse_te is not None
            and c_sparse_re is not None
            and c_dense_te is not None
            and c_dense_re is not None
            and c_sparse_te <= sparse_correct_te_cm
            and c_sparse_re <= sparse_correct_re_deg
            and (c_dense_te > dense_wrong_te_cm or c_dense_re > dense_wrong_re_deg)
        ):
            sparse_correct_dense_wrong += 1
        if c_sparse_te is not None and c_dense_te is not None and c_dense_te - c_sparse_te >= dense_worsen_margin_cm:
            candidate_dense_worsened += 1

        b_sparse_te = _safe_float(baseline.get("sparse_te_cm"))
        b_dense_te = _safe_float(baseline.get("dense_te_cm"))
        if b_sparse_te is not None and b_dense_te is not None and b_dense_te - b_sparse_te >= dense_worsen_margin_cm:
            baseline_dense_worsened += 1
    return {
        f"{stage}_regression_20cm_count": int(regression_20),
        f"{stage}_regression_50cm_count": int(regression_50),
        f"{stage}_improvement_20cm_count": int(improvement_20),
        "sparse_correct_dense_wrong_count": int(sparse_correct_dense_wrong),
        "candidate_dense_worsened_count": int(candidate_dense_worsened),
        "baseline_dense_worsened_count": int(baseline_dense_worsened),
    }


def build_hard_case_rows(cases: Sequence[Mapping[str, Any]], *, source_split: str = "unknown") -> list[dict[str, Any]]:
    hard_rows: list[dict[str, Any]] = []
    for case in cases:
        row = dict(case)
        row["source_split"] = source_split
        row["paper_safe_for_tuning"] = bool(str(source_split).lower() not in {"test", "native_eval"})
        hard_rows.append(row)
    return hard_rows


def _case_severity(case: Mapping[str, Any], category: str) -> float:
    delta = _safe_float(case.get("delta_dense_te_cm")) or 0.0
    if "better" in category or category.endswith("_gain"):
        return -delta
    if category == "candidate_dense_refinement_worsens":
        dense = _safe_float(case.get("candidate_dense_te_cm")) or 0.0
        sparse = _safe_float(case.get("candidate_sparse_te_cm")) or 0.0
        return dense - sparse
    if category == "baseline_dense_refinement_worsens":
        dense = _safe_float(case.get("baseline_dense_te_cm")) or 0.0
        sparse = _safe_float(case.get("baseline_sparse_te_cm")) or 0.0
        return dense - sparse
    return delta


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            if isinstance(flat.get("categories"), list):
                flat["categories"] = ";".join(str(item) for item in flat["categories"])
            writer.writerow(flat)


def _fmt_float(value: Any, digits: int = 3, pct: bool = False) -> str:
    number = _safe_float(value)
    if number is None:
        return "-"
    if pct:
        number *= 100.0
    return f"{number:.{digits}f}"


def _fmt_pair(pair: tuple[float, float] | None) -> str:
    if pair is None:
        return "-"
    return f"{pair[0]:.2f}/{pair[1]:.2f}"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    out.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(out)


def _write_contact_sheets(
    *,
    cases: Sequence[Mapping[str, Any]],
    output_dir: Path,
    top_k: int,
) -> list[dict[str, Any]]:
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        return [{"warning": f"PIL unavailable; skipped contact sheets: {exc}"}]

    by_category: dict[str, list[Mapping[str, Any]]] = {}
    for case in cases:
        for category in case.get("categories", []):
            by_category.setdefault(str(category), []).append(case)

    written: list[dict[str, Any]] = []
    sheet_dir = output_dir / "contact_sheets"
    thumb_dir = output_dir / "thumbnails"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    thumb_dir.mkdir(parents=True, exist_ok=True)
    thumb_w, thumb_h, caption_h = 260, 180, 72
    columns = 4
    for category, category_cases in sorted(by_category.items()):
        selected = sorted(category_cases, key=lambda row: _case_severity(row, category), reverse=True)[:top_k]
        if not selected:
            continue
        rows = math.ceil(len(selected) / columns)
        sheet = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + caption_h)), "white")
        draw = ImageDraw.Draw(sheet)
        category_thumb_dir = thumb_dir / category
        category_thumb_dir.mkdir(parents=True, exist_ok=True)
        for idx, case in enumerate(selected):
            x = (idx % columns) * thumb_w
            y = (idx // columns) * (thumb_h + caption_h)
            image_path = Path(str(case.get("image_path", "")))
            try:
                image = Image.open(image_path).convert("RGB")
                image.thumbnail((thumb_w, thumb_h))
                thumb = Image.new("RGB", (thumb_w, thumb_h), (238, 238, 238))
                ox = (thumb_w - image.width) // 2
                oy = (thumb_h - image.height) // 2
                thumb.paste(image, (ox, oy))
            except Exception:
                thumb = Image.new("RGB", (thumb_w, thumb_h), (230, 230, 230))
                ImageDraw.Draw(thumb).text((12, 78), "image unavailable", fill=(50, 50, 50))
            sheet.paste(thumb, (x, y))
            stem = f"{case.get('scene')}_{int(case.get('query_index', 0)):05d}.jpg"
            thumb.save(category_thumb_dir / stem, quality=90)
            caption = (
                f"{case.get('scene')} #{case.get('query_index')}\n"
                f"B {float(case.get('baseline_dense_te_cm') or 0):.1f}cm/"
                f"{float(case.get('baseline_dense_re_deg') or 0):.2f}deg  "
                f"C {float(case.get('candidate_dense_te_cm') or 0):.1f}cm/"
                f"{float(case.get('candidate_dense_re_deg') or 0):.2f}deg\n"
                f"dTE {float(case.get('delta_dense_te_cm') or 0):+.1f}cm"
            )
            draw.text((x + 6, y + thumb_h + 6), caption, fill=(0, 0, 0))
        path = sheet_dir / f"{category}.jpg"
        sheet.save(path, quality=92)
        written.append({"category": category, "path": str(path), "count": len(selected)})
    return written


def _copy_audit_files(roots: RunRoots, scenes: Sequence[str]) -> None:
    audit_dir = roots.output_dir / "audit_inputs"
    audit_dir.mkdir(parents=True, exist_ok=True)
    for label, root in ((roots.baseline_label, roots.baseline_root), (roots.candidate_label, roots.candidate_root)):
        for scene in scenes:
            run_dir = root / scene
            target = audit_dir / label / scene
            target.mkdir(parents=True, exist_ok=True)
            for name in ("manifest.json", "command.txt", "metrics_summary.json", "split_audit.json", "git_status.txt"):
                source = run_dir / name
                if source.exists():
                    shutil.copy2(source, target / name)


def build_report(roots: RunRoots, scenes: Sequence[str], top_k: int = 12) -> dict[str, Any]:
    roots.output_dir.mkdir(parents=True, exist_ok=True)
    runs: dict[str, dict[str, Any]] = {roots.baseline_label: {}, roots.candidate_label: {}}
    query_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for label, root in ((roots.baseline_label, roots.baseline_root), (roots.candidate_label, roots.candidate_root)):
        for scene in scenes:
            run_dir = root / scene
            runs[label][scene] = summarize_run(run_dir)
            query_rows[(label, scene)] = load_query_results(run_dir)

    all_cases: list[dict[str, Any]] = []
    for scene in scenes:
        all_cases.extend(
            build_qualitative_cases(
                scene=scene,
                baseline_rows=query_rows[(roots.baseline_label, scene)],
                candidate_rows=query_rows[(roots.candidate_label, scene)],
                data_root=roots.data_root,
            )
        )

    metric_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    for scene in scenes:
        paired = {"scene": scene}
        paired.update(
            _paired_hard_metrics(
                query_rows[(roots.baseline_label, scene)],
                query_rows[(roots.candidate_label, scene)],
                stage="dense",
            )
        )
        paired_rows.append(paired)
        for label in (roots.baseline_label, roots.candidate_label):
            summary = runs[label][scene]
            for stage in STAGE_NAMES:
                stage_summary = summary[stage]
                metric_rows.append(
                    {
                        "method": label,
                        "scene": scene,
                        "stage": stage,
                        "query_count": summary["query_count"],
                        "median_te_cm": stage_summary.get("median_te_cm"),
                        "median_re_deg": stage_summary.get("median_re_deg"),
                        "R50cm5deg": stage_summary.get("R50cm5deg"),
                        "R15cm5deg": stage_summary.get("R15cm5deg"),
                        "R10cm5deg": stage_summary.get("R10cm5deg"),
                        "R5cm5deg": stage_summary.get("R5cm5deg"),
                        "R2cm2deg": stage_summary.get("R2cm2deg"),
                        "p90_te_cm": stage_summary.get("p90_te_cm"),
                        "p95_te_cm": stage_summary.get("p95_te_cm"),
                        "cvar10_te_cm": stage_summary.get("cvar10_te_cm"),
                        "severe_rate_1m": stage_summary.get("severe_rate_1m"),
                        "catastrophic_rate_5m": stage_summary.get("catastrophic_rate_5m"),
                        "avg_inliers": stage_summary.get("avg_inliers"),
                        "sampled_count": summary.get("sampled_count"),
                        "landmark_count": summary.get("landmark_count"),
                        "map_size_mb": summary.get("map_size_mb"),
                        "audit_status": summary.get("audit_status"),
                        "run_dir": summary.get("run_dir"),
                    }
                )
        base = runs[roots.baseline_label][scene]
        cand = runs[roots.candidate_label][scene]
        for stage in STAGE_NAMES:
            row = {"scene": scene, "stage": stage}
            for key in ("median_te_cm", "median_re_deg", "R50cm5deg", "R15cm5deg", "R10cm5deg", "R5cm5deg", "R2cm2deg", "p90_te_cm", "p95_te_cm", "cvar10_te_cm", "severe_rate_1m", "catastrophic_rate_5m", "avg_inliers"):
                b = base[stage].get(key)
                c = cand[stage].get(key)
                row[f"baseline_{key}"] = b
                row[f"candidate_{key}"] = c
                row[f"delta_{key}"] = (float(c) - float(b)) if b is not None and c is not None else None
            delta_rows.append(row)

    macro: dict[str, Any] = {}
    for label in (roots.baseline_label, roots.candidate_label):
        macro[label] = {}
        for stage in STAGE_NAMES:
            macro[label][stage] = {
                key: _mean(runs[label][scene][stage].get(key) for scene in scenes)
                for key in ("median_te_cm", "median_re_deg", "R50cm5deg", "R15cm5deg", "R10cm5deg", "R5cm5deg", "R2cm2deg", "p90_te_cm", "p95_te_cm", "cvar10_te_cm", "severe_rate_1m", "catastrophic_rate_5m", "avg_inliers")
            }
    macro["delta"] = {}
    for stage in STAGE_NAMES:
        macro["delta"][stage] = {
            key: (
                float(macro[roots.candidate_label][stage][key]) - float(macro[roots.baseline_label][stage][key])
                if macro[roots.candidate_label][stage][key] is not None
                and macro[roots.baseline_label][stage][key] is not None
                else None
            )
            for key in macro[roots.baseline_label][stage]
        }

    hard_cases = build_hard_case_rows(all_cases, source_split="native_eval")
    contact_sheets = _write_contact_sheets(cases=all_cases, output_dir=roots.output_dir, top_k=top_k)
    report = {
        "protocol": {
            "scenes": list(scenes),
            "baseline_root": str(roots.baseline_root),
            "candidate_root": str(roots.candidate_root),
            "data_root": str(roots.data_root),
            "baseline_label": roots.baseline_label,
            "candidate_label": roots.candidate_label,
            "recall_thresholds": RECALL_THRESHOLDS,
            "qualitative_image_order": "sorted STDLoc eval image_name order from dataset_test.txt",
        },
        "runs": runs,
        "macro": macro,
        "metric_rows": metric_rows,
        "delta_rows": delta_rows,
        "paired_rows": paired_rows,
        "qualitative_cases": all_cases,
        "hard_cases": hard_cases,
        "contact_sheets": contact_sheets,
        "paper_reported_medians": PAPER_REPORTED_MEDIANS,
        "paper_reported_recalls": PAPER_REPORTED_RECALLS,
    }
    _write_csv(
        roots.output_dir / "cambridge_metrics.csv",
        metric_rows,
        (
            "method",
            "scene",
            "stage",
            "query_count",
            "median_te_cm",
            "median_re_deg",
            "R50cm5deg",
            "R15cm5deg",
            "R10cm5deg",
            "R5cm5deg",
            "R2cm2deg",
            "p90_te_cm",
            "p95_te_cm",
            "cvar10_te_cm",
            "severe_rate_1m",
            "catastrophic_rate_5m",
            "avg_inliers",
            "sampled_count",
            "landmark_count",
            "map_size_mb",
            "audit_status",
            "run_dir",
        ),
    )
    _write_csv(
        roots.output_dir / "cambridge_deltas.csv",
        delta_rows,
        (
            "scene",
            "stage",
            "baseline_median_te_cm",
            "candidate_median_te_cm",
            "delta_median_te_cm",
            "baseline_median_re_deg",
            "candidate_median_re_deg",
            "delta_median_re_deg",
            "baseline_R50cm5deg",
            "candidate_R50cm5deg",
            "delta_R50cm5deg",
            "baseline_R15cm5deg",
            "candidate_R15cm5deg",
            "delta_R15cm5deg",
            "baseline_R10cm5deg",
            "candidate_R10cm5deg",
            "delta_R10cm5deg",
            "baseline_R5cm5deg",
            "candidate_R5cm5deg",
            "delta_R5cm5deg",
            "baseline_R2cm2deg",
            "candidate_R2cm2deg",
            "delta_R2cm2deg",
            "baseline_p90_te_cm",
            "candidate_p90_te_cm",
            "delta_p90_te_cm",
            "baseline_p95_te_cm",
            "candidate_p95_te_cm",
            "delta_p95_te_cm",
            "baseline_cvar10_te_cm",
            "candidate_cvar10_te_cm",
            "delta_cvar10_te_cm",
            "baseline_severe_rate_1m",
            "candidate_severe_rate_1m",
            "delta_severe_rate_1m",
            "baseline_catastrophic_rate_5m",
            "candidate_catastrophic_rate_5m",
            "delta_catastrophic_rate_5m",
            "baseline_avg_inliers",
            "candidate_avg_inliers",
            "delta_avg_inliers",
        ),
    )
    _write_csv(
        roots.output_dir / "qualitative_cases.csv",
        all_cases,
        (
            "scene",
            "query_index",
            "image_name",
            "image_path",
            "categories",
            "pose_qnorm",
            "pose_qnorm_delta",
            "baseline_sparse_te_cm",
            "baseline_sparse_re_deg",
            "candidate_sparse_te_cm",
            "candidate_sparse_re_deg",
            "delta_sparse_te_cm",
            "delta_sparse_re_deg",
            "baseline_dense_te_cm",
            "baseline_dense_re_deg",
            "candidate_dense_te_cm",
            "candidate_dense_re_deg",
            "delta_dense_te_cm",
            "delta_dense_re_deg",
            "baseline_dense_inliers",
            "candidate_dense_inliers",
            "delta_dense_inliers",
        ),
    )
    _write_csv(
        roots.output_dir / "hard_cases.csv",
        hard_cases,
        (
            "scene",
            "query_index",
            "image_name",
            "image_path",
            "source_split",
            "paper_safe_for_tuning",
            "categories",
            "pose_qnorm",
            "pose_qnorm_delta",
            "baseline_dense_te_cm",
            "candidate_dense_te_cm",
            "delta_dense_te_cm",
            "baseline_dense_re_deg",
            "candidate_dense_re_deg",
            "delta_dense_re_deg",
        ),
    )
    (roots.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (roots.output_dir / "report.md").write_text(_render_markdown(report, roots), encoding="utf-8")
    _copy_audit_files(roots, scenes)
    return report


def _render_markdown(report: Mapping[str, Any], roots: RunRoots) -> str:
    scenes = list(report["protocol"]["scenes"])
    lines = [
        "# Cambridge Complete Evaluation",
        "",
        "Protocol: official Cambridge test split, fixed STDLoc-compatible inference path, "
        "median translation/rotation plus strict recall thresholds. Test results are for evaluation only, not for tuning.",
        "",
        f"Baseline: `{roots.baseline_label}` -> `{roots.baseline_root}`",
        f"Candidate: `{roots.candidate_label}` -> `{roots.candidate_root}`",
        "",
        "## Macro Metrics",
    ]
    macro_rows: list[list[str]] = []
    for stage in STAGE_NAMES:
        for label in (roots.baseline_label, roots.candidate_label):
            values = report["macro"][label][stage]
            macro_rows.append(
                [
                    label,
                    stage,
                    _fmt_float(values.get("median_te_cm"), 3),
                    _fmt_float(values.get("median_re_deg"), 4),
                    _fmt_float(values.get("R50cm5deg"), 2, pct=True),
                    _fmt_float(values.get("R15cm5deg"), 2, pct=True),
                    _fmt_float(values.get("R10cm5deg"), 2, pct=True),
                    _fmt_float(values.get("R5cm5deg"), 2, pct=True),
                    _fmt_float(values.get("R2cm2deg"), 2, pct=True),
                ]
            )
        delta = report["macro"]["delta"][stage]
        macro_rows.append(
            [
                "candidate-baseline",
                stage,
                _fmt_float(delta.get("median_te_cm"), 3),
                _fmt_float(delta.get("median_re_deg"), 4),
                _fmt_float(delta.get("R50cm5deg"), 2, pct=True),
                _fmt_float(delta.get("R15cm5deg"), 2, pct=True),
                _fmt_float(delta.get("R10cm5deg"), 2, pct=True),
                _fmt_float(delta.get("R5cm5deg"), 2, pct=True),
                _fmt_float(delta.get("R2cm2deg"), 2, pct=True),
            ]
        )
    lines.append(_markdown_table(["Method", "Stage", "TE cm", "RE deg", "R50", "R15", "R10", "R5", "R2"], macro_rows))

    lines.extend(["", "## Per-Scene Dense Metrics"])
    dense_rows = []
    for row in report["metric_rows"]:
        if row["stage"] != "dense":
            continue
        dense_rows.append(
            [
                row["method"],
                row["scene"],
                str(row["query_count"]),
                _fmt_float(row["median_te_cm"], 3),
                _fmt_float(row["median_re_deg"], 4),
                _fmt_float(row["R50cm5deg"], 2, pct=True),
                _fmt_float(row["R15cm5deg"], 2, pct=True),
                _fmt_float(row["R10cm5deg"], 2, pct=True),
                _fmt_float(row["R5cm5deg"], 2, pct=True),
                _fmt_float(row["R2cm2deg"], 2, pct=True),
                _fmt_float(row["p90_te_cm"], 2),
                _fmt_float(row["p95_te_cm"], 2),
                _fmt_float(row["cvar10_te_cm"], 2),
                _fmt_float(row["avg_inliers"], 1),
                str(row.get("audit_status") or "-"),
            ]
        )
    lines.append(_markdown_table(["Method", "Scene", "N", "TE cm", "RE deg", "R50", "R15", "R10", "R5", "R2", "P90", "P95", "CVaR10", "Inliers", "Audit"], dense_rows))

    lines.extend(["", "## Hard Regression Metrics"])
    paired_rows = []
    for row in report.get("paired_rows", []):
        paired_rows.append(
            [
                row["scene"],
                str(row.get("dense_regression_20cm_count", 0)),
                str(row.get("dense_regression_50cm_count", 0)),
                str(row.get("dense_improvement_20cm_count", 0)),
                str(row.get("sparse_correct_dense_wrong_count", 0)),
                str(row.get("candidate_dense_worsened_count", 0)),
            ]
        )
    lines.append(_markdown_table(["Scene", "Reg20", "Reg50", "Imp20", "SparseOKDenseWrong", "DenseWorse"], paired_rows))

    lines.extend(["", "## Candidate - Baseline Delta"])
    delta_rows = []
    for row in report["delta_rows"]:
        if row["stage"] != "dense":
            continue
        delta_rows.append(
            [
                row["scene"],
                _fmt_float(row["delta_median_te_cm"], 3),
                _fmt_float(row["delta_median_re_deg"], 4),
                _fmt_float(row["delta_R50cm5deg"], 2, pct=True),
                _fmt_float(row["delta_R15cm5deg"], 2, pct=True),
                _fmt_float(row["delta_R10cm5deg"], 2, pct=True),
                _fmt_float(row["delta_R5cm5deg"], 2, pct=True),
                _fmt_float(row["delta_R2cm2deg"], 2, pct=True),
            ]
        )
    lines.append(_markdown_table(["Scene", "dTE cm", "dRE deg", "dR50", "dR15", "dR10", "dR5", "dR2"], delta_rows))

    lines.extend(["", "## Paper-Reported Median Pose Comparison"])
    method_rows = []
    candidate_dense = {
        scene: report["runs"][roots.candidate_label][scene]["dense"]
        for scene in scenes
    }
    local_baseline_dense = {
        scene: report["runs"][roots.baseline_label][scene]["dense"]
        for scene in scenes
    }
    for method, values in report["paper_reported_medians"].items():
        method_rows.append(
            [
                method,
                *[_fmt_pair(values.get(scene)) for scene in scenes],
                _fmt_pair(values.get("Avg")),
            ]
        )
    method_rows.append(
        [
            f"{roots.baseline_label} local",
            *[
                f"{local_baseline_dense[scene]['median_te_cm']:.2f}/{local_baseline_dense[scene]['median_re_deg']:.2f}"
                for scene in scenes
            ],
            f"{report['macro'][roots.baseline_label]['dense']['median_te_cm']:.2f}/{report['macro'][roots.baseline_label]['dense']['median_re_deg']:.2f}",
        ]
    )
    method_rows.append(
        [
            f"{roots.candidate_label} local",
            *[
                f"{candidate_dense[scene]['median_te_cm']:.2f}/{candidate_dense[scene]['median_re_deg']:.2f}"
                for scene in scenes
            ],
            f"{report['macro'][roots.candidate_label]['dense']['median_te_cm']:.2f}/{report['macro'][roots.candidate_label]['dense']['median_re_deg']:.2f}",
        ]
    )
    lines.append(_markdown_table(["Method", *scenes, "Avg"], method_rows))

    lines.extend(["", "## Paper-Reported Average Recall Comparison"])
    recall_rows = [
        [method, f"{r50:.1f}", f"{r15:.1f}", f"{r10:.1f}"]
        for method, (r50, r15, r10) in report["paper_reported_recalls"].items()
    ]
    for label in (roots.baseline_label, roots.candidate_label):
        dense = report["macro"][label]["dense"]
        recall_rows.append(
            [
                f"{label} local",
                _fmt_float(dense.get("R50cm5deg"), 1, pct=True),
                _fmt_float(dense.get("R15cm5deg"), 1, pct=True),
                _fmt_float(dense.get("R10cm5deg"), 1, pct=True),
            ]
        )
    lines.append(_markdown_table(["Method", "Avg R50/5", "Avg R15/5", "Avg R10/5"], recall_rows))

    lines.extend(["", "## Qualitative Package"])
    lines.append("- `qualitative_cases.csv`: paired per-query improvements, regressions, threshold flips, and dense-refinement failures.")
    lines.append("- `hard_cases.csv`: fixed hard-case list; official-test rows are marked unsafe for tuning.")
    lines.append("- `contact_sheets/`: thumbnails grouped by failure/improvement category.")
    lines.append("- `audit_inputs/`: copied manifest, command, metrics, split audit, and git status files where present.")
    return "\n".join(lines) + "\n"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build complete Cambridge quantitative and qualitative eval report.")
    parser.add_argument("--baseline_root", required=True)
    parser.add_argument("--candidate_root", required=True)
    parser.add_argument("--data_root", default="/mnt/pool/sqy/Cambridge_stdloc")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--baseline_label", default="native_baseline")
    parser.add_argument("--candidate_label", default="lsf_v6_guarded512")
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--top_k", type=int, default=12)
    return parser


def main(args: argparse.Namespace | None = None) -> None:
    ns = build_argparser().parse_args() if args is None else args
    scenes = tuple(ns.scene) if ns.scene else CAMBRIDGE_SCENES
    roots = RunRoots(
        baseline_root=Path(ns.baseline_root),
        candidate_root=Path(ns.candidate_root),
        data_root=Path(ns.data_root),
        output_dir=Path(ns.output_dir),
        baseline_label=str(ns.baseline_label),
        candidate_label=str(ns.candidate_label),
    )
    report = build_report(roots, scenes=scenes, top_k=int(ns.top_k))
    print(json.dumps({"output_dir": str(roots.output_dir), "macro": report["macro"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
