from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


OFFICIAL_REPO_URL = "https://github.com/Cyril-gyd/ULF-Loc"
OFFICIAL_ARXIV_URL = "https://arxiv.org/abs/2605.04730"
EXPECTED_FILES = (
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
OFFICIAL_CAMBRIDGE_REPORTING = {
    "avg_error_cm": 8.3,
    "avg_error_deg": 0.13,
    "recall_15cm_5deg_percent": 72.0,
    "recall_10cm_5deg_percent": 62.2,
}
LOCAL_REPRODUCTION_REQUIRED = "local_reproduction_required"
PAPER_REPORTED = "paper_reported"
REFERENCE_MODES = (LOCAL_REPRODUCTION_REQUIRED, PAPER_REPORTED)


def _path_status(root: Path | None) -> dict[str, Any]:
    if root is None:
        return {
            "checkout_path": "",
            "checkout_exists": False,
            "expected_files": {name: False for name in EXPECTED_FILES},
            "missing_files": list(EXPECTED_FILES),
        }
    expected = {name: (root / name).exists() for name in EXPECTED_FILES}
    return {
        "checkout_path": str(root),
        "checkout_exists": root.exists(),
        "expected_files": expected,
        "missing_files": [name for name, exists in expected.items() if not exists],
    }


def _result_status(result_root: Path | None) -> dict[str, Any]:
    if result_root is None:
        return {
            "result_root": "",
            "result_root_exists": False,
            "scene_outputs": {scene: False for scene in CAMBRIDGE_SCENES},
            "missing_scene_outputs": list(CAMBRIDGE_SCENES),
        }
    scene_outputs = {scene: (result_root / scene).exists() for scene in CAMBRIDGE_SCENES}
    return {
        "result_root": str(result_root),
        "result_root_exists": result_root.exists(),
        "scene_outputs": scene_outputs,
        "missing_scene_outputs": [scene for scene, exists in scene_outputs.items() if not exists],
    }


def build_ulfloc_alignment_reference(
    *,
    checkout_path: str | Path | None = None,
    result_root: str | Path | None = None,
    official_head: str = "",
    code_reproduced: bool = False,
    reference_mode: str = LOCAL_REPRODUCTION_REQUIRED,
    notes: str = "",
) -> dict[str, Any]:
    """Build a paper-safety reference for ULF-Loc without running its code."""

    if reference_mode not in REFERENCE_MODES:
        raise ValueError(f"reference_mode must be one of {REFERENCE_MODES}, got {reference_mode!r}")

    checkout = _path_status(Path(checkout_path) if checkout_path else None)
    results = _result_status(Path(result_root) if result_root else None)
    checkout_ready = bool(checkout["checkout_exists"]) and not checkout["missing_files"]
    result_ready = bool(results["result_root_exists"]) and not results["missing_scene_outputs"]
    reproduced = bool(code_reproduced and checkout_ready and result_ready)
    paper_reported = reference_mode == PAPER_REPORTED

    blockers: list[str] = []
    if not checkout_ready and not paper_reported:
        blockers.append("missing or incomplete local ULF-Loc checkout")
    if not result_ready and not paper_reported:
        blockers.append("missing complete local Cambridge ULF-Loc outputs")
    if not code_reproduced and not paper_reported:
        blockers.append("ULF-Loc run not explicitly marked reproduced")

    return {
        "method": {
            "name": "ULF-Loc",
            "role": "external_sota",
            "source": OFFICIAL_REPO_URL,
            "paper": OFFICIAL_ARXIV_URL,
            "official_head": official_head,
            "code_available": True,
            "code_reproduced": reproduced,
            "reference_mode": reference_mode,
            "paper_reported_reference": paper_reported,
            "requires_local_reproduction": not paper_reported,
            "reporting_allowed": bool(paper_reported or reproduced),
        },
        "official_cambridge_reporting": dict(OFFICIAL_CAMBRIDGE_REPORTING),
        "expected_cambridge_scenes": list(CAMBRIDGE_SCENES),
        "expected_files": list(EXPECTED_FILES),
        "checkout": checkout,
        "results": results,
        "reproduction_blockers": blockers,
        "comparison_limitations": [
            "ULF-Loc numbers are paper-reported external references, not locally reproduced."
        ]
        if paper_reported and not reproduced
        else [],
        "notes": notes,
    }


def write_ulfloc_alignment_reference(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
