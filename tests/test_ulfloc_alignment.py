import json

from loc_gs.reporting.ulfloc_alignment import (
    CAMBRIDGE_SCENES,
    EXPECTED_FILES,
    PAPER_REPORTED,
    build_ulfloc_alignment_reference,
)
from loc_gs.scripts.write_ulfloc_alignment_reference import main


def test_ulfloc_alignment_reference_requires_checkout_and_results():
    report = build_ulfloc_alignment_reference(official_head="abc123")

    assert report["method"]["name"] == "ULF-Loc"
    assert report["method"]["official_head"] == "abc123"
    assert report["method"]["code_available"] is True
    assert report["method"]["code_reproduced"] is False
    assert "missing or incomplete local ULF-Loc checkout" in report["reproduction_blockers"]
    assert report["checkout"]["missing_files"] == list(EXPECTED_FILES)
    assert report["results"]["missing_scene_outputs"] == list(CAMBRIDGE_SCENES)


def test_ulfloc_alignment_reference_marks_reproduced_only_with_complete_outputs(tmp_path):
    checkout = tmp_path / "ULF-Loc"
    checkout.mkdir()
    for name in EXPECTED_FILES:
        path = checkout / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    result_root = tmp_path / "outputs" / "cambridge"
    for scene in CAMBRIDGE_SCENES:
        (result_root / scene).mkdir(parents=True)

    report = build_ulfloc_alignment_reference(
        checkout_path=checkout,
        result_root=result_root,
        code_reproduced=True,
    )

    assert report["method"]["code_reproduced"] is True
    assert report["reproduction_blockers"] == []


def test_ulfloc_alignment_reference_allows_paper_reported_external_baseline():
    report = build_ulfloc_alignment_reference(
        official_head="abc123",
        reference_mode=PAPER_REPORTED,
    )

    assert report["method"]["code_reproduced"] is False
    assert report["method"]["paper_reported_reference"] is True
    assert report["method"]["requires_local_reproduction"] is False
    assert report["method"]["reporting_allowed"] is True
    assert report["reproduction_blockers"] == []
    assert report["comparison_limitations"]


def test_write_ulfloc_alignment_reference_cli(tmp_path):
    output = tmp_path / "ulfloc_reference.json"

    assert main(["--output-json", str(output), "--official-head", "abc123"]) == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["method"]["official_head"] == "abc123"
    assert report["method"]["code_reproduced"] is False


def test_write_ulfloc_alignment_reference_cli_paper_reported_mode(tmp_path):
    output = tmp_path / "ulfloc_reference.json"

    assert main(["--output-json", str(output), "--paper-reported-ok"]) == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["method"]["reference_mode"] == PAPER_REPORTED
    assert report["method"]["reporting_allowed"] is True
