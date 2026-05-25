import json

from loc_gs.reporting.submission_alignment import build_submission_alignment_report


def test_submission_alignment_report_marks_missing_ulf_reproduction_and_reporting_fields():
    board = {
        "runs": [
            {
                "run_name": "shop_lsf",
                "scene": "ShopFacade",
                "run_role": "main_candidate",
                "paper_safe": False,
                "paper_safety_reason": "split audit unknown",
                "metrics": {
                    "dense": {
                        "median_te_cm": 3.79,
                        "median_re_deg": 0.25,
                        "recall_5cm_5deg": 0.56,
                        "recall_2cm_2deg": 0.19,
                    }
                },
            }
        ]
    }
    references = {
        "methods": [
            {
                "name": "Native STDLoc parity",
                "role": "baseline",
                "code_reproduced": True,
                "source": "third_party/stdloc",
            },
            {
                "name": "ULF-Loc",
                "role": "external_sota",
                "code_reproduced": False,
                "source": "https://arxiv.org/abs/2605.04730",
            },
        ]
    }

    report = build_submission_alignment_report(board, references=references)

    assert report["submission_ready"] is False
    assert "ULF-Loc" in report["missing_reproductions"]
    row = report["runs"][0]
    assert row["missing_dense_metrics"] == ["recall_10cm_5deg"]
    assert row["missing_reporting_fields"] == ["runtime", "memory", "map_size", "edit_budget"]
    json.dumps(report)


def test_submission_alignment_report_allows_paper_reported_external_reference():
    board = {
        "runs": [
            {
                "run_name": "shop_lsf",
                "scene": "ShopFacade",
                "run_role": "main_candidate",
                "paper_safe": True,
                "metrics": {
                    "dense": {
                        "median_te_cm": 3.79,
                        "median_re_deg": 0.25,
                        "recall_10cm_5deg": 0.9,
                        "recall_5cm_5deg": 0.56,
                        "recall_2cm_2deg": 0.19,
                    }
                },
                "reporting": {
                    "runtime": {"mean_ms": 10.0},
                    "memory": {"peak_gpu_mb": 1000},
                    "map_size": {"sampled_count": 8192},
                    "edit_budget": {"max_edits": 32},
                },
            }
        ]
    }
    references = {
        "methods": [
            {
                "name": "ULF-Loc",
                "role": "external_sota",
                "code_reproduced": False,
                "reference_mode": "paper_reported",
                "paper_reported_reference": True,
                "requires_local_reproduction": False,
            },
        ]
    }

    report = build_submission_alignment_report(board, references=references)

    assert report["missing_reproductions"] == []
    assert report["paper_reported_references"] == ["ULF-Loc"]
    assert report["submission_ready"] is True
