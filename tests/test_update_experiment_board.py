import json

from loc_gs.scripts.update_experiment_board import build_argparser, main


def _write_run(root, name, *, scene="ShopFacade", manifest=True, audit_status="passed", role=None, audit_bundle=True):
    run = root / name
    run.mkdir(parents=True)
    summary = {
        "scene": scene,
        "dense": {
            "median_te": 2.5,
            "median_ae": 0.12,
            "recall_10cm_5d": 0.9,
            "recall_5cm_5d": 0.8,
            "recall_2cm_2d": 0.3,
        },
    }
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    if audit_bundle:
        (run / "metrics_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (run / "command.txt").write_text("python -m loc_gs.scripts.eval_stdloc_native\n", encoding="utf-8")
        (run / "git_status.txt").write_text("", encoding="utf-8")
    if manifest:
        payload = {
            "scene": scene,
            "split": "selfmap_train",
            "command": ["python", "-m", "loc_gs.scripts.eval_stdloc_native"],
            "checkpoint_path": f"output/checkpoints/{name}.pth",
            "map_path": "output/stdloc/map_cambridge_spgs/ShopFacade",
            "data_roots": ["/mnt/pool/sqy/dataset/Cambridge/ShopFacade"],
            "hyperparameters": {"rho": 0.0, "alpha": 0.0},
            "rho": 0.0,
            "git_commit": "abc123",
            "timestamp_utc": "2026-05-16T00:00:00+00:00",
            "feedback_enabled": False,
            "residual_enabled": False,
            "selector_enabled": False,
        }
        if role is not None:
            payload["run_role"] = role
        (run / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    if audit_status is not None:
        (run / "split_audit.json").write_text(
            json.dumps({"audit_status": audit_status, "checks": {}}),
            encoding="utf-8",
        )
    return run


def test_update_experiment_board_writes_markdown_and_json(tmp_path):
    root = tmp_path / "results"
    _write_run(root, "main_safe", role="main_candidate")
    _write_run(root, "missing_audit", manifest=False, audit_status=None)
    _write_run(root, "failed_audit", audit_status="failed")
    md = tmp_path / "board.md"
    js = tmp_path / "board.json"
    args = build_argparser().parse_args(
        [
            "--result_roots",
            str(root),
            "--output_markdown",
            str(md),
            "--output_json",
            str(js),
        ]
    )

    assert main(args) == 0

    rows = json.loads(js.read_text(encoding="utf-8"))["runs"]
    by_name = {row["run_name"]: row for row in rows}
    assert by_name["main_safe"]["paper_safe"] is True
    assert by_name["main_safe"]["run_role"] == "main_candidate"
    assert by_name["missing_audit"]["paper_safe"] is False
    assert by_name["missing_audit"]["run_role"] == "diagnostic"
    assert "missing manifest" in by_name["missing_audit"]["paper_safety_reason"]
    assert by_name["failed_audit"]["run_role"] == "rejected"
    text = md.read_text(encoding="utf-8")
    assert "| Run | Scene | Role | Paper-safe |" in text
    assert "main_safe" in text
    assert "failed_audit" in text


def test_update_experiment_board_requires_complete_audit_bundle_for_paper_safe(tmp_path):
    root = tmp_path / "results"
    _write_run(root, "incomplete_bundle", role="main_candidate", audit_bundle=False)
    js = tmp_path / "board.json"
    args = build_argparser().parse_args(
        [
            "--result_roots",
            str(root),
            "--output_json",
            str(js),
        ]
    )

    assert main(args) == 0

    row = json.loads(js.read_text(encoding="utf-8"))["runs"][0]
    assert row["paper_safe"] is False
    assert row["run_role"] == "main_candidate"
    assert "missing metrics_summary.json" in row["paper_safety_reason"]
    assert "missing command.txt" in row["paper_safety_reason"]
    assert "missing git diff/status" in row["paper_safety_reason"]


def test_update_experiment_board_requires_complete_manifest_fields_for_paper_safe(tmp_path):
    root = tmp_path / "results"
    run = _write_run(root, "incomplete_manifest", role="main_candidate")
    (run / "manifest.json").write_text(
        json.dumps({"scene": "ShopFacade", "run_role": "main_candidate"}),
        encoding="utf-8",
    )
    js = tmp_path / "board.json"
    args = build_argparser().parse_args(
        [
            "--result_roots",
            str(root),
            "--output_json",
            str(js),
        ]
    )

    assert main(args) == 0

    row = json.loads(js.read_text(encoding="utf-8"))["runs"][0]
    assert row["paper_safe"] is False
    assert row["run_role"] == "main_candidate"
    assert "missing manifest field split" in row["paper_safety_reason"]
    assert "missing manifest field checkpoint_path" in row["paper_safety_reason"]
    assert "missing manifest field hyperparameters" in row["paper_safety_reason"]


def test_update_experiment_board_can_mark_manifest_ablation(tmp_path):
    root = tmp_path / "results"
    _write_run(root, "ablation_safe", role="ablation")
    js = tmp_path / "board.json"
    args = build_argparser().parse_args(
        [
            "--result_roots",
            str(root),
            "--output_json",
            str(js),
        ]
    )

    assert main(args) == 0
    rows = json.loads(js.read_text(encoding="utf-8"))["runs"]

    assert rows[0]["run_role"] == "ablation"
    assert rows[0]["metrics"]["dense"]["median_te_cm"] == 2.5


def test_update_experiment_board_discovers_metrics_summary_bundles(tmp_path):
    root = tmp_path / "results"
    run = root / "audit_bundle"
    run.mkdir(parents=True)
    (run / "metrics_summary.json").write_text(
        json.dumps(
            {
                "scene": "ShopFacade",
                "dense": {
                    "median_te_cm": 2.25,
                    "median_re_deg": 0.11,
                    "recall_10cm_5deg": 0.91,
                    "recall_5cm_5deg": 0.82,
                    "recall_2cm_2deg": 0.31,
                },
            }
        ),
        encoding="utf-8",
    )
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "scene": "ShopFacade",
                "split": "selfmap_train",
                "command": ["python", "-m", "loc_gs.scripts.eval_stdloc_native"],
                "checkpoint_path": "output/stdloc_hybrid/ShopFacade/latest.pth",
                "map_path": "output/stdloc/map_cambridge_spgs/ShopFacade",
                "data_roots": ["/mnt/pool/sqy/dataset/Cambridge/ShopFacade"],
                "hyperparameters": {"rho": 0.0, "alpha": 0.0},
                "rho": 0.0,
                "git_commit": "abc123",
                "timestamp_utc": "2026-05-16T00:00:00+00:00",
                "feedback_enabled": False,
                "residual_enabled": False,
                "selector_enabled": False,
            }
        ),
        encoding="utf-8",
    )
    (run / "split_audit.json").write_text(json.dumps({"audit_status": "passed"}), encoding="utf-8")
    (run / "command.txt").write_text("python -m loc_gs.scripts.eval_stdloc_native\n", encoding="utf-8")
    (run / "git_status.txt").write_text("", encoding="utf-8")
    js = tmp_path / "board.json"
    args = build_argparser().parse_args(
        [
            "--result_roots",
            str(root),
            "--output_json",
            str(js),
        ]
    )

    assert main(args) == 0

    rows = json.loads(js.read_text(encoding="utf-8"))["runs"]
    assert len(rows) == 1
    assert rows[0]["run_name"] == "audit_bundle"
    assert rows[0]["paper_safe"] is True
    assert rows[0]["metrics"]["dense"]["median_te_cm"] == 2.25


def test_update_experiment_board_prefers_metrics_summary_when_both_exist(tmp_path):
    root = tmp_path / "results"
    run = _write_run(root, "audit_bundle")
    (run / "summary.json").write_text(
        json.dumps({"scene": "ShopFacade", "dense": {"median_te": 99.0}}),
        encoding="utf-8",
    )
    (run / "metrics_summary.json").write_text(
        json.dumps({"scene": "ShopFacade", "dense": {"median_te_cm": 2.25}}),
        encoding="utf-8",
    )
    js = tmp_path / "board.json"
    args = build_argparser().parse_args(
        [
            "--result_roots",
            str(root),
            "--output_json",
            str(js),
        ]
    )

    assert main(args) == 0

    rows = json.loads(js.read_text(encoding="utf-8"))["runs"]
    assert len(rows) == 1
    assert rows[0]["metrics"]["dense"]["median_te_cm"] == 2.25
