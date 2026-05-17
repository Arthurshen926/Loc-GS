import json
import sys
from pathlib import Path

import pytest

from loc_gs.scripts import locgsctl


def _run_cli(capsys, *args: str) -> dict:
    code = locgsctl.main(list(args))
    assert code == 0
    captured = capsys.readouterr()
    return json.loads(captured.out)


def test_status_reports_environment_and_key_paths(tmp_path, monkeypatch, capsys):
    repo = tmp_path / "repo"
    (repo / "loc_gs").mkdir(parents=True)
    (repo / "third_party" / "stdloc").mkdir(parents=True)
    (repo / "docs").mkdir()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")

    payload = _run_cli(capsys, "--repo-root", str(repo), "status")

    assert payload["python_executable"] == sys.executable
    assert payload["cuda_visible_devices"] == "2"
    assert payload["paths"]["repo_root"]["exists"] is True
    assert payload["paths"]["loc_gs"]["exists"] is True
    assert payload["paths"]["third_party_stdloc"]["exists"] is True
    assert payload["paths"]["docs"]["exists"] is True
    assert "git_commit" in payload


def test_summarize_compacts_summary_metrics(tmp_path, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "model_path": "output/stdloc/map_cambridge_spgs/ShopFacade",
                "dense": {
                    "median_te": 2.5,
                    "median_ae": 0.12,
                    "recall_10cm_5d": 0.91,
                    "recall_5cm_5d": 0.80,
                    "recall_2cm_2d": 0.30,
                },
                "sparse": {
                    "median_te_cm": 4.0,
                    "median_re_deg": 0.2,
                    "recall_5cm_5deg": 0.5,
                },
            }
        ),
        encoding="utf-8",
    )

    payload = _run_cli(capsys, "summarize", str(run_dir))

    assert payload["source"] == str(run_dir / "summary.json")
    assert payload["model_path"] == "output/stdloc/map_cambridge_spgs/ShopFacade"
    assert payload["dense"] == {
        "median_te_cm": 2.5,
        "median_re_deg": 0.12,
        "recall_10cm_5deg": 0.91,
        "recall_5cm_5deg": 0.80,
        "recall_2cm_2deg": 0.30,
    }
    assert payload["sparse"]["median_te_cm"] == 4.0
    assert payload["sparse"]["recall_5cm_5deg"] == 0.5


def test_summarize_prefers_metrics_summary_for_audit_bundle_directories(tmp_path, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps({"scene": "ShopFacade", "dense": {"median_te": 99.0}}),
        encoding="utf-8",
    )
    (run_dir / "metrics_summary.json").write_text(
        json.dumps({"scene": "ShopFacade", "dense": {"median_te_cm": 2.25}}),
        encoding="utf-8",
    )

    payload = _run_cli(capsys, "summarize", str(run_dir))

    assert payload["source"] == str(run_dir / "metrics_summary.json")
    assert payload["dense"]["median_te_cm"] == 2.25


def test_compare_reports_candidate_minus_baseline_deltas(tmp_path, capsys):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    (baseline / "summary.json").write_text(
        json.dumps(
            {
                "dense": {
                    "median_te": 9.0,
                    "median_ae": 0.15,
                    "recall_10cm_5d": 0.50,
                    "recall_5cm_5d": 0.30,
                    "recall_2cm_2d": 0.10,
                }
            }
        ),
        encoding="utf-8",
    )
    (candidate / "summary.json").write_text(
        json.dumps(
            {
                "dense": {
                    "median_te_cm": 8.5,
                    "median_re_deg": 0.14,
                    "recall_10cm_5deg": 0.55,
                    "recall_5cm_5deg": 0.35,
                    "recall_2cm_2deg": 0.12,
                }
            }
        ),
        encoding="utf-8",
    )

    payload = _run_cli(capsys, "compare", str(baseline), str(candidate))

    assert payload["stage"] == "dense"
    assert payload["baseline"]["median_te_cm"] == 9.0
    assert payload["candidate"]["median_te_cm"] == 8.5
    assert payload["delta"]["median_te_cm"] == pytest.approx(-0.5)
    assert payload["delta"]["median_re_deg"] == pytest.approx(-0.01)
    assert payload["delta"]["recall_10cm_5deg"] == pytest.approx(0.05)
    assert payload["delta"]["recall_5cm_5deg"] == pytest.approx(0.05)
    assert payload["delta"]["recall_2cm_2deg"] == pytest.approx(0.02)


def test_list_scenes_uses_stdloc_cambridge_default_root(capsys):
    payload = _run_cli(capsys, "list-scenes")

    assert payload["scenes"][0]["data_root"].startswith("/mnt/pool/sqy/Cambridge_stdloc/")


def test_manifest_includes_experiment_audit_fields(tmp_path, capsys):
    output = tmp_path / "manifest.json"

    payload = _run_cli(
        capsys,
        "manifest",
        "--scene",
        "ShopFacade",
        "--split",
        "selfmap_train",
        "--checkpoint",
        "output/stdloc_hybrid/ShopFacade/latest.pth",
        "--map",
        "output/stdloc/map_cambridge_spgs/ShopFacade",
        "--data-root",
        "/mnt/pool/sqy/dataset/Cambridge/ShopFacade",
        "--hyperparameters",
        '{"rho": 0.25, "alpha": 0.0}',
        "--feedback-enabled",
        "--rho",
        "0.25",
        "--output",
        str(output),
        "--command",
        "--",
        "python",
        "-m",
        "loc_gs.scripts.eval_stdloc_native",
    )

    assert payload["scene"] == "ShopFacade"
    assert payload["checkpoint_path"] == "output/stdloc_hybrid/ShopFacade/latest.pth"
    assert payload["map_path"] == "output/stdloc/map_cambridge_spgs/ShopFacade"
    assert payload["data_roots"] == ["/mnt/pool/sqy/dataset/Cambridge/ShopFacade"]
    assert payload["hyperparameters"] == {"rho": 0.25, "alpha": 0.0}
    assert payload["feedback_enabled"] is True
    assert payload["command"] == ["python", "-m", "loc_gs.scripts.eval_stdloc_native"]
    assert json.loads(output.read_text(encoding="utf-8"))["hyperparameters"]["rho"] == 0.25
