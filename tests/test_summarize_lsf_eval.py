import json
import subprocess
import sys


def test_summarize_lsf_eval_cli_writes_protocol_report(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "results.json").write_text(
        json.dumps(
            [
                {
                    "image_name": "a.png",
                    "sparse_TE": 20.0,
                    "sparse_AE": 2.0,
                    "dense_TE": 4.0,
                    "dense_AE": 1.0,
                    "sparse": {"inliers": 10},
                    "dense": [{"inliers": 40}],
                },
                {
                    "image_name": "b.png",
                    "sparse_TE": 80.0,
                    "sparse_AE": 3.0,
                    "dense_TE": 16.0,
                    "dense_AE": 4.0,
                    "sparse": {"inliers": 8},
                    "dense": [{"inliers": 35}],
                },
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "timing_profile.json").write_text(
        json.dumps({"latency_ms": {"total": {"mean": 100.0}}, "fps": {"mean_latency": 10.0}}),
        encoding="utf-8",
    )
    output_path = tmp_path / "lsf_report.json"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.summarize_lsf_eval",
            "--run_dir",
            str(run_dir),
            "--scene",
            "ToyScene",
            "--method",
            "toy_lsf",
            "--landmark_count",
            "12288",
            "--output_json",
            str(output_path),
        ],
        check=True,
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["scene"] == "ToyScene"
    assert report["pose"]["dense"]["recall_15cm_5deg"] == 0.5
    assert report["dense_transition"]["improved_count"] == 2
    assert report["online_timing"]["latency_ms"]["total"]["mean"] == 100.0

