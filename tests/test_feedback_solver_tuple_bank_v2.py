import json
import pickle
import subprocess
import sys

import torch

from loc_gs.diagnostics.feedback_solver_tuple_bank import compare_feedback_solver_tuple_banks
from loc_gs.feedback.io import save_feedback_bank


def _write_v2_bank(path):
    records = []
    xy = [(0.0, 0.0), (0.0, 10.0), (10.0, 0.0), (10.0, 10.0)]
    for kp, point in enumerate(xy):
        for gid in (kp, kp + 4):
            records.append(
                {
                    "scene": "ToyScene",
                    "query_id": f"seq1/frame00001.png::kp_{kp:06d}",
                    "image_id": "seq1/frame00001.png",
                    "keypoint_id": f"kp_{kp:06d}",
                    "keypoint_xy": [point[1], point[0]],
                    "matched_landmark_id": str(gid),
                    "matched_gaussian_id": str(gid),
                    "descriptor_score": 0.9,
                    "match_rank": 1,
                    "pnp_inlier": True,
                    "reprojection_error_px": 1.0,
                    "dense_transition": "improved",
                }
            )
    save_feedback_bank(
        path,
        records,
        {
            "scene": "ToyScene",
            "split_name": "selfmap_train",
            "schema_version": "feedback_bank_v2",
            "query_id_source": "image_id",
            "split_audit": {"audit_status": "passed"},
        },
    )


def test_feedback_solver_tuple_bank_groups_by_real_image_id_and_compares_selection(tmp_path):
    bank_path = tmp_path / "feedback_bank.jsonl"
    _write_v2_bank(bank_path)
    xyz = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 3.0],
            [1.0, 1.0, 5.0],
            [0.0, 0.0, 1.0],
            [0.01, 0.0, 1.0],
            [0.0, 0.01, 1.0],
            [0.01, 0.01, 1.0],
        ],
        dtype=torch.float32,
    )

    report = compare_feedback_solver_tuple_banks(
        bank_path,
        source_idx=torch.tensor([0, 1, 2, 3]),
        candidate_idx=torch.tensor([4, 5, 6, 7]),
        xyz=xyz,
        tuple_size=4,
        max_tuples_per_query=32,
        min_logdet_h=-5.0,
        min_spread_3d=0.05,
    )

    assert report["source"]["summary"]["query_group_mode"] == "feedback_bank_v2:image_id"
    assert report["source"]["summary"]["query_count"] == 1
    assert report["source"]["queries"][0]["query_id"] == "seq1/frame00001.png"
    assert report["summary_delta"]["support_count_sum"] == 0
    assert report["summary_delta"]["mean_logdet_H"] < 0
    assert report["candidate"]["summary"]["viable_tuple_count"] == 0
    assert report["source"]["summary"]["viable_tuple_count"] == 1


def test_feedback_solver_tuple_bank_canonicalizes_v2_query_id_to_image_group(tmp_path):
    bank_path = tmp_path / "feedback_bank.jsonl"
    _write_v2_bank(bank_path)

    report = compare_feedback_solver_tuple_banks(
        bank_path,
        source_idx=torch.tensor([0, 1, 2, 3]),
        candidate_idx=torch.tensor([4, 5, 6, 7]),
        group_field="query_id",
        tuple_size=4,
        max_tuples_per_query=32,
    )

    assert report["source"]["summary"]["query_group_mode"] == "feedback_bank_v2:query_id"
    assert report["source"]["summary"]["query_count"] == 1
    assert report["source"]["queries"][0]["query_id"] == "seq1/frame00001.png"
    assert report["source"]["summary"]["viable_tuple_count"] == 1


def test_build_solver_tuple_bank_v2_cli_writes_real_group_report(tmp_path):
    bank_path = tmp_path / "feedback_bank.jsonl"
    _write_v2_bank(bank_path)
    source_path = tmp_path / "source.pkl"
    candidate_path = tmp_path / "candidate.pkl"
    xyz_path = tmp_path / "xyz.pt"
    output_json = tmp_path / "solver_tuple_v2.json"
    output_md = tmp_path / "solver_tuple_v2.md"
    with source_path.open("wb") as handle:
        pickle.dump(torch.tensor([0, 1, 2, 3]), handle)
    with candidate_path.open("wb") as handle:
        pickle.dump(torch.tensor([4, 5, 6, 7]), handle)
    torch.save(
        torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 2.0],
                [0.0, 1.0, 3.0],
                [1.0, 1.0, 5.0],
                [0.0, 0.0, 1.0],
                [0.01, 0.0, 1.0],
                [0.0, 0.01, 1.0],
                [0.01, 0.01, 1.0],
            ],
            dtype=torch.float32,
        ),
        xyz_path,
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_solver_tuple_bank_v2",
            "--feedback_bank",
            str(bank_path),
            "--source_idx",
            str(source_path),
            "--candidate_idx",
            str(candidate_path),
            "--xyz_path",
            str(xyz_path),
            "--output_json",
            str(output_json),
            "--output_report",
            str(output_md),
            "--min_logdet_h",
            "-5",
            "--min_spread_3d",
            "0.05",
        ],
        check=True,
    )

    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["source"]["summary"]["query_group_mode"] == "feedback_bank_v2:image_id"
    assert report["summary_delta"]["mean_logdet_H"] < 0
    assert "seq1/frame00001.png" in output_md.read_text(encoding="utf-8")
