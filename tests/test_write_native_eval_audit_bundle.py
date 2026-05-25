import json
import pickle

from loc_gs.scripts import write_native_eval_audit_bundle as audit_bundle_module
from loc_gs.scripts.write_native_eval_audit_bundle import main, write_native_eval_audit_bundle


def test_write_native_eval_audit_bundle_from_stdloc_summary(tmp_path):
    run_dir = tmp_path / "ShopFacade"
    run_dir.mkdir()
    map_dir = tmp_path / "maps" / "ShopFacade"
    detector_dir = map_dir / "detector"
    detector_dir.mkdir(parents=True)
    with (detector_dir / "sampled_idx.pkl").open("wb") as handle:
        pickle.dump([10, 20, 30], handle)
    (map_dir / "payload.bin").write_bytes(b"12345")
    (map_dir / "manifest.json").write_text(
        json.dumps(
            {
                "same_budget": True,
                "branch_selection": False,
                "single_path_deployment": True,
                "solver_aware": {"max_edits": 32, "added_non_native_count": 3},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "model_path": str(map_dir),
                "eval_split": "train",
                "test_stride": 1,
                "max_test_cameras": 20,
                "sparse": {
                    "median_ae": 0.2,
                    "median_te": 3.0,
                    "recall_10cm_5d": 0.9,
                    "recall_5cm_5d": 0.7,
                    "recall_2cm_2d": 0.4,
                    "avg_inliers": 12.0,
                },
                "dense": {
                    "median_ae": 0.1,
                    "median_te": 2.5,
                    "recall_10cm_5d": 0.95,
                    "recall_5cm_5d": 0.8,
                    "recall_2cm_2d": 0.5,
                    "avg_inliers": 120.0,
                },
            }
        ),
        encoding="utf-8",
    )

    paths = write_native_eval_audit_bundle(
        run_dir,
        scene="ShopFacade",
        split="train",
        data_root=str(tmp_path / "missing_cambridge" / "ShopFacade"),
        checkpoint_path="output/stdloc_hybrid/ShopFacade/latest.pth",
        map_path=str(map_dir),
        command=["python", "-m", "loc_gs.scripts.launch_stdloc_native_cambridge"],
        feedback_split="selfmap_train",
        quality_gate_mode="disabled",
        notes="train-dev smoke",
    )

    assert set(paths) >= {
        "manifest",
        "command",
        "metrics_summary",
        "split_audit",
        "git_status",
    }
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    split_audit = json.loads((run_dir / "split_audit.json").read_text(encoding="utf-8"))

    assert manifest["scene"] == "ShopFacade"
    assert manifest["split"] == "train"
    assert manifest["map_path"] == str(map_dir)
    assert manifest["hyperparameters"]["max_test_cameras"] == 20
    assert manifest["feedback"]["selector"] is False
    assert manifest["rho"] is False
    assert manifest["single_path_evaluator"] is True
    assert manifest["sampled_count"] == 3
    assert manifest["landmark_count"] == 3
    assert manifest["map_size_bytes"] > 0
    assert manifest["solver_aware"]["max_edits"] == 32
    assert manifest["same_budget"] is True
    assert manifest["branch_selection"] is False
    assert manifest["single_path_deployment"] is True
    assert manifest["map_same_budget"] is True
    assert manifest["map_branch_selection"] is False
    assert metrics["dense"]["median_te_cm"] == 2.5
    assert metrics["dense"]["median_re_deg"] == 0.1
    assert metrics["dense"]["recall_10cm_5deg"] == 0.95
    assert metrics["sparse"]["recall_5cm_5deg"] == 0.7
    assert metrics["sampled_count"] == 3
    assert metrics["landmark_count"] == 3
    assert metrics["map_size_bytes"] > 0
    assert split_audit["audit_status"] == "unknown"
    assert split_audit["checks"]["feedback_bank_split"]["status"] == "passed"


def test_write_native_eval_audit_bundle_records_actual_copied_cfg(tmp_path):
    run_dir = tmp_path / "ShopFacade"
    run_dir.mkdir()
    (run_dir / "stdloc_cambridge.yaml").write_text(
        "sparse:\n  solver: poselib\n  landmark_path: detector/sampled_idx.pkl\n"
        "dense:\n  solver: poselib\n",
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps({"eval_split": "train", "dense": {"median_ae": 0.1, "median_te": 2.0}}),
        encoding="utf-8",
    )

    write_native_eval_audit_bundle(
        run_dir,
        scene="ShopFacade",
        split="train",
        data_root="",
        checkpoint_path="",
        map_path="",
        command=["python", "-m", "loc_gs.scripts.eval_stdloc_native"],
    )

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["hyperparameters"]["cfg"] == str(run_dir / "stdloc_cambridge.yaml")
    assert manifest["hyperparameters"]["sparse_solver"] == "poselib"
    assert manifest["hyperparameters"]["dense_solver"] == "poselib"
    assert manifest["hyperparameters"]["sparse_landmark_path"] == "detector/sampled_idx.pkl"


def test_write_native_eval_audit_bundle_marks_modified_stdloc_evaluator(tmp_path, monkeypatch):
    run_dir = tmp_path / "ShopFacade"
    run_dir.mkdir()
    (run_dir / "stdloc_cambridge.yaml").write_text(
        "sparse:\n  solver: poselib\n"
        "dense:\n  solver: poselib\n",
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps({"eval_split": "train", "dense": {"median_ae": 0.1, "median_te": 2.0}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(audit_bundle_module, "_third_party_stdloc_evaluator_modified", lambda: True)

    write_native_eval_audit_bundle(
        run_dir,
        scene="ShopFacade",
        split="train",
        data_root="",
        checkpoint_path="",
        map_path="",
        command=["python", "-m", "loc_gs.scripts.eval_stdloc_native"],
    )

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["evaluator_safety"]["third_party_stdloc_evaluator_modified"] is True
    assert manifest["evaluator_safety"]["paper_safe_fixed_poselib_candidate"] is False
    assert manifest["evaluator_safety"]["official_stdloc_parity_candidate"] is False


def test_write_native_eval_audit_bundle_marks_opencv_cfg_as_non_official_parity(
    tmp_path, monkeypatch
):
    run_dir = tmp_path / "ShopFacade"
    run_dir.mkdir()
    (run_dir / "stdloc_cambridge_opencv.yaml").write_text(
        "sparse:\n  solver: opencv_prosac_magsac\n"
        "dense:\n  solver: opencv_prosac_magsac\n",
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps({"eval_split": "train", "dense": {"median_ae": 0.1, "median_te": 2.0}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(audit_bundle_module, "_third_party_stdloc_evaluator_modified", lambda: False)

    write_native_eval_audit_bundle(
        run_dir,
        scene="ShopFacade",
        split="train",
        data_root="",
        checkpoint_path="",
        map_path="",
        command=[
            "python",
            "-m",
            "loc_gs.scripts.eval_stdloc_native",
            "--cfg",
            "configs/stdloc_cambridge_opencv.yaml",
        ],
    )

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["evaluator_safety"]["cfg_is_vendored_stdloc_default"] is False
    assert manifest["evaluator_safety"]["official_stdloc_parity_candidate"] is False
    assert manifest["evaluator_safety"]["paper_safe_fixed_poselib_candidate"] is False
    assert manifest["evaluator_safety"]["uses_opencv_variant"] is True
    assert manifest["evaluator_safety"]["evaluator_variant"] == "opencv_prosac_magsac"


def test_write_native_eval_audit_bundle_marks_poselib_method_cfg_as_paper_safe_not_baseline(
    tmp_path, monkeypatch
):
    run_dir = tmp_path / "ShopFacade"
    run_dir.mkdir()
    (run_dir / "stdloc_spgs_cambridge_dense1.yaml").write_text(
        "sparse:\n"
        "  solver: poselib\n"
        "  landmark_score_path: detector/sampled_scores.pkl\n"
        "  landmark_prior_weight: 0.05\n"
        "dense:\n"
        "  solver: poselib\n"
        "  locability_prior_weight: 0.05\n",
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps({"eval_split": "train", "dense": {"median_ae": 0.1, "median_te": 2.0}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(audit_bundle_module, "_third_party_stdloc_evaluator_modified", lambda: False)

    write_native_eval_audit_bundle(
        run_dir,
        scene="ShopFacade",
        split="train",
        data_root="",
        checkpoint_path="",
        map_path="",
        command=["python", "-m", "loc_gs.scripts.eval_stdloc_native"],
    )

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    safety = manifest["evaluator_safety"]
    assert safety["cfg_is_vendored_stdloc_default"] is False
    assert safety["official_stdloc_parity_candidate"] is False
    assert safety["uses_poselib_single_path"] is True
    assert safety["paper_safe_fixed_poselib_candidate"] is True
    assert safety["evaluator_variant"] == "poselib"


def test_write_native_eval_audit_bundle_reads_soft_prior_map_manifest_split_audit(tmp_path):
    run_dir = tmp_path / "ShopFacade"
    run_dir.mkdir()
    map_dir = tmp_path / "maps" / "ShopFacade_soft"
    (map_dir / "detector").mkdir(parents=True)
    with (map_dir / "detector" / "sampled_idx.pkl").open("wb") as handle:
        pickle.dump([1, 2], handle)
    split_audit_payload = {
        "audit_status": "passed",
        "checks": {
            "feedback_bank_split": {"status": "passed", "split_name": "selfmap_train"},
            "image_id_disjointness": {"status": "passed", "overlap": []},
            "quality_gate": {
                "status": "passed",
                "mode": "disabled",
                "per_query_branch_selection": False,
            },
        },
    }
    (map_dir / "soft_prior_manifest.json").write_text(
        json.dumps({"rho": 0.25, "split_audit": split_audit_payload}),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "model_path": str(map_dir),
                "eval_split": "train",
                "dense": {"median_ae": 0.1, "median_te": 2.0},
            }
        ),
        encoding="utf-8",
    )

    write_native_eval_audit_bundle(
        run_dir,
        scene="ShopFacade",
        split="train",
        data_root="",
        checkpoint_path="",
        map_path=str(map_dir),
        command=["python", "-m", "loc_gs.scripts.eval_stdloc_native"],
    )

    split_audit = json.loads((run_dir / "split_audit.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert split_audit["audit_status"] == "passed"
    assert manifest["map_manifest_type"] == "soft_prior_manifest.json"
    assert manifest["map_rho"] == 0.25


def test_cli_accepts_command_separator(tmp_path, capsys):
    run_dir = tmp_path / "ShopFacade"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "model_path": "output/stdloc/map_cambridge_spgs/ShopFacade",
                "eval_split": "train",
                "max_test_cameras": 1,
                "test_stride": 1,
                "dense": {"median_ae": 0.1, "median_te": 2.0},
            }
        ),
        encoding="utf-8",
    )

    code = main(
        [
            str(run_dir),
            "--scene",
            "ShopFacade",
            "--command",
            "--",
            "python",
            "-m",
            "loc_gs.scripts.launch_stdloc_native_cambridge",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"].endswith("command.txt")
    assert (run_dir / "command.txt").read_text(encoding="utf-8") == (
        "python -m loc_gs.scripts.launch_stdloc_native_cambridge\n"
    )


def test_write_native_eval_audit_bundle_fills_r10_from_results_json(tmp_path):
    run_dir = tmp_path / "OldHospital"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "eval_split": "train",
                "max_test_cameras": 2,
                "test_stride": 1,
                "dense": {"median_ae": 0.1, "median_te": 2.0},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "results.json").write_text(
        json.dumps(
            [
                {"dense_AE": 4.0, "dense_TE": 9.0, "sparse_AE": 1.0, "sparse_TE": 1.0},
                {"dense_AE": 7.0, "dense_TE": 9.0, "sparse_AE": 1.0, "sparse_TE": 20.0},
            ]
        ),
        encoding="utf-8",
    )

    write_native_eval_audit_bundle(
        run_dir,
        scene="OldHospital",
        split="train",
        data_root="",
        checkpoint_path="",
        map_path="",
        command=["python", "-m", "loc_gs.scripts.eval_stdloc_native"],
    )

    metrics = json.loads((run_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    assert metrics["dense"]["recall_10cm_5deg"] == 0.5
    assert metrics["dense"]["recall_5cm_5deg"] == 0.0
    assert metrics["sparse"]["recall_10cm_5deg"] == 0.5


def test_write_native_eval_audit_bundle_uses_upstream_feedback_split_audit(tmp_path):
    run_dir = tmp_path / "ShopFacade"
    run_dir.mkdir()
    map_dir = tmp_path / "maps" / "ShopFacade"
    detector_dir = map_dir / "detector"
    detector_dir.mkdir(parents=True)
    with (detector_dir / "sampled_idx.pkl").open("wb") as handle:
        pickle.dump([10, 20, 30], handle)
    proposal_dir = tmp_path / "proposal" / "ShopFacade"
    proposal_dir.mkdir(parents=True)
    feedback_dir = tmp_path / "feedback" / "ShopFacade"
    feedback_dir.mkdir(parents=True)
    feedback_bank = feedback_dir / "feedback_bank.jsonl"
    feedback_bank.write_text("", encoding="utf-8")
    (feedback_dir / "manifest.json").write_text(
        json.dumps(
            {
                "pair_cache_metadata": {
                    "split_audit": {
                        "audit_status": "passed",
                        "checks": {
                            "feedback_bank_split": {
                                "status": "passed",
                                "split_name": "selfmap_train_rendered",
                            },
                            "image_id_disjointness": {"status": "passed", "overlap": []},
                            "quality_gate": {
                                "status": "passed",
                                "mode": "disabled",
                                "per_query_branch_selection": False,
                            },
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (proposal_dir / "manifest.json").write_text(
        json.dumps({"support_metadata": {"source_feedback_bank": str(feedback_bank)}}),
        encoding="utf-8",
    )
    (map_dir / "manifest.json").write_text(
        json.dumps({"selector_path": str(proposal_dir / "selector.pt")}),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "model_path": str(map_dir),
                "eval_split": "train",
                "max_test_cameras": 1,
                "test_stride": 1,
                "dense": {"median_ae": 0.1, "median_te": 2.0},
            }
        ),
        encoding="utf-8",
    )

    write_native_eval_audit_bundle(
        run_dir,
        scene="ShopFacade",
        split="train",
        data_root="",
        checkpoint_path="",
        map_path=str(map_dir),
        command=["python", "-m", "loc_gs.scripts.eval_stdloc_native"],
        run_role="rejected",
    )

    split_audit = json.loads((run_dir / "split_audit.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert split_audit["audit_status"] == "passed"
    assert split_audit["checks"]["image_id_disjointness"]["status"] == "passed"
    assert manifest["split_audit"]["audit_status"] == "passed"
    assert manifest["run_role"] == "rejected"


def test_write_native_eval_audit_bundle_audits_native_train_eval_split(tmp_path):
    scene_root = tmp_path / "Cambridge" / "ShopFacade"
    scene_root.mkdir(parents=True)
    (scene_root / "dataset_train.txt").write_text(
        "Visual Landmark Dataset V1\nImageFile, Camera Position [X Y Z W P Q R]\n\n"
        "seq2/frame00001.png 0 0 0 1 0 0 0\n",
        encoding="utf-8",
    )
    (scene_root / "dataset_test.txt").write_text(
        "Visual Landmark Dataset V1\nImageFile, Camera Position [X Y Z W P Q R]\n\n"
        "seq1/frame00001.png 0 0 0 1 0 0 0\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "ShopFacade"
    run_dir.mkdir()
    map_dir = tmp_path / "maps" / "ShopFacade"
    detector_dir = map_dir / "detector"
    detector_dir.mkdir(parents=True)
    with (detector_dir / "sampled_idx.pkl").open("wb") as handle:
        pickle.dump([10, 20, 30], handle)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "model_path": str(map_dir),
                "eval_split": "train",
                "max_test_cameras": 1,
                "test_stride": 1,
                "dense": {"median_ae": 0.1, "median_te": 2.0},
            }
        ),
        encoding="utf-8",
    )

    write_native_eval_audit_bundle(
        run_dir,
        scene="ShopFacade",
        split="train",
        data_root=str(scene_root),
        checkpoint_path="",
        map_path=str(map_dir),
        command=["python", "-m", "loc_gs.scripts.eval_stdloc_native"],
    )

    split_audit = json.loads((run_dir / "split_audit.json").read_text(encoding="utf-8"))
    assert split_audit["audit_status"] == "passed"
    assert split_audit["source"] == "dataset_train_test_disjoint_native_eval"
    assert split_audit["checks"]["feedback_bank_split"]["split_name"] == "native_eval_train"


def test_write_native_eval_audit_bundle_does_not_pass_native_test_eval_split(tmp_path):
    scene_root = tmp_path / "Cambridge" / "ShopFacade"
    scene_root.mkdir(parents=True)
    (scene_root / "dataset_train.txt").write_text("seq2/frame00001.png 0 0 0 1 0 0 0\n", encoding="utf-8")
    (scene_root / "dataset_test.txt").write_text("seq1/frame00001.png 0 0 0 1 0 0 0\n", encoding="utf-8")
    run_dir = tmp_path / "ShopFacade"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps({"eval_split": "test", "dense": {"median_ae": 0.1, "median_te": 2.0}}),
        encoding="utf-8",
    )

    write_native_eval_audit_bundle(
        run_dir,
        scene="ShopFacade",
        split="test",
        data_root=str(scene_root),
        checkpoint_path="",
        map_path="",
        command=["python", "-m", "loc_gs.scripts.eval_stdloc_native"],
    )

    split_audit = json.loads((run_dir / "split_audit.json").read_text(encoding="utf-8"))
    assert split_audit["audit_status"] == "unknown"


def test_write_native_eval_audit_bundle_passes_test_eval_when_map_cameras_are_train_only(tmp_path):
    scene_root = tmp_path / "Cambridge" / "ShopFacade"
    scene_root.mkdir(parents=True)
    (scene_root / "dataset_train.txt").write_text("seq2/frame00001.png 0 0 0 1 0 0 0\n", encoding="utf-8")
    (scene_root / "dataset_test.txt").write_text("seq1/frame00001.png 0 0 0 1 0 0 0\n", encoding="utf-8")
    map_dir = tmp_path / "maps" / "ShopFacade"
    (map_dir / "detector").mkdir(parents=True)
    (map_dir / "cameras.json").write_text(
        json.dumps([{"img_name": "seq2/frame00001.png"}]),
        encoding="utf-8",
    )
    run_dir = tmp_path / "ShopFacade"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps({"model_path": str(map_dir), "eval_split": "test", "dense": {"median_ae": 0.1, "median_te": 2.0}}),
        encoding="utf-8",
    )

    write_native_eval_audit_bundle(
        run_dir,
        scene="ShopFacade",
        split="test",
        data_root=str(scene_root),
        checkpoint_path="",
        map_path=str(map_dir),
        command=["python", "-m", "loc_gs.scripts.eval_stdloc_native"],
    )

    split_audit = json.loads((run_dir / "split_audit.json").read_text(encoding="utf-8"))
    assert split_audit["audit_status"] == "passed"
    assert split_audit["source"] == "map_cameras_vs_cambridge_test_split"
    assert split_audit["checks"]["image_id_disjointness"]["status"] == "passed"
    assert split_audit["checks"]["map_camera_train_membership"]["status"] == "passed"


def test_write_native_eval_audit_bundle_fails_test_eval_when_map_cameras_include_test_ids(tmp_path):
    scene_root = tmp_path / "Cambridge" / "ShopFacade"
    scene_root.mkdir(parents=True)
    (scene_root / "dataset_train.txt").write_text("seq2/frame00001.png 0 0 0 1 0 0 0\n", encoding="utf-8")
    (scene_root / "dataset_test.txt").write_text("seq1/frame00001.png 0 0 0 1 0 0 0\n", encoding="utf-8")
    map_dir = tmp_path / "maps" / "ShopFacade"
    (map_dir / "detector").mkdir(parents=True)
    (map_dir / "cameras.json").write_text(
        json.dumps([{"img_name": "seq1/frame00001.png"}, {"img_name": "seq2/frame00001.png"}]),
        encoding="utf-8",
    )
    run_dir = tmp_path / "ShopFacade"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps({"model_path": str(map_dir), "eval_split": "test", "dense": {"median_ae": 0.1, "median_te": 2.0}}),
        encoding="utf-8",
    )

    write_native_eval_audit_bundle(
        run_dir,
        scene="ShopFacade",
        split="test",
        data_root=str(scene_root),
        checkpoint_path="",
        map_path=str(map_dir),
        command=["python", "-m", "loc_gs.scripts.eval_stdloc_native"],
    )

    split_audit = json.loads((run_dir / "split_audit.json").read_text(encoding="utf-8"))
    assert split_audit["audit_status"] == "failed"
    assert split_audit["source"] == "map_cameras_vs_cambridge_test_split"
    assert split_audit["checks"]["image_id_disjointness"]["status"] == "failed"
    assert split_audit["checks"]["map_camera_train_membership"]["status"] == "failed"
    assert split_audit["checks"]["image_id_disjointness"]["overlap"] == ["seq1/frame00001.png"]


def test_write_native_eval_audit_bundle_combines_upstream_audit_with_map_camera_audit(tmp_path):
    scene_root = tmp_path / "Cambridge" / "ShopFacade"
    scene_root.mkdir(parents=True)
    (scene_root / "dataset_train.txt").write_text("seq2/frame00001.png 0 0 0 1 0 0 0\n", encoding="utf-8")
    (scene_root / "dataset_test.txt").write_text("seq1/frame00001.png 0 0 0 1 0 0 0\n", encoding="utf-8")
    map_dir = tmp_path / "maps" / "ShopFacade_lsf"
    (map_dir / "detector").mkdir(parents=True)
    (map_dir / "cameras.json").write_text(json.dumps([{"img_name": "seq1/frame00001.png"}]), encoding="utf-8")
    (map_dir / "manifest.json").write_text(
        json.dumps(
            {
                "split_audit": {
                    "audit_status": "passed",
                    "checks": {"upstream_split_audit": {"status": "passed", "split_name": "selfmap_train_rendered"}},
                },
                "branch_selection": False,
            }
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "ShopFacade"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps({"model_path": str(map_dir), "eval_split": "test", "dense": {"median_ae": 0.1, "median_te": 2.0}}),
        encoding="utf-8",
    )

    write_native_eval_audit_bundle(
        run_dir,
        scene="ShopFacade",
        split="test",
        data_root=str(scene_root),
        checkpoint_path="",
        map_path=str(map_dir),
        command=["python", "-m", "loc_gs.scripts.eval_stdloc_native"],
    )

    split_audit = json.loads((run_dir / "split_audit.json").read_text(encoding="utf-8"))
    assert split_audit["audit_status"] == "failed"
    assert split_audit["source"] == "upstream_plus_map_cameras_vs_cambridge_test_split"
    assert split_audit["checks"]["upstream_split_audit"]["status"] == "passed"
    assert split_audit["checks"]["image_id_disjointness"]["status"] == "failed"
