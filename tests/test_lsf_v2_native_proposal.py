import json
import pickle
import subprocess
import sys

import pytest
import torch

from loc_gs.diagnostics.lsf_v2_native_proposal import build_lsf_v2_native_proposal


def _dump_pickle(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def _write_source_map(root):
    _dump_pickle(root / "detector" / "sampled_idx.pkl", torch.tensor([1, 3], dtype=torch.long))
    _dump_pickle(
        root / "detector" / "sampled_scores.pkl",
        {
            "sampled_scores": torch.tensor([0.8, 0.6], dtype=torch.float32),
            "score_avg": torch.tensor([0.1, 0.9, 0.2, 0.4, 0.8, 0.7], dtype=torch.float32),
        },
    )


def _write_support(path, *, split="selfmap_train"):
    torch.save(
        {
            "support_score": torch.tensor([0.0, 0.75, 0.0, 0.65, 0.10, 0.0], dtype=torch.float32),
            "hard_negative_risk": torch.tensor([0.0, 0.10, 0.0, 0.20, 0.0, 0.95], dtype=torch.float32),
            "dense_worsen_risk": torch.tensor([0.0, 0.00, 0.0, 0.10, 0.0, 0.00], dtype=torch.float32),
            "metadata": {
                "schema": "solver_consensus_support_v1",
                "scene": "ToyScene",
                "split": split,
                "source_feedback_bank": "feedback_bank.jsonl",
            },
        },
        path,
    )


def test_lsf_v2_native_proposal_writes_candidate_safe_core_and_manifest(tmp_path):
    source_map = tmp_path / "source"
    support_path = tmp_path / "solver_consensus_support.pt"
    output_dir = tmp_path / "proposal"
    _write_source_map(source_map)
    _write_support(support_path)

    summary = build_lsf_v2_native_proposal(
        source_map=source_map,
        solver_consensus_support_path=support_path,
        output_dir=output_dir,
        candidate_pool_size=2,
        safe_support_threshold=0.5,
        max_hard_negative_risk=0.5,
        max_dense_worsen_risk=0.5,
        min_native_rank=0.0,
    )

    selector = torch.load(output_dir / "selector_lsf_v2_native_proposal.pt", map_location="cpu")
    candidates = torch.load(output_dir / "candidate_pool_lsf_v2_native_top2.pt", map_location="cpu")
    safe_core = torch.load(output_dir / "safe_core_lsf_v2_source.pt", map_location="cpu")
    risk = torch.load(output_dir / "hard_negative_risk_lsf_v2.pt", map_location="cpu")
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert selector.shape == (6,)
    assert candidates.tolist() == [4, 2]
    assert safe_core.tolist() == [1, 3]
    assert risk.tolist() == pytest.approx([0.0, 0.1, 0.0, 0.2, 0.0, 0.95])
    assert summary["candidate_pool_count"] == 2
    assert summary["safe_core_count"] == 2
    assert summary["test_split_used"] is False
    assert manifest["method"] == "lsf_v2_native_proposal"
    assert manifest["branch_selection"] is False
    assert manifest["single_path_deployment"] is True
    assert manifest["score_source"]["kind"] == "detector_score_avg"


def test_lsf_v2_native_proposal_can_rank_normalize_scene_support(tmp_path):
    source_map = tmp_path / "source"
    support_path = tmp_path / "solver_consensus_support.pt"
    output_dir = tmp_path / "proposal"
    _write_source_map(source_map)
    torch.save(
        {
            "support_score": torch.tensor([0.0, 0.03, 0.0, 0.02, 0.04, 0.0], dtype=torch.float32),
            "observed_count": torch.tensor([0, 3, 0, 2, 4, 0], dtype=torch.long),
            "hard_negative_risk": torch.zeros(6, dtype=torch.float32),
            "dense_worsen_risk": torch.zeros(6, dtype=torch.float32),
            "metadata": {
                "schema": "solver_consensus_support_v1",
                "scene": "ToyScene",
                "split": "selfmap_train",
            },
        },
        support_path,
    )

    summary = build_lsf_v2_native_proposal(
        source_map=source_map,
        solver_consensus_support_path=support_path,
        output_dir=output_dir,
        candidate_pool_size=2,
        safe_support_threshold=0.5,
        min_native_rank=0.0,
        support_score_mode="rank_observed",
        native_weight=0.0,
        support_weight=1.0,
    )

    selector = torch.load(output_dir / "selector_lsf_v2_native_proposal.pt", map_location="cpu")
    safe_core = torch.load(output_dir / "safe_core_lsf_v2_source.pt", map_location="cpu")
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert selector.tolist() == pytest.approx([0.0, 0.5, 0.0, 0.0, 1.0, 0.0])
    assert safe_core.tolist() == [1]
    assert summary["support_score_mode"] == "rank_observed"
    assert manifest["hyperparameters"]["support_score_mode"] == "rank_observed"


def test_lsf_v2_native_proposal_uses_canonical_lsf_solver_utility(tmp_path):
    source_map = tmp_path / "source"
    support_path = tmp_path / "solver_consensus_support.pt"
    lsf_path = tmp_path / "localization_support_field.pt"
    output_dir = tmp_path / "proposal"
    _write_source_map(source_map)
    _write_support(support_path)
    torch.save(
        {
            "support_for_selection": torch.tensor([0.0, 0.1, 0.1, 0.1, 0.2, 0.0], dtype=torch.float32),
            "support_score": torch.tensor([0.0, 0.1, 0.1, 0.1, 0.2, 0.0], dtype=torch.float32),
            "hard_negative_risk": torch.zeros(6, dtype=torch.float32),
            "dense_worsen_risk": torch.zeros(6, dtype=torch.float32),
            "solver_set_utility": torch.tensor([0.0, 0.8, 0.1, 0.0, 1.0, 0.0], dtype=torch.float32),
            "pose_information": torch.zeros(6, dtype=torch.float32),
            "ambiguity_risk": torch.zeros(6, dtype=torch.float32),
            "metadata": {
                "schema": "localization_support_field_v1",
                "scene": "ToyScene",
                "split_name": "selfmap_train",
            },
        },
        lsf_path,
    )

    summary = build_lsf_v2_native_proposal(
        source_map=source_map,
        solver_consensus_support_path=support_path,
        localization_support_field_path=lsf_path,
        output_dir=output_dir,
        candidate_pool_size=2,
        min_native_rank=0.0,
        native_weight=0.0,
        support_weight=0.0,
        solver_utility_weight=1.0,
    )

    candidates = torch.load(output_dir / "candidate_pool_lsf_v2_native_top2.pt", map_location="cpu")
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert candidates.tolist()[0] == 4
    assert summary["localization_support_field_path"] == str(lsf_path)
    assert manifest["hyperparameters"]["solver_utility_weight"] == 1.0
    assert manifest["localization_support_field"]["schema"] == "localization_support_field_v1"


def test_lsf_v2_native_proposal_rejects_test_split_support(tmp_path):
    source_map = tmp_path / "source"
    support_path = tmp_path / "solver_consensus_support.pt"
    _write_source_map(source_map)
    _write_support(support_path, split="test")

    with pytest.raises(ValueError, match="test split"):
        build_lsf_v2_native_proposal(
            source_map=source_map,
            solver_consensus_support_path=support_path,
            output_dir=tmp_path / "proposal",
        )


def test_lsf_v2_native_proposal_rejects_mismatched_full_score_avg(tmp_path):
    source_map = tmp_path / "source"
    support_path = tmp_path / "solver_consensus_support.pt"
    _write_source_map(source_map)
    torch.save(
        {
            "support_score": torch.zeros(5, dtype=torch.float32),
            "hard_negative_risk": torch.zeros(5, dtype=torch.float32),
            "dense_worsen_risk": torch.zeros(5, dtype=torch.float32),
            "metadata": {"schema": "solver_consensus_support_v1", "split": "selfmap_train"},
        },
        support_path,
    )

    with pytest.raises(ValueError, match="score_avg length 6 does not match support size 5"):
        build_lsf_v2_native_proposal(
            source_map=source_map,
            solver_consensus_support_path=support_path,
            output_dir=tmp_path / "proposal",
        )


def test_lsf_v2_native_proposal_can_force_point_cloud_score_source(tmp_path):
    source_map = tmp_path / "source"
    support_path = tmp_path / "solver_consensus_support.pt"
    _write_source_map(source_map)
    _write_support(support_path)

    with pytest.raises(FileNotFoundError, match="point_cloud locability_logit"):
        build_lsf_v2_native_proposal(
            source_map=source_map,
            solver_consensus_support_path=support_path,
            output_dir=tmp_path / "proposal",
            score_source="point_cloud",
        )


def test_lsf_v2_native_proposal_cli_writes_artifacts(tmp_path):
    source_map = tmp_path / "source"
    support_path = tmp_path / "solver_consensus_support.pt"
    output_dir = tmp_path / "proposal"
    _write_source_map(source_map)
    _write_support(support_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_lsf_v2_native_proposal",
            "--source_map",
            str(source_map),
            "--solver_consensus_support_path",
            str(support_path),
            "--output_dir",
            str(output_dir),
            "--candidate_pool_size",
            "2",
            "--safe_support_threshold",
            "0.5",
            "--min_native_rank",
            "0.0",
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["candidate_pool_count"] == 2
    assert (output_dir / "selector_lsf_v2_native_proposal.pt").exists()
    assert (output_dir / "candidate_pool_lsf_v2_native_top2.pt").exists()
    assert (output_dir / "summary.json").exists()
