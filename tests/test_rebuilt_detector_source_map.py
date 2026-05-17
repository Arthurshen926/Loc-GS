import pickle
import subprocess
import sys

import torch

from loc_gs.stdloc_native.clean_source_map import build_clean_detector_source_map, materialize_detector_support_files


def _dump_pickle(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def test_build_clean_detector_source_map_replaces_polluted_detector_with_rebuilt_payload(tmp_path):
    source = tmp_path / "source"
    detector = source / "detector"
    rebuilt = source / "detector_rebuilt_16384"
    detector.mkdir(parents=True)
    rebuilt.mkdir(parents=True)
    (source / "cfg_args").write_text("cfg", encoding="utf-8")
    (source / "point_cloud" / "iteration_30000").mkdir(parents=True)
    (source / "point_cloud" / "iteration_30000" / "point_cloud.ply").write_text("ply", encoding="utf-8")
    _dump_pickle(detector / "sampled_idx.pkl", torch.tensor([9, 8], dtype=torch.long))
    _dump_pickle(detector / "sampled_scores.pkl", {"sampled_scores": torch.tensor([1.5, 1.5])})
    _dump_pickle(rebuilt / "sampled_idx.pkl", torch.tensor([1, 2, 3], dtype=torch.long))
    _dump_pickle(
        rebuilt / "sampled_scores.pkl",
        {
            "sampled_scores": torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
            "score_avg": torch.tensor([0.0, 0.1, 0.2, 0.3], dtype=torch.float32),
        },
    )
    (rebuilt / "30000_detector.pth").write_bytes(b"detector")

    manifest = build_clean_detector_source_map(
        source_map=source,
        rebuilt_detector_dir=rebuilt,
        output_map=tmp_path / "clean" / "Scene",
        scene="Scene",
    )

    output = tmp_path / "clean" / "Scene"
    with (output / "detector" / "sampled_idx.pkl").open("rb") as handle:
        out_idx = pickle.load(handle)
    with (output / "detector" / "sampled_scores.pkl").open("rb") as handle:
        out_scores = pickle.load(handle)
    with (source / "detector" / "sampled_idx.pkl").open("rb") as handle:
        source_idx = pickle.load(handle)

    assert out_idx.tolist() == [1, 2, 3]
    assert torch.allclose(out_scores["sampled_scores"], torch.tensor([0.1, 0.2, 0.3]))
    assert source_idx.tolist() == [9, 8]
    assert not (output / "detector" / "sampled_idx.pkl").is_symlink()
    assert not (output / "detector" / "sampled_scores.pkl").is_symlink()
    assert (output / "cfg_args").is_symlink()
    assert manifest["scene"] == "Scene"
    assert manifest["source_detector_excluded"] is True
    assert manifest["rebuilt_sampled_count"] == 3
    assert manifest["single_path_deployment"] is True


def test_build_clean_detector_source_map_preserves_source_detector_weights_when_rebuilt_payload_is_scores_only(tmp_path):
    source = tmp_path / "source"
    detector = source / "detector"
    rebuilt = tmp_path / "rebuilt_detector"
    detector.mkdir(parents=True)
    rebuilt.mkdir(parents=True)
    (source / "cfg_args").write_text("cfg", encoding="utf-8")
    (detector / "30000_detector.pth").write_bytes(b"native-detector")
    _dump_pickle(detector / "sampled_idx.pkl", torch.tensor([9], dtype=torch.long))
    _dump_pickle(detector / "sampled_scores.pkl", {"sampled_scores": torch.tensor([9.0])})
    _dump_pickle(rebuilt / "sampled_idx.pkl", torch.tensor([1, 2], dtype=torch.long))
    _dump_pickle(rebuilt / "sampled_scores.pkl", {"sampled_scores": torch.tensor([0.1, 0.2])})

    manifest = build_clean_detector_source_map(
        source_map=source,
        rebuilt_detector_dir=rebuilt,
        output_map=tmp_path / "out",
        scene="Scene",
    )

    output = tmp_path / "out"
    assert (output / "detector" / "30000_detector.pth").read_bytes() == b"native-detector"
    with (output / "detector" / "sampled_idx.pkl").open("rb") as handle:
        assert pickle.load(handle).tolist() == [1, 2]
    assert manifest["source_detector_support_files_materialized"] == 1


def test_materialize_detector_support_files_keeps_target_sampling_payload(tmp_path):
    source = tmp_path / "source_detector"
    target = tmp_path / "target_detector"
    source.mkdir()
    target.mkdir()
    (source / "30000_detector.pth").write_bytes(b"detector")
    _dump_pickle(source / "sampled_idx.pkl", torch.tensor([9], dtype=torch.long))
    _dump_pickle(target / "sampled_idx.pkl", torch.tensor([1, 2], dtype=torch.long))
    _dump_pickle(target / "sampled_scores.pkl", {"sampled_scores": torch.tensor([0.1, 0.2])})

    manifest = materialize_detector_support_files(source, target)

    assert (target / "30000_detector.pth").read_bytes() == b"detector"
    with (target / "sampled_idx.pkl").open("rb") as handle:
        assert pickle.load(handle).tolist() == [1, 2]
    assert manifest["source_detector_support_files_materialized"] == 1


def test_build_clean_detector_source_map_cli(tmp_path):
    source = tmp_path / "source"
    rebuilt = source / "detector_rebuilt_16384"
    (source / "detector").mkdir(parents=True)
    rebuilt.mkdir(parents=True)
    (source / "cfg_args").write_text("cfg", encoding="utf-8")
    _dump_pickle(source / "detector" / "sampled_idx.pkl", torch.tensor([9], dtype=torch.long))
    _dump_pickle(rebuilt / "sampled_idx.pkl", torch.tensor([4, 5], dtype=torch.long))
    _dump_pickle(rebuilt / "sampled_scores.pkl", {"sampled_scores": torch.tensor([0.4, 0.5])})
    output = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_clean_detector_source_map",
            "--source_map",
            str(source),
            "--rebuilt_detector_dir",
            str(rebuilt),
            "--output_map",
            str(output),
            "--scene",
            "Scene",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (output / "clean_source_manifest.json").exists()
    assert not (output / "detector" / "sampled_idx.pkl").is_symlink()
