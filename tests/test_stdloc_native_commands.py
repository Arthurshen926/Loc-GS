from pathlib import Path

from loc_gs.stdloc_native.commands import (
    StdlocEvalConfig,
    StdlocTrainConfig,
    build_eval_job,
    build_train_job,
    command_to_shell,
    resolve_scene_images,
)


def test_eval_command_matches_official_cambridge_defaults(tmp_path):
    cfg = StdlocEvalConfig(
        scene="ShopFacade",
        data_root=Path("/data/cambridge"),
        map_root=Path("/maps/stdloc"),
        output_dir=Path("/out/eval_shop"),
        repo_root=tmp_path,
        python_bin="python-test",
    )

    job = build_eval_job(cfg)

    assert job.cwd == tmp_path / "third_party/stdloc"
    assert job.command == [
        "python-test",
        "stdloc.py",
        "-s",
        "/data/cambridge/ShopFacade",
        "-m",
        "/maps/stdloc/ShopFacade",
        "-r",
        "1",
        "-f",
        "sp",
        "-g",
        "3dgs",
        "--images",
        "processed",
        "--data_device",
        "cpu",
        "--cfg",
        "configs/stdloc_cambridge.yaml",
        "--eval_split",
        "test",
        "--output_path",
        "/out/eval_shop",
    ]


def test_eval_command_allows_map_scene_override(tmp_path):
    cfg = StdlocEvalConfig(
        scene="GreatCourt",
        map_scene="GreatCourt_stream_stable2",
        data_root=Path("/data/cambridge"),
        map_root=Path("/maps/stdloc"),
        repo_root=tmp_path,
        python_bin="python-test",
    )

    job = build_eval_job(cfg)

    assert job.command[job.command.index("-s") + 1] == "/data/cambridge/GreatCourt"
    assert job.command[job.command.index("-m") + 1] == "/maps/stdloc/GreatCourt_stream_stable2"


def test_eval_command_resolves_relative_paths_against_repo_root(tmp_path):
    cfg = StdlocEvalConfig(
        scene="ShopFacade",
        data_root=Path("datasets/cambridge"),
        map_root=Path("output/stdloc/map_cambridge_spgs"),
        output_dir=Path("output/stdloc_native/results/ShopFacade"),
        repo_root=tmp_path,
        python_bin="python-test",
    )

    job = build_eval_job(cfg)

    assert job.command[job.command.index("-s") + 1] == str(tmp_path / "datasets/cambridge/ShopFacade")
    assert job.command[job.command.index("-m") + 1] == str(
        tmp_path / "output/stdloc/map_cambridge_spgs/ShopFacade"
    )
    assert job.command[job.command.index("--output_path") + 1] == str(
        tmp_path / "output/stdloc_native/results/ShopFacade"
    )


def test_train_command_matches_official_cambridge_defaults(tmp_path):
    cfg = StdlocTrainConfig(
        scene="KingsCollege",
        data_root=Path("/data/cambridge"),
        map_root=Path("/maps/stdloc"),
        repo_root=tmp_path,
        python_bin="python-test",
        iterations=30000,
        detector_iterations=30000,
    )

    job = build_train_job(cfg)

    assert job.cwd == tmp_path / "third_party/stdloc"
    assert job.command[:15] == [
        "python-test",
        "train.py",
        "-s",
        "/data/cambridge/KingsCollege",
        "-m",
        "/maps/stdloc/KingsCollege",
        "-r",
        "1",
        "-f",
        "sp",
        "-g",
        "3dgs",
        "--iterations",
        "30000",
        "--data_device",
    ]
    assert "--train_detector" in job.command
    assert job.command[job.command.index("--train_detector_iterations") + 1] == "30000"
    assert job.command[job.command.index("--densify_grad_threshold") + 1] == "0.0004"
    assert job.command[job.command.index("--position_lr_init") + 1] == "0.000016"
    assert job.command[job.command.index("--scaling_lr") + 1] == "0.001"


def test_job_gpu_environment_and_shell_rendering(tmp_path):
    cfg = StdlocEvalConfig(
        scene="GreatCourt",
        data_root=Path("/data"),
        map_root=Path("/maps"),
        repo_root=tmp_path,
        python_bin="python-test",
    )

    job = build_eval_job(cfg).with_gpu("2")

    assert job.env["CUDA_VISIBLE_DEVICES"] == "2"
    rendered = command_to_shell(job)
    assert "CUDA_VISIBLE_DEVICES=2" in rendered
    assert "python-test stdloc.py" in rendered


def test_resolve_scene_images_falls_back_when_processed_is_absent(tmp_path):
    data_root = tmp_path / "cambridge"
    (data_root / "ShopFacade" / "processed").mkdir(parents=True)
    (data_root / "KingsCollege").mkdir(parents=True)

    assert resolve_scene_images(data_root, "ShopFacade", "processed") == "processed"
    assert resolve_scene_images(data_root, "KingsCollege", "processed") == "."
    assert resolve_scene_images(data_root / "missing", "GreatCourt", "processed") == "processed"
