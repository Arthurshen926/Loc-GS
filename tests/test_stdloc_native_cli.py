from pathlib import Path

from loc_gs.stdloc_native.commands import CommandJob
from loc_gs.scripts import eval_stdloc_native
from loc_gs.scripts import launch_stdloc_native_cambridge
from loc_gs.scripts import launch_stdloc_native_lff_cambridge
from loc_gs.scripts import launch_stdloc_native_soft_prior_cambridge
from loc_gs.scripts import train_stdloc_native


def test_eval_stdloc_native_dry_run_prints_command(capsys):
    args = eval_stdloc_native.build_argparser().parse_args(
        [
            "--scene",
            "ShopFacade",
            "--data_root",
            "/data/cambridge",
            "--map_root",
            "/maps/stdloc",
            "--output_dir",
            "/out/eval_shop",
            "--python_bin",
            "python-test",
            "--dry_run",
        ]
    )

    eval_stdloc_native.main(args)

    out = capsys.readouterr().out
    assert "python-test stdloc.py" in out
    assert "--cfg configs/stdloc_cambridge.yaml" in out
    assert "--output_path /out/eval_shop" in out


def test_train_stdloc_native_dry_run_prints_command(capsys):
    args = train_stdloc_native.build_argparser().parse_args(
        [
            "--scene",
            "GreatCourt",
            "--data_root",
            "/data/cambridge",
            "--map_root",
            "/maps/stdloc",
            "--python_bin",
            "python-test",
            "--dry_run",
        ]
    )

    train_stdloc_native.main(args)

    out = capsys.readouterr().out
    assert "python-test train.py" in out
    assert "--train_detector" in out
    assert "--position_lr_init 0.000016" in out


def test_launcher_assigns_scenes_round_robin_to_gpus(tmp_path):
    assignments = launch_stdloc_native_cambridge.assign_scene_gpus(
        ["GreatCourt", "KingsCollege", "OldHospital"],
        ["0", "1"],
    )

    assert assignments == [
        ("GreatCourt", "0"),
        ("KingsCollege", "1"),
        ("OldHospital", "0"),
    ]


def test_launcher_dry_run_uses_requested_phase(capsys, tmp_path):
    args = launch_stdloc_native_cambridge.build_argparser().parse_args(
        [
            "--scenes",
            "ShopFacade",
            "GreatCourt",
            "--gpus",
            "0",
            "1",
            "--phase",
            "eval",
            "--data_root",
            "/data/cambridge",
            "--map_root",
            str(tmp_path / "maps"),
            "--output_root",
            str(tmp_path / "results"),
            "--python_bin",
            "python-test",
            "--dry_run",
        ]
    )

    launch_stdloc_native_cambridge.main(args)

    out = capsys.readouterr().out
    assert "CUDA_VISIBLE_DEVICES=0" in out
    assert "CUDA_VISIBLE_DEVICES=1" in out
    assert "stdloc.py" in out
    assert "train.py" not in out


def test_launcher_dry_run_applies_map_name_overrides(capsys, tmp_path):
    args = launch_stdloc_native_cambridge.build_argparser().parse_args(
        [
            "--scenes",
            "GreatCourt",
            "StMarysChurch",
            "--gpus",
            "0",
            "1",
            "--phase",
            "eval",
            "--data_root",
            "/data/cambridge",
            "--map_root",
            "/maps/stdloc",
            "--map_name_overrides",
            "GreatCourt=GreatCourt_stream_stable2",
            "StMarysChurch=StMarysChurch_stream_fastsave",
            "--dry_run",
        ]
    )

    launch_stdloc_native_cambridge.main(args)

    out = capsys.readouterr().out
    assert "-m /maps/stdloc/GreatCourt_stream_stable2" in out
    assert "-m /maps/stdloc/StMarysChurch_stream_fastsave" in out


def test_launcher_dry_run_auto_falls_back_to_scene_root_images(capsys, tmp_path):
    data_root = tmp_path / "cambridge"
    (data_root / "ShopFacade" / "processed").mkdir(parents=True)
    (data_root / "KingsCollege").mkdir(parents=True)
    args = launch_stdloc_native_cambridge.build_argparser().parse_args(
        [
            "--scenes",
            "ShopFacade",
            "KingsCollege",
            "--gpus",
            "0",
            "--phase",
            "eval",
            "--data_root",
            str(data_root),
            "--map_root",
            str(tmp_path / "maps"),
            "--output_root",
            str(tmp_path / "results"),
            "--dry_run",
        ]
    )

    launch_stdloc_native_cambridge.main(args)

    lines = capsys.readouterr().out.strip().splitlines()
    assert "--images processed" in lines[0]
    assert "--images ." in lines[1]


def test_launcher_reuses_first_free_gpu_for_pending_jobs(monkeypatch, tmp_path):
    launches = []

    class FakeProc:
        def __init__(self, command):
            self.command = command
            self.polls = 0

        def poll(self):
            self.polls += 1
            if self.command[-1] == "long" and self.polls < 3:
                return None
            return 0

    def fake_popen(command, cwd, env, stdout, stderr, text):
        launches.append((command[-1], env.get("CUDA_VISIBLE_DEVICES", "")))
        return FakeProc(command)

    monkeypatch.setattr(launch_stdloc_native_cambridge.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(launch_stdloc_native_cambridge.time, "sleep", lambda _seconds: None)
    jobs = [
        ("eval", "GreatCourt", CommandJob(["python", "long"], tmp_path).with_gpu("0")),
        ("eval", "KingsCollege", CommandJob(["python", "short1"], tmp_path).with_gpu("1")),
        ("eval", "ShopFacade", CommandJob(["python", "short2"], tmp_path).with_gpu("0")),
    ]

    launch_stdloc_native_cambridge._launch_jobs(
        jobs,
        tmp_path / "logs",
        gpus=["0", "1"],
    )

    assert launches == [
        ("long", "0"),
        ("short1", "1"),
        ("short2", "1"),
    ]


def test_soft_prior_launcher_dry_run_uses_per_scene_generated_maps_and_cfgs(capsys, tmp_path):
    args = launch_stdloc_native_soft_prior_cambridge.build_argparser().parse_args(
        [
            "--scenes",
            "ShopFacade",
            "GreatCourt",
            "--gpus",
            "0",
            "1",
            "--source_map_root",
            "/maps/stdloc",
            "--output_map_root",
            str(tmp_path / "soft_maps"),
            "--output_root",
            str(tmp_path / "results"),
            "--map_name_overrides",
            "GreatCourt=GreatCourt_stream_stable2",
            "--calibrated_matchability_template",
            "/calib/{scene}/stdloc_bank_query_like.pt",
            "--selfmap_reliability_template",
            "/selfmap/{scene}/summary.json",
            "--python_bin",
            "python-test",
            "--dry_run",
        ]
    )

    launch_stdloc_native_soft_prior_cambridge.main(args)

    out = capsys.readouterr().out
    assert "-m " + str(tmp_path / "soft_maps/ShopFacade_native_soft_prior") in out
    assert "-m " + str(tmp_path / "soft_maps/GreatCourt_native_soft_prior") in out
    assert "--cfg " + str(tmp_path / "soft_maps/ShopFacade_native_soft_prior/stdloc_soft_prior.yaml") in out
    assert "CUDA_VISIBLE_DEVICES=0" in out
    assert "CUDA_VISIBLE_DEVICES=1" in out


def test_soft_prior_launcher_defaults_to_boosted_r5_tempered_dense_prior():
    args = launch_stdloc_native_soft_prior_cambridge.build_argparser().parse_args(
        [
            "--calibrated_matchability_template",
            "/calib/{scene}.pt",
        ]
    )

    assert args.fusion_mode == "boost"
    assert args.prior_blend == 0.25
    assert args.selfmap_reliability_r5_center == 0.5


def test_lff_launcher_dry_run_uses_generated_maps_checkpoints_and_cfgs(capsys, tmp_path):
    args = launch_stdloc_native_lff_cambridge.build_argparser().parse_args(
        [
            "--scenes",
            "ShopFacade",
            "GreatCourt",
            "--gpus",
            "0",
            "1",
            "--source_map_root",
            "/maps/stdloc",
            "--output_map_root",
            str(tmp_path / "lff_maps"),
            "--output_root",
            str(tmp_path / "results"),
            "--map_name_overrides",
            "GreatCourt=GreatCourt_stream_stable2",
            "--checkpoint_template",
            "/ckpts/{scene}/latest.pth",
            "--calibrated_matchability_template",
            "/calib/{scene}/stdloc_bank_query_like.pt",
            "--selfmap_reliability_template",
            "/selfmap/{scene}/summary.json",
            "--python_bin",
            "python-test",
            "--dry_run",
        ]
    )

    launch_stdloc_native_lff_cambridge.main(args)

    out = capsys.readouterr().out
    assert "-m " + str(tmp_path / "lff_maps/ShopFacade_native_lff") in out
    assert "-m " + str(tmp_path / "lff_maps/GreatCourt_native_lff") in out
    assert "--cfg " + str(tmp_path / "lff_maps/ShopFacade_native_lff/stdloc_lff.yaml") in out
    assert "CUDA_VISIBLE_DEVICES=0" in out
    assert "CUDA_VISIBLE_DEVICES=1" in out


def test_lff_launcher_defaults_to_bounded_descriptor_residual():
    args = launch_stdloc_native_lff_cambridge.build_argparser().parse_args(
        [
            "--checkpoint_template",
            "/ckpts/{scene}.pth",
            "--calibrated_matchability_template",
            "/calib/{scene}.pt",
        ]
    )

    assert args.descriptor_alpha_max == 0.03
    assert args.locability_fusion_mode == "boost"
    assert args.base_sparse_prior_weight == 0.0
    assert args.selfmap_reliability_r5_center == 0.5
    assert args.selector_mode == "reliability_boost"
    assert args.selector_matchability_weight == 1.0
    assert args.selector_locability_weight == 0.0
