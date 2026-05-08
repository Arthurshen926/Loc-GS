from loc_gs.scripts.launch_dim_matcher_experiments import build_eval_command, select_idle_gpus


def test_select_idle_gpus_filters_memory_and_utilization():
    rows = [
        {"index": 0, "memory_used": 100, "utilization": 0},
        {"index": 1, "memory_used": 12000, "utilization": 0},
        {"index": 2, "memory_used": 100, "utilization": 80},
    ]

    assert select_idle_gpus(rows, max_memory_used_mb=1000, max_utilization=10) == [0]


def test_build_eval_command_sets_gpu_and_lightglue_options():
    cmd, env = build_eval_command(
        gpu_id=3,
        checkpoint="output/stdloc_hybrid/ShopFacade_full_sota/origteacher_e2_nocache/latest.pth",
        output_dir="output/stdloc_hybrid/ShopFacade_dim_lightglue/eval_q5",
        scene="ShopFacade",
        sparse_matcher="lightglue",
        dense_matcher="lightglue_rendered",
        dim_pipeline="superpoint+lightglue",
        max_queries=5,
    )

    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert cmd[:3] == ["python", "-m", "loc_gs.scripts.eval_cambridge_hybrid"]
    assert "--sparse_matcher" in cmd
    assert "lightglue" in cmd
    assert "--dense_matcher" in cmd
    assert "lightglue_rendered" in cmd
    assert "--dim_pipeline" in cmd
    assert "superpoint+lightglue" in cmd
