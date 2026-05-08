#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import time
from pathlib import Path


def query_gpus() -> list[dict[str, int]]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception:
        return []
    rows = []
    for row in csv.reader(out.splitlines()):
        if len(row) < 3:
            continue
        rows.append(
            {
                "index": int(row[0].strip()),
                "memory_used": int(row[1].strip()),
                "utilization": int(row[2].strip()),
            }
        )
    return rows


def select_idle_gpus(
    rows: list[dict[str, int]],
    max_memory_used_mb: int = 1000,
    max_utilization: int = 10,
) -> list[int]:
    return [
        int(row["index"])
        for row in rows
        if int(row["memory_used"]) <= int(max_memory_used_mb)
        and int(row["utilization"]) <= int(max_utilization)
    ]


def build_eval_command(
    gpu_id: int,
    checkpoint: str,
    output_dir: str,
    scene: str,
    sparse_matcher: str,
    dense_matcher: str,
    dim_pipeline: str,
    max_queries: int = 0,
) -> tuple[list[str], dict[str, str]]:
    cmd = [
        "python",
        "-m",
        "loc_gs.scripts.eval_cambridge_hybrid",
        "--checkpoint",
        checkpoint,
        "--scene",
        scene,
        "--output_dir",
        output_dir,
        "--eval_pose_source",
        "cameras_json",
        "--query_detector",
        "stdloc",
        "--query_feature_source",
        "original",
        "--landmark_source",
        "stdloc_detector",
        "--descriptor_source",
        "hybrid_ply_blend",
        "--ply_loc_feature_weight",
        "0.9",
        "--sparse_matcher",
        sparse_matcher,
        "--dense_matcher",
        dense_matcher,
        "--dim_pipeline",
        dim_pipeline,
        "--rendered_keypoint_source",
        "locability",
        "--dense_iters",
        "2",
        "--dense_full_render",
        "--subpixel_refine",
        "--mnn",
        "--solver",
        "poselib",
        "--poselib_refine",
        "--sparse_reprojection_error",
        "12",
        "--dense_reprojection_error",
        "8",
        "--pnp_confidence",
        "0.99999",
        "--pnp_min_iterations",
        "1000",
    ]
    if max_queries > 0:
        cmd.extend(["--max_queries", str(int(max_queries))])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(int(gpu_id))
    return cmd, env


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch DIM/LightGlue matcher ablations on idle GPUs.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scene", default="ShopFacade")
    parser.add_argument("--output_root", default="output/stdloc_hybrid")
    parser.add_argument("--max_queries", type=int, default=5)
    parser.add_argument("--max_memory_used_mb", type=int, default=1000)
    parser.add_argument("--max_utilization", type=int, default=10)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    experiments = [
        ("stdloc_parity", "stdloc_parity", "superpoint+lightglue", "parity"),
        ("lightglue", "lightglue_rendered", "superpoint+lightglue", "lightglue"),
        ("dim", "lightglue_rendered", "aliked+lightglue", "aliked_lightglue"),
    ]
    idle = select_idle_gpus(
        query_gpus(),
        max_memory_used_mb=args.max_memory_used_mb,
        max_utilization=args.max_utilization,
    )
    if not idle:
        idle = [0]
    procs = []
    for idx, (sparse, dense, pipeline, name) in enumerate(experiments[: len(idle)]):
        out_dir = Path(args.output_root) / f"{args.scene}_dim_{name}" / f"eval_q{args.max_queries or 'full'}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd, env = build_eval_command(
            gpu_id=idle[idx],
            checkpoint=args.checkpoint,
            output_dir=str(out_dir),
            scene=args.scene,
            sparse_matcher=sparse,
            dense_matcher=dense,
            dim_pipeline=pipeline,
            max_queries=args.max_queries,
        )
        print(" ".join(cmd), f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
        if not args.dry_run:
            log_path = out_dir / "launcher.log"
            log = log_path.open("w", encoding="utf-8")
            procs.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT))
    while procs:
        procs = [proc for proc in procs if proc.poll() is None]
        time.sleep(5)


if __name__ == "__main__":
    main()
