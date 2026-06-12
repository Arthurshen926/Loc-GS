import json

import pytest

from loc_gs.scripts.eval_sparse_distilled_cambridge import (
    build_argparser,
    build_sparse_distilled_manifest,
)


def test_sparse_distilled_cli_rejects_dense_inference_flag():
    parser = build_argparser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--scene", "GreatCourt", "--enable_dense_inference"])


def test_sparse_distilled_manifest_rejects_test_split():
    with pytest.raises(ValueError, match="test split"):
        build_sparse_distilled_manifest(scene="GreatCourt", split_name="test", command=["cmd"])


def test_sparse_distilled_manifest_records_sparse_only_teacher_boundary():
    manifest = build_sparse_distilled_manifest(
        scene="GreatCourt",
        split_name="train_dev",
        command=["python", "-m", "loc_gs.scripts.eval_sparse_distilled_cambridge"],
        dense_teacher_enabled=False,
    )

    assert manifest["method"] == "sparse_dense_distilled_internal"
    assert manifest["inference_stage"] == "sparse_only"
    assert manifest["dense_teacher_enabled"] is False
    assert manifest["external_runtime_dependency"] == "forbidden"
    json.dumps(manifest)
