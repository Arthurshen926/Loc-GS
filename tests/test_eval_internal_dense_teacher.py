import json
from pathlib import Path

from loc_gs.scripts.eval_internal_dense_teacher import create_dense_teacher_context


def test_create_dense_teacher_context_writes_internal_manifest(tmp_path: Path):
    map_root = tmp_path / "maps"
    data_root = tmp_path / "Cambridge"
    cfg = tmp_path / "dense.yaml"
    cfg.write_text("sparse: {}\ndense: {}\n", encoding="utf-8")
    (map_root / "GreatCourt").mkdir(parents=True)
    (data_root / "GreatCourt").mkdir(parents=True)

    run_dir = create_dense_teacher_context(
        context_root=tmp_path / "ctx",
        scene="GreatCourt",
        split_name="train_dev_seed13_20p",
        map_root=map_root,
        data_root=data_root,
        cfg_path=cfg,
        command=["python", "-m", "loc_gs.scripts.eval_internal_dense_teacher"],
    )

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    split_audit = json.loads((run_dir / "split_audit.json").read_text(encoding="utf-8"))
    assert manifest["method"] == "internal_sparse_dense_teacher"
    assert manifest["scene"] == "GreatCourt"
    assert manifest["split_name"] == "train_dev_seed13_20p"
    assert manifest["map_path"] == str(map_root / "GreatCourt")
    assert manifest["data_roots"] == [str(data_root / "GreatCourt")]
    assert manifest["hyperparameters"]["cfg"] == str(cfg)
    assert split_audit["official_test_used"] is False

