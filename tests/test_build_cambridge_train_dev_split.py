import json
from argparse import Namespace
from pathlib import Path

import pytest

from loc_gs.scripts.build_cambridge_train_dev_split import main, split_train_rows


def test_split_train_rows_is_deterministic_and_disjoint():
    rows = [f"seq1/{idx:05d}.png 0 0 0 1 0 0 0" for idx in range(20)]

    selfmap_a, dev_a = split_train_rows(rows, dev_fraction=0.2, min_dev=3, max_dev=0, seed=7)
    selfmap_b, dev_b = split_train_rows(rows, dev_fraction=0.2, min_dev=3, max_dev=0, seed=7)

    assert dev_a == dev_b
    assert selfmap_a == selfmap_b
    assert len(dev_a) == 4
    assert set(selfmap_a).isdisjoint(dev_a)
    assert set(selfmap_a) | set(dev_a) == set(rows)


def test_build_cambridge_train_dev_split_cli_writes_audited_lists(tmp_path):
    source = tmp_path / "ShopFacade"
    source.mkdir()
    train_rows = [f"seq1/train_{idx:05d}.png 0 0 0 1 0 0 0" for idx in range(30)]
    (source / "dataset_train.txt").write_text("\n".join(train_rows) + "\n", encoding="utf-8")
    (source / "dataset_test.txt").write_text("seq2/test_00001.png 0 0 0 1 0 0 0\n", encoding="utf-8")
    output = tmp_path / "split"

    rc = main(
        Namespace(
            scene="ShopFacade",
            source_path=source,
            output_dir=output,
            dev_fraction=0.2,
            min_dev=5,
            max_dev=0,
            seed=3,
        )
    )

    assert rc == 0
    metrics = json.loads((output / "metrics_summary.json").read_text(encoding="utf-8"))
    audit = json.loads((output / "split_audit.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert metrics["train_count"] == 30
    assert metrics["train_dev_count"] == 6
    assert audit["audit_status"] == "passed"
    assert audit["paper_safe_for_tuning"] is True
    assert Path(manifest["selfmap_train_list"]).exists()
    assert Path(manifest["train_dev_list"]).exists()


def test_build_cambridge_train_dev_split_rejects_official_test_overlap(tmp_path):
    source = tmp_path / "ShopFacade"
    source.mkdir()
    train_rows = [f"seq1/train_{idx:05d}.png 0 0 0 1 0 0 0" for idx in range(10)]
    (source / "dataset_train.txt").write_text("\n".join(train_rows) + "\n", encoding="utf-8")
    (source / "dataset_test.txt").write_text("seq1/train_00000.png 0 0 0 1 0 0 0\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="audit failed"):
        main(
            Namespace(
                scene="ShopFacade",
                source_path=source,
                output_dir=tmp_path / "split",
                dev_fraction=0.9,
                min_dev=1,
                max_dev=0,
                seed=0,
            )
        )
